// Kernels of the image-generation transformer (src/diffusion). Appended to
// hybrid.metal, so its headers, `block_sum` and the quantized GEMMs are in
// scope. The residual stream and the projections' outputs are F32; whatever a
// matrix product reads (layernorm, attention and SwiGLU outputs) and Q, K and V
// are F16.

// out = silu(x)
kernel void i_silu(constant EltArgs & a [[buffer(2)]], device const float * x [[buffer(0)]],
                   device float * y [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i < a.n) y[i] = x[i] / (1.f + exp(-x[i]));
}

// out = silu(gate) * up, as F16 for the matrix product that reads it
kernel void i_swiglu(constant EltArgs & a [[buffer(3)]], device const float * g [[buffer(0)]],
                     device const float * u [[buffer(1)]], device half * y [[buffer(2)]],
                     uint i [[thread_position_in_grid]]) {
    if (i < a.n) y[i] = half(g[i] / (1.f + exp(-g[i])) * u[i]);
}

// out = gelu(x), tanh approximation
kernel void i_gelu_tanh(constant EltArgs & a [[buffer(2)]], device const float * x [[buffer(0)]],
                        device float * y [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i < a.n) {
        const float v = x[i];
        y[i] = 0.5f * v * (1.f + tanh(0.7978845608028654f * (v + 0.044715f * v * v * v)));
    }
}

struct LnArgs { uint n; float eps; float offset; };

// y = layernorm(x) * (offset + scale), no affine parameters, one threadgroup
// (256 threads) per row. `scale` is one row of the shared modulation.
kernel void i_ln_mod(constant LnArgs & a [[buffer(3)]], device const float * x [[buffer(0)]],
                     device const float * scale [[buffer(1)]], device half * y [[buffer(2)]],
                     uint row [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float scratch[8];
    device const float * xr = x + (ulong)row * a.n;
    float s = 0.f;
    for (uint i = tid; i < a.n; i += 256) s += xr[i];
    const float mean = block_sum(s, scratch, tid, 256) / float(a.n);
    float ss = 0.f;
    for (uint i = tid; i < a.n; i += 256) {
        const float d = xr[i] - mean;
        ss += d * d;
    }
    const float inv = rsqrt(block_sum(ss, scratch, tid, 256) / float(a.n) + a.eps);
    device half * yr = y + (ulong)row * a.n;
    for (uint i = tid; i < a.n; i += 256) yr[i] = half((xr[i] - mean) * inv * (a.offset + scale[i]));
}

struct GateArgs { uint n; uint cols; };

// x += tanh(gate) * y over rows of `cols` values; `gate` is one row of the
// modulation.
kernel void i_gate_add(constant GateArgs & a [[buffer(3)]], device const float * y [[buffer(0)]],
                       device const float * gate [[buffer(1)]], device float * x [[buffer(2)]],
                       uint i [[thread_position_in_grid]]) {
    if (i < a.n) x[i] += tanh(gate[i % a.cols]) * y[i];
}

struct EulerArgs { uint n; float dt; };

// x += dt * v
kernel void i_euler(constant EulerArgs & a [[buffer(2)]], device const float * v [[buffer(0)]],
                    device float * x [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i < a.n) x[i] += a.dt * v[i];
}

struct PrepArgs { uint rows; uint heads; uint pairs; float eps; float q_scale; };

// Per-head RMSNorm of Q and K, rotary embedding on both (consecutive (even,
// odd) pairs, as complex numbers), Q pre-scaled by 1/sqrt(head width), and the
// three cast to F16. One threadgroup of 128 threads per (row, head); thread d
// owns element d, so a rotary pair is two neighbouring lanes.
kernel void i_qkv_prep(constant PrepArgs & a [[buffer(10)]], device const float * q [[buffer(0)]],
                       device const float * k [[buffer(1)]], device const float * v [[buffer(2)]],
                       device const float * qw [[buffer(3)]], device const float * kw [[buffer(4)]],
                       device const float * cs [[buffer(5)]], device const float * sn [[buffer(6)]],
                       device half * qo [[buffer(7)]], device half * ko [[buffer(8)]],
                       device half * vo [[buffer(9)]],
                       uint2 g [[threadgroup_position_in_grid]], ushort d [[thread_index_in_threadgroup]]) {
    threadgroup float scratch[4];
    const ulong at = ((ulong)g.x * a.heads + g.y) * 128 + d;
    const float qv = q[at], kv = k[at];
    const float qs = block_sum(qv * qv, scratch, d, 128);
    const float ks = block_sum(kv * kv, scratch, d, 128);
    const float qn = qv * rsqrt(qs / 128.f + a.eps) * qw[d];
    const float kn = kv * rsqrt(ks / 128.f + a.eps) * kw[d];
    const float qp = simd_shuffle_xor(qn, 1), kp = simd_shuffle_xor(kn, 1);
    const uint pair = d / 2;
    const float c = cs[(ulong)g.x * a.pairs + pair], s = sn[(ulong)g.x * a.pairs + pair];
    const bool even = (d & 1) == 0;
    const float qr = even ? qn * c - qp * s : qn * c + qp * s;
    const float kr = even ? kn * c - kp * s : kn * c + kp * s;
    qo[at] = half(qr * a.q_scale);
    ko[at] = half(kr);
    vo[at] = half(v[at]);
}

struct DitAttnArgs { uint tokens; uint heads; uint prefix; uint keys; uint causal; };

// Flash attention for the image transformer (h_flash_attn_mpp_staged_d128
// without gates, sinks, windows or a ring): 64 query rows of one head per
// threadgroup, keys in blocks of 64. Keys [0, prefix) come from the cached
// text K/V, the rest from the current rows' own K/V; each is a segment of its
// own, so a block never straddles the two and K and V are read straight from
// device memory by the matrix units. `causal` masks key > row (the text
// prefix); the image rows attend to every key. K, V and Q rows are
// `heads * 128` wide (one K/V head per query head). Q is F16 and pre-scaled;
// the output is F16.
// grid (ceil(tokens/64), heads); threads 128; 64*64*4 + 64*64*2 + 3*64*4 bytes.
kernel void i_attn(constant DitAttnArgs & p [[buffer(6)]], device half * q [[buffer(0)]],
                   device const half * kp [[buffer(1)]], device const half * vp [[buffer(2)]],
                   device const half * kc [[buffer(3)]], device const half * vc [[buffer(4)]],
                   device half * out [[buffer(5)]], threadgroup char * shared [[threadgroup(0)]],
                   uint2 tgpig [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int D = 128, BR = 64, BC = 64;
    threadgroup uint rescale;
    threadgroup float * Ss = (threadgroup float *)shared;                 // [BR][BC]
    threadgroup half  * Ps = (threadgroup half *)(Ss + BR * BC);          // [BR][BC]
    threadgroup float * corr = (threadgroup float *)(Ps + BR * BC);       // [BR]
    threadgroup float * mrow = corr + BR;                                 // [BR]
    threadgroup float * lrow = mrow + BR;                                 // [BR]
    const uint h = tgpig.y;
    const int t0 = int(tgpig.x) * BR;
    const int rows = min(BR, int(p.tokens) - t0);
    const int stride = int(p.heads) * D;
    const ulong head_off = (ulong)h * D;
    const uint last_key = p.causal != 0 ? uint(t0 + rows - 1) : p.keys - 1;

    auto tQ = tensor(q + (ulong)t0 * stride + h * D, dextents<int32_t, 2>(D, rows), array<int, 2>({1, stride}));
    auto tS = tensor(Ss, dextents<int32_t, 2>(BC, BR));
    auto tP = tensor(Ps, dextents<int32_t, 2>(BC, BR));
    auto tKp = tensor((device half *)kp + head_off, dextents<int32_t, 2>(D, int(p.prefix)), array<int, 2>({1, stride}));
    auto tVp = tensor((device half *)vp + head_off, dextents<int32_t, 2>(D, int(p.prefix)), array<int, 2>({1, stride}));
    auto tKc = tensor((device half *)kc + head_off, dextents<int32_t, 2>(D, int(p.keys - p.prefix)), array<int, 2>({1, stride}));
    auto tVc = tensor((device half *)vc + head_off, dextents<int32_t, 2>(D, int(p.keys - p.prefix)), array<int, 2>({1, stride}));

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(BR, BC, D, false, true, true,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm_s;
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(BR, D, BC, false, false, true,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm_o;
    auto cS = mm_s.template get_destination_cooperative_tensor<decltype(tQ), decltype(tKp), float>();
    auto cO = mm_o.template get_destination_cooperative_tensor<decltype(tP), decltype(tVp), float>();
    for (uint i = 0; i < cO.get_capacity(); i++) cO[i] = 0.f;
    if (tid < BR) { mrow[tid] = -INFINITY; lrow[tid] = 0.f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int my_row = tid / 2;
    const int c0 = (tid % 2) * 32;
    for (int segment = 0; segment < 2; segment++) {
        const uint first = segment == 0 ? 0 : p.prefix;
        const uint length = segment == 0 ? p.prefix : p.keys - p.prefix;
        for (uint j0 = 0; j0 < length && first + j0 <= last_key; j0 += BC) {
            for (uint i = 0; i < cS.get_capacity(); i++) cS[i] = 0.f;
            auto mK = segment == 0 ? tKp.slice(0, j0) : tKc.slice(0, j0);
            mm_s.run(tQ, mK, cS);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            cS.store(tS);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float s[32];
            float mx = -INFINITY;
            for (int c = 0; c < 32; c++) {
                const uint at = j0 + uint(c0 + c);
                const uint key = first + at;
                const bool ok = my_row < rows && at < length && (p.causal == 0 || key <= uint(t0 + my_row));
                s[c] = ok ? Ss[my_row * BC + c0 + c] : -INFINITY;
                mx = max(mx, s[c]);
            }
            mx = max(mx, simd_shuffle_xor(mx, 1));
            const float m_old = mrow[my_row];
            const float m_new = max(m_old, mx);
            const float cr = m_new == -INFINITY ? 1.f : exp(m_old - m_new);
            float sum = 0.f;
            for (int c = 0; c < 32; c++) {
                const float e = m_new == -INFINITY ? 0.f : exp(s[c] - m_new);
                sum += e;
                Ps[my_row * BC + c0 + c] = half(e);
            }
            sum += simd_shuffle_xor(sum, 1);
            if (tid == 0) rescale = 0;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid % 2 == 0) {
                mrow[my_row] = m_new;
                lrow[my_row] = lrow[my_row] * cr + sum;
                corr[my_row] = cr;
                if (cr != 1.f) rescale = 1;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (rescale) {
                for (uint i = 0; i < cO.get_capacity(); i++) {
                    if (cO.is_valid_element(i)) {
                        auto idx = cO.get_multidimensional_index(i);
                        cO[i] *= corr[idx[1]];
                    }
                }
            }
            auto mV = segment == 0 ? tVp.slice(0, j0) : tVc.slice(0, j0);
            mm_o.run(tP, mV, cO);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    for (uint i = 0; i < cO.get_capacity(); i++) {
        if (!cO.is_valid_element(i)) continue;
        auto idx = cO.get_multidimensional_index(i);
        const int r = idx[1], dcol = idx[0];
        if (r >= rows) continue;
        const float l = lrow[r];
        out[((ulong)(t0 + r) * p.heads + h) * D + dcol] = half(l > 0.f ? cO[i] / l : 0.f);
    }
}

// ------------------------------------------------------------------ GEMM

// The hybrid engine's TensorOps GEMM (`mm_impl`) with F16 activations: the
// matrix product then runs half by half, which is the rate the matrix units
// are built for, and the activations are half the bytes. Accumulation and
// output stay F32. Grid (ceil(m/128), ceil(n/64)); threads 128; 64*NK*2 bytes
// of threadgroup memory.
template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread half4x4 &), int NK>
void mmh_impl(constant ProjArgs & a, device const char * w, device const half * x, device float * y,
              threadgroup half * sa, uint3 tgpig, ushort tiitg) {
    const int K = a.k;
    const int M = a.n;
    const int N = a.m;
    const int ra = tgpig.y * MM_NRA;
    const int rb = tgpig.x * MM_NRB;
    constexpr int CHUNKS = NK / 16;
    auto tA = tensor(sa, dextents<int32_t, 2>(NK, MM_NRA));
    auto tB = tensor((device half *)x, dextents<int32_t, 2>(K, N), array<int, 2>({1, int(a.x_stride)}));
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            MM_NRB, MM_NRA, NK, false, true, true,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();
    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        for (int work = tiitg; work < MM_NRA * CHUNKS; work += 128) {
            const int row = work / CHUNKS;
            const int k_chunk = work % CHUNKS;
            const int k_pos = loop_k + k_chunk * 16;
            if (ra + row < M) {
                const int block_idx = k_pos / (16 * nl);
                const short il = (k_pos / 16) % nl;
                device const block_q * row_ptr = (device const block_q *)(w + (ulong)a.row_bytes * (ra + row));
                half4x4 temp_a;
                dequantize_func(row_ptr + block_idx, il, temp_a);
                FOR_UNROLL (short i = 0; i < 16; i++) sa[row * NK + k_chunk*16 + i] = temp_a[i/4][i%4];
            } else {
                FOR_UNROLL (short i = 0; i < 16; i++) sa[row * NK + k_chunk*16 + i] = 0.h;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mA = tA.slice(0, 0);
        auto mB = tB.slice(loop_k, rb);
        mm.run(mB, mA, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto tD = tensor(y, dextents<int32_t, 2>(M, N), array<int, 2>({1, int(a.y_stride)}));
    cT.store(tD.slice(ra, rb));
}

constant constexpr int IMM_NK = 64;  // K per step
#define IMM_KERNEL(NAME, BLOCK, NL, DEQ) \
kernel void NAME(constant ProjArgs & a [[buffer(3)]], device const char * w [[buffer(0)]], \
                 device const half * x [[buffer(1)]], device float * y [[buffer(2)]], \
                 threadgroup half * sa [[threadgroup(0)]], \
                 uint3 tgpig [[threadgroup_position_in_grid]], ushort tiitg [[thread_index_in_threadgroup]]) { \
    mmh_impl<BLOCK, NL, DEQ<half4x4>, IMM_NK>(a, w, x, y, sa, tgpig, tiitg); \
}
IMM_KERNEL(i_mm_bf16, block_bf16, 1, dequantize_bf16)
IMM_KERNEL(i_mm_q8_0, block_q8_0, 2, dequantize_q8_0)
IMM_KERNEL(i_mm_q4_k, block_q4_K, 16, dequantize_q4_K)
IMM_KERNEL(i_mm_q5_k, block_q5_K, 16, dequantize_q5_K)
IMM_KERNEL(i_mm_q6_k, block_q6_K, 16, dequantize_q6_K)
IMM_KERNEL(i_mm_q4_0, block_q4_0, 2, dequantize_q4_0)
