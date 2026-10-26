// Kernels of the image-generation transformer (src/diffusion). Appended to
// hybrid.metal, so its headers, `block_sum` and the quantized GEMMs are in
// scope. Activations are F32 except Q, K and V, which attention reads as F16.

// out = silu(x)
kernel void i_silu(constant EltArgs & a [[buffer(2)]], device const float * x [[buffer(0)]],
                   device float * y [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i < a.n) y[i] = x[i] / (1.f + exp(-x[i]));
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
                     device const float * scale [[buffer(1)]], device float * y [[buffer(2)]],
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
    device float * yr = y + (ulong)row * a.n;
    for (uint i = tid; i < a.n; i += 256) yr[i] = (xr[i] - mean) * inv * (a.offset + scale[i]);
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
// text K/V, the rest from the current rows' own K/V. `causal` masks key > row
// (the text prefix); the image rows attend to every key. K, V and Q rows are
// `heads * 128` wide (one K/V head per query head). Q is F16 and pre-scaled.
// grid (ceil(tokens/64), heads); threads 128; 64*64*4 + 64*64*2 + 3*64*4 bytes.
inline void i_stage_kv(threadgroup half * Hs, device const half * pre, device const half * cur,
                       uint prefix, uint j0, uint total, uint stride, ulong head_off, uint tid) {
    for (uint i = tid; i < 64 * 16; i += 128) {
        const uint kr = i / 16, dd = (i % 16) * 8;
        const uint key = j0 + kr;
        threadgroup half4 * dst = (threadgroup half4 *)(Hs + kr * 128 + dd);
        if (key >= total) {
            dst[0] = 0.h;
            dst[1] = 0.h;
            continue;
        }
        device const half * row = key < prefix ? pre + (ulong)key * stride : cur + (ulong)(key - prefix) * stride;
        device const half4 * src = (device const half4 *)(row + head_off + dd);
        dst[0] = src[0];
        dst[1] = src[1];
    }
}

kernel void i_attn(constant DitAttnArgs & p [[buffer(6)]], device half * q [[buffer(0)]],
                   device const half * kp [[buffer(1)]], device const half * vp [[buffer(2)]],
                   device const half * kc [[buffer(3)]], device const half * vc [[buffer(4)]],
                   device float * out [[buffer(5)]], threadgroup char * shared [[threadgroup(0)]],
                   uint2 tgpig [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int D = 128, BR = 64, BC = 64;
    threadgroup uint rescale;
    threadgroup float * Ss = (threadgroup float *)shared;                 // [BR][BC]
    threadgroup half  * Hs = (threadgroup half *)shared;                  // [BC][D], aliases Ss
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
    auto tH = tensor(Hs, dextents<int32_t, 2>(D, BC));
    auto tS = tensor(Ss, dextents<int32_t, 2>(BC, BR));
    auto tP = tensor(Ps, dextents<int32_t, 2>(BC, BR));

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(BR, BC, D, false, true, true,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm_s;
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(BR, D, BC, false, false, true,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm_o;
    auto cS = mm_s.template get_destination_cooperative_tensor<decltype(tQ), decltype(tH), float>();
    auto cO = mm_o.template get_destination_cooperative_tensor<decltype(tP), decltype(tH), float>();
    for (uint i = 0; i < cO.get_capacity(); i++) cO[i] = 0.f;
    if (tid < BR) { mrow[tid] = -INFINITY; lrow[tid] = 0.f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int my_row = tid / 2;
    const int c0 = (tid % 2) * 32;
    for (uint j0 = 0; j0 < p.keys && j0 <= last_key; j0 += BC) {
        for (uint i = 0; i < cS.get_capacity(); i++) cS[i] = 0.f;
        i_stage_kv(Hs, kp, kc, p.prefix, j0, p.keys, stride, head_off, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mm_s.run(tQ, tH, cS);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        cS.store(tS);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float s[32];
        float mx = -INFINITY;
        for (int c = 0; c < 32; c++) {
            const uint key = j0 + uint(c0 + c);
            const bool ok = my_row < rows && key < p.keys && (p.causal == 0 || key <= uint(t0 + my_row));
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
        i_stage_kv(Hs, vp, vc, p.prefix, j0, p.keys, stride, head_off, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mm_o.run(tP, tH, cO);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < cO.get_capacity(); i++) {
        if (!cO.is_valid_element(i)) continue;
        auto idx = cO.get_multidimensional_index(i);
        const int r = idx[1], dcol = idx[0];
        if (r >= rows) continue;
        const float l = lrow[r];
        out[((ulong)(t0 + r) * p.heads + h) * D + dcol] = l > 0.f ? cO[i] / l : 0.f;
    }
}
