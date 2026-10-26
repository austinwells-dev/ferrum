// Kernels of the image decoder (src/diffusion/vae.rs). Appended to
// hybrid.metal. Activations are F16 in NHWC order: pixel-major rows of
// `channels` values, so a latent token row is already one pixel. Accumulation
// and normalisation are F32.

struct PrepArgs2 { uint n; uint channels; };

// z = x * std + mean per channel, F32 [pixels][channels] to F16.
kernel void vae_denorm(constant PrepArgs2 & a [[buffer(4)]], device const float * x [[buffer(0)]],
                     device const float * mean [[buffer(1)]], device const float * stdev [[buffer(2)]],
                     device half * y [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < a.n) {
        const uint c = i % a.channels;
        y[i] = half(x[i] * stdev[c] + mean[c]);
    }
}

struct RmsArgs { uint pixels; uint channels; uint silu; };

// y = x / rms(x over channels) * gamma, optionally followed by SiLU
// (F.normalize(x) * sqrt(C) * gamma). One simdgroup per pixel, four per threadgroup.
kernel void vae_chan_rms(constant RmsArgs & a [[buffer(3)]], device const half * x [[buffer(0)]],
                       device const float * gamma [[buffer(1)]], device half * y [[buffer(2)]],
                       uint g [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    const uint pixel = g * 4 + tid / 32;
    if (pixel >= a.pixels) return;
    const ushort lane = tid % 32;
    device const half * xr = x + (ulong)pixel * a.channels;
    float ss = 0.f;
    for (uint c = lane; c < a.channels; c += 32) {
        const float v = float(xr[c]);
        ss += v * v;
    }
    ss = simd_sum(ss);
    const float inv = sqrt(float(a.channels)) / max(sqrt(ss), 1e-12f);
    device half * yr = y + (ulong)pixel * a.channels;
    for (uint c = lane; c < a.channels; c += 32) {
        float v = float(xr[c]) * inv * gamma[c];
        if (a.silu != 0) v = v / (1.f + exp(-v));
        yr[c] = half(v);
    }
}

struct VaeConvArgs {
    uint h; uint w;          // output size
    uint cin; uint cout;
    uint taps;               // 1 (pointwise) or 9 (3x3, padding 1)
    uint up;                 // input is (h/up, w/up) and read nearest
    uint res_mode;           // 0 none, 1 add res (same layout), 2 add the pixel-shuffled shortcut
    uint res_cin;            // mode 2: channels of the (h/2, w/2) shortcut source
    uint dup_t;              // mode 2: frame factor (the last frame is read) and channel repeats
    uint dup_repeats;
};

// Convolution as an implicit GEMM: the 128-pixel x NK-channel input tile of
// each (tap, channel block) is gathered into threadgroup memory with zero
// padding and optional nearest 2x upsampling, and multiplied by 64 output
// channels of F16 weights `[cout][taps * cin]` (tap-major). The epilogue adds
// the bias and a residual. grid (ceil(h*w/128), ceil(cout/64)); threads 128.
template <int NK>
void conv_impl(constant VaeConvArgs & a, device half * x, device half * wt, device const half * bias,
               device const half * res, device half * y, threadgroup half * tb, uint2 tg, ushort tid) {
    constexpr int NRB = 128, NRA = 64;
    const int pixels = int(a.h * a.w);
    const int p0 = int(tg.x) * NRB, n0 = int(tg.y) * NRA;
    const int K = int(a.taps * a.cin);
    const int nrows = min(NRA, int(a.cout) - n0);
    const int in_w = int(a.w / a.up);

    auto tB = tensor(tb, dextents<int32_t, 2>(NK, NRB));
    auto tA = tensor(wt + (ulong)n0 * K, dextents<int32_t, 2>(K, nrows), array<int, 2>({1, K}));
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(NRB, NRA, NK, false, true, true,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();
    for (uint i = 0; i < cT.get_capacity(); i++) cT[i] = 0.f;

    constexpr int UNITS = NRB * (NK / 8);
    for (int tap = 0; tap < int(a.taps); tap++) {
        const int dy = a.taps == 9 ? tap / 3 - 1 : 0, dx = a.taps == 9 ? tap % 3 - 1 : 0;
        for (int c0 = 0; c0 < int(a.cin); c0 += NK) {
            for (int u = tid; u < UNITS; u += 128) {
                const int m = u / (NK / 8), cc = (u % (NK / 8)) * 8;
                const int p = p0 + m;
                half4 v0 = 0.h, v1 = 0.h;
                if (p < pixels) {
                    const int iy = p / int(a.w) + dy, ix = p % int(a.w) + dx;
                    if (iy >= 0 && iy < int(a.h) && ix >= 0 && ix < int(a.w)) {
                        const int sy = a.up == 2 ? iy >> 1 : iy, sx = a.up == 2 ? ix >> 1 : ix;
                        device const half4 * src = (device const half4 *)(x + ((ulong)sy * in_w + sx) * a.cin + c0 + cc);
                        v0 = src[0];
                        v1 = src[1];
                    }
                }
                threadgroup half4 * dst = (threadgroup half4 *)(tb + m * NK + cc);
                dst[0] = v0;
                dst[1] = v1;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            auto mB = tB.slice(0, 0);
            auto mA = tA.slice(tap * int(a.cin) + c0, 0);
            mm.run(mB, mA, cT);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    for (uint i = 0; i < cT.get_capacity(); i++) {
        if (!cT.is_valid_element(i)) continue;
        auto idx = cT.get_multidimensional_index(i);
        const int p = p0 + idx[1], n = n0 + idx[0];
        if (p >= pixels || n >= int(a.cout)) continue;
        float v = cT[i] + float(bias[n]);
        if (a.res_mode == 1) {
            v += float(res[(ulong)p * a.cout + n]);
        } else if (a.res_mode == 2) {
            const int Y = p / int(a.w), X = p % int(a.w);
            const int j = ((n * int(a.dup_t) + int(a.dup_t) - 1) * 2 + (Y & 1)) * 2 + (X & 1);
            const int src_c = j / int(a.dup_repeats);
            v += float(res[((ulong)(Y >> 1) * (a.w >> 1) + (X >> 1)) * a.res_cin + src_c]);
        }
        y[(ulong)p * a.cout + n] = half(v);
    }
}

#define CONV_KERNEL(NAME, NK)                                                                     \
kernel void NAME(constant VaeConvArgs & a [[buffer(5)]], device half * x [[buffer(0)]],              \
                 device half * wt [[buffer(1)]], device const half * bias [[buffer(2)]],          \
                 device const half * res [[buffer(3)]], device half * y [[buffer(4)]],            \
                 threadgroup half * tb [[threadgroup(0)]], uint2 tg [[threadgroup_position_in_grid]], \
                 ushort tid [[thread_index_in_threadgroup]]) {                                    \
    conv_impl<NK>(a, x, wt, bias, res, y, tb, tg, tid);                                           \
}
CONV_KERNEL(vae_conv_k32, 32)
CONV_KERNEL(vae_conv_k16, 16)

struct GemmArgs {
    uint m; uint n; uint k;
    uint lda;                // A row stride (elements)
    uint bk; uint bn;        // B element strides along k and along n
    uint ldc;
};

// C[m][n] = sum_k A[m][k] * B[n][k], A and B F16, via tensor views so B may
// be read transposed (strides); the attention of the decoder's middle block
// runs on it. grid (ceil(m/64), ceil(n/64)); threads 128.
template <typename T>
void gemm_impl(constant GemmArgs & a, device half * A, device half * B, device T * C, uint2 tg) {
    constexpr int NK = 32;
    const int m0 = int(tg.x) * 64, n0 = int(tg.y) * 64;
    const int rows = min(64, int(a.m) - m0), cols = min(64, int(a.n) - n0);
    auto tA = tensor(A + (ulong)m0 * a.lda, dextents<int32_t, 2>(int(a.k), rows), array<int, 2>({1, int(a.lda)}));
    auto tB = tensor(B + (ulong)n0 * a.bn, dextents<int32_t, 2>(int(a.k), cols), array<int, 2>({int(a.bk), int(a.bn)}));
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(64, 64, NK, false, true, true,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();
    for (uint i = 0; i < cT.get_capacity(); i++) cT[i] = 0.f;
    for (int k0 = 0; k0 < int(a.k); k0 += NK) {
        auto mA = tA.slice(k0, 0);
        auto mB = tB.slice(k0, 0);
        mm.run(mA, mB, cT);
    }
    for (uint i = 0; i < cT.get_capacity(); i++) {
        if (!cT.is_valid_element(i)) continue;
        auto idx = cT.get_multidimensional_index(i);
        const int r = idx[1], c = idx[0];
        if (r >= rows || c >= cols) continue;
        C[(ulong)(m0 + r) * a.ldc + n0 + c] = T(cT[i]);
    }
}

kernel void vae_gemm_f32(constant GemmArgs & a [[buffer(3)]], device half * A [[buffer(0)]],
                       device half * B [[buffer(1)]], device float * C [[buffer(2)]],
                       uint2 tg [[threadgroup_position_in_grid]]) {
    gemm_impl<float>(a, A, B, C, tg);
}
kernel void vae_gemm_f16(constant GemmArgs & a [[buffer(3)]], device half * A [[buffer(0)]],
                       device half * B [[buffer(1)]], device half * C [[buffer(2)]],
                       uint2 tg [[threadgroup_position_in_grid]]) {
    gemm_impl<half>(a, A, B, C, tg);
}

struct SoftmaxArgs { uint rows; uint cols; float scale; };

// Row softmax of F32 scores (times `scale`) into F16 weights; one threadgroup
// of 256 threads per row.
kernel void vae_softmax_rows(constant SoftmaxArgs & a [[buffer(2)]], device const float * s [[buffer(0)]],
                           device half * p [[buffer(1)]], uint row [[threadgroup_position_in_grid]],
                           ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float scratch[8];
    device const float * sr = s + (ulong)row * a.cols;
    float mx = -INFINITY;
    for (uint i = tid; i < a.cols; i += 256) mx = max(mx, sr[i] * a.scale);
    mx = simd_max(mx);
    if (tid % 32 == 0) scratch[tid / 32] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx = scratch[0];
    for (ushort i = 1; i < 8; i++) mx = max(mx, scratch[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0.f;
    for (uint i = tid; i < a.cols; i += 256) sum += exp(sr[i] * a.scale - mx);
    sum = block_sum(sum, scratch, tid, 256);
    device half * pr = p + (ulong)row * a.cols;
    for (uint i = tid; i < a.cols; i += 256) pr[i] = half(exp(sr[i] * a.scale - mx) / sum);
}

struct TransposeArgs { uint rows; uint cols; uint ld; };

// out[c][r] = in[r][c] for F16 `in` with row stride `ld`; the TensorOps gemm
// wants unit stride along k, which the decoder's attention needs for V.
kernel void vae_transpose(constant TransposeArgs & a [[buffer(2)]], device const half * x [[buffer(0)]],
                        device half * y [[buffer(1)]], uint2 g [[thread_position_in_grid]]) {
    if (g.x < a.cols && g.y < a.rows) y[(ulong)g.x * a.rows + g.y] = x[(ulong)g.y * a.ld + g.x];
}
