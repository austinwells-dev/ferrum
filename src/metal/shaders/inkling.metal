// Inkling 2 kernels. Appended to hybrid.metal at compile time (block_sum,
// kv_row, kv_store and kv_scales come from there).

// ---------------------------------------------------------------- depth attention

// Sums `N` per-thread values across a threadgroup of `threads` (<= 256) threads.
template <int N>
inline void block_sums(thread float * v, threadgroup float * scratch, ushort tid, ushort threads) {
    for (int k = 0; k < N; k++) {
        const float s = simd_sum(v[k]);
        if (tid % 32 == 0) scratch[k * 8 + tid / 32] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int k = 0; k < N; k++) {
        float total = 0.f;
        for (ushort i = 0; i < threads / 32; i++) total += scratch[k * 8 + i];
        v[k] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

struct InkMixArgs { uint hidden; uint tokens; uint sources; uint queries; float eps; };

// Attention over depth: each of `queries` learned vectors (query[j * hidden])
// scores the RMS-normed rows of `sources` streams (s0 .. s5, at most 6) with
// dot(q, x) / sqrt(hidden), and out[j][t] is the softmax-weighted sum of the
// normed rows. One threadgroup of 256 threads per token row.
kernel void h_ink_mix(constant InkMixArgs & p [[buffer(8)]],
                      device const float * s0 [[buffer(0)]], device const float * s1 [[buffer(1)]],
                      device const float * s2 [[buffer(2)]], device const float * s3 [[buffer(3)]],
                      device const float * s4 [[buffer(4)]], device const float * s5 [[buffer(5)]],
                      device const float * query [[buffer(6)]], device float * out [[buffer(7)]],
                      uint t [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float scratch[5 * 8];
    device const float * src[6] = {s0, s1, s2, s3, s4, s5};
    const uint H = p.hidden;
    float inv[6];
    float score[4][6];
    for (uint s = 0; s < p.sources; s++) {
        device const float * x = src[s] + (ulong)t * H;
        float v[5] = {0.f, 0.f, 0.f, 0.f, 0.f};
        for (uint i = tid; i < H; i += 256) {
            const float xi = x[i];
            v[0] += xi * xi;
            for (uint j = 0; j < p.queries; j++) v[1 + j] += xi * query[j * H + i];
        }
        block_sums<5>(v, scratch, tid, 256);
        inv[s] = rsqrt(v[0] / float(H) + p.eps);
        for (uint j = 0; j < p.queries; j++) score[j][s] = v[1 + j] * inv[s] * rsqrt(float(H));
    }
    // Softmax over the sources, folded with each source's norm.
    float w[4][6];
    for (uint j = 0; j < p.queries; j++) {
        float mx = -INFINITY;
        for (uint s = 0; s < p.sources; s++) mx = max(mx, score[j][s]);
        float total = 0.f;
        for (uint s = 0; s < p.sources; s++) {
            w[j][s] = exp(score[j][s] - mx);
            total += w[j][s];
        }
        for (uint s = 0; s < p.sources; s++) w[j][s] *= inv[s] / total;
    }
    for (uint i = tid; i < H; i += 256) {
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        for (uint s = 0; s < p.sources; s++) {
            const float xi = src[s][(ulong)t * H + i];
            for (uint j = 0; j < p.queries; j++) acc[j] += w[j][s] * xi;
        }
        for (uint j = 0; j < p.queries; j++) out[((ulong)j * p.tokens + t) * H + i] = acc[j];
    }
}

// ---------------------------------------------------------------- attention prep

struct InkPrepArgs {
    uint tokens;
    uint heads;
    uint kv_heads;
    uint position;      // cache position of token 0
    uint cache_stride;  // halfs between cached positions
    uint kv_rows;
    uint ring;
    uint global;        // full-attention layer: q and r scale with the position
    float qk_clip;
    float rel_clip;
    float eps;
    float q_scale;      // 1 / sqrt(head_dim), folded into the F16 query
    float log_floor;
    float log_alpha;
};

// L2 clip: scale x down to a norm of at most thr.
inline float clip_factor(float norm, float thr) {
    return clamp(thr / (norm + 1e-6f), 0.f, 1.f);
}

// Per (slot, token), 128 threads (one per head dimension):
//   slots [0, heads): q = clip(q * q_clip) (* tau on global layers), F16 and
//     pre-scaled, from the q half of each [q | gate] head pair in qg;
//   [heads, heads + kv): k = clip(k) into the K cache;
//   [heads + kv, heads + 2kv): v = rmsnorm(v) (no gain) into the V cache;
//   [heads + 2kv, 2heads + 2kv): the 16-wide position vector
//     r = clip(r * rel_clip) (* tau) into rel_out.
// tau = 1 + alpha * log(max(1, (pos + 1) / floor)).
template <bool Q8>
void ink_prep_impl(constant InkPrepArgs & p, device const float * qg, device const float * k,
                   device const float * v, device const float * q_clip, device const float * r,
                   device const float * rel_clip, device half * q_out, device half * kc,
                   device half * vc, device float * rel_out, uint2 group, ushort tid,
                   threadgroup float * scratch) {
    constexpr uint D = 128, R = 16;
    const uint slot = group.x, t = group.y;
    const uint kv = p.kv_heads;
    const float pos = float(p.position + t);
    const float tau = p.global != 0 ? 1.f + p.log_alpha * log(max(1.f, (pos + 1.f) / p.log_floor)) : 1.f;
    if (slot < p.heads) {
        const uint h = slot;
        const float x = qg[((ulong)t * p.heads + h) * 2 * D + tid] * q_clip[h * D + tid];
        const float norm = sqrt(block_sum(x * x, scratch, tid, D));
        q_out[((ulong)t * p.heads + h) * D + tid] = half(x * clip_factor(norm, p.qk_clip) * tau * p.q_scale);
    } else if (slot < p.heads + kv) {
        const uint h = slot - p.heads;
        const float x = k[((ulong)t * kv + h) * D + tid];
        const float norm = sqrt(block_sum(x * x, scratch, tid, D));
        const ulong e = (ulong)kv_row(p.position + t, p.ring) * p.cache_stride + h * D + tid;
        kv_store<Q8>(kc, p.kv_rows, p.cache_stride, e, x * clip_factor(norm, p.qk_clip));
    } else if (slot < p.heads + 2 * kv) {
        const uint h = slot - p.heads - kv;
        const float x = v[((ulong)t * kv + h) * D + tid];
        const float ss = block_sum(x * x, scratch, tid, D);
        const ulong e = (ulong)kv_row(p.position + t, p.ring) * p.cache_stride + h * D + tid;
        kv_store<Q8>(vc, p.kv_rows, p.cache_stride, e, x * rsqrt(ss / float(D) + p.eps));
    } else if (tid < 32) {
        const uint h = slot - p.heads - 2 * kv;
        const float x = tid < R ? r[((ulong)t * p.heads + h) * R + tid] * rel_clip[h * R + tid] : 0.f;
        const float norm = sqrt(simd_sum(x * x));
        if (tid < R) rel_out[((ulong)t * p.heads + h) * R + tid] = x * clip_factor(norm, p.rel_clip) * tau;
    }
}

#define INK_PREP(name, q8)                                                                      \
kernel void name(constant InkPrepArgs & p [[buffer(10)]],                                      \
                 device const float * qg [[buffer(0)]], device const float * k [[buffer(1)]],  \
                 device const float * v [[buffer(2)]], device const float * q_clip [[buffer(3)]], \
                 device const float * r [[buffer(4)]], device const float * rel_clip [[buffer(5)]], \
                 device half * q_out [[buffer(6)]], device half * kc [[buffer(7)]],            \
                 device half * vc [[buffer(8)]], device float * rel_out [[buffer(9)]],         \
                 uint2 group [[threadgroup_position_in_grid]],                                 \
                 ushort tid [[thread_index_in_threadgroup]]) {                                 \
    threadgroup float scratch[8];                                                              \
    ink_prep_impl<q8>(p, qg, k, v, q_clip, r, rel_clip, q_out, kc, vc, rel_out, group, tid, scratch); \
}
INK_PREP(h_ink_prep, false)
INK_PREP(h_ink_prep_q8, true)

// ---------------------------------------------------------------- short convolutions

struct InkConvArgs {
    uint tokens;
    uint channels;      // channels of this convolution
    uint x_stride;      // floats between rows of x
    uint state_width;   // channels of one state row (all four convolutions)
    uint offset;        // this convolution's first channel in a state row
    uint heads;         // input-dependent taps: heads of `head_width` channels (0 = none)
    uint head_width;
    uint tape;          // verify: record x in tape, leave the state alone
    float scale;        // 1 / sqrt(kernel * (dynamic ? 2 : 1) + 1)
};

// Causal depthwise convolution with kernel 4 and a residual:
//   y[t, c] = scale * (x[t, c] + sum_k w[c * 4 + k] * x[t - 3 + k, c]
//                      + sum_lag dyn[t, (c / head_width) * 4 + lag] * x[t - lag, c]).
// state is [3][state_width], oldest row first: the previous inputs.
// One thread per channel.
kernel void h_ink_conv(constant InkConvArgs & p [[buffer(6)]],
                       device const float * x [[buffer(0)]], device const float * w [[buffer(1)]],
                       device const float * dyn [[buffer(2)]], device float * y [[buffer(3)]],
                       device float * state [[buffer(4)]], device float * tape [[buffer(5)]],
                       uint c [[thread_position_in_grid]]) {
    if (c >= p.channels) return;
    const uint S = p.state_width, sc = p.offset + c;
    float h0 = state[sc], h1 = state[S + sc], h2 = state[2 * S + sc];
    const float4 wc = float4(w[c * 4], w[c * 4 + 1], w[c * 4 + 2], w[c * 4 + 3]);
    const uint hd = p.heads != 0 ? c / p.head_width : 0;
    for (uint t = 0; t < p.tokens; t++) {
        const float xt = x[(ulong)t * p.x_stride + c];
        float acc = xt + wc.x * h0 + wc.y * h1 + wc.z * h2 + wc.w * xt;
        if (p.heads != 0) {
            device const float * d = dyn + ((ulong)t * p.heads + hd) * 4;
            acc += d[0] * xt + d[1] * h2 + d[2] * h1 + d[3] * h0;
        }
        y[(ulong)t * p.channels + c] = acc * p.scale;
        if (p.tape != 0) tape[(ulong)t * S + sc] = xt;
        h0 = h1;
        h1 = h2;
        h2 = xt;
    }
    if (p.tape == 0) {
        state[sc] = h0;
        state[S + sc] = h1;
        state[2 * S + sc] = h2;
    }
}

// ---------------------------------------------------------------- loop controllers

struct InkCtrlArgs { uint hidden; uint tokens; uint rank; float eps; };

// z_in[t] = [rmsnorm(x[t]); rmsnorm(m[t])] (no gain). 256 threads per row.
kernel void h_ink_ctrl_in(constant InkCtrlArgs & p [[buffer(3)]],
                          device const float * x [[buffer(0)]], device const float * m [[buffer(1)]],
                          device float * z [[buffer(2)]],
                          uint t [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float scratch[2 * 8];
    const uint H = p.hidden;
    device const float * xr = x + (ulong)t * H;
    device const float * mr = m + (ulong)t * H;
    float v[2] = {0.f, 0.f};
    for (uint i = tid; i < H; i += 256) {
        v[0] += xr[i] * xr[i];
        v[1] += mr[i] * mr[i];
    }
    block_sums<2>(v, scratch, tid, 256);
    const float ix = rsqrt(v[0] / float(H) + p.eps), im = rsqrt(v[1] / float(H) + p.eps);
    device float * zr = z + (ulong)t * 2 * H;
    for (uint i = tid; i < H; i += 256) {
        zr[i] = xr[i] * ix;
        zr[H + i] = mr[i] * im;
    }
}

// y = x - tanh(F z) * x + tanh(G z) * m, with gain G and forget F [hidden][rank]
// and z [tokens][rank]. One thread per element.
kernel void h_ink_ctrl_out(constant InkCtrlArgs & p [[buffer(6)]],
                           device const float * x [[buffer(0)]], device const float * m [[buffer(1)]],
                           device const float * z [[buffer(2)]], device const float * gain [[buffer(3)]],
                           device const float * forget [[buffer(4)]], device float * y [[buffer(5)]],
                           uint i [[thread_position_in_grid]]) {
    if (i >= p.tokens * p.hidden) return;
    const uint t = i / p.hidden, c = i % p.hidden;
    device const float4 * zr = (device const float4 *)(z + (ulong)t * p.rank);
    device const float4 * gr = (device const float4 *)(gain + (ulong)c * p.rank);
    device const float4 * fr = (device const float4 *)(forget + (ulong)c * p.rank);
    float g = 0.f, f = 0.f;
    for (uint k = 0; k < p.rank / 4; k++) {
        g += dot(gr[k], zr[k]);
        f += dot(fr[k], zr[k]);
    }
    y[i] = x[i] - tanh(f) * x[i] + tanh(g) * m[i];
}

// ---------------------------------------------------------------- experts

struct InkRouteArgs { uint experts; uint used; uint tokens; float scale; };

// Sigmoid routing: experts are picked by sigmoid(logit) + bias (ties to the
// lower index); their weights are the unbiased sigmoids divided by their sum
// (+ 1e-20) and multiplied by `scale`. One threadgroup of `experts` threads per token.
kernel void h_ink_route(constant InkRouteArgs & p [[buffer(4)]],
                        device const float * logits [[buffer(0)]], device const float * bias [[buffer(1)]],
                        device uint * ids [[buffer(2)]], device float * weights [[buffer(3)]],
                        uint t [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                        ushort threads [[threads_per_threadgroup]]) {
    threadgroup float best_v[32];
    threadgroup uint best_i[32];
    threadgroup float picked[64];
    const float prob = 1.f / (1.f + exp(-logits[(ulong)t * p.experts + tid]));
    float key = prob + bias[tid];
    for (uint k = 0; k < p.used; k++) {
        float bv = key;
        uint bi = tid;
        for (ushort off = 16; off > 0; off /= 2) {
            const float ov = simd_shuffle_down(bv, off);
            const uint oi = simd_shuffle_down(bi, off);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (tid % 32 == 0) { best_v[tid / 32] = bv; best_i[tid / 32] = bi; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bv = best_v[0]; bi = best_i[0];
        for (ushort i = 1; i < threads / 32; i++) {
            if (best_v[i] > bv || (best_v[i] == bv && best_i[i] < bi)) { bv = best_v[i]; bi = best_i[i]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == bi) {
            ids[(ulong)t * p.used + k] = bi;
            picked[k] = prob;
            key = -INFINITY;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < p.used) {
        float total = 0.f;
        for (uint k = 0; k < p.used; k++) total += picked[k];
        weights[(ulong)t * p.used + tid] = picked[tid] / (total + 1e-20f) * p.scale;
    }
}

struct InkSituArgs { uint width; uint rows; uint sorted; float beta; float linear_beta; uint scaled; };

// act = beta tanh(g / beta) sigmoid(g) * lb tanh(u / lb) of the fused
// [gate | up] rows (each 2 * width wide). With `scaled`, g and u are first
// multiplied by their expert's gate_up scale and the route's weight by its
// expert's down scale. Row r is route sorted[r] when `sorted`, else route r.
kernel void h_ink_situ(constant InkSituArgs & p [[buffer(7)]],
                       device const float * gu [[buffer(0)]], device const uint * ids [[buffer(1)]],
                       device const uint * sorted [[buffer(2)]], device const float * gu_scale [[buffer(3)]],
                       device const float * down_scale [[buffer(4)]], device float * act [[buffer(5)]],
                       device float * weights [[buffer(6)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= p.rows * p.width) return;
    const uint r = i / p.width, c = i % p.width;
    const uint route = p.sorted != 0 ? sorted[r] : r;
    const float s = p.scaled != 0 ? gu_scale[ids[route]] : 1.f;
    const float g = gu[(ulong)r * 2 * p.width + c] * s;
    const float u = gu[(ulong)r * 2 * p.width + p.width + c] * s;
    const float sig = 1.f / (1.f + exp(-g));
    act[i] = p.beta * tanh(g / p.beta) * sig * p.linear_beta * tanh(u / p.linear_beta);
    if (p.scaled != 0 && c == 0) weights[route] *= down_scale[ids[route]];
}

// ---------------------------------------------------------------- elementwise

struct InkEltArgs { uint n; float a; };

// y += x
kernel void h_ink_acc(constant InkEltArgs & p [[buffer(2)]], device const float * x [[buffer(0)]],
                      device float * y [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i < p.n) y[i] += x[i];
}

// y = a * tanh(y / a)
kernel void h_ink_softcap(constant InkEltArgs & p [[buffer(1)]], device float * y [[buffer(0)]],
                          uint i [[thread_position_in_grid]]) {
    if (i < p.n) y[i] = p.a * tanh(y[i] / p.a);
}
