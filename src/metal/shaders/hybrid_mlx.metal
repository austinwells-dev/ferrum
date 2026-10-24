// Copyright 2026 Nace AI (Kapitsa, https://github.com/nace-ai/kapitsa)
// Copyright 2026 Austin Wells (Ferrum changes)
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
// on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for
// the specific language governing permissions and limitations under the License.
//
// This file is Apache-2.0 only (not dual-licensed with the rest of Ferrum); see THIRD_PARTY_NOTICES.
//
// MLX affine 4-bit (group 64) projections for the hybrid engine. The decode GEMV is adapted from Kapitsa's
// src/kernels/qmv.metal and src/kernels/quant.h (qmv3_impl, load_x, dot_row at be57c8f): the shift-free dequant (the
// code at slot j of a word is masked in place, worth q * 2^(4j), and x is pre-scaled by 2^-(4j)) and the per-group
// epilogue y += s * dot + b * sum(x). Changes: Ferrum's projection ABI (ProjArgs, one weight buffer, F32 rows
// `x_stride` apart, the activation row from the grid), and Ferrum's per-row weight layout: each row holds its K/8
// code words, then its K/64 bf16 scales, then its K/64 bf16 biases (`row_bytes` = K/2 + K/16 bytes), so rows are
// self-contained (embedding gathers and LM-head prefixes work as for GGUF formats). The batched GEMV, the TensorOps
// tile with an MLX dequant step and the embedding gather are Ferrum's, written for this layout.
//
// Appended to hybrid.metal at compile time (ProjArgs, GetRowsArgs and FOR_UNROLL come from there).

// Row r's code words, scales and biases (Ferrum's per-row MLX layout).
struct MlxRow {
    device const uint * codes;
    device const bfloat * scales;
    device const bfloat * biases;
};
inline MlxRow mlx_row(device const char * w, uint row, uint row_bytes, uint k) {
    device const char * base = w + (ulong)row * row_bytes;
    MlxRow r;
    r.codes = (device const uint *)base;
    r.scales = (device const bfloat *)(base + k / 2);
    r.biases = r.scales + k / 64;
    return r;
}

// 16 activations from k0 pre-scaled for the shift-free dot (slot j of a word worth 2^(4j)), and their sum.
inline void mlx_load_x16(device const float * x, thread float (&xs)[16], thread float & xsum) {
    xsum = 0.f;
    FOR_UNROLL (short i = 0; i < 4; ++i) {
        const float4 v = ((device const float4 *)x)[i];
        xsum += v.x + v.y + v.z + v.w;
        FOR_UNROLL (short e = 0; e < 4; ++e) {
            const short j = (i * 4 + e) % 8;
            xs[i * 4 + e] = v[e] * float(1.0f / float(1u << (4 * j)));
        }
    }
}

// dot(codes of 16 values at k0, prescaled x) for one row, and its group's scale and bias applied.
inline float mlx_dot16(MlxRow r, uint k0, thread const float (&xs)[16], float xsum) {
    const uint2 wv = *(device const uint2 *)(r.codes + k0 / 8);
    float d = 0.f;
    FOR_UNROLL (short j = 0; j < 8; ++j) d += float(wv.x & (0xFu << (4 * j))) * xs[j];
    FOR_UNROLL (short j = 0; j < 8; ++j) d += float(wv.y & (0xFu << (4 * j))) * xs[8 + j];
    const uint g = k0 / 64;
    return float(r.scales[g]) * d + float(r.biases[g]) * xsum;
}

// M=1 GEMV (also each row of FERRUM_HYBRID_EXACT's per-row path), adapted from Kapitsa's qmv3: the activation row
// is staged once per threadgroup in 4096-wide tiles, then each SIMD group's lanes take VPL values of each of its ROWS
// rows per step. Grid (ceil(n / (ROWS * NSG)), m); threads (32, NSG). Kapitsa tunes 2 rows x 8 SIMD groups of 16 values
// on the M4 Pro; on the M5 one row per SIMD group of 32 values (16-byte loads) measured 2-5% faster on every 27B shape.
template <short ROWS, short VPL>
void mv_mlx4_impl(constant ProjArgs & a, device const char * w, device const float * x, device float * y,
                  threadgroup float * xs, uint3 tgpig, ushort lane, ushort sg, ushort nsg) {
    constexpr short WORDS = VPL / 8;
    const uint row0 = (tgpig.x * nsg + sg) * ROWS;
    const uint tid = uint(sg) * 32 + lane, ntg = uint(nsg) * 32;
    device const float * xv = x + (ulong)tgpig.y * a.x_stride;
    MlxRow rows[ROWS];
    float acc[ROWS];
    FOR_UNROLL (short r = 0; r < ROWS; ++r) {
        // Rows past n are clamped (computed redundantly, never stored) so lanes never diverge.
        rows[r] = mlx_row(w, min(row0 + r, a.n - 1), a.row_bytes, a.k);
        acc[r] = 0.f;
    }
    for (uint kt = 0; kt < a.k; kt += 4096) {
        const uint tile = min(4096u, a.k - kt);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < tile / 4; i += ntg) ((threadgroup float4 *)xs)[i] = ((device const float4 *)(xv + kt))[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (row0 >= a.n) continue;
        for (uint k0 = lane * VPL; k0 < tile; k0 += 32 * VPL) {
            float xq[VPL], xsum = 0.f;
            FOR_UNROLL (short i = 0; i < VPL / 4; ++i) {
                const float4 v = ((threadgroup const float4 *)(xs + k0))[i];
                xsum += v.x + v.y + v.z + v.w;
                FOR_UNROLL (short e = 0; e < 4; ++e) xq[i * 4 + e] = v[e] * float(1.0f / float(1u << (4 * ((i * 4 + e) % 8))));
            }
            const uint kg = kt + k0, g = kg / 64;
            FOR_UNROLL (short r = 0; r < ROWS; ++r) {
                uint wv[WORDS];
                if (WORDS == 4) {
                    const uint4 v = *(device const uint4 *)(rows[r].codes + kg / 8);
                    wv[0] = v.x; wv[1] = v.y; wv[2] = v.z; wv[3] = v.w;
                } else {
                    const uint2 v = *(device const uint2 *)(rows[r].codes + kg / 8);
                    wv[0] = v.x; wv[1] = v.y;
                }
                float d = 0.f;
                FOR_UNROLL (short i = 0; i < WORDS; ++i)
                    FOR_UNROLL (short j = 0; j < 8; ++j) d += float(wv[i] & (0xFu << (4 * j))) * xq[i * 8 + j];
                acc[r] += float(rows[r].scales[g]) * d + float(rows[r].biases[g]) * xsum;
            }
        }
    }
    if (row0 >= a.n) return;
    FOR_UNROLL (short r = 0; r < ROWS; ++r) {
        const float v = simd_sum(acc[r]);
        if (lane == 0 && row0 + r < a.n) y[(ulong)tgpig.y * a.y_stride + row0 + r] = v;
    }
}
#define MV_MLX4(NAME, ROWS, VPL)                                                                              \
kernel void NAME(constant ProjArgs & a [[buffer(3)]], device const char * w [[buffer(0)]],                   \
                 device const float * x [[buffer(1)]], device float * y [[buffer(2)]],                       \
                 uint3 tgpig [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]],    \
                 ushort sg [[simdgroup_index_in_threadgroup]], ushort3 ntg [[threads_per_threadgroup]]) {   \
    threadgroup float xs[4096];                                                                               \
    mv_mlx4_impl<ROWS, VPL>(a, w, x, y, xs, tgpig, lane, sg, ntg.y);                                          \
}
MV_MLX4(h_mv_mlx4, 1, 32)

// Batched GEMV for 2..4 activation rows (one weight read for all of them), the h_mvb_* geometry: grid
// (ceil(n / (2 * NSG)), ceil(m / 4)); threads (32, NSG); 2 output rows per SIMD group.
kernel void h_mvb_mlx4(constant ProjArgs & a [[buffer(3)]], device const char * w [[buffer(0)]],
                       device const float * x [[buffer(1)]], device float * y [[buffer(2)]],
                       uint3 tgpig [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]],
                       ushort sg [[simdgroup_index_in_threadgroup]], ushort3 ntg [[threads_per_threadgroup]]) {
    constexpr short NR = 2, R = 4;
    const uint row0 = (tgpig.x * ntg.y + sg) * NR;
    if (row0 >= a.n) return;
    const uint t0 = tgpig.y * R;
    const short rrows = short(min(uint(R), a.m - t0));
    MlxRow rows[NR];
    float acc[NR][R];
    FOR_UNROLL (short r = 0; r < NR; ++r) {
        rows[r] = mlx_row(w, min(row0 + r, a.n - 1), a.row_bytes, a.k);
        FOR_UNROLL (short t = 0; t < R; ++t) acc[r][t] = 0.f;
    }
    for (uint k0 = lane * 16; k0 < a.k; k0 += 32 * 16) {
        float xs[R][16], xsum[R];
        FOR_UNROLL (short t = 0; t < R; ++t) {
            if (t < rrows) mlx_load_x16(x + (ulong)(t0 + t) * a.x_stride + k0, xs[t], xsum[t]);
        }
        FOR_UNROLL (short r = 0; r < NR; ++r) {
            const uint2 wv = *(device const uint2 *)(rows[r].codes + k0 / 8);
            const uint g = k0 / 64;
            const float s = float(rows[r].scales[g]), b = float(rows[r].biases[g]);
            FOR_UNROLL (short t = 0; t < R; ++t) {
                if (t >= rrows) continue;
                float d = 0.f;
                FOR_UNROLL (short j = 0; j < 8; ++j) d += float(wv.x & (0xFu << (4 * j))) * xs[t][j];
                FOR_UNROLL (short j = 0; j < 8; ++j) d += float(wv.y & (0xFu << (4 * j))) * xs[t][8 + j];
                acc[r][t] += s * d + b * xsum[t];
            }
        }
    }
    FOR_UNROLL (short r = 0; r < NR; ++r) {
        FOR_UNROLL (short t = 0; t < R; ++t) {
            const float v = simd_sum(acc[r][t]);
            if (lane == 0 && t < rrows && row0 + r < a.n) y[(ulong)(t0 + t) * a.y_stride + row0 + r] = v;
        }
    }
}

// 16 dequantized values of row `r` from k0 (a multiple of 16), as half.
inline void mlx_dequant16(MlxRow r, uint k0, threadgroup half * dst) {
    const uint2 wv = *(device const uint2 *)(r.codes + k0 / 8);
    const uint g = k0 / 64;
    const float s = float(r.scales[g]), b = float(r.biases[g]);
    FOR_UNROLL (short j = 0; j < 8; ++j) dst[j] = half(s * float((wv.x >> (4 * j)) & 0xFu) + b);
    FOR_UNROLL (short j = 0; j < 8; ++j) dst[8 + j] = half(s * float((wv.y >> (4 * j)) & 0xFu) + b);
}

// TensorOps GEMM over MLX rows, as mm_impl: a 64-row weight tile dequantized to F16 in threadgroup memory, NRB
// activation rows read as F32 from device memory, F32 accumulation. Grid (ceil(m / NRB), ceil(n / 64)); threads 128;
// 64 * NK * 2 bytes of threadgroup memory.
template <int NRB, int NK>
void mm_mlx4_impl(constant ProjArgs & a, device const char * w, device const float * x, device float * y,
                  threadgroup half * sa, uint3 tgpig, ushort tiitg) {
    constexpr int NRA = 64;
    constexpr int CHUNKS = NK / 16;
    const int K = a.k, M = a.n, N = a.m;
    const int ra = tgpig.y * NRA;
    const int rb = tgpig.x * NRB;
    auto tA = tensor(sa, dextents<int32_t, 2>(NK, NRA));
    auto tB = tensor((device float *)x, dextents<int32_t, 2>(K, N), array<int, 2>({1, int(a.x_stride)}));
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            NRB, NRA, NK, false, true, true,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();
    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        for (int work = tiitg; work < NRA * CHUNKS; work += 128) {
            const int row = work / CHUNKS;
            const int k_chunk = work % CHUNKS;
            threadgroup half * dst = sa + row * NK + k_chunk * 16;
            if (ra + row < M) {
                mlx_dequant16(mlx_row(w, uint(ra + row), a.row_bytes, a.k), uint(loop_k + k_chunk * 16), dst);
            } else {
                FOR_UNROLL (short i = 0; i < 16; i++) dst[i] = 0.h;
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
kernel void h_mm_mlx4(constant ProjArgs & a [[buffer(3)]], device const char * w [[buffer(0)]],
                      device const float * x [[buffer(1)]], device float * y [[buffer(2)]],
                      threadgroup half * sa [[threadgroup(0)]],
                      uint3 tgpig [[threadgroup_position_in_grid]], ushort tiitg [[thread_index_in_threadgroup]]) {
    mm_mlx4_impl<128, 32>(a, w, x, y, sa, tgpig, tiitg);
}
kernel void h_mms_mlx4(constant ProjArgs & a [[buffer(3)]], device const char * w [[buffer(0)]],
                       device const float * x [[buffer(1)]], device float * y [[buffer(2)]],
                       threadgroup half * sa [[threadgroup(0)]],
                       uint3 tgpig [[threadgroup_position_in_grid]], ushort tiitg [[thread_index_in_threadgroup]]) {
    mm_mlx4_impl<16, 64>(a, w, x, y, sa, tgpig, tiitg);
}

// Embedding rows: grid (ceil(k / 16 / 32), tokens); threads 32; each thread dequantizes 16 values.
kernel void h_get_rows_mlx4(constant GetRowsArgs & a [[buffer(3)]], device const char * w [[buffer(0)]],
                            device const uint * ids [[buffer(1)]], device float * y [[buffer(2)]],
                            uint3 tgpig [[threadgroup_position_in_grid]], ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint chunk = tgpig.x * 32 + tiisg;
    if (chunk * 16 >= a.k) return;
    const MlxRow r = mlx_row(w, ids[tgpig.y], a.row_bytes, a.k);
    const uint k0 = chunk * 16;
    const uint2 wv = *(device const uint2 *)(r.codes + k0 / 8);
    const float s = float(r.scales[k0 / 64]), b = float(r.biases[k0 / 64]);
    device float * out = y + (ulong)tgpig.y * a.k + k0;
    FOR_UNROLL (short j = 0; j < 8; ++j) out[j] = s * float((wv.x >> (4 * j)) & 0xFu) + b;
    FOR_UNROLL (short j = 0; j < 8; ++j) out[8 + j] = s * float((wv.y >> (4 * j)) & 0xFu) + b;
}
