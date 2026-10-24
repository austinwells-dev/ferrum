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

// ---------------------------------------------------------------- verify widths on the matrix units (M5)
//
// Adapted from Kapitsa's src/kernels/tensor/verify8t.metal (qmm8t_impl, qmm8t2_impl at be57c8f): per 64-weight group,
// one tensor_ops matmul2d of the integer 4-bit codes (uint8, the left operand, filled from registers) by the half
// inputs (a "T table", the right operand), then the fp32 epilogue y += s * dot + b * sum(x). The codes are exact, so
// the only roundings are the half inputs and the matrix unit's sums. qmm8t takes token rows in 8-row tiles (one
// matmul per tile and weight load); qmm8t2 takes two tiles (16 rows) per weight load, for tree verifies. Changes:
// one weight matrix per dispatch (Kapitsa fuses up to four), Ferrum's per-row MLX layout, Ferrum's T-table producer
// (h_mlx_ttable, from already-normalized F32 rows) and parameter block.

// T table (h_mlx_ttable): token row t's half inputs [K] with each 64-group's inputs permuted so that slot
// 4c + 16jb + kk holds input 16c + 8(jb >> 1) + 2kk + (jb & 1) (the matrix unit's operand layout, probed by Kapitsa);
// rows m .. rows_pad - 1 are zero. sums[(t / 8) * G * 8 + g * 8 + t % 8] = the group's sum of the half inputs.
struct MlxTableArgs { uint k; uint m; uint x_stride; uint rows; };
inline uint mlx_input_of_slot(uint s) {
    const uint c = (s >> 2) & 3, jb = (s >> 4) & 3, kk = s & 3;
    return 16 * c + 8 * (jb >> 1) + 2 * kk + (jb & 1);
}
// grid (K / 64, rows); threads 64.
kernel void h_mlx_ttable(constant MlxTableArgs & a [[buffer(3)]], device const float * x [[buffer(0)]],
                         device half * table [[buffer(1)]], device float * sums [[buffer(2)]],
                         uint2 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                         ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[2];
    const uint g = tg.x, t = tg.y, groups = a.k / 64;
    const half h = t < a.m ? half(x[(ulong)t * a.x_stride + g * 64 + mlx_input_of_slot(tid)]) : 0.h;
    table[(ulong)t * a.k + g * 64 + tid] = h;
    const float s = simd_sum(float(h));
    if (lane == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) sums[((t / 8) * groups + g) * 8 + t % 8] = part[0] + part[1];
}

struct MlxQmm8tArgs {
    uint k;
    uint m;          // token rows R (live)
    uint ks;         // k-slices per threadgroup (one SIMD group each)
    uint n;
    uint y_stride;
    uint row_bytes;
    uint per;        // threadgroups per tile (or pair): ceil(n / NT)
    uint pad;
};

// qmm8t: grid (per * tiles); threads 32 * ks; tiles fastest.
template <int NT, int PF, int MAXKS, int TROWS>
inline void mlx_qmm8t_impl(constant MlxQmm8tArgs & p, device const char * w, device const half * T,
                           device const float * sums, device float * Y, threadgroup float * red,
                           uint tg, uint sg, uint lane, uint ntgs) {
    // TROWS = 8: one tile per threadgroup group; 16: a tile pair (qmm8t2).
    const uint nt = ntgs > p.per ? ntgs / p.per : 1u;
    const uint tile = tg % nt;
    tg /= nt;
    const uint gpr = p.k / 64;
    T += (ulong)tile * TROWS * p.k;
    const uint n0 = tg * NT;
    const uint gper = (gpr + p.ks - 1) / p.ks, g0 = min(sg * gper, gpr), g1 = min(g0 + gper, gpr);
    const uint c = (lane & 1) | (((lane >> 3) & 1) << 1), r = ((lane >> 1) & 3) | (((lane >> 4) & 1) << 2);
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(NT, TROWS, 64, false, true, false,
                                                               mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
    mpp::tensor_ops::matmul2d<desc, execution_simdgroup> op;
    auto Lw = op.template get_left_input_cooperative_tensor<uint8_t, half, float>();
    auto tX = tensor<device half, dextents<int32_t, 2>, tensor_inline>((device half *)T, dextents<int32_t, 2>(p.k, TROWS));
    auto Rx0 = tX.template slice<64, TROWS>(0, 0);
    auto acc = op.template get_destination_cooperative_tensor<decltype(Lw), decltype(Rx0), float>();
    FOR_UNROLL (uint16_t i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
    constexpr int NJ = NT / 8;
    device const uint * wp[NJ];
    device const ushort * sp[NJ];
    device const ushort * bp[NJ];
    FOR_UNROLL (int j = 0; j < NJ; ++j) {
        const MlxRow row = mlx_row(w, min(n0 + r + 8 * j, p.n - 1), p.row_bytes, p.k);
        wp[j] = row.codes + 2 * c;
        sp[j] = (device const ushort *)row.scales;
        bp[j] = (device const ushort *)row.biases;
    }
    // A pair's second tile (lanes with bit 3 set) reads its own sums block.
    const uint t0 = TROWS == 16 ? 4 * (lane & 1) + 8 * ((lane >> 3) & 1) : 4 * (lane & 1);
    if (TROWS == 16) {
        sums += (ulong)(2 * tile + ((lane >> 3) & 1)) * gpr * 8 + 4 * (lane & 1);
    } else {
        sums += (ulong)tile * gpr * 8 + t0;
    }
    uint2 wn[NJ];
    ushort sn[NJ], bn[NJ];
    auto load = [&](uint g) {
        FOR_UNROLL (int j = 0; j < NJ; ++j) {
            wn[j] = *(device const uint2 *)(wp[j] + g * 8);
            sn[j] = sp[j][g];
            bn[j] = bp[j][g];
        }
    };
    if (PF && g0 < g1) load(g0);
    for (uint g = g0; g < g1; ++g) {
        if (!PF) load(g);
        uint2 wv[NJ];
        ushort sc[NJ], bc[NJ];
        FOR_UNROLL (int j = 0; j < NJ; ++j) wv[j] = wn[j], sc[j] = sn[j], bc[j] = bn[j];
        if (PF && g + 1 < g1) load(g + 1);
        FOR_UNROLL (int j = 0; j < NJ; ++j) {
            FOR_UNROLL (int jb = 0; jb < 4; ++jb) {
                const uint u = ((jb >> 1 ? wv[j].y : wv[j].x) >> (4 * (jb & 1))) & 0x0F0F0F0Fu;
                const int i = (j >> 1) * 32 + jb * 8 + (j & 1) * 4;
                Lw[i] = uchar(u);
                Lw[i + 1] = uchar(u >> 8);
                Lw[i + 2] = uchar(u >> 16);
                Lw[i + 3] = uchar(u >> 24);
            }
        }
        auto Rx = tX.template slice<64, TROWS>(g * 64, 0);
        auto d = op.template get_destination_cooperative_tensor<decltype(Lw), decltype(Rx), float>();
        op.run(Lw, Rx, d);
        const float4 sx = *(device const float4 *)(sums + g * 8);
        FOR_UNROLL (int j = 0; j < NJ; ++j) {
            const float s = as_type<float>(uint(sc[j]) << 16), b = as_type<float>(uint(bc[j]) << 16);
            FOR_UNROLL (int kk = 0; kk < 4; ++kk) acc[j * 4 + kk] = fma(s, d[j * 4 + kk], fma(b, sx[kk], acc[j * 4 + kk]));
        }
    }
    // Fixed-order sum of the k-slices (SIMD group 0 adds 1, 2, ... in turn).
    if (p.ks > 1) {
        // A lane holds NT / 2 outputs either way: a pair's second tile fills the lanes an 8-row tile pads.
        constexpr uint cap = NT / 2;
        if (sg) {
            FOR_UNROLL (uint i = 0; i < cap; ++i) red[((sg - 1) * cap + i) * 32 + lane] = acc[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg) return;
        for (uint j = 1; j < p.ks; ++j) {
            FOR_UNROLL (uint i = 0; i < cap; ++i) acc[i] += red[((j - 1) * cap + i) * 32 + lane];
        }
    }
    if (TROWS == 8 && ((lane >> 3) & 1)) return;  // padding rows of an 8-row tile
    FOR_UNROLL (int j = 0; j < NJ; ++j) {
        const uint n = n0 + r + 8 * j;
        if (n >= p.n) continue;
        FOR_UNROLL (int kk = 0; kk < 4; ++kk) {
            const uint t = TROWS * tile + t0 + kk;
            if (t < p.m) Y[(ulong)t * p.y_stride + n] = acc[j * 4 + kk];
        }
    }
}
#define MLX_QMM8T(NAME, NT, PF, TROWS)                                                                         \
kernel void NAME(constant MlxQmm8tArgs & p [[buffer(4)]], device const char * w [[buffer(0)]],               \
                 device const half * T [[buffer(1)]], device const float * sums [[buffer(2)]],               \
                 device float * Y [[buffer(3)]], uint tg [[threadgroup_position_in_grid]],                    \
                 uint ntgs [[threadgroups_per_grid]], uint sg [[simdgroup_index_in_threadgroup]],             \
                 uint lane [[thread_index_in_simdgroup]]) {                                                   \
    threadgroup float red[7 * (NT / 2) * 32];                                                                 \
    mlx_qmm8t_impl<NT, PF, 8, TROWS>(p, w, T, sums, Y, red, tg, sg, lane, ntgs);                             \
}
MLX_QMM8T(h_mlx_qmm8t_nt16, 16, 1, 8)
MLX_QMM8T(h_mlx_qmm8t_nt32, 32, 1, 8)
MLX_QMM8T(h_mlx_qmm8t2_nt16, 16, 1, 16)
MLX_QMM8T(h_mlx_qmm8t2_nt32, 32, 1, 16)
