// Kernels for the Qwen3.5-family hybrid (Gated DeltaNet + gated attention) engine.
// Residual activations are F32. Quantized projections read F32 activations;
// TensorOps GEMMs dequantize weight tiles to F16 and accumulate in F32, the
// same precision as llama.cpp's Metal backend. Block layouts, dequantizers and
// the M=1 dot-product kernels follow llama.cpp's ggml-metal.metal (MIT).
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

#define QK_K 256
#define FOR_UNROLL _Pragma("clang loop unroll(full)") for

// ---------------------------------------------------------------- blocks

struct block_q4_0 { half d; uchar qs[16]; };
struct block_q8_0 { half d; char qs[32]; };
struct block_q4_K { half d; half dmin; uchar scales[12]; uchar qs[128]; };
struct block_q5_K { half d; half dmin; uchar scales[12]; uchar qh[32]; uchar qs[128]; };
struct block_q6_K { uchar ql[128]; uchar qh[64]; char scales[16]; half d; };
struct block_iq4_xs { half d; ushort scales_h; uchar scales_l[4]; uchar qs[128]; };
struct block_iq3_s { half d; uchar qs[64]; uchar qh[8]; uchar signs[32]; uchar scales[4]; };

constexpr constant static float kvalues_iq4nl_f[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
    1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f
};

constexpr constant static uint iq3s_grid[512] = {
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
};
constexpr constant static uchar kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};

static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
    return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                 : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)), uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
}

// Each dequantizer writes 16 consecutive values of one block; `il` selects
// which 16 (0..nl-1 where a block holds 16*nl values).
template <typename type4x4>
void dequantize_q4_0(device const block_q4_0 * xb, short il, thread type4x4 & reg) {
    device const ushort * qs = ((device const ushort *)xb + 1);
    const float d1 = il ? (xb->d / 16.h) : xb->d;
    const float d2 = d1 / 256.f;
    const float md = -8.h * xb->d;
    const ushort mask0 = il ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;
    float4x4 reg_f;
    for (int i = 0; i < 8; i++) {
        reg_f[i/2][2*(i%2) + 0] = d1 * (qs[i] & mask0) + md;
        reg_f[i/2][2*(i%2) + 1] = d2 * (qs[i] & mask1) + md;
    }
    reg = (type4x4) reg_f;
}

template <typename type4x4>
void dequantize_q8_0(device const block_q8_0 * xb, short il, thread type4x4 & reg) {
    device const char * qs = xb->qs;
    const float d = xb->d;
    float4x4 reg_f;
    for (int i = 0; i < 16; i++) {
        reg_f[i/4][i%4] = (qs[i + 16*il] * d);
    }
    reg = (type4x4) reg_f;
}

template <typename type4x4>
void dequantize_q4_K(device const block_q4_K * xb, short il, thread type4x4 & reg) {
    device const uchar * q = xb->qs;
    short is = (il/4) * 2;
    q = q + (il/4) * 32 + 16 * (il&1);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    const float d   = il < 2 ? xb->d : xb->d / 16.h;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];
    const ushort mask = il < 2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - ml;
    }
}

template <typename type4x4>
void dequantize_q5_K(device const block_q5_K * xb, short il, thread type4x4 & reg) {
    device const uchar * q  = xb->qs;
    device const uchar * qh = xb->qh;
    short is = (il/4) * 2;
    q  = q + 32 * (il/4) + 16 * (il&1);
    qh = qh + 16 * (il&1);
    uchar ul = 1 << (il/2);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    const float d = il < 2 ? xb->d : xb->d / 16.f;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];
    const ushort mask  = il<2 ? 0x0F : 0xF0;
    const float qh_val = il<2 ? 16.f : 256.f;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * ((q[i] & mask) + (qh[i] & ul ? qh_val : 0)) - ml;
    }
}

template <typename type4x4>
void dequantize_q6_K(device const block_q6_K * xb, short il, thread type4x4 & reg) {
    const half d_all = xb->d;
    device const ushort * ql = (device const ushort *)xb->ql;
    device const ushort * qh = (device const ushort *)xb->qh;
    device const char * scales = (device const char *)xb->scales;
    ql = ql + 32*(il/8) + 16*((il/2)&1) + 8*(il&1);
    qh = qh + 16*(il/8) + 8*(il&1);
    float sc = scales[(il%2) + 2 * ((il/2))];
    il = (il/2) & 3;
    const uint kmask1 = il>1 ? (il>2 ? 0xC0C0C0C0 : 0x30303030) : (il>0 ? 0x0C0C0C0C : 0x03030303);
    const uint kmask2 = il>1 ? 0xF0F0F0F0                       : 0x0F0F0F0F;
    const float ml = d_all * sc * 32.f;
    const float dl0 = d_all * sc;
    const float dl1 = dl0 / 256.f;
    const float dl2 = dl0 / (256.f * 256.f);
    const float dl3 = dl0 / (256.f * 256.f * 256.f);
    const uchar shr_h = il>2 ? 2 : 0;
    const uchar shl_h = il>1 ? 0 : (il>0 ? 2 : 4);
    const uchar shr_l = il>1 ? 4 : 0;
    for (int i = 0; i < 4; ++i) {
        const uint  low = (ql[2*i] | (uint)(ql[2*i+1] << 16)) & kmask2;
        const uint high = (qh[2*i] | (uint)(qh[2*i+1] << 16)) & kmask1;
        const uint q = ((high << shl_h) >> shr_h) | (low >> shr_l);
        reg[i][0] = dl0 *  ((half)(q & 0xFF))       - ml;
        reg[i][1] = dl1 * ((float)(q & 0xFF00))     - ml;
        reg[i][2] = dl2 * ((float)(q & 0xFF0000))   - ml;
        reg[i][3] = dl3 * ((float)(q & 0xFF000000)) - ml;
    }
}

template <typename type4x4>
void dequantize_iq4_xs(device const block_iq4_xs * xb, short il, thread type4x4 & reg) {
    const int ib32 = il/2;
    il = il%2;
    device const uint * q4 = (device const uint *)xb->qs + 4*ib32;
    const int ls = ((xb->scales_l[ib32/2] >> 4*(ib32%2)) & 0xf) | (((xb->scales_h >> 2*ib32) & 3) << 4);
    const float d = (float)xb->d * (ls - 32);
    uint aux32;
    thread const uchar * q8 = (thread const uchar *)&aux32;
    for (int i = 0; i < 4; ++i) {
        aux32 = (q4[i] >> 4*il) & 0x0f0f0f0f;
        reg[i][0] = d * kvalues_iq4nl_f[q8[0]];
        reg[i][1] = d * kvalues_iq4nl_f[q8[1]];
        reg[i][2] = d * kvalues_iq4nl_f[q8[2]];
        reg[i][3] = d * kvalues_iq4nl_f[q8[3]];
    }
}

template <typename type4x4>
void dequantize_iq3_s(device const block_iq3_s * xb, short il, thread type4x4 & reg) {
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    device const uchar * qs = xb->qs + 8*ib32;
    device const uchar * signs = xb->signs + 4*ib32 + 2*il;
    const uchar qh = xb->qh[ib32] >> 4*il;
    const float dl = d * (1 + 2*((xb->scales[ib32/2] >> 4*(ib32%2)) & 0xf));
    constant uchar * grid1 = (constant uchar *)(iq3s_grid + (qs[4*il+0] | ((qh << 8) & 256)));
    constant uchar * grid2 = (constant uchar *)(iq3s_grid + (qs[4*il+1] | ((qh << 7) & 256)));
    for (int i = 0; i < 4; ++i) {
        reg[0][i] = dl * grid1[i] * select(1, -1, signs[0] & kmask_iq2xs[i+0]);
        reg[1][i] = dl * grid2[i] * select(1, -1, signs[0] & kmask_iq2xs[i+4]);
    }
    grid1 = (constant uchar *)(iq3s_grid + (qs[4*il+2] | ((qh << 6) & 256)));
    grid2 = (constant uchar *)(iq3s_grid + (qs[4*il+3] | ((qh << 5) & 256)));
    for (int i = 0; i < 4; ++i) {
        reg[2][i] = dl * grid1[i] * select(1, -1, signs[1] & kmask_iq2xs[i+0]);
        reg[3][i] = dl * grid2[i] * select(1, -1, signs[1] & kmask_iq2xs[i+4]);
    }
}

template <typename type4x4>
void dequantize_f32(device const float4x4 * src, short il, thread type4x4 & reg) {
    reg = (type4x4)(*src);
}

// ---------------------------------------------------------------- projections

// y[m][n] = sum_k x[m][k] * W[n][k]; W rows are `row_bytes` apart.
struct ProjArgs {
    uint k;
    uint n;
    uint m;
    uint row_bytes;
    uint x_stride;   // elements between activation rows
    uint y_stride;   // elements between output rows
};

// M=1 kernels: grid (ceil(n / (NSG*NR0)), m); threads (32, NSG). Each SIMD
// group owns NR0 output rows; activation row tgpig.y.

template<short NR0, short NSG>
void mv_q4_K(constant ProjArgs & a, device const char * w, device const float * x, device float * y,
             uint3 tgpig, ushort tiisg, ushort sgitg) {
    constexpr ushort kmask1 = 0x3f3f;
    constexpr ushort kmask2 = 0x0f0f;
    constexpr ushort kmask3 = 0xc0c0;
    const short ix = tiisg/8;
    const short it = tiisg%8;
    const short iq = it/4;
    const short ir = it%4;
    const int nb = a.k/QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * NR0;
    const uint r1 = tgpig.y;
    device const block_q4_K * xq = (device const block_q4_K *) (w + (ulong)first_row*a.row_bytes);
    device const float * yv = x + (ulong)r1*a.x_stride;
    float yl[16];
    float yh[16];
    float sumf[NR0]={0.f};
    device const float * y4 = yv + ix * QK_K + 64 * iq + 8 * ir;
    ushort sc16[4];
    thread const uchar * sc8 = (thread const uchar *)sc16;
    const int nrows = min(int(NR0), int(a.n) - first_row);
    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
        }
        device const ushort * sc = (device const ushort *)xq[ib].scales + iq;
        device const ushort * q1 = (device const ushort *)xq[ib].qs + 16 * iq + 4 * ir;
        device const half   * dh = &xq[ib].d;
        for (short row = 0; row < NR0; row++) {
            if (row >= nrows) break;
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
            device const ushort * q2 = q1 + 32;
            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            FOR_UNROLL (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2*i + 0] * (q1[i] & 0x000F);
                acc1[1] += yl[2*i + 1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2*i + 8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2*i + 9] * (q1[i] & 0xF000);
                acc2[0] += yh[2*i + 0] * (q2[i] & 0x000F);
                acc2[1] += yh[2*i + 1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2*i + 8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2*i + 9] * (q2[i] & 0xF000);
            }
            sumf[row] += dh[0] * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
                                  (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
                                  (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                                  (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -
                         dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);
            q1 += a.row_bytes/2;
            sc += a.row_bytes/2;
            dh += a.row_bytes/2;
        }
        y4 += 4 * QK_K;
    }
    device float * out = y + (ulong)r1*a.y_stride;
    for (int row = 0; row < nrows; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) out[first_row + row] = sum_all;
    }
}

template<short NR0, short NSG>
void mv_q5_K(constant ProjArgs & a, device const char * w, device const float * x, device float * y,
             uint3 tgpig, ushort tiisg, ushort sgitg) {
    const int nb = a.k/QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * NR0;
    const uint r1 = tgpig.y;
    device const block_q5_K * xq = (device const block_q5_K *) (w + (ulong)first_row*a.row_bytes);
    device const float * yy = x + (ulong)r1*a.x_stride;
    float sumf[NR0]={0.f};
    float yl[16], yh[16];
    constexpr ushort kmask1 = 0x3f3f;
    constexpr ushort kmask2 = 0x0f0f;
    constexpr ushort kmask3 = 0xc0c0;
    const short tid = tiisg/4;
    const short ix  = tiisg%4;
    const short iq  = tid/4;
    const short ir  = tid%4;
    const short l0 = 8*ir;
    const short q_offset = 32*iq + l0;
    const short y_offset = 64*iq + l0;
    const uchar hm1 = 1u << (2*iq);
    const uchar hm2 = hm1 << 1;
    const uchar hm3 = hm1 << 4;
    const uchar hm4 = hm2 << 4;
    ushort sc16[4];
    thread const uchar * sc8 = (thread const uchar *)sc16;
    const int nrows = min(int(NR0), int(a.n) - first_row);
    device const float * y1 = yy + ix*QK_K + y_offset;
    for (int i = ix; i < nb; i += 4) {
        device const uchar * q1 = xq[i].qs + q_offset;
        device const uchar * qh = xq[i].qh + l0;
        device const half * dh = &xq[i].d;
        device const ushort * sa = (device const ushort *)xq[i].scales + iq;
        device const float * y2 = y1 + 128;
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short l = 0; l < 8; ++l) {
            yl[l+0] = y1[l+ 0]; sumy[0] += yl[l+0];
            yl[l+8] = y1[l+32]; sumy[1] += yl[l+8];
            yh[l+0] = y2[l+ 0]; sumy[2] += yh[l+0];
            yh[l+8] = y2[l+32]; sumy[3] += yh[l+8];
        }
        for (short row = 0; row < NR0; ++row) {
            if (row >= nrows) break;
            device const uchar * q2 = q1 + 64;
            sc16[0] = sa[0] & kmask1;
            sc16[1] = sa[2] & kmask1;
            sc16[2] = ((sa[4] >> 0) & kmask2) | ((sa[0] & kmask3) >> 2);
            sc16[3] = ((sa[4] >> 4) & kmask2) | ((sa[2] & kmask3) >> 2);
            float4 acc1 = {0.f};
            float4 acc2 = {0.f};
            FOR_UNROLL (short l = 0; l < 8; ++l) {
                uchar h = qh[l];
                acc1[0] += yl[l+0] * (q1[l] & 0x0F);
                acc1[1] += yl[l+8] * (q1[l] & 0xF0);
                acc1[2] += yh[l+0] * (q2[l] & 0x0F);
                acc1[3] += yh[l+8] * (q2[l] & 0xF0);
                acc2[0] += h & hm1 ? yl[l+0] : 0.f;
                acc2[1] += h & hm2 ? yl[l+8] : 0.f;
                acc2[2] += h & hm3 ? yh[l+0] : 0.f;
                acc2[3] += h & hm4 ? yh[l+8] : 0.f;
            }
            sumf[row] += dh[0] * (sc8[0] * (acc1[0]      + 16.f*acc2[0]) +
                                  sc8[1] * (acc1[1]/16.f + 16.f*acc2[1]) +
                                  sc8[4] * (acc1[2]      + 16.f*acc2[2]) +
                                  sc8[5] * (acc1[3]/16.f + 16.f*acc2[3])) -
                         dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);
            q1 += a.row_bytes;
            qh += a.row_bytes;
            dh += a.row_bytes/2;
            sa += a.row_bytes/2;
        }
        y1 += 4 * QK_K;
    }
    device float * out = y + (ulong)r1*a.y_stride;
    for (int row = 0; row < nrows; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0) out[first_row + row] = tot;
    }
}

template<short NR0, short NSG>
void mv_q6_K(constant ProjArgs & a, device const char * w, device const float * x, device float * y,
             uint3 tgpig, ushort tiisg, ushort sgitg) {
    constexpr uchar kmask1 = 0x03;
    constexpr uchar kmask2 = 0x0C;
    constexpr uchar kmask3 = 0x30;
    constexpr uchar kmask4 = 0xC0;
    const int nb = a.k/QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * NR0;
    const uint r1 = tgpig.y;
    device const block_q6_K * xq = (device const block_q6_K *) (w + (ulong)first_row*a.row_bytes);
    device const float * yy = x + (ulong)r1*a.x_stride;
    float sumf[NR0] = { 0.f };
    float yl[16];
    const short tid = tiisg/2;
    const short ix  = tiisg%2;
    const short ip  = tid/8;
    const short il  = tid%8;
    const short l0  = 4*il;
    const short is  = 8*ip + l0/16;
    const short y_offset   = 128*ip + l0;
    const short q_offset_l =  64*ip + l0;
    const short q_offset_h =  32*ip + l0;
    const int nrows = min(int(NR0), int(a.n) - first_row);
    for (int i = ix; i < nb; i += 2) {
        device const uchar * q1 = xq[i].ql + q_offset_l;
        device const uchar * q2 = q1 + 32;
        device const uchar * qh = xq[i].qh + q_offset_h;
        device const char  * sc = xq[i].scales + is;
        device const half  * dh = &xq[i].d;
        device const float * yp = yy + i * QK_K + y_offset;
        for (short l = 0; l < 4; ++l) {
            yl[4*l + 0] = yp[l +  0];
            yl[4*l + 1] = yp[l + 32];
            yl[4*l + 2] = yp[l + 64];
            yl[4*l + 3] = yp[l + 96];
        }
        for (short row = 0; row < NR0; ++row) {
            if (row >= nrows) break;
            float4 sums = {0.f, 0.f, 0.f, 0.f};
            FOR_UNROLL (short l = 0; l < 4; ++l) {
                sums[0] += yl[4*l + 0] * ((char)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4*l + 1] * ((char)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4*l + 2] * ((char)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4*l + 3] * ((char)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
            }
            sumf[row] += dh[0] * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);
            q1 += a.row_bytes;
            q2 += a.row_bytes;
            qh += a.row_bytes;
            sc += a.row_bytes;
            dh += a.row_bytes/2;
        }
    }
    device float * out = y + (ulong)r1*a.y_stride;
    for (int row = 0; row < nrows; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) out[first_row + row] = sum_all;
    }
}

template<short NR0, short NSG>
void mv_iq4_xs(constant ProjArgs & a, device const char * w, device const float * x, device float * y,
               threadgroup float * shmem_f32, uint3 tgpig, ushort tiisg, ushort sgitg) {
    const int first_row = (tgpig.x * NSG + sgitg) * NR0;
    const uint r1 = tgpig.y;
    device const block_iq4_xs * xq = (device const block_iq4_xs *) (w + (ulong)first_row*a.row_bytes);
    device const float * yv = x + (ulong)r1*a.x_stride;
    const int nb = a.k/QK_K;
    const int ns01 = a.row_bytes/sizeof(block_iq4_xs);
    const short ix = tiisg/16;
    const short it = tiisg%16;
    const short ib = it/2;
    const short il = it%2;
    if (sgitg == 0) shmem_f32[tiisg] = kvalues_iq4nl_f[tiisg%16];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float4 yl[4];
    float sumf[NR0]={0.f};
    device const float * yb = yv + ix * QK_K + ib * 32 + il * 8;
    uint aux32[2];
    thread const uchar * q8 = (thread const uchar *)aux32;
    float4 qf1, qf2;
    const int nrows = min(int(NR0), int(a.n) - first_row);
    for (int ibl = ix; ibl < nb; ibl += 2) {
        device const float4 * y4 = (device const float4 *)yb;
        yl[0] = y4[0];
        yl[1] = y4[4];
        yl[2] = y4[1];
        yl[3] = y4[5];
        for (short row = 0; row < NR0; ++row) {
            if (row >= nrows) break;
            device const block_iq4_xs & xb = xq[row*ns01 + ibl];
            device const uint * q4 = (device const uint *)(xb.qs + 16*ib + 8*il);
            float4 acc1 = {0.f}, acc2 = {0.f};
            aux32[0] = (q4[0]     ) & 0x0f0f0f0f;
            aux32[1] = (q4[0] >> 4) & 0x0f0f0f0f;
            qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
            qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
            acc1 += yl[0] * qf1;
            acc2 += yl[1] * qf2;
            aux32[0] = (q4[1]     ) & 0x0f0f0f0f;
            aux32[1] = (q4[1] >> 4) & 0x0f0f0f0f;
            qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
            qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
            acc1 += yl[2] * qf1;
            acc2 += yl[3] * qf2;
            acc1 += acc2;
            const int ls = (((xb.scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((xb.scales_h >> 2*ib) & 3) << 4)) - 32;
            sumf[row] += (float)xb.d * ls * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);
        }
        yb += 2 * QK_K;
    }
    device float * out = y + (ulong)r1*a.y_stride;
    for (int row = 0; row < nrows; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) out[first_row + row] = sum_all;
    }
}

// IQ3_S: the 512-entry grid is staged in threadgroup memory (NSG must be 2).
template<short NR0, short NSG>
void mv_iq3_s(constant ProjArgs & a, device const char * w, device const float * x, device float * y,
              threadgroup uint * svalues, uint3 tgpig, ushort tiisg, ushort sgitg) {
    const int nb = a.k/QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * NR0;
    const uint r1 = tgpig.y;
    device const block_iq3_s * xq = (device const block_iq3_s *) (w + (ulong)first_row*a.row_bytes);
    device const float * yv = x + (ulong)r1*a.x_stride;
    float yl[32];
    float sumf[NR0]={0.f};
    const int nb32 = nb * (QK_K / 32);
    {
        const int pos = (32*sgitg + tiisg)*8;
        for (int i = 0; i < 8; ++i) svalues[pos + i] = iq3s_grid[pos + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const int nrows = min(int(NR0), int(a.n) - first_row);
    const int ix = tiisg;
    device const float * y4 = yv + 32 * ix;
    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        for (short i = 0; i < 32; ++i) yl[i] = y4[i];
        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);
        device const block_iq3_s * xr = xq + ibl;
        device const uchar * qs = xr->qs + 8 * ib;
        device const uchar * qh = xr->qh + ib;
        device const uchar * sc = xr->scales + (ib/2);
        device const uchar * signs = xr->signs + 4 * ib;
        device const half * dh = &xr->d;
        for (short row = 0; row < NR0; row++) {
            if (row >= nrows) break;
            const float db = dh[0];
            const float d = db * (1 + 2*((sc[0] >> 4*(ib%2)) & 0xf));
            float2 sum = {0};
            for (short l = 0; l < 4; ++l) {
                const threadgroup uint * table1 = qh[0] & kmask_iq2xs[2*l+0] ? svalues + 256 : svalues;
                const threadgroup uint * table2 = qh[0] & kmask_iq2xs[2*l+1] ? svalues + 256 : svalues;
                const threadgroup uchar * grid1 = (const threadgroup uchar *)(table1 + qs[2*l+0]);
                const threadgroup uchar * grid2 = (const threadgroup uchar *)(table2 + qs[2*l+1]);
                for (short j = 0; j < 4; ++j) {
                    sum[0] += yl[8*l + j + 0] * grid1[j] * select(1, -1, signs[l] & kmask_iq2xs[j+0]);
                    sum[1] += yl[8*l + j + 4] * grid2[j] * select(1, -1, signs[l] & kmask_iq2xs[j+4]);
                }
            }
            sumf[row] += d * (sum[0] + sum[1]);
            dh    += a.row_bytes/2;
            qs    += a.row_bytes;
            qh    += a.row_bytes;
            sc    += a.row_bytes;
            signs += a.row_bytes;
        }
        y4 += 32 * 32;
    }
    device float * out = y + (ulong)r1*a.y_stride;
    for (int row = 0; row < nrows; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) out[first_row + row] = sum_all;
    }
}

// Q8_0: NSG SIMD groups split K; one threadgroup owns NR0 rows.
// grid (ceil(n/NR0), m); threads (32, NSG); 32*NR0 floats of threadgroup memory.
template<short NR0, short NSG>
void mv_q8_0(constant ProjArgs & a, device const char * w, device const float * x, device float * y,
             threadgroup float * shmem, uint3 tgpig, ushort tiisg, ushort sgitg) {
    constexpr short NQ = 8;
    const int nb = a.k/32;
    const int r0 = tgpig.x*NR0;
    const uint r1 = tgpig.y;
    device const float * yv = x + (ulong)r1*a.x_stride;
    device const block_q8_0 * ax[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const int rr = min(r0 + row, int(a.n) - 1);
        ax[row] = (device const block_q8_0 *) (w + (ulong)rr*a.row_bytes);
    }
    float sumf[NR0] = { 0.f };
    const short ix = tiisg/(32/NQ);
    const short il = tiisg%(32/NQ);
    const int ib0 = sgitg*NQ + ix;
    float yl[NQ];
    device const float * yb = yv + ib0*32 + il*NQ;
    for (int ib = ib0; ib < nb; ib += NSG*NQ) {
        for (short i = 0; i < NQ; ++i) yl[i] = yb[i];
        for (short row = 0; row < NR0; row++) {
            device const char * qs = ax[row][ib].qs + il*NQ;
            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NQ; ++i) sumq += qs[i] * yl[i];
            sumf[row] += sumq*ax[row][ib].d;
        }
        yb += NSG*NQ*32;
    }
    for (short row = 0; row < NR0; ++row) {
        if (sgitg == 0) shmem[32*row + tiisg] = 0.f;
        sumf[row] = simd_sum(sumf[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short row = 0; row < NR0; ++row) {
        if (tiisg == 0) shmem[32*row + sgitg] = sumf[row];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float * out = y + (ulong)r1*a.y_stride;
    for (short row = 0; row < NR0 && r0 + row < int(a.n); ++row) {
        float tot = simd_sum(shmem[32*row + tiisg]);
        if (tiisg == 0 && sgitg == 0) out[r0 + row] = tot;
    }
}

// Q4_0 (MTP-free trunk never uses it; kept for small auxiliary tensors).
template<short NR0, short NSG>
void mv_q4_0(constant ProjArgs & a, device const char * w, device const float * x, device float * y,
             uint3 tgpig, ushort tiisg, ushort sgitg) {
    const int nb = a.k/32;
    const int first_row = (tgpig.x * NSG + sgitg) * NR0;
    const uint r1 = tgpig.y;
    device const float * yv = x + (ulong)r1*a.x_stride;
    float sumf[NR0] = {0.f};
    const int nrows = min(int(NR0), int(a.n) - first_row);
    for (int ib = tiisg; ib < nb; ib += 32) {
        float yl[32];
        for (int i = 0; i < 32; i++) yl[i] = yv[ib*32 + i];
        for (short row = 0; row < NR0; row++) {
            if (row >= nrows) break;
            device const block_q4_0 * b = (device const block_q4_0 *)(w + (ulong)(first_row + row)*a.row_bytes) + ib;
            float s = 0.f;
            for (int j = 0; j < 16; j++) {
                s += yl[j] * (float(b->qs[j] & 15) - 8.f) + yl[j+16] * (float(b->qs[j] >> 4) - 8.f);
            }
            sumf[row] += s * float(b->d);
        }
    }
    device float * out = y + (ulong)r1*a.y_stride;
    for (int row = 0; row < nrows; ++row) {
        float t = simd_sum(sumf[row]);
        if (tiisg == 0) out[first_row + row] = t;
    }
}

// Dense F32 weights (small recurrent-gate projections).
template<short NR0, short NSG>
void mv_f32(constant ProjArgs & a, device const char * w, device const float * x, device float * y,
            uint3 tgpig, ushort tiisg, ushort sgitg) {
    const int first_row = (tgpig.x * NSG + sgitg) * NR0;
    const uint r1 = tgpig.y;
    device const float4 * yv = (device const float4 *)(x + (ulong)r1*a.x_stride);
    const int nrows = min(int(NR0), int(a.n) - first_row);
    float sumf[NR0] = {0.f};
    for (uint i = tiisg; i < a.k/4; i += 32) {
        float4 xv = yv[i];
        for (short row = 0; row < NR0; row++) {
            if (row >= nrows) break;
            device const float4 * wr = (device const float4 *)(w + (ulong)(first_row + row)*a.row_bytes);
            sumf[row] += dot(xv, wr[i]);
        }
    }
    device float * out = y + (ulong)r1*a.y_stride;
    for (int row = 0; row < nrows; ++row) {
        float t = simd_sum(sumf[row]);
        if (tiisg == 0) out[first_row + row] = t;
    }
}

#define MV_KERNEL(NAME, IMPL, NR0, NSG) \
kernel void NAME(constant ProjArgs & a [[buffer(3)]], device const char * w [[buffer(0)]], \
                 device const float * x [[buffer(1)]], device float * y [[buffer(2)]], \
                 uint3 tgpig [[threadgroup_position_in_grid]], ushort tiisg [[thread_index_in_simdgroup]], \
                 ushort sgitg [[simdgroup_index_in_threadgroup]]) { \
    IMPL<NR0, NSG>(a, w, x, y, tgpig, tiisg, sgitg); \
}
MV_KERNEL(h_mv_q4_k, mv_q4_K, 2, 2)
MV_KERNEL(h_mv_q4_k_r1s2, mv_q4_K, 1, 2)
MV_KERNEL(h_mv_q4_k_r1s4, mv_q4_K, 1, 4)
MV_KERNEL(h_mv_q4_k_r2s4, mv_q4_K, 2, 4)
MV_KERNEL(h_mv_q4_k_r4s2, mv_q4_K, 4, 2)
MV_KERNEL(h_mv_q4_k_r4s1, mv_q4_K, 4, 1)
MV_KERNEL(h_mv_q6_k_r1s2, mv_q6_K, 1, 2)
MV_KERNEL(h_mv_q6_k_r1s4, mv_q6_K, 1, 4)
MV_KERNEL(h_mv_q6_k_r2s4, mv_q6_K, 2, 4)
MV_KERNEL(h_mv_q6_k_r4s2, mv_q6_K, 4, 2)
MV_KERNEL(h_mv_q5_k, mv_q5_K, 1, 2)
MV_KERNEL(h_mv_q6_k, mv_q6_K, 2, 2)
MV_KERNEL(h_mv_q4_0, mv_q4_0, 4, 2)
MV_KERNEL(h_mv_f32, mv_f32, 4, 2)

kernel void h_mv_q8_0(constant ProjArgs & a [[buffer(3)]], device const char * w [[buffer(0)]],
                      device const float * x [[buffer(1)]], device float * y [[buffer(2)]],
                      threadgroup float * shmem [[threadgroup(0)]],
                      uint3 tgpig [[threadgroup_position_in_grid]], ushort tiisg [[thread_index_in_simdgroup]],
                      ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    mv_q8_0<2, 4>(a, w, x, y, shmem, tgpig, tiisg, sgitg);
}
kernel void h_mv_iq3_s(constant ProjArgs & a [[buffer(3)]], device const char * w [[buffer(0)]],
                       device const float * x [[buffer(1)]], device float * y [[buffer(2)]],
                       threadgroup uint * shmem [[threadgroup(0)]],
                       uint3 tgpig [[threadgroup_position_in_grid]], ushort tiisg [[thread_index_in_simdgroup]],
                       ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    mv_iq3_s<4, 2>(a, w, x, y, shmem, tgpig, tiisg, sgitg);
}
kernel void h_mv_iq4_xs(constant ProjArgs & a [[buffer(3)]], device const char * w [[buffer(0)]],
                        device const float * x [[buffer(1)]], device float * y [[buffer(2)]],
                        threadgroup float * shmem [[threadgroup(0)]],
                        uint3 tgpig [[threadgroup_position_in_grid]], ushort tiisg [[thread_index_in_simdgroup]],
                        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    mv_iq4_xs<2, 2>(a, w, x, y, shmem, tgpig, tiisg, sgitg);
}

// TensorOps GEMM after llama.cpp's kernel_mul_mm (tensor path): a 64-row
// weight tile is dequantized to F16 in threadgroup memory; 128 activation
// rows are read as F32 directly from device memory; accumulation is F32.
// grid (ceil(m/128), ceil(n/64)); threads 128; 64*32*2 bytes threadgroup memory.
constant constexpr int MM_NRA = 64;   // weight rows per tile
constant constexpr int MM_NRB = 128;  // activation rows per tile
constant constexpr int MM_NK = 32;    // K per step (two 16-value chunks)

template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread half4x4 &)>
void mm_impl(constant ProjArgs & a, device const char * w, device const float * x, device float * y,
             threadgroup half * sa, uint3 tgpig, ushort tiitg) {
    const int K = a.k;
    const int M = a.n;      // output features
    const int N = a.m;      // activation rows
    const int ra = tgpig.y * MM_NRA;
    const int rb = tgpig.x * MM_NRB;
    auto tA = tensor(sa, dextents<int32_t, 2>(MM_NK, MM_NRA));
    auto tB = tensor((device float *)x, dextents<int32_t, 2>(K, N), array<int, 2>({1, int(a.x_stride)}));
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            MM_NRB, MM_NRA, MM_NK, false, true, true,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;
    auto cT = mm.get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();
    for (int loop_k = 0; loop_k < K; loop_k += MM_NK) {
        for (int work = tiitg; work < MM_NRA * 2; work += 128) {
            const int row = work / 2;
            const int k_chunk = work % 2;
            const int k_pos = loop_k + k_chunk * 16;
            if (ra + row < M) {
                const int block_idx = k_pos / (16 * nl);
                const short il = (k_pos / 16) % nl;
                device const block_q * row_ptr = (device const block_q *)(w + (ulong)a.row_bytes * (ra + row));
                half4x4 temp_a;
                dequantize_func(row_ptr + block_idx, il, temp_a);
                FOR_UNROLL (short i = 0; i < 16; i++) sa[row * MM_NK + k_chunk*16 + i] = temp_a[i/4][i%4];
            } else {
                FOR_UNROLL (short i = 0; i < 16; i++) sa[row * MM_NK + k_chunk*16 + i] = 0.h;
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

#define MM_KERNEL(NAME, BLOCK, NL, DEQ) \
kernel void NAME(constant ProjArgs & a [[buffer(3)]], device const char * w [[buffer(0)]], \
                 device const float * x [[buffer(1)]], device float * y [[buffer(2)]], \
                 threadgroup half * sa [[threadgroup(0)]], \
                 uint3 tgpig [[threadgroup_position_in_grid]], ushort tiitg [[thread_index_in_threadgroup]]) { \
    mm_impl<BLOCK, NL, DEQ<half4x4>>(a, w, x, y, sa, tgpig, tiitg); \
}
MM_KERNEL(h_mm_q4_k, block_q4_K, 16, dequantize_q4_K)
MM_KERNEL(h_mm_q5_k, block_q5_K, 16, dequantize_q5_K)
MM_KERNEL(h_mm_q6_k, block_q6_K, 16, dequantize_q6_K)
MM_KERNEL(h_mm_q8_0, block_q8_0, 2, dequantize_q8_0)
MM_KERNEL(h_mm_q4_0, block_q4_0, 2, dequantize_q4_0)
MM_KERNEL(h_mm_iq4_xs, block_iq4_xs, 16, dequantize_iq4_xs)
MM_KERNEL(h_mm_iq3_s, block_iq3_s, 16, dequantize_iq3_s)
MM_KERNEL(h_mm_f32, float4x4, 1, dequantize_f32)

// ---------------------------------------------------------------- embedding

struct GetRowsArgs { uint k; uint row_bytes; uint tokens; };

// grid (ceil(k/16/32), tokens); threads 32. Each thread dequantizes 16 values.
template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread float4x4 &)>
void get_rows_impl(constant GetRowsArgs & a, device const char * w, device const uint * ids, device float * y,
                   uint3 tgpig, ushort tiisg) {
    const uint chunk = tgpig.x * 32 + tiisg;
    if (chunk * 16 >= a.k) return;
    const uint token = tgpig.y;
    const uint id = ids[token];
    device const block_q * row = (device const block_q *)(w + (ulong)id * a.row_bytes);
    float4x4 v;
    dequantize_func(row + chunk / nl, chunk % nl, v);
    device float * out = y + (ulong)token * a.k + chunk * 16;
    for (short i = 0; i < 16; i++) out[i] = v[i/4][i%4];
}
#define GET_ROWS_KERNEL(NAME, BLOCK, NL, DEQ) \
kernel void NAME(constant GetRowsArgs & a [[buffer(3)]], device const char * w [[buffer(0)]], \
                 device const uint * ids [[buffer(1)]], device float * y [[buffer(2)]], \
                 uint3 tgpig [[threadgroup_position_in_grid]], ushort tiisg [[thread_index_in_simdgroup]]) { \
    get_rows_impl<BLOCK, NL, DEQ<float4x4>>(a, w, ids, y, tgpig, tiisg); \
}
GET_ROWS_KERNEL(h_get_rows_q4_k, block_q4_K, 16, dequantize_q4_K)
GET_ROWS_KERNEL(h_get_rows_q5_k, block_q5_K, 16, dequantize_q5_K)
GET_ROWS_KERNEL(h_get_rows_q6_k, block_q6_K, 16, dequantize_q6_K)
GET_ROWS_KERNEL(h_get_rows_q8_0, block_q8_0, 2, dequantize_q8_0)
GET_ROWS_KERNEL(h_get_rows_q4_0, block_q4_0, 2, dequantize_q4_0)
GET_ROWS_KERNEL(h_get_rows_iq4_xs, block_iq4_xs, 16, dequantize_iq4_xs)

// ---------------------------------------------------------------- norms and elementwise

inline float block_sum(float v, threadgroup float * scratch, ushort tid, ushort threads) {
    v = simd_sum(v);
    if (tid % 32 == 0) scratch[tid / 32] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.f;
    for (ushort i = 0; i < threads / 32; i++) total += scratch[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

struct NormArgs { uint n; float eps; uint rows; };

// y = rmsnorm(x) * w, one threadgroup (256 threads) per row.
kernel void h_rmsnorm(constant NormArgs & a [[buffer(3)]], device const float * x [[buffer(0)]],
                      device const float * w [[buffer(1)]], device float * y [[buffer(2)]],
                      uint row [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float scratch[8];
    device const float * xr = x + (ulong)row * a.n;
    float ss = 0.f;
    for (uint i = tid; i < a.n; i += 256) ss += xr[i] * xr[i];
    ss = block_sum(ss, scratch, tid, 256);
    const float scale = rsqrt(ss / float(a.n) + a.eps);
    device float * yr = y + (ulong)row * a.n;
    for (uint i = tid; i < a.n; i += 256) yr[i] = xr[i] * scale * w[i];
}

// h = x + r (written to h), y = rmsnorm(h) * w.
kernel void h_add_rmsnorm(constant NormArgs & a [[buffer(5)]], device const float * x [[buffer(0)]],
                          device const float * r [[buffer(1)]], device const float * w [[buffer(2)]],
                          device float * h [[buffer(3)]], device float * y [[buffer(4)]],
                          uint row [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float scratch[8];
    device const float * xr = x + (ulong)row * a.n;
    device const float * rr = r + (ulong)row * a.n;
    device float * hr = h + (ulong)row * a.n;
    float ss = 0.f;
    for (uint i = tid; i < a.n; i += 256) {
        float v = xr[i] + rr[i];
        hr[i] = v;
        ss += v * v;
    }
    ss = block_sum(ss, scratch, tid, 256);
    const float scale = rsqrt(ss / float(a.n) + a.eps);
    device float * yr = y + (ulong)row * a.n;
    for (uint i = tid; i < a.n; i += 256) yr[i] = hr[i] * scale * w[i];
}

struct EltArgs { uint n; };

kernel void h_add(constant EltArgs & a [[buffer(3)]], device const float * x [[buffer(0)]],
                  device const float * r [[buffer(1)]], device float * y [[buffer(2)]],
                  uint i [[thread_position_in_grid]]) {
    if (i < a.n) y[i] = x[i] + r[i];
}

// out = silu(gate) * up
kernel void h_swiglu(constant EltArgs & a [[buffer(3)]], device const float * g [[buffer(0)]],
                     device const float * u [[buffer(1)]], device float * y [[buffer(2)]],
                     uint i [[thread_position_in_grid]]) {
    if (i < a.n) {
        float x = g[i];
        y[i] = x / (1.f + exp(-x)) * u[i];
    }
}

// ---------------------------------------------------------------- Gated DeltaNet

struct GdnGateArgs { uint heads; uint tokens; };

// alpha, beta: [tokens][heads] projections. g = softplus(alpha + dt_bias) * a,
// beta' = sigmoid(beta). One thread per (token, head).
kernel void h_gdn_gates(constant GdnGateArgs & p [[buffer(6)]],
                        device const float * alpha [[buffer(0)]], device const float * beta [[buffer(1)]],
                        device const float * dt_bias [[buffer(2)]], device const float * a_log [[buffer(3)]],
                        device float * g [[buffer(4)]], device float * b [[buffer(5)]],
                        uint i [[thread_position_in_grid]]) {
    if (i >= p.heads * p.tokens) return;
    const uint h = i % p.heads;
    const float z = alpha[i] + dt_bias[h];
    const float sp = z > 20.f ? z : log(1.f + exp(z));
    g[i] = sp * a_log[h];
    b[i] = 1.f / (1.f + exp(-beta[i]));
}

struct ConvArgs {
    uint channels;     // conv channels (2*key_dim + value_dim)
    uint tokens;
    uint key_dim;      // channels [0, 2*key_dim) are Q then K and are L2-normalized per head
    uint head_dim;     // 128
    float eps;
};

// Causal depthwise conv (kernel 4) + SiLU + per-head L2 norm of Q and K.
// One threadgroup of head_dim threads per channel group; loops over tokens in
// order so the rolling 3-row state stays in registers. `state` [3][channels]
// is read and rewritten in place.
kernel void h_gdn_conv(constant ConvArgs & p [[buffer(4)]],
                       device const float * x [[buffer(0)]],        // [tokens][channels]
                       device const float * w [[buffer(1)]],        // [channels][4]
                       device float * y [[buffer(2)]],              // [tokens][channels]
                       device float * state [[buffer(3)]],          // [3][channels]
                       uint group [[threadgroup_position_in_grid]],
                       ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float scratch[8];
    const uint c = group * p.head_dim + tid;
    const bool normalize = c < 2 * p.key_dim;
    float4 wc = ((device const float4 *)w)[c];
    float s0 = state[c], s1 = state[p.channels + c], s2 = state[2 * p.channels + c];
    for (uint t = 0; t < p.tokens; t++) {
        const float xt = x[(ulong)t * p.channels + c];
        float v = wc[0] * s0 + wc[1] * s1 + wc[2] * s2 + wc[3] * xt;
        v = v / (1.f + exp(-v));
        s0 = s1; s1 = s2; s2 = xt;
        if (normalize) {
            // All threads of the group take the same branch (group never straddles Q/K/V).
            float ss = block_sum(v * v, scratch, tid, p.head_dim);
            v = v * rsqrt(max(ss, p.eps * p.eps));
        }
        y[(ulong)t * p.channels + c] = v;
    }
    state[c] = s0;
    state[p.channels + c] = s1;
    state[2 * p.channels + c] = s2;
}

struct DeltaArgs {
    uint tokens;
    uint v_heads;      // 48
    uint k_heads;      // 16
    uint channels;     // row stride of the conv output
    uint key_dim;      // offset of K within a row (Q at 0, V at 2*key_dim)
    float scale;       // 1/sqrt(head_dim)
};

// Sequential gated delta rule after llama.cpp's kernel_gated_delta_net_impl.
// State per V head is stored transposed: M[i][j] = S[j][i], row i contiguous.
// grid (128/NSG, v_heads); threads (32, NSG). Each SIMD group owns one row i
// (a V channel); each lane owns NSG consecutive K entries of that row.
template<short NSG>
void gdn_recurrent_impl(constant DeltaArgs & p, device const float * qkv, device const float * g,
                        device const float * b, device float * out, device float * state,
                        uint3 tgpig, uint3 tpitg) {
    constexpr uint S = 128;
    const uint tx = tpitg.x;
    const uint ty = tpitg.y;
    const uint h = tgpig.y;
    const uint i = tgpig.x * NSG + ty;
    const uint kh = h % p.k_heads;
    device float * s_ptr = state + ((ulong)h * S + i) * S;
    float ls[NSG];
    FOR_UNROLL (short j = 0; j < NSG; j++) ls[j] = s_ptr[tx * NSG + j];
    for (uint t = 0; t < p.tokens; t++) {
        device const float * row = qkv + (ulong)t * p.channels;
        device const float * q = row + kh * S;
        device const float * k = row + p.key_dim + kh * S;
        const float v = row[2 * p.key_dim + h * S + i];
        const float ge = exp(g[t * p.v_heads + h]);
        const float beta = b[t * p.v_heads + h];
        float s_k = 0.f;
        FOR_UNROLL (short j = 0; j < NSG; j++) {
            ls[j] *= ge;
            s_k += ls[j] * k[tx * NSG + j];
        }
        s_k = simd_sum(s_k);
        const float d = (v - s_k) * beta;
        float yv = 0.f;
        FOR_UNROLL (short j = 0; j < NSG; j++) {
            ls[j] += k[tx * NSG + j] * d;
            yv += ls[j] * q[tx * NSG + j];
        }
        yv = simd_sum(yv);
        if (tx == 0) out[((ulong)t * p.v_heads + h) * S + i] = yv * p.scale;
    }
    FOR_UNROLL (short j = 0; j < NSG; j++) s_ptr[tx * NSG + j] = ls[j];
}

kernel void h_gdn_recurrent(constant DeltaArgs & p [[buffer(5)]],
                            device const float * qkv [[buffer(0)]], device const float * g [[buffer(1)]],
                            device const float * b [[buffer(2)]], device float * out [[buffer(3)]],
                            device float * state [[buffer(4)]],
                            uint3 tgpig [[threadgroup_position_in_grid]], uint3 tpitg [[thread_position_in_threadgroup]]) {
    gdn_recurrent_impl<4>(p, qkv, g, b, out, state, tgpig, tpitg);
}

struct GatedNormArgs { uint heads; uint head_dim; float eps; };

// y = rmsnorm(o) * w * silu(z) per (token, head); one threadgroup of head_dim threads.
kernel void h_gated_rmsnorm(constant GatedNormArgs & p [[buffer(4)]],
                            device const float * o [[buffer(0)]], device const float * z [[buffer(1)]],
                            device const float * w [[buffer(2)]], device float * y [[buffer(3)]],
                            uint row [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float scratch[8];
    const ulong idx = (ulong)row * p.head_dim + tid;
    const float v = o[idx];
    const float ss = block_sum(v * v, scratch, tid, p.head_dim);
    const float scale = rsqrt(ss / float(p.head_dim) + p.eps);
    const float zv = z[idx];
    y[idx] = v * scale * w[tid] * (zv / (1.f + exp(-zv)));
}

// ---------------------------------------------------------------- gated attention

struct AttnPrepArgs {
    uint tokens;
    uint heads;        // query heads
    uint kv_heads;
    uint head_dim;     // 256
    uint rope_dims;    // 64
    uint position;     // absolute position of token 0
    float theta;
    float eps;
    uint cache_stride; // elements between cached positions (kv_heads*head_dim)
    float q_scale;     // softmax scale folded into the F16 query
};

// Query heads: rmsnorm + partial NEOX RoPE of q (first half of each [q|gate]
// pair in qg) into q_out [tokens][heads][head_dim]. KV heads: rmsnorm + RoPE of
// k and a copy of v into the F16 cache at `position + token`.
// grid (heads + kv_heads, tokens); threads head_dim.
kernel void h_attn_prep(constant AttnPrepArgs & p [[buffer(7)]],
                        device const float * qg [[buffer(0)]], device const float * k [[buffer(1)]],
                        device const float * v [[buffer(2)]], device const float * q_norm [[buffer(3)]],
                        device const float * k_norm [[buffer(4)]],
                        device half * q_out [[buffer(5)]], device half * kv [[buffer(6)]],
                        uint2 group [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float scratch[8];
    threadgroup float normed[256];
    const uint head = group.x;
    const uint t = group.y;
    const bool is_q = head < p.heads;
    const uint kvh = head - p.heads;
    const float x = is_q ? qg[((ulong)t * p.heads + head) * 2 * p.head_dim + tid]
                         : k[((ulong)t * p.kv_heads + kvh) * p.head_dim + tid];
    const float ss = block_sum(x * x, scratch, tid, p.head_dim);
    const float scale = rsqrt(ss / float(p.head_dim) + p.eps);
    const float nv = x * scale * (is_q ? q_norm[tid] : k_norm[tid]);
    normed[tid] = nv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float outv = nv;
    const uint half_rot = p.rope_dims / 2;
    if (tid < p.rope_dims) {
        const uint pair = tid % half_rot;
        const float pos = float(p.position + t);
        const float freq = pow(p.theta, -2.f * float(pair) / float(p.rope_dims));
        const float angle = pos * freq;
        const float c = cos(angle), s = sin(angle);
        if (tid < half_rot) outv = nv * c - normed[tid + half_rot] * s;
        else outv = normed[tid - half_rot] * s + nv * c;
    }
    if (is_q) {
        q_out[((ulong)t * p.heads + head) * p.head_dim + tid] = half(outv * p.q_scale);
    } else {
        device half * kc = kv + (ulong)(p.position + t) * p.cache_stride + kvh * p.head_dim;
        kc[tid] = half(outv);
    }
}

// V is stored in its own cache buffer with the same geometry.
kernel void h_store_v(constant AttnPrepArgs & p [[buffer(2)]], device const float * v [[buffer(0)]],
                      device half * vc [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    const uint row = p.kv_heads * p.head_dim;
    if (i >= p.tokens * row) return;
    const uint t = i / row, r = i % row;
    vc[(ulong)(p.position + t) * p.cache_stride + r] = half(v[i]);
}

struct AttnArgs {
    uint tokens;       // query rows
    uint heads;
    uint kv_heads;
    uint head_dim;     // 256
    uint position;     // absolute position of query 0; query t attends to [0, position + t]
    uint cache_stride;
    float scale;
};

// Reference-grade attention: one threadgroup (head_dim threads) per (query, head),
// online softmax over keys. Output is multiplied by sigmoid(gate) from qg.
kernel void h_attention(constant AttnArgs & p [[buffer(5)]],
                        device const half * q [[buffer(0)]], device const half * kc [[buffer(1)]],
                        device const half * vc [[buffer(2)]], device const float * qg [[buffer(3)]],
                        device float * out [[buffer(4)]],
                        uint2 group [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float scratch[8];
    const uint h = group.x, t = group.y;
    const uint kvh = h / (p.heads / p.kv_heads);
    const float qv = float(q[((ulong)t * p.heads + h) * p.head_dim + tid]);
    const uint keys = p.position + t + 1;
    float m = -INFINITY, l = 0.f, acc = 0.f;
    for (uint j = 0; j < keys; j++) {
        const ulong base = (ulong)j * p.cache_stride + kvh * p.head_dim + tid;
        const float s = block_sum(qv * float(kc[base]), scratch, tid, p.head_dim);
        const float m_new = max(m, s);
        const float corr = exp(m - m_new);
        const float e = exp(s - m_new);
        l = l * corr + e;
        acc = acc * corr + e * float(vc[base]);
        m = m_new;
    }
    const float gate = qg[((ulong)t * p.heads + h) * 2 * p.head_dim + p.head_dim + tid];
    out[((ulong)t * p.heads + h) * p.head_dim + tid] = acc / l / (1.f + exp(-gate));
}

// ---------------------------------------------------------------- sampling

struct ArgmaxArgs { uint n; };

// One threadgroup of 1024 threads; writes the index of the (first) maximum.
kernel void h_argmax(constant ArgmaxArgs & p [[buffer(2)]], device const float * x [[buffer(0)]],
                     device uint * out [[buffer(1)]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float best_v[32];
    threadgroup uint best_i[32];
    float bv = -INFINITY;
    uint bi = 0;
    for (uint i = tid; i < p.n; i += 1024) {
        const float v = x[i];
        if (v > bv) { bv = v; bi = i; }
    }
    for (ushort off = 16; off > 0; off /= 2) {
        const float ov = simd_shuffle_down(bv, off);
        const uint oi = simd_shuffle_down(bi, off);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if (tid % 32 == 0) { best_v[tid / 32] = bv; best_i[tid / 32] = bi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        bv = best_v[tid]; bi = best_i[tid];
        for (ushort off = 16; off > 0; off /= 2) {
            const float ov = simd_shuffle_down(bv, off);
            const uint oi = simd_shuffle_down(bi, off);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (tid == 0) out[0] = bi;
    }
}

kernel void h_zero(constant EltArgs & a [[buffer(1)]], device float * y [[buffer(0)]],
                   uint i [[thread_position_in_grid]]) {
    if (i < a.n) y[i] = 0.f;
}

// ---------------------------------------------------------------- mixture of experts

struct MoeRouteArgs { uint experts; uint used; uint hidden; uint tokens; };

// Softmax over expert logits, top-k selection (ties to the lower index),
// renormalized weights, and the sigmoid shared-expert gate dot(x, gate_w).
// One threadgroup of `experts` threads per token.
kernel void h_moe_route(constant MoeRouteArgs & p [[buffer(6)]],
                        device const float * logits [[buffer(0)]], device const float * x [[buffer(1)]],
                        device const float * shared_w [[buffer(2)]],
                        device uint * ids [[buffer(3)]], device float * weights [[buffer(4)]],
                        device float * shared_gate [[buffer(5)]],
                        uint t [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                        ushort threads [[threads_per_threadgroup]]) {
    threadgroup float scratch[32];
    threadgroup float best_v[32];
    threadgroup uint best_i[32];
    const float l = logits[(ulong)t * p.experts + tid];
    // max
    float mx = simd_max(l);
    if (tid % 32 == 0) scratch[tid / 32] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx = -INFINITY;
    for (ushort i = 0; i < threads / 32; i++) mx = max(mx, scratch[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float e = exp(l - mx);
    const float total = block_sum(e, scratch, tid, threads);
    float prob = e / total;
    float picked_sum = 0.f;
    for (uint k = 0; k < p.used; k++) {
        float bv = prob;
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
        picked_sum += bv;
        if (tid == 0) {
            ids[(ulong)t * p.used + k] = bi;
            weights[(ulong)t * p.used + k] = bv;
        }
        if (tid == bi) prob = -1.f;
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (tid < p.used) weights[(ulong)t * p.used + tid] /= picked_sum;
    // Shared-expert gate.
    float dot = 0.f;
    for (uint i = tid; i < p.hidden; i += threads) dot += x[(ulong)t * p.hidden + i] * shared_w[i];
    dot = block_sum(dot, scratch, tid, threads);
    if (tid == 0) shared_gate[t] = 1.f / (1.f + exp(-dot));
}

struct MoeMapArgs { uint experts; uint routes; };

// Deterministic expert-major ordering of routes. One threadgroup, one thread
// per expert: offsets[e]..offsets[e+1] are expert e's rows in sorted order,
// sorted[p] is the route at row p and position[r] the row of route r.
kernel void h_moe_map(constant MoeMapArgs & p [[buffer(4)]], device const uint * ids [[buffer(0)]],
                      device uint * offsets [[buffer(1)]], device uint * sorted [[buffer(2)]],
                      device uint * position [[buffer(3)]],
                      ushort tid [[thread_index_in_threadgroup]], ushort threads [[threads_per_threadgroup]]) {
    threadgroup uint counts[1024];
    for (uint e = tid; e < p.experts; e += threads) {
        uint n = 0;
        for (uint r = 0; r < p.routes; r++) n += ids[r] == e ? 1u : 0u;
        counts[e] = n;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint acc = 0;
        for (uint e = 0; e < p.experts; e++) {
            const uint n = counts[e];
            counts[e] = acc;
            offsets[e] = acc;
            acc += n;
        }
        offsets[p.experts] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < p.experts; e += threads) {
        uint at = counts[e];
        for (uint r = 0; r < p.routes; r++) {
            if (ids[r] == e) {
                sorted[at] = r;
                position[r] = at;
                at++;
            }
        }
    }
}

struct MoeGatherArgs { uint hidden; uint used; uint routes; };

// xg[row] = x[token of sorted[row]]; grid (ceil(hidden/256), routes).
kernel void h_moe_gather(constant MoeGatherArgs & p [[buffer(3)]], device const float * x [[buffer(0)]],
                         device const uint * sorted [[buffer(1)]], device float * xg [[buffer(2)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.hidden) return;
    const uint token = sorted[gid.y] / p.used;
    xg[(ulong)gid.y * p.hidden + gid.x] = x[(ulong)token * p.hidden + gid.x];
}

struct MoeCombineArgs { uint hidden; uint used; uint tokens; uint identity; };

// out[t] = sum_s w[t][s] * down[row(t,s)] + shared_gate[t] * shared[t].
kernel void h_moe_combine(constant MoeCombineArgs & p [[buffer(6)]],
                          device const float * down [[buffer(0)]], device const float * weights [[buffer(1)]],
                          device const uint * position [[buffer(2)]], device const float * shared [[buffer(3)]],
                          device const float * shared_gate [[buffer(4)]], device float * out [[buffer(5)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= p.tokens * p.hidden) return;
    const uint t = gid / p.hidden, i = gid % p.hidden;
    float acc = 0.f;
    for (uint s = 0; s < p.used; s++) {
        const uint r = t * p.used + s;
        const uint row = p.identity ? r : position[r];
        acc += weights[r] * down[(ulong)row * p.hidden + i];
    }
    out[gid] = acc + shared_gate[t] * shared[gid];
}

// Expert GEMV: route r (grid y) multiplies activation row r / x_div by the
// rows of expert ids[r]; output row r.
struct MoeMvArgs { ProjArgs p; uint expert_rows; uint x_div; };

#define MV_ID_KERNEL(NAME, IMPL, NR0, NSG) \
kernel void NAME(constant MoeMvArgs & a [[buffer(4)]], device const char * w [[buffer(0)]], \
                 device const float * x [[buffer(1)]], device const uint * ids [[buffer(2)]], \
                 device float * y [[buffer(3)]], \
                 uint3 tgpig [[threadgroup_position_in_grid]], ushort tiisg [[thread_index_in_simdgroup]], \
                 ushort sgitg [[simdgroup_index_in_threadgroup]]) { \
    const uint r = tgpig.y; \
    device const char * we = w + (ulong)ids[r] * a.expert_rows * a.p.row_bytes; \
    IMPL<NR0, NSG>(a.p, we, x + (ulong)(r / a.x_div) * a.p.x_stride, y + (ulong)r * a.p.y_stride, \
                   uint3(tgpig.x, 0, 0), tiisg, sgitg); \
}
#define MV_ID_KERNEL_SHMEM(NAME, IMPL, NR0, NSG, T) \
kernel void NAME(constant MoeMvArgs & a [[buffer(4)]], device const char * w [[buffer(0)]], \
                 device const float * x [[buffer(1)]], device const uint * ids [[buffer(2)]], \
                 device float * y [[buffer(3)]], threadgroup T * shmem [[threadgroup(0)]], \
                 uint3 tgpig [[threadgroup_position_in_grid]], ushort tiisg [[thread_index_in_simdgroup]], \
                 ushort sgitg [[simdgroup_index_in_threadgroup]]) { \
    const uint r = tgpig.y; \
    device const char * we = w + (ulong)ids[r] * a.expert_rows * a.p.row_bytes; \
    IMPL<NR0, NSG>(a.p, we, x + (ulong)(r / a.x_div) * a.p.x_stride, y + (ulong)r * a.p.y_stride, \
                   shmem, uint3(tgpig.x, 0, 0), tiisg, sgitg); \
}
MV_ID_KERNEL(h_mv_id_q4_k, mv_q4_K, 2, 2)
MV_ID_KERNEL(h_mv_id_q5_k, mv_q5_K, 1, 2)
MV_ID_KERNEL(h_mv_id_q6_k, mv_q6_K, 2, 2)
MV_ID_KERNEL(h_mv_id_q4_0, mv_q4_0, 4, 2)
MV_ID_KERNEL_SHMEM(h_mv_id_q8_0, mv_q8_0, 2, 4, float)
MV_ID_KERNEL_SHMEM(h_mv_id_iq4_xs, mv_iq4_xs, 2, 2, float)
MV_ID_KERNEL_SHMEM(h_mv_id_iq3_s, mv_iq3_s, 4, 2, uint)

// Expert GEMM over expert-sorted rows: z = expert; rows offsets[z]..offsets[z+1]
// of x (gathered) and y. grid (ceil(max_rows/128), ceil(expert_rows/64), experts).
struct MoeMmArgs { ProjArgs p; uint expert_rows; };
// Experts see few rows each (~routes/experts), so the activation tile is 32.
constant constexpr int MM_ID_NRB = 32;

template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread half4x4 &)>
void mm_id_impl(constant MoeMmArgs & a, device const char * w, device const float * x,
                device const uint * offsets, device float * y,
                threadgroup half * sa, uint3 tgpig, ushort tiitg) {
    const uint e = tgpig.z;
    const uint first = offsets[e];
    const int count = int(offsets[e + 1] - first);
    if (int(tgpig.x) * MM_ID_NRB >= count) return;
    const int K = a.p.k;
    const int M = a.expert_rows;
    const int ra = tgpig.y * MM_NRA;
    const int rb = tgpig.x * MM_ID_NRB;
    device const char * we = w + (ulong)e * a.expert_rows * a.p.row_bytes;
    auto tA = tensor(sa, dextents<int32_t, 2>(MM_NK, MM_NRA));
    auto tB = tensor((device float *)(x + (ulong)first * a.p.x_stride), dextents<int32_t, 2>(K, count),
                     array<int, 2>({1, int(a.p.x_stride)}));
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            MM_ID_NRB, MM_NRA, MM_NK, false, true, true,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;
    auto cT = mm.get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();
    for (int loop_k = 0; loop_k < K; loop_k += MM_NK) {
        for (int work = tiitg; work < MM_NRA * 2; work += 128) {
            const int row = work / 2;
            const int k_chunk = work % 2;
            const int k_pos = loop_k + k_chunk * 16;
            if (ra + row < M) {
                device const block_q * row_ptr = (device const block_q *)(we + (ulong)a.p.row_bytes * (ra + row));
                half4x4 temp_a;
                dequantize_func(row_ptr + k_pos / (16 * nl), (k_pos / 16) % nl, temp_a);
                FOR_UNROLL (short i = 0; i < 16; i++) sa[row * MM_NK + k_chunk*16 + i] = temp_a[i/4][i%4];
            } else {
                FOR_UNROLL (short i = 0; i < 16; i++) sa[row * MM_NK + k_chunk*16 + i] = 0.h;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mA = tA.slice(0, 0);
        auto mB = tB.slice(loop_k, rb);
        mm.run(mB, mA, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto tD = tensor(y + (ulong)first * a.p.y_stride, dextents<int32_t, 2>(M, count),
                     array<int, 2>({1, int(a.p.y_stride)}));
    cT.store(tD.slice(ra, rb));
}
#define MM_ID_KERNEL(NAME, BLOCK, NL, DEQ) \
kernel void NAME(constant MoeMmArgs & a [[buffer(4)]], device const char * w [[buffer(0)]], \
                 device const float * x [[buffer(1)]], device const uint * offsets [[buffer(2)]], \
                 device float * y [[buffer(3)]], threadgroup half * sa [[threadgroup(0)]], \
                 uint3 tgpig [[threadgroup_position_in_grid]], ushort tiitg [[thread_index_in_threadgroup]]) { \
    mm_id_impl<BLOCK, NL, DEQ<half4x4>>(a, w, x, offsets, y, sa, tgpig, tiitg); \
}
MM_ID_KERNEL(h_mm_id_q4_k, block_q4_K, 16, dequantize_q4_K)
MM_ID_KERNEL(h_mm_id_q5_k, block_q5_K, 16, dequantize_q5_K)
MM_ID_KERNEL(h_mm_id_q6_k, block_q6_K, 16, dequantize_q6_K)
MM_ID_KERNEL(h_mm_id_q8_0, block_q8_0, 2, dequantize_q8_0)
MM_ID_KERNEL(h_mm_id_q4_0, block_q4_0, 2, dequantize_q4_0)
MM_ID_KERNEL(h_mm_id_iq4_xs, block_iq4_xs, 16, dequantize_iq4_xs)
MM_ID_KERNEL(h_mm_id_iq3_s, block_iq3_s, 16, dequantize_iq3_s)

// ---------------------------------------------------------------- flash attention

struct FaArgs {
    uint tokens;         // query tokens
    uint heads;          // query heads
    uint kv_heads;
    uint group;          // heads / kv_heads
    uint position;       // absolute position of query token 0
    uint cache_stride;   // halfs between cached positions
    uint keys_per_split;
    uint splits;
};

// Causal attention over the F16 cache for query rows r = t*group + g of one
// KV head. Each SIMD group owns 8 rows; keys stream in blocks of 32 with an
// online softmax. Q arrives in F16, pre-scaled by 1/sqrt(head_dim). With one
// split the result is normalized, multiplied by sigmoid(gate) and written to
// `out`; otherwise unnormalized partials and (max, sum) go to `po`/`pml`.
// grid (ceil(tokens*group / (8*NSG)), kv_heads, splits); threads 32*NSG.
template<short NSG>
void flash_attn_impl(constant FaArgs & p, device const half * q, device const half * kc,
                     device const half * vc, device const float * qg,
                     device float * out, device float * po, device float * pml,
                     threadgroup half * shared, uint3 tgpig, ushort sg, ushort lane) {
    constexpr int D = 256, DT = D / 8, C = 32;
    threadgroup half  * Qs = shared + sg * (8 * D);
    threadgroup float * Ss = (threadgroup float *)(shared + NSG * 8 * D) + sg * (8 * C);
    threadgroup half  * Ps = (threadgroup half *)((threadgroup float *)(shared + NSG * 8 * D) + NSG * 8 * C) + sg * (8 * C);
    threadgroup float * Ds = (threadgroup float *)((threadgroup half *)((threadgroup float *)(shared + NSG * 8 * D) + NSG * 8 * C) + NSG * 8 * C) + sg * 64;

    const uint kvh = tgpig.y;
    const uint split = tgpig.z;
    const uint rows_total = p.tokens * p.group;
    const uint r0 = (tgpig.x * NSG + sg) * 8;
    const uint total_keys = p.position + p.tokens;
    // Stage this SIMD group's 8 query rows (zero beyond the valid rows).
    for (uint i = lane; i < 8 * D; i += 32) {
        const uint r = r0 + i / D, d = i % D;
        half v = 0.h;
        if (r < rows_total) {
            const uint t = r / p.group, g = r % p.group;
            v = q[((ulong)t * p.heads + kvh * p.group + g) * D + d];
        }
        Qs[i] = v;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 O[DT];
    for (short i = 0; i < DT; i++) O[i] = make_filled_simdgroup_matrix<float, 8>(0.f);

    const uint my_row = lane / 4;              // softmax: 4 lanes per row
    const uint row = r0 + my_row;
    const bool row_valid = row < rows_total;
    const uint row_last_key = row_valid ? p.position + row / p.group : 0;
    const uint sg_last_key = p.position + min(r0 + 7, rows_total - 1) / p.group;
    float m = -INFINITY, l = 0.f;

    const uint key_begin = split * p.keys_per_split;
    const uint key_end = min(key_begin + p.keys_per_split, total_keys);
    const ulong head_off = (ulong)kvh * D;
    for (uint j0 = key_begin; j0 < key_end && j0 <= sg_last_key; j0 += C) {
        simdgroup_float8x8 S[C / 8];
        for (short kt = 0; kt < C / 8; kt++) S[kt] = make_filled_simdgroup_matrix<float, 8>(0.f);
        for (short dt = 0; dt < DT; dt++) {
            simdgroup_half8x8 qa;
            simdgroup_load(qa, Qs + dt * 8, D);
            for (short kt = 0; kt < C / 8; kt++) {
                simdgroup_half8x8 kb;
                simdgroup_load(kb, kc + (ulong)(j0 + kt * 8) * p.cache_stride + head_off + dt * 8,
                               p.cache_stride, ulong2(0), true);
                simdgroup_multiply_accumulate(S[kt], qa, kb, S[kt]);
            }
        }
        for (short kt = 0; kt < C / 8; kt++) simdgroup_store(S[kt], Ss + kt * 8, C);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        // Online softmax: lane owns row my_row, columns (lane%4)*8 .. +8.
        const uint c0 = (lane % 4) * 8;
        float s[8];
        float mx = -INFINITY;
        for (short c = 0; c < 8; c++) {
            const uint key = j0 + c0 + c;
            const bool ok = row_valid && key < key_end && key <= row_last_key;
            s[c] = ok ? Ss[my_row * C + c0 + c] : -INFINITY;
            mx = max(mx, s[c]);
        }
        mx = max(mx, simd_shuffle_xor(mx, 1));
        mx = max(mx, simd_shuffle_xor(mx, 2));
        const float m_new = max(m, mx);
        const float corr = m_new == -INFINITY ? 1.f : exp(m - m_new);
        float sum = 0.f;
        for (short c = 0; c < 8; c++) {
            const float e = m_new == -INFINITY ? 0.f : exp(s[c] - m_new);
            sum += e;
            Ps[my_row * C + c0 + c] = half(e);
        }
        sum += simd_shuffle_xor(sum, 1);
        sum += simd_shuffle_xor(sum, 2);
        l = l * corr + sum;
        m = m_new;
        // Diagonal rescale of the running output.
        for (uint i = lane; i < 64; i += 32) Ds[i] = 0.f;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane % 4 == 0) Ds[my_row * 9] = corr;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 diag;
        simdgroup_load(diag, Ds, 8);
        for (short dt = 0; dt < DT; dt++) simdgroup_multiply(O[dt], diag, O[dt]);
        // O += P V
        simdgroup_half8x8 pa[C / 8];
        for (short kt = 0; kt < C / 8; kt++) simdgroup_load(pa[kt], Ps + kt * 8, C);
        for (short dt = 0; dt < DT; dt++) {
            for (short kt = 0; kt < C / 8; kt++) {
                simdgroup_half8x8 vb;
                simdgroup_load(vb, vc + (ulong)(j0 + kt * 8) * p.cache_stride + head_off + dt * 8,
                               p.cache_stride);
                simdgroup_multiply_accumulate(O[dt], pa[kt], vb, O[dt]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (p.splits == 1) {
        for (uint i = lane; i < 64; i += 32) Ds[i] = 0.f;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane % 4 == 0) Ds[my_row * 9] = l > 0.f ? 1.f / l : 0.f;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 diag;
        simdgroup_load(diag, Ds, 8);
        threadgroup float * Os = Ss;   // reuse: 8x8 staging
        for (short dt = 0; dt < DT; dt++) {
            simdgroup_multiply(O[dt], diag, O[dt]);
            simdgroup_store(O[dt], Os, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = lane; i < 64; i += 32) {
                const uint r = r0 + i / 8;
                if (r < rows_total) {
                    const uint t = r / p.group, g = r % p.group, h = kvh * p.group + g;
                    const uint d = dt * 8 + i % 8;
                    const float gate = qg[((ulong)t * p.heads + h) * 2 * D + D + d];
                    out[((ulong)t * p.heads + h) * D + d] = Os[i] / (1.f + exp(-gate));
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        threadgroup float * Os = Ss;
        for (short dt = 0; dt < DT; dt++) {
            simdgroup_store(O[dt], Os, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = lane; i < 64; i += 32) {
                const uint r = r0 + i / 8;
                if (r < rows_total) {
                    const uint t = r / p.group, g = r % p.group, h = kvh * p.group + g;
                    const ulong grow = (ulong)split * p.tokens * p.heads + (ulong)t * p.heads + h;
                    po[grow * D + dt * 8 + i % 8] = Os[i];
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane % 4 == 0 && row_valid) {
            const uint t = row / p.group, g = row % p.group, h = kvh * p.group + g;
            const ulong grow = (ulong)split * p.tokens * p.heads + (ulong)t * p.heads + h;
            pml[grow * 2] = m;
            pml[grow * 2 + 1] = l;
        }
    }
}

kernel void h_flash_attn_4(constant FaArgs & p [[buffer(7)]],
                           device const half * q [[buffer(0)]], device const half * kc [[buffer(1)]],
                           device const half * vc [[buffer(2)]], device const float * qg [[buffer(3)]],
                           device float * out [[buffer(4)]], device float * po [[buffer(5)]],
                           device float * pml [[buffer(6)]], threadgroup half * shared [[threadgroup(0)]],
                           uint3 tgpig [[threadgroup_position_in_grid]],
                           ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    flash_attn_impl<4>(p, q, kc, vc, qg, out, po, pml, shared, tgpig, sg, lane);
}
kernel void h_flash_attn_1(constant FaArgs & p [[buffer(7)]],
                           device const half * q [[buffer(0)]], device const half * kc [[buffer(1)]],
                           device const half * vc [[buffer(2)]], device const float * qg [[buffer(3)]],
                           device float * out [[buffer(4)]], device float * po [[buffer(5)]],
                           device float * pml [[buffer(6)]], threadgroup half * shared [[threadgroup(0)]],
                           uint3 tgpig [[threadgroup_position_in_grid]],
                           ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    flash_attn_impl<1>(p, q, kc, vc, qg, out, po, pml, shared, tgpig, sg, lane);
}

// Merge split partials: one threadgroup of head_dim threads per (token, head).
kernel void h_flash_reduce(constant FaArgs & p [[buffer(4)]],
                           device const float * po [[buffer(0)]], device const float * pml [[buffer(1)]],
                           device const float * qg [[buffer(2)]], device float * out [[buffer(3)]],
                           uint row [[threadgroup_position_in_grid]], ushort d [[thread_index_in_threadgroup]]) {
    constexpr uint D = 256;
    const ulong rows = (ulong)p.tokens * p.heads;
    float mx = -INFINITY;
    for (uint s = 0; s < p.splits; s++) mx = max(mx, pml[(s * rows + row) * 2]);
    float l = 0.f, acc = 0.f;
    for (uint s = 0; s < p.splits; s++) {
        const float ms = pml[(s * rows + row) * 2];
        if (ms == -INFINITY) continue;
        const float w = exp(ms - mx);
        l += w * pml[(s * rows + row) * 2 + 1];
        acc += w * po[(s * rows + row) * D + d];
    }
    const uint t = row / p.heads, h = row % p.heads;
    const float gate = qg[((ulong)t * p.heads + h) * 2 * D + D + d];
    out[(ulong)row * D + d] = (l > 0.f ? acc / l : 0.f) / (1.f + exp(-gate));
}

// Bandwidth probe: each threadgroup streams a contiguous span with 16-byte loads.
kernel void h_bw_read(constant EltArgs & a [[buffer(2)]], device const uint4 * x [[buffer(0)]],
                      device float * y [[buffer(1)]], uint tg [[threadgroup_position_in_grid]],
                      ushort tid [[thread_index_in_threadgroup]], ushort threads [[threads_per_threadgroup]]) {
    const uint per_group = a.n;   // uint4 elements per threadgroup
    device const uint4 * p = x + (ulong)tg * per_group;
    uint4 acc = 0;
    for (uint i = tid; i < per_group; i += threads) acc ^= p[i];
    if (acc.x == 0x12345678u) y[tg] = float(acc.y);
}

// TensorOps flash attention for prompt chunks. One threadgroup = 64 query
// tokens of one query head; keys stream in blocks of 64. S = Q Kᵀ and
// O += P V run as matmul2d; the online softmax runs over S staged in
// threadgroup memory. Q is F16 pre-scaled; K/V come from the F16 cache.
// grid (ceil(tokens/64), heads); threads 128; 64*64*4 + 64*64*2 + 3*64*4 bytes.
kernel void h_flash_attn_mpp(constant FaArgs & p [[buffer(5)]],
                             device half * q [[buffer(0)]], device half * kc [[buffer(1)]],
                             device half * vc [[buffer(2)]], device const float * qg [[buffer(3)]],
                             device float * out [[buffer(4)]],
                             threadgroup char * shared [[threadgroup(0)]],
                             uint2 tgpig [[threadgroup_position_in_grid]],
                             ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int D = 256, BR = 64, BC = 64;
    threadgroup float * Ss = (threadgroup float *)shared;                 // [BR][BC]
    threadgroup half  * Ps = (threadgroup half *)(Ss + BR * BC);          // [BR][BC]
    threadgroup float * corr = (threadgroup float *)(Ps + BR * BC);       // [BR]
    threadgroup float * mrow = corr + BR;                                 // [BR]
    threadgroup float * lrow = mrow + BR;                                 // [BR]
    const uint h = tgpig.y;
    const uint kvh = h / p.group;
    const int t0 = int(tgpig.x) * BR;
    const int rows = min(BR, int(p.tokens) - t0);
    const uint total_keys = p.position + p.tokens;
    const uint last_key = p.position + uint(t0 + rows - 1);
    const int q_stride = int(p.heads) * D;
    const int kv_stride = int(p.cache_stride);

    auto tQ = tensor(q + (ulong)t0 * q_stride + h * D, dextents<int32_t, 2>(D, rows), array<int, 2>({1, q_stride}));
    auto tK = tensor(kc + kvh * D, dextents<int32_t, 2>(D, int(total_keys)), array<int, 2>({1, kv_stride}));
    auto tV = tensor(vc + kvh * D, dextents<int32_t, 2>(D, int(total_keys)), array<int, 2>({1, kv_stride}));
    auto tS = tensor(Ss, dextents<int32_t, 2>(BC, BR));
    auto tP = tensor(Ps, dextents<int32_t, 2>(BC, BR));

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(BR, BC, D, false, true, true,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply),
        execution_simdgroups<4>> mm_s;
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(BR, D, BC, false, false, true,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm_o;
    auto cS = mm_s.get_destination_cooperative_tensor<decltype(tQ), decltype(tK), float>();
    auto cO = mm_o.get_destination_cooperative_tensor<decltype(tP), decltype(tV), float>();
    for (uint i = 0; i < cO.get_capacity(); i++) cO[i] = 0.f;
    if (tid < BR) { mrow[tid] = -INFINITY; lrow[tid] = 0.f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Softmax work split: thread tid owns row tid/2, columns (tid%2)*32 .. +32.
    const int my_row = tid / 2;
    const int c0 = (tid % 2) * 32;
    const uint row_last_key = p.position + uint(t0 + my_row);
    for (uint j0 = 0; j0 < total_keys && j0 <= last_key; j0 += BC) {
        auto mK = tK.slice(0, int(j0));
        mm_s.run(tQ, mK, cS);
        cS.store(tS);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float s[32];
        float mx = -INFINITY;
        for (int c = 0; c < 32; c++) {
            const uint key = j0 + uint(c0 + c);
            const bool ok = my_row < rows && key < total_keys && key <= row_last_key;
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
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid % 2 == 0) {
            mrow[my_row] = m_new;
            lrow[my_row] = lrow[my_row] * cr + sum;
            corr[my_row] = cr;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < cO.get_capacity(); i++) {
            if (cO.is_valid_element(i)) {
                auto idx = cO.get_multidimensional_index(i);
                cO[i] *= corr[idx[1]];
            }
        }
        auto mV = tV.slice(0, int(j0));
        mm_o.run(tP, mV, cO);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < cO.get_capacity(); i++) {
        if (!cO.is_valid_element(i)) continue;
        auto idx = cO.get_multidimensional_index(i);
        const int r = idx[1], dcol = idx[0];
        if (r >= rows) continue;
        const uint t = uint(t0 + r);
        const float l = lrow[r];
        const float gate = qg[((ulong)t * p.heads + h) * 2 * D + D + dcol];
        out[((ulong)t * p.heads + h) * D + dcol] = (l > 0.f ? cO[i] / l : 0.f) / (1.f + exp(-gate));
    }
}

// Single-token decode attention for one KV head and all `group` (<= 8) query
// heads sharing it. QKᵀ is key-parallel (lane = key, full 256-wide dot
// products against Q staged in threadgroup memory); PV is dimension-parallel
// (lane owns 8 of 256 dims, V rows load coalesced). Four SIMD groups take
// interleaved 32-key blocks of this split's range and merge before writing a
// partial (or, with one split, the gated output).
// grid (kv_heads, splits); threads 128; 8*256*2 + 4*8*2*4 + 2*8*256*4 bytes.
kernel void h_attn_decode(constant FaArgs & p [[buffer(7)]],
                          device const half * q [[buffer(0)]], device const half * kc [[buffer(1)]],
                          device const half * vc [[buffer(2)]], device const float * qg [[buffer(3)]],
                          device float * out [[buffer(4)]], device float * po [[buffer(5)]],
                          device float * pml [[buffer(6)]],
                          threadgroup char * shared [[threadgroup(0)]],
                          uint2 tgpig [[threadgroup_position_in_grid]],
                          ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int D = 256, G = 8, NSG = 4;
    threadgroup half * Qs = (threadgroup half *)shared;                     // [G][D]
    threadgroup float * Ms = (threadgroup float *)(Qs + G * D);             // [NSG][G] max
    threadgroup float * Ls = Ms + NSG * G;                                  // [NSG][G] sum
    threadgroup float * As = Ls + NSG * G;                                  // [2][G][D]
    const uint kvh = tgpig.x;
    const uint split = tgpig.y;
    const uint group = p.group;
    const uint tid = sg * 32 + lane;
    for (uint i = tid; i < G * D; i += 128) {
        const uint g = i / D;
        Qs[i] = g < group ? q[(kvh * group + g) * D + i % D] : 0.h;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint total = p.position + 1;
    const uint begin = split * p.keys_per_split;
    const uint end = min(begin + p.keys_per_split, total);
    const ulong head = (ulong)kvh * D;
    float m[G], l[G], acc[G][8];
    for (int g = 0; g < G; g++) {
        m[g] = -INFINITY; l[g] = 0.f;
        for (int j = 0; j < 8; j++) acc[g][j] = 0.f;
    }
    for (uint j0 = begin + sg * 32; j0 < end; j0 += NSG * 32) {
        const uint key = j0 + lane;
        const bool ok = key < end;
        float s[G];
        for (int g = 0; g < G; g++) s[g] = 0.f;
        if (ok) {
            device const half4 * kr = (device const half4 *)(kc + (ulong)key * p.cache_stride + head);
            for (int d4 = 0; d4 < D / 4; d4++) {
                const float4 kv = float4(kr[d4]);
                for (int g = 0; g < G; g++) {
                    const float4 qv = float4(((threadgroup const half4 *)(Qs + g * D))[d4]);
                    s[g] += dot(qv, kv);
                }
            }
        }
        float pr[G];
        for (int g = 0; g < G; g++) {
            const float sv = ok ? s[g] : -INFINITY;
            const float bm = simd_max(sv);
            const float m_new = max(m[g], bm);
            const float corr = m_new == -INFINITY ? 1.f : exp(m[g] - m_new);
            pr[g] = ok ? exp(sv - m_new) : 0.f;
            l[g] = l[g] * corr + simd_sum(pr[g]);
            m[g] = m_new;
            for (int j = 0; j < 8; j++) acc[g][j] *= corr;
        }
        const uint count = min(32u, end - j0);
        for (uint k = 0; k < count; k++) {
            device const half * vr = vc + (ulong)(j0 + k) * p.cache_stride + head + lane * 8;
            const float4 v0 = float4(*(device const half4 *)vr);
            const float4 v1 = float4(*(device const half4 *)(vr + 4));
            for (int g = 0; g < G; g++) {
                const float pk = simd_shuffle(pr[g], ushort(k));
                acc[g][0] += pk * v0.x; acc[g][1] += pk * v0.y; acc[g][2] += pk * v0.z; acc[g][3] += pk * v0.w;
                acc[g][4] += pk * v1.x; acc[g][5] += pk * v1.y; acc[g][6] += pk * v1.z; acc[g][7] += pk * v1.w;
            }
        }
    }
    // Merge the four SIMD groups pairwise (2+3 into 0+1, then 1 into 0).
    for (int round = 0; round < 2; round++) {
        const uint writers = round == 0 ? 2u : 1u;      // SIMD groups >= writers publish
        if (sg >= writers && sg < 2 * writers) {
            const uint slot = sg - writers;
            for (int g = 0; g < G; g++) {
                if (lane == 0) { Ms[slot * G + g] = m[g]; Ls[slot * G + g] = l[g]; }
                for (int j = 0; j < 8; j++) As[(slot * G + g) * D + lane * 8 + j] = acc[g][j];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg < writers) {
            for (int g = 0; g < G; g++) {
                const float m2 = Ms[sg * G + g];
                const float m_new = max(m[g], m2);
                if (m_new == -INFINITY) continue;
                const float w1 = exp(m[g] - m_new), w2 = exp(m2 - m_new);
                l[g] = l[g] * w1 + Ls[sg * G + g] * w2;
                for (int j = 0; j < 8; j++) {
                    acc[g][j] = acc[g][j] * w1 + As[(sg * G + g) * D + lane * 8 + j] * w2;
                }
                m[g] = m_new;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (sg == 0) {
        for (uint g = 0; g < group; g++) {
            const uint h = kvh * group + g;
            for (int j = 0; j < 8; j++) {
                const uint dcol = lane * 8 + j;
                if (p.splits == 1) {
                    const float gate = qg[(ulong)h * 2 * D + D + dcol];
                    out[(ulong)h * D + dcol] = (l[g] > 0.f ? acc[g][j] / l[g] : 0.f) / (1.f + exp(-gate));
                } else {
                    const ulong grow = (ulong)split * p.heads + h;
                    po[grow * D + dcol] = acc[g][j];
                    if (dcol == 0) { pml[grow * 2] = m[g]; pml[grow * 2 + 1] = l[g]; }
                }
            }
        }
    }
}
