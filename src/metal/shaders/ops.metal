#include <metal_stdlib>
using namespace metal;
// ABI: length, row width, K, N, dtype (0 f32 / 1 f16 / 2 bf16), position,
// head dimension, epsilon bits, theta bits. All indexing validated to fit uint.
inline float load(device const uchar* p, uint i, uint dtype) {
    if (dtype == 0) return ((device const float*)p)[i];
    if (dtype == 1) return float(((device const half*)p)[i]);
    return as_type<float>(uint(((device const ushort*)p)[i]) << 16);
}
inline void store(device uchar* p, uint i, uint dtype, float x) {
    if (dtype == 0) ((device float*)p)[i] = x;
    else if (dtype == 1) ((device half*)p)[i] = half(x);
    else {
        uint bits = as_type<uint>(x);
        // Round to nearest, ties to even; preserve NaN rather than rounding it to infinity.
        ushort b = isnan(x) ? ushort((bits >> 16) | 0x40) : ushort((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
        ((device ushort*)p)[i] = b;
    }
}
#define ARGS device const uchar* a [[buffer(0)]], device const uchar* b [[buffer(1)]], device uchar* c [[buffer(2)]], constant uint* p [[buffer(3)]]
kernel void add(ARGS, uint i [[thread_position_in_grid]]) { if(i<p[0]) store(c,i,p[4],load(a,i,p[4])+load(b,i,p[4])); }
kernel void mul(ARGS, uint i [[thread_position_in_grid]]) { if(i<p[0]) store(c,i,p[4],load(a,i,p[4])*load(b,i,p[4])); }
kernel void silu(ARGS, uint i [[thread_position_in_grid]]) { if(i<p[0]) { float x=load(a,i,p[4]); float s=x>=0 ? 1.f/(1.f+exp(-x)) : exp(x)/(1.f+exp(x)); store(c,i,p[4],x*s); } }
kernel void rmsnorm(ARGS, uint row [[thread_position_in_grid]]) {
    uint w=p[1], base=row*w; if(base>=p[0]) return;
    float sum=0; for(uint j=0;j<w;j++) { float x=load(a,base+j,p[4]); sum+=x*x; }
    float inv=rsqrt(sum/float(w)+as_type<float>(p[7]));
    for(uint j=0;j<w;j++) store(c,base+j,p[4],load(a,base+j,p[4])*inv*load(b,j,p[4]));
}
kernel void softmax(ARGS, uint row [[thread_position_in_grid]]) {
    uint w=p[1], base=row*w; if(base>=p[0]) return;
    float mx=-INFINITY; for(uint j=0;j<w;j++) mx=max(mx,load(a,base+j,p[4]));
    float sum=0; for(uint j=0;j<w;j++) sum+=exp(load(a,base+j,p[4])-mx);
    for(uint j=0;j<w;j++) store(c,base+j,p[4],exp(load(a,base+j,p[4])-mx)/sum);
}
// Adjacent-pair (interleaved) RoPE. The same supplied position applies to all heads.
kernel void rope(ARGS, uint pair [[thread_position_in_grid]]) {
    uint i=pair*2; if(i>=p[0]) return;
    float angle=float(p[5])*pow(as_type<float>(p[8]),-float(i%p[6])/float(p[6]));
    float co=cos(angle), si=sin(angle), x=load(a,i,p[4]), y=load(a,i+1,p[4]);
    store(c,i,p[4],x*co-y*si); store(c,i+1,p[4],x*si+y*co);
}
kernel void matmul(ARGS, uint2 tid [[thread_position_in_threadgroup]], uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup float aa[16][16], bb[16][16];
    uint row=group.y*16+tid.y, col=group.x*16+tid.x, k=p[2], n=p[3], m=p[1];
    float sum=0;
    for(uint tile=0;tile<k;tile+=16) {
        aa[tid.y][tid.x]=(row<m && tile+tid.x<k)?load(a,row*k+tile+tid.x,p[4]):0;
        bb[tid.y][tid.x]=(col<n && tile+tid.y<k)?load(b,(tile+tid.y)*n+col,p[4]):0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint j=0;j<16;j++) sum+=aa[tid.y][j]*bb[j][tid.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(row<m && col<n) store(c,row*n+col,p[4],sum);
}
