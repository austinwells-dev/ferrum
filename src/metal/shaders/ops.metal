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
kernel void rmsnorm(ARGS, uint tid [[thread_index_in_threadgroup]], uint row [[threadgroup_position_in_grid]]) {
    threadgroup float partial[8];
    uint w=p[1], base=row*w, lane=tid%32, simd=tid/32;
    float sum=0;
    for(uint j=tid;j<w;j+=256) { float x=load(a,base+j,p[4]); sum+=x*x; }
    sum=simd_sum(sum);
    if(lane==0) partial[simd]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum=simd_sum(lane<8?partial[lane]:0.f);
    float inv=rsqrt(sum/float(w)+as_type<float>(p[7]));
    // BF16 midpoint sensitivity: only rows close to a storage rounding boundary
    // need the legacy ascending sum to preserve the established model path.
    threadgroup uint ambiguous[8];
    threadgroup float ordered_inv;
    uint near=0;
    if(p[4]==2) for(uint j=tid;j<w;j+=256) {
        float value=load(a,base+j,p[4])*inv*load(b,j,p[4]);
        uint fraction=as_type<uint>(value)&65535;
        near |= uint(abs(int(fraction)-32768)<=8);
    }
    near=simd_max(near);
    if(lane==0) ambiguous[simd]=near;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    near=simd_max(lane<8?ambiguous[lane]:0u);
    if(near && tid==0) {
        float ordered=0;
        for(uint j=0;j<w;j++) {float x=load(a,base+j,p[4]);ordered+=x*x;}
        ordered_inv=rsqrt(ordered/float(w)+as_type<float>(p[7]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(near) inv=ordered_inv;
    for(uint j=tid;j<w;j+=256) store(c,base+j,p[4],load(a,base+j,p[4])*inv*load(b,j,p[4]));

}
kernel void softmax(ARGS, uint tid [[thread_index_in_threadgroup]], uint row [[threadgroup_position_in_grid]]) {
    threadgroup float partial[8];
    uint w=p[1], base=row*w, lane=tid%32, simd=tid/32;
    float mx=-INFINITY;
    for(uint j=tid;j<w;j+=256) mx=max(mx,load(a,base+j,p[4]));
    mx=simd_max(mx);
    if(lane==0) partial[simd]=mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx=simd_max(lane<8?partial[lane]:-INFINITY);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum=0;
    for(uint j=tid;j<w;j+=256) sum+=exp(load(a,base+j,p[4])-mx);
    sum=simd_sum(sum);
    if(lane==0) partial[simd]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum=simd_sum(lane<8?partial[lane]:0.f);
    for(uint j=tid;j<w;j+=256) store(c,base+j,p[4],exp(load(a,base+j,p[4])-mx)/sum);
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
// Phase 2 copies never alias their inputs. p[1..3,5..8] are operation-specific.
kernel void copy_range(ARGS, uint i [[thread_position_in_grid]]) {
    store(c,i,p[4],load(a,p[1]+i,p[4]));
}
kernel void concat_flat(ARGS, uint i [[thread_position_in_grid]]) {
    store(c,i,p[4],i<p[0]?load(a,i,p[4]):load(b,i-p[0],p[4]));
}
kernel void transpose2(ARGS, uint i [[thread_position_in_grid]]) {
    store(c,i,p[4],load(a,(i%p[1])*p[2]+i/p[1],p[4]));
}
// [A,B,C] -> [B,A,C]
kernel void swap01(ARGS, uint i [[thread_position_in_grid]]) {
    uint z=i%p[3], x=(i/p[3])%p[1], y=i/(p[3]*p[1]);
    store(c,i,p[4],load(a,(x*p[2]+y)*p[3]+z,p[4]));
}
// [sequence,heads,dim] -> [sequence,dim], selecting one head.
kernel void select_head(ARGS, uint i [[thread_position_in_grid]]) {
    store(c,i,p[4],load(a,(i/p[3]*p[2]+p[1])*p[3]+i%p[3],p[4]));
}
kernel void bias_add(ARGS, uint i [[thread_position_in_grid]]) {
    store(c,i,p[4],load(a,i,p[4])+load(b,i%p[1],p[4]));
}
kernel void scale(ARGS, uint i [[thread_position_in_grid]]) {
    store(c,i,p[4],load(a,i,p[4])*as_type<float>(p[7]));
}
kernel void causal_mask(ARGS, uint i [[thread_position_in_grid]]) {
    store(c,i,p[4],i%p[1]>p[5]+i/p[1]?-INFINITY:load(a,i,p[4]));
}
// Split-half pairing, sequence-major [S,H,D], absolute position offset + token.
kernel void rope_split(ARGS, uint pair [[thread_position_in_grid]]) {
    uint halfdim=p[6]/2, head=pair/halfdim, j=pair%halfdim;
    uint i=head*p[6]+j;
    float angle=float(p[5]+head/p[1])*pow(as_type<float>(p[8]),-float(2*j)/float(p[6]));
    float co=cos(angle), si=sin(angle), x=load(a,i,p[4]), y=load(a,i+halfdim,p[4]);
    store(c,i,p[4],x*co-y*si); store(c,i+halfdim,p[4],x*si+y*co);
}
// IDs are validated on the host and uploaded as little-endian u32 words.
// Their private carrier tensor has the weight dtype only to use the common dispatcher.
kernel void embedding_gather(ARGS, uint i [[thread_position_in_grid]]) {
    uint token=((device const uint*)b)[i/p[1]];
    store(c,i,p[4],load(a,token*p[1]+i%p[1],p[4]));
}
// Direct grouped layout products. Each dot retains the original ascending-K sum
// and separate storage rounding; scaling/masking/softmax remain separate kernels.
kernel void attention_scores(ARGS, uint i [[thread_position_in_grid]]) {
    uint t=i%p[2], s=(i/p[2])%p[1], h=i/(p[1]*p[2]);
    uint kv=h/(p[3]/p[5]); float sum=0;
    for(uint j=0;j<p[6];j++) sum+=load(a,(s*p[3]+h)*p[6]+j,p[4])*load(b,(t*p[5]+kv)*p[6]+j,p[4]);
    store(c,i,p[4],sum);
}
kernel void attention_mask(ARGS, uint i [[thread_position_in_grid]]) {
    uint s=(i/p[2])%p[1], t=i%p[2];
    store(c,i,p[4],t>p[5]+s?-INFINITY:load(a,i,p[4]));
}
kernel void attention_context(ARGS, uint i [[thread_position_in_grid]]) {
    uint j=i%p[6], h=(i/p[6])%p[3], s=i/(p[3]*p[6]);
    uint kv=h/(p[3]/p[5]); float sum=0;
    for(uint t=0;t<p[2];t++) sum+=load(a,(h*p[1]+s)*p[2]+t,p[4])*load(b,(t*p[5]+kv)*p[6]+j,p[4]);
    store(c,i,p[4],sum);
}
// Row-major [N,K] weights; one SIMD group per output row, four groups per TG.
kernel void gemv(ARGS, uint tid [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]]) {
    uint lane=tid%32, row=group*4+tid/32;
    float sum=0;
    if(row<p[3]) for(uint j=lane;j<p[2];j+=32) sum+=load(a,j,p[4])*load(b,row*p[2]+j,p[4]);
    sum=simd_sum(sum);
    if(lane==0 && row<p[3]) store(c,row,p[4],sum);
}
kernel void matmul_nt(ARGS, uint2 tid [[thread_position_in_threadgroup]], uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup float aa[16][16], bb[16][16];
    uint row=group.y*16+tid.y, col=group.x*16+tid.x, k=p[2], n=p[3], m=p[1];
    float sum=0;
    for(uint tile=0;tile<k;tile+=16) {
        aa[tid.y][tid.x]=(row<m && tile+tid.x<k)?load(a,row*k+tile+tid.x,p[4]):0;
        bb[tid.y][tid.x]=(col<n && tile+tid.y<k)?load(b,col*k+tile+tid.y,p[4]):0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint j=0;j<16;j++) sum+=aa[tid.y][j]*bb[j][tid.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(row<m && col<n) store(c,row*n+col,p[4],sum);
}

// Diagnostic-only ordered reduction; never selected for production BF16 inference.
kernel void rmsnorm_ordered(ARGS, uint row [[thread_position_in_grid]]) {
    uint w=p[1], base=row*w; if(base>=p[0]) return;
    float sum=0; for(uint j=0;j<w;j++) {float x=load(a,base+j,p[4]);sum+=x*x;}
    float inv=rsqrt(sum/float(w)+as_type<float>(p[7]));
    for(uint j=0;j<w;j++) store(c,base+j,p[4],load(a,base+j,p[4])*inv*load(b,j,p[4]));
}

kernel void softmax_ordered(ARGS, uint row [[thread_position_in_grid]]) {
    uint w=p[1], base=row*w; if(base>=p[0]) return;
    float mx=-INFINITY; for(uint j=0;j<w;j++) mx=max(mx,load(a,base+j,p[4]));
    float sum=0; for(uint j=0;j<w;j++) sum+=exp(load(a,base+j,p[4])-mx);
    for(uint j=0;j<w;j++) store(c,base+j,p[4],exp(load(a,base+j,p[4])-mx)/sum);
}
// Experimental native storage SIMD-group GEMM. Capability gated on the host.
template<typename T>
void native_project(device const uchar* a, device const uchar* b, device uchar* c, constant uint* p,
                    uint tid, uint2 group, threadgroup T* aa, threadgroup T* bb, threadgroup float* cc) {
    uint row=group.y*8, col=group.x*8, k=p[2], n=p[3], m=p[1];
    simdgroup_float8x8 sum(0.f);
    for(uint tile=0;tile<k;tile+=8) {
        for(uint i=tid;i<64;i+=32) {
            uint r=i/8, j=i%8;
            aa[i]=(row+r<m && tile+j<k)?((device const T*)a)[(row+r)*k+tile+j]:T(0);
            bb[i]=(col+j<n && tile+r<k)?((device const T*)b)[(col+j)*k+tile+r]:T(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_matrix<T,8,8> x,y;
        simdgroup_load(x,aa,8); simdgroup_load(y,bb,8);
        simdgroup_multiply_accumulate(sum,x,y,sum);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(sum,cc,8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<64;i+=32) if(row+i/8<m && col+i%8<n) store(c,(row+i/8)*n+col+i%8,p[4],cc[i]);
}
kernel void project_bf16(ARGS, uint tid [[thread_index_in_threadgroup]], uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup bfloat aa[64],bb[64]; threadgroup float cc[64];
    native_project<bfloat>(a,b,c,p,tid,group,aa,bb,cc);
}
kernel void project_f16(ARGS, uint tid [[thread_index_in_threadgroup]], uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup half aa[64],bb[64]; threadgroup float cc[64];
    native_project<half>(a,b,c,p,tid,group,aa,bb,cc);
}
