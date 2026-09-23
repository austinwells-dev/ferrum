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
inline float round_storage(float x, uint dtype) {
    if(dtype==1) return float(half(x));
    if(dtype==2) {
        uint bits=as_type<uint>(x);
        uint rounded=isnan(x)?((bits>>16)|0x40):((bits+0x7fff+((bits>>16)&1))>>16);
        return as_type<float>(rounded<<16);
    }
    return x;
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
// Preserve the storage rounding of the separate scale kernel before reduction.
inline float attention_scaled(device const uchar* a, uint i, uint j, uint row, constant uint* p) {
    if(j>p[5]+row%p[2]) return -INFINITY;
    float x=load(a,i,p[4])*as_type<float>(p[7]);
    if(p[4]==1) return float(half(x));
    if(p[4]==2) {
        uint bits=as_type<uint>(x);
        uint rounded=isnan(x)?((bits>>16)|0x40):((bits+0x7fff+((bits>>16)&1))>>16);
        return as_type<float>(rounded<<16);
    }
    return x;
}
kernel void attention_softmax(ARGS, uint tid [[thread_index_in_threadgroup]], uint row [[threadgroup_position_in_grid]]) {
    threadgroup float partial[8];
    uint w=p[1], base=row*w, lane=tid%32, simd=tid/32;
    float mx=-INFINITY;
    for(uint j=tid;j<w;j+=256) mx=max(mx,attention_scaled(a,base+j,j,row,p));
    mx=simd_max(mx);
    if(lane==0) partial[simd]=mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx=simd_max(lane<8?partial[lane]:-INFINITY);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum=0;
    for(uint j=tid;j<w;j+=256) sum+=exp(attention_scaled(a,base+j,j,row,p)-mx);
    sum=simd_sum(sum);
    if(lane==0) partial[simd]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum=simd_sum(lane<8?partial[lane]:0.f);
    for(uint j=tid;j<w;j+=256) store(c,base+j,p[4],exp(attention_scaled(a,base+j,j,row,p)-mx)/sum);
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
inline float q4_0_weight(device const uchar* weights, uint row, uint column, uint k) {
    uint block_index=column>>5, within=column&31, blocks=k>>5;
    device const uchar* block=weights+(row*blocks+block_index)*18;
    float scale=float(*((device const half*)block));
    uchar packed=block[2+(within&15)];
    int q=within<16 ? int(packed&15)-8 : int(packed>>4)-8;
    return scale*float(q);
}
// GGML Q4_0 uses { f16 d, 16 bytes of paired low/high nibbles } per 32 values.
kernel void q4_0_gemv(ARGS, uint tid [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]]) {
    uint lane=tid%32, simd=tid/32, row=group*4+simd;
    uint k=p[2], blocks=k/32;
    float sum0=0.f, sum1=0.f, sum2=0.f, sum3=0.f;
    if(row<p[3]) {
        for(uint tile=0;tile<blocks/4;tile++) {
            uint base=tile*4*32+lane;
            sum0+=q4_0_weight(b,row,base+0*32,k)*load(a,base+0*32,p[4]);
            sum1+=q4_0_weight(b,row,base+1*32,k)*load(a,base+1*32,p[4]);
            sum2+=q4_0_weight(b,row,base+2*32,k)*load(a,base+2*32,p[4]);
            sum3+=q4_0_weight(b,row,base+3*32,k)*load(a,base+3*32,p[4]);
        }
        for(uint block=blocks/4*4;block<blocks;block++) {
            uint column=block*32+lane;
            sum0+=q4_0_weight(b,row,column,k)*load(a,column,p[4]);
        }
    }
    float sum=simd_sum((sum0+sum1)+(sum2+sum3));
    if(lane==0 && row<p[3]) store(c,row,p[4],sum);
}
kernel void q4_0_gemm(ARGS, uint tid [[thread_index_in_threadgroup]], uint2 group [[threadgroup_position_in_grid]]) {
    uint lane=tid%32, simd=tid/32, row=group.x*4+simd, batch=group.y*4;
    uint k=p[2], blocks=k/32;
    float sum0=0.f, sum1=0.f, sum2=0.f, sum3=0.f;
    if(row<p[3]) for(uint block=0;block<blocks;block++) {
        uint column=block*32+lane;
        float w=q4_0_weight(b,row,column,k);
        if(batch+0<p[1]) sum0+=w*load(a,(batch+0)*k+column,p[4]);
        if(batch+1<p[1]) sum1+=w*load(a,(batch+1)*k+column,p[4]);
        if(batch+2<p[1]) sum2+=w*load(a,(batch+2)*k+column,p[4]);
        if(batch+3<p[1]) sum3+=w*load(a,(batch+3)*k+column,p[4]);
    }
    sum0=simd_sum(sum0); sum1=simd_sum(sum1); sum2=simd_sum(sum2); sum3=simd_sum(sum3);
    if(lane==0 && row<p[3]) {
        if(batch+0<p[1]) store(c,(batch+0)*p[3]+row,p[4],sum0);
        if(batch+1<p[1]) store(c,(batch+1)*p[3]+row,p[4],sum1);
        if(batch+2<p[1]) store(c,(batch+2)*p[3]+row,p[4],sum2);
        if(batch+3<p[1]) store(c,(batch+3)*p[3]+row,p[4],sum3);
    }
}
kernel void embedding_gather_q4_0(ARGS, uint i [[thread_position_in_grid]]) {
    if(i>=p[0]) return;
    uint width=p[1], token=i/width, column=i%width;
    uint id=((device const uint*)b)[token];
    store(c,i,p[4],q4_0_weight(a,id,column,width));
}
inline float q6_k_weight(device const uchar* weights, uint row, uint column, uint k) {
    uint block_index=column>>8, within=column&255, blocks=k>>8;
    device const uchar* block=weights+(row*blocks+block_index)*210;
    uint group=within>>5, lane=within&31, half_block=group>>2, slice=group&3;
    uint ql_index=half_block*64+((slice&1)*32)+lane;
    uchar packed=block[ql_index];
    uint low=slice<2 ? uint(packed&15) : uint(packed>>4);
    uint high=(uint(block[128+half_block*32+lane])>>(slice*2))&3;
    int q=int(low|(high<<4))-32;
    uint scale_index=192+half_block*8+slice*2+(lane>>4);
    char scale=((device const char*)block)[scale_index];
    float d=float(*((device const half*)(block+208)));
    return d*float(scale)*float(q);
}
// GGML Q6_K: 256 values, ql[128], qh[64], signed scales[16], f16 d.
kernel void q6_k_gemv(ARGS, uint tid [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]]) {
    uint lane=tid%32, simd=tid/32, row=group*4+simd;
    uint k=p[2], chunks=k/32;
    float sum0=0.f, sum1=0.f, sum2=0.f, sum3=0.f;
    if(row<p[3]) {
        for(uint tile=0;tile<chunks/4;tile++) {
            uint base=tile*4*32+lane;
            sum0+=q6_k_weight(b,row,base+0*32,k)*load(a,base+0*32,p[4]);
            sum1+=q6_k_weight(b,row,base+1*32,k)*load(a,base+1*32,p[4]);
            sum2+=q6_k_weight(b,row,base+2*32,k)*load(a,base+2*32,p[4]);
            sum3+=q6_k_weight(b,row,base+3*32,k)*load(a,base+3*32,p[4]);
        }
        for(uint chunk=chunks/4*4;chunk<chunks;chunk++) {
            uint column=chunk*32+lane;
            sum0+=q6_k_weight(b,row,column,k)*load(a,column,p[4]);
        }
    }
    float sum=simd_sum((sum0+sum1)+(sum2+sum3));
    if(lane==0 && row<p[3]) store(c,row,p[4],sum);
}
kernel void q6_k_gemm(ARGS, uint tid [[thread_index_in_threadgroup]], uint2 group [[threadgroup_position_in_grid]]) {
    uint lane=tid%32, simd=tid/32, row=group.x*4+simd, batch=group.y*4;
    uint k=p[2], chunks=k/32;
    float sum0=0.f, sum1=0.f, sum2=0.f, sum3=0.f;
    if(row<p[3]) for(uint chunk=0;chunk<chunks;chunk++) {
        uint column=chunk*32+lane;
        float w=q6_k_weight(b,row,column,k);
        if(batch+0<p[1]) sum0+=w*load(a,(batch+0)*k+column,p[4]);
        if(batch+1<p[1]) sum1+=w*load(a,(batch+1)*k+column,p[4]);
        if(batch+2<p[1]) sum2+=w*load(a,(batch+2)*k+column,p[4]);
        if(batch+3<p[1]) sum3+=w*load(a,(batch+3)*k+column,p[4]);
    }
    sum0=simd_sum(sum0); sum1=simd_sum(sum1); sum2=simd_sum(sum2); sum3=simd_sum(sum3);
    if(lane==0 && row<p[3]) {
        if(batch+0<p[1]) store(c,(batch+0)*p[3]+row,p[4],sum0);
        if(batch+1<p[1]) store(c,(batch+1)*p[3]+row,p[4],sum1);
        if(batch+2<p[1]) store(c,(batch+2)*p[3]+row,p[4],sum2);
        if(batch+3<p[1]) store(c,(batch+3)*p[3]+row,p[4],sum3);
    }
}
kernel void embedding_gather_q6_k(ARGS, uint i [[thread_position_in_grid]]) {
    if(i>=p[0]) return;
    uint width=p[1], token=i/width, column=i%width;
    uint id=((device const uint*)b)[token];
    store(c,i,p[4],q6_k_weight(a,id,column,width));
}
// GGML Q8_0 rows are packed as repeated { half scale, 32 signed bytes } blocks.
// One SIMD group reduces one output row; four rows share a 128-thread group.
kernel void q8_0_gemv(ARGS, uint tid [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]]) {
    uint lane=tid%32, simd=tid/32, row=group*4+simd;
    uint k=p[2], blocks=k/32;
    float sum0=0.f, sum1=0.f, sum2=0.f, sum3=0.f;
    if(row<p[3]) {
        device const uchar* row_weights=b+row*blocks*34;
        uint groups=blocks/4, block=0;
        for(uint tile=0;tile<groups;tile++) {
            block=tile*4;
            device const uchar* packed0=row_weights+(block+0)*34;
            device const uchar* packed1=row_weights+(block+1)*34;
            device const uchar* packed2=row_weights+(block+2)*34;
            device const uchar* packed3=row_weights+(block+3)*34;
            float scale0=float(*((device const half*)packed0));
            float scale1=float(*((device const half*)packed1));
            float scale2=float(*((device const half*)packed2));
            float scale3=float(*((device const half*)packed3));
            char q0=((device const char*)(packed0+2))[lane];
            char q1=((device const char*)(packed1+2))[lane];
            char q2=((device const char*)(packed2+2))[lane];
            char q3=((device const char*)(packed3+2))[lane];
            sum0+=(scale0*float(q0))*load(a,(block+0)*32+lane,p[4]);
            sum1+=(scale1*float(q1))*load(a,(block+1)*32+lane,p[4]);
            sum2+=(scale2*float(q2))*load(a,(block+2)*32+lane,p[4]);
            sum3+=(scale3*float(q3))*load(a,(block+3)*32+lane,p[4]);
        }
        for(uint tail=groups*4;tail<blocks;tail++) {
            device const uchar* packed=row_weights+tail*34;
            float scale=float(*((device const half*)packed));
            char q=((device const char*)(packed+2))[lane];
            sum0+=(scale*float(q))*load(a,tail*32+lane,p[4]);
        }
    }
    float sum=simd_sum((sum0+sum1)+(sum2+sum3));
    if(lane==0 && row<p[3]) store(c,row,p[4],sum);
}
// One SIMD group handles one output channel across four independent sequence
// rows. The packed weight and its scale are loaded once for four outputs.
kernel void q8_0_gemm(ARGS, uint tid [[thread_index_in_threadgroup]], uint2 group [[threadgroup_position_in_grid]]) {
    uint lane=tid%32, simd=tid/32, row=group.x*4+simd, batch=group.y*4;
    uint k=p[2], blocks=k/32;
    // Keep four independent K reduction chains per prompt row. Besides hiding
    // dependent-add latency, the block-major loop reuses each packed weight
    // across all four prompt rows in this SIMD group.
    float sum00=0.f, sum01=0.f, sum02=0.f, sum03=0.f;
    float sum10=0.f, sum11=0.f, sum12=0.f, sum13=0.f;
    float sum20=0.f, sum21=0.f, sum22=0.f, sum23=0.f;
    float sum30=0.f, sum31=0.f, sum32=0.f, sum33=0.f;
    if(row<p[3]) {
        device const uchar* row_weights=b+row*blocks*34;
        uint groups=blocks/4;
        for(uint tile=0;tile<groups;tile++) {
            uint block=tile*4;
            device const uchar* packed0=row_weights+(block+0)*34;
            device const uchar* packed1=row_weights+(block+1)*34;
            device const uchar* packed2=row_weights+(block+2)*34;
            device const uchar* packed3=row_weights+(block+3)*34;
            float w0=float(*((device const half*)packed0))*float(((device const char*)(packed0+2))[lane]);
            float w1=float(*((device const half*)packed1))*float(((device const char*)(packed1+2))[lane]);
            float w2=float(*((device const half*)packed2))*float(((device const char*)(packed2+2))[lane]);
            float w3=float(*((device const half*)packed3))*float(((device const char*)(packed3+2))[lane]);
            uint column=block*32+lane;
            if(batch+0<p[1]) {
                sum00+=w0*load(a,(batch+0)*k+column+0*32,p[4]);
                sum01+=w1*load(a,(batch+0)*k+column+1*32,p[4]);
                sum02+=w2*load(a,(batch+0)*k+column+2*32,p[4]);
                sum03+=w3*load(a,(batch+0)*k+column+3*32,p[4]);
            }
            if(batch+1<p[1]) {
                sum10+=w0*load(a,(batch+1)*k+column+0*32,p[4]);
                sum11+=w1*load(a,(batch+1)*k+column+1*32,p[4]);
                sum12+=w2*load(a,(batch+1)*k+column+2*32,p[4]);
                sum13+=w3*load(a,(batch+1)*k+column+3*32,p[4]);
            }
            if(batch+2<p[1]) {
                sum20+=w0*load(a,(batch+2)*k+column+0*32,p[4]);
                sum21+=w1*load(a,(batch+2)*k+column+1*32,p[4]);
                sum22+=w2*load(a,(batch+2)*k+column+2*32,p[4]);
                sum23+=w3*load(a,(batch+2)*k+column+3*32,p[4]);
            }
            if(batch+3<p[1]) {
                sum30+=w0*load(a,(batch+3)*k+column+0*32,p[4]);
                sum31+=w1*load(a,(batch+3)*k+column+1*32,p[4]);
                sum32+=w2*load(a,(batch+3)*k+column+2*32,p[4]);
                sum33+=w3*load(a,(batch+3)*k+column+3*32,p[4]);
            }
        }
        for(uint block=groups*4;block<blocks;block++) {
            device const uchar* packed=row_weights+block*34;
            float w=float(*((device const half*)packed))*float(((device const char*)(packed+2))[lane]);
            uint column=block*32+lane;
            if(batch+0<p[1]) sum00+=w*load(a,(batch+0)*k+column,p[4]);
            if(batch+1<p[1]) sum10+=w*load(a,(batch+1)*k+column,p[4]);
            if(batch+2<p[1]) sum20+=w*load(a,(batch+2)*k+column,p[4]);
            if(batch+3<p[1]) sum30+=w*load(a,(batch+3)*k+column,p[4]);
        }
    }
    float sum0=simd_sum((sum00+sum01)+(sum02+sum03));
    float sum1=simd_sum((sum10+sum11)+(sum12+sum13));
    float sum2=simd_sum((sum20+sum21)+(sum22+sum23));
    float sum3=simd_sum((sum30+sum31)+(sum32+sum33));
    if(lane==0 && row<p[3]) {
        if(batch+0<p[1]) store(c,(batch+0)*p[3]+row,p[4],sum0);
        if(batch+1<p[1]) store(c,(batch+1)*p[3]+row,p[4],sum1);
        if(batch+2<p[1]) store(c,(batch+2)*p[3]+row,p[4],sum2);
        if(batch+3<p[1]) store(c,(batch+3)*p[3]+row,p[4],sum3);
    }
}
kernel void embedding_gather_q8_0(ARGS, uint i [[thread_position_in_grid]]) {
    if(i>=p[0]) return;
    uint width=p[1], token=i/width, column=i%width;
    uint id=((device const uint*)b)[token];
    uint blocks=width/32, block=column/32, lane=column%32;
    device const uchar* packed=a+id*blocks*34+block*34;
    float scale=float(*((device const half*)packed));
    char q=((device const char*)(packed+2))[lane];
    store(c,i,p[4],scale*float(q));
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

// Aligned four-element loads expose independent accumulation chains and reduce
// loop/addressing overhead. Host selects only aligned, K-divisible-by-four inputs.
template<typename T>
void gemv_vector_impl(device const vec<T,4>* a, device const vec<T,4>* b, device uchar* c,
                      constant uint* p, uint tid, uint group) {
    uint lane=tid%32, row=group*4+tid/32, k4=p[2]/4;
    float4 sum=0.f;
    if(row<p[3]) for(uint j=lane;j<k4;j+=32) sum+=float4(a[j])*float4(b[row*k4+j]);
    float total=simd_sum((sum.x+sum.y)+(sum.z+sum.w));
    if(lane==0 && row<p[3]) store(c,row,p[4],total);
}
kernel void gemv_vector(ARGS, uint tid [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]]) {
    if(p[4]==2) gemv_vector_impl<bfloat>((device const bfloat4*)a,(device const bfloat4*)b,c,p,tid,group);
    else gemv_vector_impl<half>((device const half4*)a,(device const half4*)b,c,p,tid,group);
}

// Four SIMD groups split K for each output row. This trades additional active
// lanes and one small cross-SIMD reduction for less serial weight streaming.
template<typename T>
void gemv_wide_impl(device const vec<T,4>* a, device const vec<T,4>* b,
                    device uchar* c, constant uint* p, uint tid, uint row,
                    threadgroup float* partial) {
    uint lane=tid%32, simd=tid/32, k4=p[2]/4;
    float4 sum=0.f;
    if(row<p[3]) for(uint j=simd*32+lane;j<k4;j+=128)
        sum+=float4(a[j])*float4(b[row*k4+j]);
    float chunk=simd_sum((sum.x+sum.y)+(sum.z+sum.w));
    if(lane==0) partial[simd]=chunk;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(simd==0) {
        float total=lane<4?partial[lane]:0.f;
        total=simd_sum(total);
        if(lane==0 && row<p[3]) store(c,row,p[4],total);
    }
}
kernel void gemv_wide(ARGS, uint tid [[thread_index_in_threadgroup]], uint row [[threadgroup_position_in_grid]]) {
    threadgroup float partial[4];
    if(p[4]==2) gemv_wide_impl<bfloat>((device const bfloat4*)a,(device const bfloat4*)b,c,p,tid,row,partial);
    else gemv_wide_impl<half>((device const half4*)a,(device const half4*)b,c,p,tid,row,partial);
}

// Grouped attention products: retain explicit score/probability storage boundaries.
template<typename T, bool context>
void attention_matrix(device const uchar* a, device const uchar* b, device uchar* c,
                      constant uint* p, uint tid, uint2 group,
                      threadgroup T* aa, threadgroup T* bb, threadgroup float* cc) {
    uint s=p[1], t=p[2], heads=p[3], kv=p[5], d=p[6];
    uint blocks=(s+7)/8, h=group.y/blocks, row=(group.y%blocks)*8, col=group.x*8;
    uint kh=h/(heads/kv), inner=context?t:d, width=context?d:t;
    simdgroup_float8x8 sum(0.f);
    for(uint tile=0;tile<inner;tile+=8) {
        for(uint i=tid;i<64;i+=32) {
            uint r=i/8,j=i%8;
            if(context) {
                aa[i]=(row+r<s && tile+j<t)?((device const T*)a)[(h*s+row+r)*t+tile+j]:T(0);
                bb[i]=(tile+r<t && col+j<d)?((device const T*)b)[(tile+r)*kv*d+kh*d+col+j]:T(0);
            } else {
                aa[i]=(row+r<s && tile+j<d)?((device const T*)a)[(row+r)*heads*d+h*d+tile+j]:T(0);
                bb[i]=(col+j<t && tile+r<d)?((device const T*)b)[(col+j)*kv*d+kh*d+tile+r]:T(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_matrix<T,8,8> x,y;
        simdgroup_load(x,aa,8); simdgroup_load(y,bb,8);
        simdgroup_multiply_accumulate(sum,x,y,sum);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(sum,cc,8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<64;i+=32) if(row+i/8<s && col+i%8<width) {
        uint dest=context?((row+i/8)*heads*d+h*d+col+i%8):((h*s+row+i/8)*t+col+i%8);
        store(c,dest,p[4],cc[i]);
    }
}
kernel void attention_scores_matrix(ARGS, uint tid [[thread_index_in_threadgroup]], uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup bfloat aa[64],bb[64]; threadgroup float cc[64];
    attention_matrix<bfloat,false>(a,b,c,p,tid,group,aa,bb,cc);
}
kernel void attention_context_matrix(ARGS, uint tid [[thread_index_in_threadgroup]], uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup bfloat aa[64],bb[64]; threadgroup float cc[64];
    attention_matrix<bfloat,true>(a,b,c,p,tid,group,aa,bb,cc);
}

// Four SIMD groups share a 16x32 output tile and K=32 staging. Each SIMD
// accumulates two 8x8 tiles; input reuse amortizes threadgroup barriers.
template<typename T>
void project_wide(device const uchar* a, device const uchar* b, device uchar* c,
                  constant uint* p, uint tid, uint2 group,
                  threadgroup T* aa, threadgroup T* bb, threadgroup float* cc) {
    uint m=p[1],k=p[2],n=p[3],row=group.y*16,col=group.x*32;
    uint sg=tid/32,rr=(sg/2)*8,cr=(sg%2)*16;
    simdgroup_float8x8 sum0(0.f),sum1(0.f);
    for(uint tile=0;tile<k;tile+=32) {
        for(uint i=tid;i<512;i+=128) {
            uint r=i/32,j=i%32;
            aa[i]=(row+r<m && tile+j<k)?((device const T*)a)[(row+r)*k+tile+j]:T(0);
        }
        for(uint i=tid;i<1024;i+=128) {
            uint r=i/32,j=i%32;
            bb[j*32+r]=(col+r<n && tile+j<k)?((device const T*)b)[(col+r)*k+tile+j]:T(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint j=0;j<32;j+=8) {
            simdgroup_matrix<T,8,8> x,y0,y1;
            simdgroup_load(x,aa+rr*32+j,32);
            simdgroup_load(y0,bb+j*32+cr,32);
            simdgroup_load(y1,bb+j*32+cr+8,32);
            simdgroup_multiply_accumulate(sum0,x,y0,sum0);
            simdgroup_multiply_accumulate(sum1,x,y1,sum1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(sum0,cc+rr*32+cr,32);
    simdgroup_store(sum1,cc+rr*32+cr+8,32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<512;i+=128) if(row+i/32<m && col+i%32<n)
        store(c,(row+i/32)*n+col+i%32,p[4],cc[i]);
}
kernel void project_wide_bf16(ARGS, uint tid [[thread_index_in_threadgroup]], uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup bfloat aa[512],bb[1024]; threadgroup float cc[512];
    project_wide<bfloat>(a,b,c,p,tid,group,aa,bb,cc);
}
kernel void project_wide_f16(ARGS, uint tid [[thread_index_in_threadgroup]], uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup half aa[512],bb[1024]; threadgroup float cc[512];
    project_wide<half>(a,b,c,p,tid,group,aa,bb,cc);
}

// Decode context: 32 adjacent output columns, eight independent T partitions.
// Neighboring lanes read adjacent V elements; shared storage combines partials.
kernel void attention_context_decode(ARGS, uint tid [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]]) {
    uint d=p[6], tiles=(d+31)/32, h=group/tiles, col=(group%tiles)*32+tid%32;
    uint part=tid/32, kv=h/(p[3]/p[5]);
    float sum=0;
    if(col<d) for(uint t=part;t<p[2];t+=8)
        sum+=load(a,h*p[2]+t,p[4])*load(b,(t*p[5]+kv)*d+col,p[4]);
    threadgroup float partial[256];
    partial[tid]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid<32 && col<d) {
        sum=0;
        for(uint j=0;j<8;j++) sum+=partial[j*32+tid];
        store(c,h*d+col,p[4],sum);
    }
}

// Preserve the SiLU output's storage rounding before the gated multiply.
kernel void silu_mul(ARGS, uint i [[thread_position_in_grid]]) {
    if(i>=p[0]) return;
    float x=load(a,i,p[4]);
    float sigmoid=x>=0?1.f/(1.f+exp(-x)):exp(x)/(1.f+exp(x));
    store(c,i,p[4],round_storage(x*sigmoid,p[4])*load(b,i,p[4]));
}
