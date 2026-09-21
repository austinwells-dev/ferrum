// Four independent rows per SIMD group share activation loads and expose
// independent accumulation chains. Each lane retains its ascending stride-32 sum.
template<typename T>
void gemv_rows(device const T* a, device const T* b, device uchar* c, constant uint* p, uint tid, uint group) {
    uint lane=tid%32, base=(group*4+tid/32)*4;
    float sums[4]={0.f,0.f,0.f,0.f};
    for(uint j=lane;j<p[2];j+=32) {
        float x=float(a[j]);
        #pragma unroll
        for(uint r=0;r<4;r++) if(base+r<p[3]) sums[r]+=x*float(b[(base+r)*p[2]+j]);
    }
    #pragma unroll
    for(uint r=0;r<4;r++) {
        float sum=simd_sum(sums[r]);
        if(lane==0 && base+r<p[3]) store(c,base+r,p[4],sum);
    }
}
kernel void gemv(ARGS, uint tid [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]]) {
    if(p[4]==2) gemv_rows<bfloat>((device const bfloat*)a,(device const bfloat*)b,c,p,tid,group);
    else if(p[4]==1) gemv_rows<half>((device const half*)a,(device const half*)b,c,p,tid,group);
    else gemv_rows<float>((device const float*)a,(device const float*)b,c,p,tid,group);
}
