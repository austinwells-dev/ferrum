#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// Raw-buffer views keep the existing checked binding/lifetime ABI. MPP is an
// inline shader primitive, not a host inference engine. Input views are read-only
// by contract (MPP requires non-const element types). Accumulation is FP32
// with relaxed_precision=false; the final store keeps Ferrum's BF16 rounding.
kernel void project_mpp(device bfloat* a [[buffer(0)]],
                        device bfloat* b [[buffer(1)]],
                        device ushort* c [[buffer(2)]],
                        constant uint* p [[buffer(3)]],
                        uint2 group [[threadgroup_position_in_grid]]) {
    int m=int(p[1]), k=int(p[2]), n=int(p[3]);
    tensor<device bfloat, dextents<int,2>, tensor_inline> A(a,dextents<int,2>(k,m));
    tensor<device bfloat, dextents<int,2>, tensor_inline> B(b,dextents<int,2>(k,n));
    auto left=A.slice(0,int(group.y)*64);
    auto right=B.slice(0,int(group.x)*64);
    constexpr auto desc=matmul2d_descriptor(64,64,static_cast<int>(dynamic_extent),false,true,false,matmul2d_descriptor::mode::multiply);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto result=op.get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
    op.run(left,right,result);
    #pragma unroll
    for(uint i=0;i<result.get_capacity();i++) {
        {
            auto coord=result.get_multidimensional_index(i);
            uint col=group.x*64+coord[0], row=group.y*64+coord[1];
            if(row<uint(m) && col<uint(n)) {
                float x=result[i]; uint bits=as_type<uint>(x);
                c[row*uint(n)+col]=isnan(x)?ushort((bits>>16)|0x40):ushort((bits+0x7fff+((bits>>16)&1))>>16);
            }
        }
    }
}

// Q8_0 weights are expanded only into a 64x128 threadgroup tile immediately
// before MPP consumes it. Each packed weight tile is reused across 64 prompt
// rows and remains packed in model storage.
kernel void q8_0_gemm_mpp(device bfloat* a [[buffer(0)]],
                          device const uchar* packed [[buffer(1)]],
                          device ushort* c [[buffer(2)]],
                          constant uint* p [[buffer(3)]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint2 group [[threadgroup_position_in_grid]]) {
    constexpr int TILE_M=64, TILE_N=64, TILE_K=128;
    threadgroup bfloat dequantized[TILE_N*TILE_K];
    int m=int(p[1]), k=int(p[2]), n=int(p[3]);
    int blocks=k/32;
    tensor<device bfloat, dextents<int,2>, tensor_inline> A(a,dextents<int,2>(k,m));
    tensor<threadgroup bfloat, dextents<int,2>, tensor_inline> B(
        dequantized,dextents<int,2>(TILE_K,TILE_N));
    auto left=A.slice(0,int(group.y)*TILE_M);
    constexpr auto desc=matmul2d_descriptor(
        TILE_M,TILE_N,TILE_K,false,true,false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto result=op.get_destination_cooperative_tensor<decltype(left),decltype(B),float>();
    int k_tiles=(k+TILE_K-1)/TILE_K;
    for(int kt=0;kt<k_tiles;kt++) {
        if(kt>0) threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint i=tid;i<uint(TILE_N*TILE_K);i+=128) {
            uint out_col=i/TILE_K, in_col=i%TILE_K;
            int channel=int(group.x)*TILE_N+int(out_col);
            int source_k=kt*TILE_K+int(in_col);
            bfloat value=bfloat(0.0f);
            if(channel<n && source_k<k) {
                uint block=uint(source_k/32);
                device const uchar* qblock=packed+(uint(channel)*uint(blocks)+block)*34;
                float scale=float(*((device const half*)qblock));
                char q=((device const char*)(qblock+2))[source_k%32];
                value=bfloat(scale*float(q));
            }
            dequantized[i]=value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto right=B;
        auto left_k=A.slice(kt*TILE_K,int(group.y)*TILE_M);
        op.run(left_k,right,result);
    }
    for(uint i=0;i<result.get_capacity();i++) {
        auto coord=result.get_multidimensional_index(i);
        uint col=group.x*TILE_N+coord[0], row=group.y*TILE_M+coord[1];
        if(row<uint(m) && col<uint(n)) {
            float x=result[i]; uint bits=as_type<uint>(x);
            c[row*uint(n)+col]=isnan(x)?ushort((bits>>16)|0x40):ushort((bits+0x7fff+((bits>>16)&1))>>16);
        }
    }
}

// Strided tensor views describe grouped sequence-major Q/K/V without copies.
template<bool Context>
void grouped_mpp(device bfloat* a, device bfloat* b, device ushort* c,
                 constant uint* p, uint3 group) {
    int s=int(p[1]), t=int(p[2]), heads=int(p[3]), kv=int(p[5]), d=int(p[6]);
    int h=int(group.z), kh=h/(heads/kv);
    tensor<device bfloat,dextents<int,2>,tensor_inline> A(
        a+(Context?h*s*t:h*d),dextents<int,2>(Context?t:d,s),
        array<int,2>{1,Context?t:heads*d});
    tensor<device bfloat,dextents<int,2>,tensor_inline> B(
        b+kh*d,dextents<int,2>(d,t),array<int,2>{1,kv*d});
    auto left=A.slice(0,int(group.y)*64);
    auto right=Context?B.slice(int(group.x)*64,0):B.slice(0,int(group.x)*64);
    constexpr auto desc=matmul2d_descriptor(64,64,static_cast<int>(dynamic_extent),false,!Context,false,matmul2d_descriptor::mode::multiply);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto result=op.template get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
    op.run(left,right,result);
    #pragma unroll
    for(uint i=0;i<result.get_capacity();i++) {
        auto coord=result.get_multidimensional_index(i);
        uint col=group.x*64+coord[0], row=group.y*64+coord[1];
        if(row<uint(s) && col<uint(Context?d:t)) {
            uint dest=Context?((row*uint(heads)+uint(h))*uint(d)+col):((uint(h)*uint(s)+row)*uint(t)+col);
            float x=result[i]; uint bits=as_type<uint>(x);
            c[dest]=isnan(x)?ushort((bits>>16)|0x40):ushort((bits+0x7fff+((bits>>16)&1))>>16);
        }
    }
}
kernel void attention_scores_mpp(device bfloat* a [[buffer(0)]], device bfloat* b [[buffer(1)]], device ushort* c [[buffer(2)]], constant uint* p [[buffer(3)]], uint3 group [[threadgroup_position_in_grid]]) {
    grouped_mpp<false>(a,b,c,p,group);
}
kernel void attention_context_mpp(device bfloat* a [[buffer(0)]], device bfloat* b [[buffer(1)]], device ushort* c [[buffer(2)]], constant uint* p [[buffer(3)]], uint3 group [[threadgroup_position_in_grid]]) {
    grouped_mpp<true>(a,b,c,p,group);
}
