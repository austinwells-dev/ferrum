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

// Smaller K tiles reduce threadgroup staging and are tested against the
// production 64x128 tile for Q8_0 prompt GEMM.
kernel void q8_0_gemm_mpp_k64(device bfloat* a [[buffer(0)]],
                              device const uchar* packed [[buffer(1)]],
                              device ushort* c [[buffer(2)]],
                              constant uint* p [[buffer(3)]],
                              uint tid [[thread_index_in_threadgroup]],
                              uint2 group [[threadgroup_position_in_grid]]) {
    constexpr int TILE_M=64, TILE_N=64, TILE_K=64;
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

kernel void q4_0_gemm_mpp(device bfloat* a [[buffer(0)]],
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
        // One lane loads a packed byte and expands both nibbles. The two
        // values occupy the low and high halves of the GGML block, so keeping
        // this mapping explicit avoids a second scale and byte load per pair.
        for(uint i=tid;i<uint(TILE_N*(TILE_K/2));i+=128) {
            uint out_col=i/(TILE_K/2), packed_index=i%(TILE_K/2);
            uint block_in_tile=packed_index/16, byte_in_block=packed_index%16;
            int channel=int(group.x)*TILE_N+int(out_col);
            int source_k=kt*TILE_K+int(block_in_tile*32+byte_in_block);
            if(channel<n && source_k<k) {
                uint block_index=uint(source_k)/32;
                device const uchar* block=packed+(uint(channel)*uint(blocks)+block_index)*18;
                float scale=float(*((device const half*)block));
                uchar q=block[2+byte_in_block];
                int low=int(q&15)-8, high=int(q>>4)-8;
                uint tile_index=out_col*TILE_K+block_in_tile*32+byte_in_block;
                dequantized[tile_index]=bfloat(scale*float(low));
                dequantized[tile_index+16]=bfloat(scale*float(high));
            } else {
                uint tile_index=out_col*TILE_K+block_in_tile*32+byte_in_block;
                dequantized[tile_index]=bfloat(0.0f);
                dequantized[tile_index+16]=bfloat(0.0f);
            }
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

kernel void q5_0_gemm_mpp(device bfloat* a [[buffer(0)]],
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
        for(uint i=tid;i<uint(TILE_N*(TILE_K/2));i+=128) {
            uint out_col=i/(TILE_K/2), packed_index=i%(TILE_K/2);
            uint block_in_tile=packed_index/16, byte_in_block=packed_index%16;
            int channel=int(group.x)*TILE_N+int(out_col);
            int source_k=kt*TILE_K+int(block_in_tile*32+byte_in_block);
            uint tile_index=out_col*TILE_K+block_in_tile*32+byte_in_block;
            if(channel<n && source_k<k) {
                uint block_index=uint(source_k)/32;
                device const uchar* block=packed+(uint(channel)*uint(blocks)+block_index)*22;
                float d=float(*((device const half*)block));
                uchar qs=block[6+byte_in_block];
                uint qh0=(uint(block[2+(byte_in_block>>3)])>>(byte_in_block&7))&1;
                uint upper=byte_in_block+16;
                uint qh1=(uint(block[2+(upper>>3)])>>(upper&7))&1;
                int q0=int(qs&15)+int(qh0<<4)-16;
                int q1=int(qs>>4)+int(qh1<<4)-16;
                dequantized[tile_index]=bfloat(d*float(q0));
                dequantized[tile_index+16]=bfloat(d*float(q1));
            } else {
                dequantized[tile_index]=bfloat(0.0f);
                dequantized[tile_index+16]=bfloat(0.0f);
            }
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

kernel void q5_1_gemm_mpp(device bfloat* a [[buffer(0)]],
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
        for(uint i=tid;i<uint(TILE_N*(TILE_K/2));i+=128) {
            uint out_col=i/(TILE_K/2), packed_index=i%(TILE_K/2);
            uint block_in_tile=packed_index/16, byte_in_block=packed_index%16;
            int channel=int(group.x)*TILE_N+int(out_col);
            int source_k=kt*TILE_K+int(block_in_tile*32+byte_in_block);
            uint tile_index=out_col*TILE_K+block_in_tile*32+byte_in_block;
            if(channel<n && source_k<k) {
                uint block_index=uint(source_k)/32;
                device const uchar* block=packed+(uint(channel)*uint(blocks)+block_index)*24;
                float d=float(*((device const half*)block));
                float minimum=float(*((device const half*)(block+2)));
                uchar qs=block[8+byte_in_block];
                uint qh0=(uint(block[4+(byte_in_block>>3)])>>(byte_in_block&7))&1;
                uint upper=byte_in_block+16;
                uint qh1=(uint(block[4+(upper>>3)])>>(upper&7))&1;
                uint q0=uint(qs&15)+(qh0<<4);
                uint q1=uint(qs>>4)+(qh1<<4);
                dequantized[tile_index]=bfloat(d*float(q0)+minimum);
                dequantized[tile_index+16]=bfloat(d*float(q1)+minimum);
            } else {
                dequantized[tile_index]=bfloat(0.0f);
                dequantized[tile_index+16]=bfloat(0.0f);
            }
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

inline uint qk4_scale_mpp(device const uchar* scales, uint group) {
    if(group<4) return uint(scales[group]&63);
    return uint(scales[group+4]&15) | (uint(scales[group-4]>>6)<<4);
}
inline uint qk4_min_mpp(device const uchar* scales, uint group) {
    if(group<4) return uint(scales[group+4]&63);
    return uint(scales[group+4]>>4) | (uint(scales[group]>>6)<<4);
}

kernel void q4_k_gemm_mpp(device bfloat* a [[buffer(0)]],
                          device const uchar* packed [[buffer(1)]],
                          device ushort* c [[buffer(2)]],
                          constant uint* p [[buffer(3)]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint2 group [[threadgroup_position_in_grid]]) {
    constexpr int TILE_M=64, TILE_N=64, TILE_K=128;
    threadgroup bfloat dequantized[TILE_N*TILE_K];
    int m=int(p[1]), k=int(p[2]), n=int(p[3]);
    int blocks=k/256;
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
        for(uint i=tid;i<uint(TILE_N*(TILE_K/2));i+=128) {
            uint out_col=i/(TILE_K/2), pair_index=i%(TILE_K/2);
            uint half_tile=pair_index/32, lane=pair_index%32;
            int channel=int(group.x)*TILE_N+int(out_col);
            int source_k=kt*TILE_K+int(half_tile*64+lane);
            uint tile_index=out_col*TILE_K+half_tile*64+lane;
            if(channel<n && source_k<k) {
                uint block_index=uint(source_k)>>8, within=uint(source_k)&255;
                uint group0=within>>5, chunk=within>>6;
                device const uchar* block=packed+(uint(channel)*uint(blocks)+block_index)*144;
                float d=float(*((device const half*)block));
                float dmin=float(*((device const half*)(block+2)));
                device const uchar* scales=block+4;
                uchar qs=block[16+chunk*32+lane];
                uint scale0=qk4_scale_mpp(scales,group0);
                uint scale1=qk4_scale_mpp(scales,group0+1);
                uint min0=qk4_min_mpp(scales,group0);
                uint min1=qk4_min_mpp(scales,group0+1);
                dequantized[tile_index]=bfloat(d*float(scale0)*float(qs&15)-dmin*float(min0));
                dequantized[tile_index+32]=bfloat(d*float(scale1)*float(qs>>4)-dmin*float(min1));
            } else {
                dequantized[tile_index]=bfloat(0.0f);
                dequantized[tile_index+32]=bfloat(0.0f);
            }
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

kernel void q5_k_gemm_mpp(device bfloat* a [[buffer(0)]],
                          device const uchar* packed [[buffer(1)]],
                          device ushort* c [[buffer(2)]],
                          constant uint* p [[buffer(3)]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint2 group [[threadgroup_position_in_grid]]) {
    constexpr int TILE_M=64, TILE_N=64, TILE_K=128;
    threadgroup bfloat dequantized[TILE_N*TILE_K];
    int m=int(p[1]), k=int(p[2]), n=int(p[3]);
    int blocks=k/256;
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
        for(uint i=tid;i<uint(TILE_N*(TILE_K/2));i+=128) {
            uint out_col=i/(TILE_K/2), pair_index=i%(TILE_K/2);
            uint half_tile=pair_index/32, lane=pair_index%32;
            int channel=int(group.x)*TILE_N+int(out_col);
            int source_k=kt*TILE_K+int(half_tile*64+lane);
            uint tile_index=out_col*TILE_K+half_tile*64+lane;
            if(channel<n && source_k<k) {
                uint block_index=uint(source_k)>>8, within=uint(source_k)&255;
                uint group0=within>>5, chunk=within>>6;
                device const uchar* block=packed+(uint(channel)*uint(blocks)+block_index)*176;
                float d=float(*((device const half*)block));
                float dmin=float(*((device const half*)(block+2)));
                device const uchar* scales=block+4;
                uchar qs=block[48+chunk*32+lane];
                uchar qh=block[16+lane];
                uint high0=(uint(qh)>>group0)&1;
                uint high1=(uint(qh)>>(group0+1))&1;
                uint scale0=qk4_scale_mpp(scales,group0);
                uint scale1=qk4_scale_mpp(scales,group0+1);
                uint min0=qk4_min_mpp(scales,group0);
                uint min1=qk4_min_mpp(scales,group0+1);
                uint q0=uint(qs&15)|(high0<<4), q1=uint(qs>>4)|(high1<<4);
                dequantized[tile_index]=bfloat(d*float(scale0)*float(q0)-dmin*float(min0));
                dequantized[tile_index+32]=bfloat(d*float(scale1)*float(q1)-dmin*float(min1));
            } else {
                dequantized[tile_index]=bfloat(0.0f);
                dequantized[tile_index+32]=bfloat(0.0f);
            }
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

kernel void mlx_affine4_gemm_mpp(device half* a [[buffer(0)]],
                                 device const uchar* packed [[buffer(1)]],
                                 device ushort* c [[buffer(2)]],
                                 constant uint* p [[buffer(3)]],
                                 uint tid [[thread_index_in_threadgroup]],
                                 uint2 group [[threadgroup_position_in_grid]]) {
    constexpr int TILE_M=64, TILE_N=64, TILE_K=128;
    threadgroup half dequantized[TILE_N*TILE_K];
    int m=int(p[1]), k=int(p[2]), n=int(p[3]);
    int groups=k/64;
    tensor<device half, dextents<int,2>, tensor_inline> A(a,dextents<int,2>(k,m));
    tensor<threadgroup half, dextents<int,2>, tensor_inline> B(
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
        for(uint i=tid;i<uint(TILE_N*(TILE_K/2));i+=128) {
            uint out_col=i/(TILE_K/2), pair_index=i%(TILE_K/2);
            uint group_in_tile=pair_index/32, lane=pair_index%32;
            int channel=int(group.x)*TILE_N+int(out_col);
            int source_k=kt*TILE_K+int(group_in_tile*64+lane*2);
            uint tile_index=out_col*TILE_K+group_in_tile*64+lane*2;
            if(channel<n && source_k<k) {
                uint block_index=uint(source_k)/64;
                device const uchar* block=packed+(uint(channel)*uint(groups)+block_index)*36;
                uchar qs=block[4+lane];
                float scale=float(*((device const half*)block));
                float bias=float(*((device const half*)(block+2)));
                dequantized[tile_index]=half(scale*float(qs&15)+bias);
                dequantized[tile_index+1]=half(scale*float(qs>>4)+bias);
            } else {
                dequantized[tile_index]=half(0.0f);
                dequantized[tile_index+1]=half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto right=B;
        auto left_k=A.slice(kt*TILE_K,int(group.y)*TILE_M);
        op.run(left_k,right,result);
    }
    for(uint i=0;i<result.get_capacity();i++) {
        auto coord=result.get_multidimensional_index(i);
        uint col=group.x*TILE_N+coord[0], row=group.y*TILE_M+coord[1];
        if(row<uint(m) && col<uint(n)) c[row*uint(n)+col]=as_type<ushort>(half(result[i]));
    }
}

// Experimental M5 tile-width sweep. Each tile contains one MLX 64-value Q4
// metadata group; the production K=128 kernel reuses the same 64x64 output tile
// while loading two adjacent groups per MPP step.
kernel void mlx_affine4_gemm_mpp_k64(device half* a [[buffer(0)]],
                                     device const uchar* packed [[buffer(1)]],
                                     device ushort* c [[buffer(2)]],
                                     constant uint* p [[buffer(3)]],
                                     uint tid [[thread_index_in_threadgroup]],
                                     uint2 group [[threadgroup_position_in_grid]]) {
    constexpr int TILE_M=64, TILE_N=64, TILE_K=64;
    threadgroup half dequantized[TILE_N*TILE_K];
    int m=int(p[1]), k=int(p[2]), n=int(p[3]);
    int groups=k/64;
    tensor<device half, dextents<int,2>, tensor_inline> A(a,dextents<int,2>(k,m));
    tensor<threadgroup half, dextents<int,2>, tensor_inline> B(
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
        for(uint i=tid;i<uint(TILE_N*(TILE_K/2));i+=128) {
            uint out_col=i/(TILE_K/2), lane=i%(TILE_K/2);
            int channel=int(group.x)*TILE_N+int(out_col);
            int source_k=kt*TILE_K+int(lane*2);
            uint tile_index=out_col*TILE_K+lane*2;
            if(channel<n && source_k<k) {
                uint block_index=uint(source_k)/64;
                device const uchar* block=packed+(uint(channel)*uint(groups)+block_index)*36;
                uchar qs=block[4+lane];
                float scale=float(*((device const half*)block));
                float bias=float(*((device const half*)(block+2)));
                dequantized[tile_index]=half(scale*float(qs&15)+bias);
                dequantized[tile_index+1]=half(scale*float(qs>>4)+bias);
            } else {
                dequantized[tile_index]=half(0.0f);
                dequantized[tile_index+1]=half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto right=B;
        auto left_k=A.slice(kt*TILE_K,int(group.y)*TILE_M);
        op.run(left_k,right,result);
    }
    for(uint i=0;i<result.get_capacity();i++) {
        auto coord=result.get_multidimensional_index(i);
        uint col=group.x*TILE_N+coord[0], row=group.y*TILE_M+coord[1];
        if(row<uint(m) && col<uint(n)) c[row*uint(n)+col]=as_type<ushort>(half(result[i]));
    }
}

inline float q6_k_weight_mpp(device const uchar* weights, uint row, uint column, uint k) {
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

kernel void q6_k_gemm_mpp(device bfloat* a [[buffer(0)]],
                          device const uchar* packed [[buffer(1)]],
                          device ushort* c [[buffer(2)]],
                          constant uint* p [[buffer(3)]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint2 group [[threadgroup_position_in_grid]]) {
    constexpr int TILE_M=64, TILE_N=64, TILE_K=128;
    threadgroup bfloat dequantized[TILE_N*TILE_K];
    int m=int(p[1]), k=int(p[2]), n=int(p[3]);
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
        // A ql byte carries two Q6_K values 64 positions apart. Decode them
        // together so each lane fetches ql/qh once and shares their bit-plane
        // extraction across the pair.
        for(uint i=tid;i<uint(TILE_N*(TILE_K/2));i+=128) {
            uint out_col=i/(TILE_K/2), byte_index=i%(TILE_K/2);
            uint slice=byte_index/32, lane=byte_index%32;
            int channel=int(group.x)*TILE_N+int(out_col);
            uint tile_base=uint(kt*TILE_K);
            uint block_index=tile_base>>8, half_block=(tile_base>>7)&1;
            uint tile_index=out_col*TILE_K+slice*32+lane;
            if(channel<n && tile_base<uint(k)) {
                uint blocks=uint(k)>>8;
                device const uchar* block=packed+(uint(channel)*blocks+block_index)*210;
                uchar ql=block[half_block*64+slice*32+lane];
                uchar qh=block[128+half_block*32+lane];
                uint high_shift=slice*2;
                uint q0=(uint(ql&15)|(((uint(qh)>>high_shift)&3)<<4));
                uint q1=(uint(ql>>4)|(((uint(qh)>>(high_shift+4))&3)<<4));
                char scale0=((device const char*)block)[192+half_block*8+slice*2+(lane>>4)];
                char scale1=((device const char*)block)[192+half_block*8+4+slice*2+(lane>>4)];
                float d=float(*((device const half*)(block+208)));
                dequantized[tile_index]=bfloat(d*float(scale0)*float(int(q0)-32));
                dequantized[tile_index+64]=bfloat(d*float(scale1)*float(int(q1)-32));
            } else {
                dequantized[tile_index]=bfloat(0.0f);
                dequantized[tile_index+64]=bfloat(0.0f);
            }
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

inline bfloat q4_k_expert_tile_value(device const uchar* row, uint column) {
    uint within=column&255, group=within>>5, lane=within&31, chunk=within>>6;
    uchar packed=row[16+chunk*32+lane];
    uint q=(group&1)==0?uint(packed&15):uint(packed>>4);
    float d=float(*((device const half*)row));
    float dmin=float(*((device const half*)(row+2)));
    return bfloat(d*float(qk4_scale_mpp(row+4,group))*float(q)
                  -dmin*float(qk4_min_mpp(row+4,group)));
}

inline bfloat q5_k_expert_tile_value(device const uchar* row, uint column) {
    uint within=column&255, group=within>>5, lane=within&31, chunk=within>>6;
    uchar packed=row[48+chunk*32+lane];
    uint nibble=(group&1)==0?uint(packed&15):uint(packed>>4);
    uint high=(uint(row[16+lane])>>group)&1;
    float d=float(*((device const half*)row));
    float dmin=float(*((device const half*)(row+2)));
    return bfloat(d*float(qk4_scale_mpp(row+4,group))*float(nibble|(high<<4))
                  -dmin*float(qk4_min_mpp(row+4,group)));
}

inline bfloat q6_k_expert_tile_value(device const uchar* row, uint column) {
    uint within=column&255, group=within>>5, lane=within&31;
    uint half_block=group>>2, slice=group&3;
    uint ql_index=half_block*64+((slice&1)*32)+lane;
    uchar packed=row[ql_index];
    uint low=slice<2?uint(packed&15):uint(packed>>4);
    uint high=(uint(row[128+half_block*32+lane])>>(slice*2))&3;
    int q=int(low|(high<<4))-32;
    uint scale_index=192+half_block*8+slice*2+(lane>>4);
    char scale=((device const char*)row)[scale_index];
    float d=float(*((device const half*)(row+208)));
    return bfloat(d*float(scale)*float(q));
}

// Routed rows are compacted into expert-major 32-row segments before this
// kernel. TensorOps then reuses each quantized expert tile across its prompt
// rows; output rows scatter back to assignment order for existing MoE kernels.
void expert_project_k_mpp_impl(
    device bfloat* input [[buffer(0)]],
    device const uchar* packed [[buffer(1)]],
    device const uint* assignment_ids [[buffer(2)]],
    device const uint* counts [[buffer(3)]],
    device const uint* bases [[buffer(4)]],
    device ushort* output [[buffer(5)]],
    threadgroup bfloat* dequantized,
    constant uint* p [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    constexpr uint TILE_M=32, TILE_N=64, TILE_K=128;
    uint k=p[2], n=p[3], experts=p[5], type=p[7];
    uint expert=group.z;
    if(expert>=experts) return;
    uint count=counts[expert], base=bases[expert], m_tile=group.y*TILE_M;
    if(m_tile>=count) return;
    uint padded_rows=((count+TILE_M-1)/TILE_M)*TILE_M;
    uint blocks=k/256;
    uint bytes_per_block=type==12?144:(type==13?176:210);
    uint bytes_per_row=blocks*bytes_per_block;
    device const uchar* expert_weights=packed+expert*n*bytes_per_row;
    tensor<device bfloat,dextents<int,2>,tensor_inline> A(
        input+base*k,dextents<int,2>(int(k),int(padded_rows)));
    tensor<threadgroup bfloat,dextents<int,2>,tensor_inline> B(
        dequantized,dextents<int,2>(int(TILE_K),int(TILE_N)));
    auto left=A.slice(0,int(m_tile));
    constexpr auto desc=matmul2d_descriptor(
        32,64,128,false,true,false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto result=op.get_destination_cooperative_tensor<decltype(left),decltype(B),float>();
    uint k_tiles=(k+TILE_K-1)/TILE_K;
    for(uint kt=0;kt<k_tiles;kt++) {
        if(kt>0) threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint i=tid;i<TILE_N*TILE_K;i+=128) {
            uint out_col=i/TILE_K, in_col=i%TILE_K;
            uint column=group.x*TILE_N+out_col, source_k=kt*TILE_K+in_col;
            bfloat value=bfloat(0.0f);
            if(column<n && source_k<k) {
                device const uchar* row=expert_weights+column*bytes_per_row;
                if(type==12) value=q4_k_expert_tile_value(
                    row+(source_k>>8)*144,source_k);
                else if(type==13) value=q5_k_expert_tile_value(
                    row+(source_k>>8)*176,source_k);
                else value=q6_k_expert_tile_value(
                    row+(source_k>>8)*210,source_k);
            }
            dequantized[i]=value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto right=B;
        auto left_k=A.slice(int(kt*TILE_K),int(m_tile));
        op.run(left_k,right,result);
    }
    for(uint i=0;i<result.get_capacity();i++) {
        auto coord=result.get_multidimensional_index(i);
        uint column=group.x*TILE_N+coord[0], row=m_tile+coord[1];
        if(row<count && column<n) {
            uint assignment=assignment_ids[base+row];
            float x=result[i]; uint bits=as_type<uint>(x);
            output[assignment*n+column]=isnan(x)
                ?ushort((bits>>16)|0x40)
                :ushort((bits+0x7fff+((bits>>16)&1))>>16);
        }
    }
}

kernel void expert_project_q4_k_mpp(
    device bfloat* a [[buffer(0)]], device const uchar* b [[buffer(1)]],
    device const uint* ids [[buffer(2)]], device const uint* counts [[buffer(3)]],
    device const uint* bases [[buffer(4)]], device ushort* c [[buffer(5)]],
    constant uint* p [[buffer(6)]], uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup bfloat dequantized[64*128];
    expert_project_k_mpp_impl(a,b,ids,counts,bases,c,dequantized,p,tid,group);
}
kernel void expert_project_q5_k_mpp(
    device bfloat* a [[buffer(0)]], device const uchar* b [[buffer(1)]],
    device const uint* ids [[buffer(2)]], device const uint* counts [[buffer(3)]],
    device const uint* bases [[buffer(4)]], device ushort* c [[buffer(5)]],
    constant uint* p [[buffer(6)]], uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup bfloat dequantized[64*128];
    expert_project_k_mpp_impl(a,b,ids,counts,bases,c,dequantized,p,tid,group);
}
kernel void expert_project_q6_k_mpp(
    device bfloat* a [[buffer(0)]], device const uchar* b [[buffer(1)]],
    device const uint* ids [[buffer(2)]], device const uint* counts [[buffer(3)]],
    device const uint* bases [[buffer(4)]], device ushort* c [[buffer(5)]],
    constant uint* p [[buffer(6)]], uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup bfloat dequantized[64*128];
    expert_project_k_mpp_impl(a,b,ids,counts,bases,c,dequantized,p,tid,group);
}

// Strided tensor views describe grouped sequence-major Q/K/V without copies.
template<typename T, bool Context, bool BFloat>
void grouped_mpp(device T* a, device T* b, device ushort* c,
                 constant uint* p, uint3 group) {
    int s=int(p[1]), t=int(p[2]), heads=int(p[3]), kv=int(p[5]), d=int(p[6]);
    int h=int(group.z), kh=h/(heads/kv);
    tensor<device T,dextents<int,2>,tensor_inline> A(
        a+(Context?h*s*t:h*d),dextents<int,2>(Context?t:d,s),
        array<int,2>{1,Context?t:heads*d});
    tensor<device T,dextents<int,2>,tensor_inline> B(
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
            float x=result[i];
            if (BFloat) {
                uint bits=as_type<uint>(x);
                c[dest]=isnan(x)?ushort((bits>>16)|0x40):ushort((bits+0x7fff+((bits>>16)&1))>>16);
            } else {
                c[dest]=as_type<ushort>(half(x));
            }
        }
    }
}
kernel void attention_scores_mpp(device bfloat* a [[buffer(0)]], device bfloat* b [[buffer(1)]], device ushort* c [[buffer(2)]], constant uint* p [[buffer(3)]], uint3 group [[threadgroup_position_in_grid]]) {
    grouped_mpp<bfloat,false,true>(a,b,c,p,group);
}
kernel void attention_context_mpp(device bfloat* a [[buffer(0)]], device bfloat* b [[buffer(1)]], device ushort* c [[buffer(2)]], constant uint* p [[buffer(3)]], uint3 group [[threadgroup_position_in_grid]]) {
    grouped_mpp<bfloat,true,true>(a,b,c,p,group);
}
kernel void attention_scores_mpp_f16(device half* a [[buffer(0)]], device half* b [[buffer(1)]], device ushort* c [[buffer(2)]], constant uint* p [[buffer(3)]], uint3 group [[threadgroup_position_in_grid]]) {
    grouped_mpp<half,false,false>(a,b,c,p,group);
}
kernel void attention_context_mpp_f16(device half* a [[buffer(0)]], device half* b [[buffer(1)]], device ushort* c [[buffer(2)]], constant uint* p [[buffer(3)]], uint3 group [[threadgroup_position_in_grid]]) {
    grouped_mpp<half,true,false>(a,b,c,p,group);
}
