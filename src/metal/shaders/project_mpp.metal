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
