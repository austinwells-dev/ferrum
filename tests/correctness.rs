use ferrum::{
    DType, Error, MetalDevice, Tensor, reference as cpu,
    tensor::{Layout, Shape},
};

fn validate(dtype: DType, name: &str, actual: &Tensor, expected: Vec<f32>) {
    // Oracles consume quantized inputs and round the result to the output storage dtype.
    let expected: Vec<_> = expected.into_iter().map(|x| dtype.round(x)).collect();
    let (atol, rtol) = match dtype {
        // F32 RoPE phases amplify rounding at position 2048; oracle uses F64.
        DType::F32 if name == "rope" => (5e-4, 3e-5),
        DType::F32 => (3e-5, 3e-5),
        DType::F16 => (2e-3, 2e-3),
        DType::BF16 => (1.6e-2, 1e-2),
    };
    let error = cpu::check(&actual.to_f32(), &expected, atol, rtol)
        .unwrap_or_else(|e| panic!("{name} {dtype:?} {:?}: {e}", actual.shape().dimensions()));
    println!(
        "{name} {dtype:?} {:?}: max_abs={error:.6e}",
        actual.shape().dimensions()
    );
}
fn tensor(d: &MetalDevice, shape: Vec<usize>, dtype: DType) -> Tensor {
    let n = shape.iter().product();
    Tensor::from_f32(d, shape, dtype, &cpu::deterministic(n)).unwrap()
}
#[test]
fn elementwise_all_dtypes_and_edges() {
    let d = MetalDevice::new().unwrap();
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        for shape in [
            vec![],
            vec![0],
            vec![1],
            vec![3, 7],
            vec![257],
            vec![2, 3, 4096],
        ] {
            let a = tensor(&d, shape.clone(), dtype);
            let x = a.to_f32();
            let b = Tensor::from_f32(
                &d,
                shape,
                dtype,
                &x.iter().rev().copied().collect::<Vec<_>>(),
            )
            .unwrap();
            let y = b.to_f32();
            validate(
                dtype,
                "add",
                &d.add(&a, &b).unwrap().tensor,
                cpu::add(&x, &y),
            );
            validate(
                dtype,
                "mul",
                &d.mul(&a, &b).unwrap().tensor,
                cpu::mul(&x, &y),
            );
            validate(dtype, "silu", &d.silu(&a).unwrap().tensor, cpu::silu(&x));
        }
    }
}
#[test]
fn row_operations_all_dtypes() {
    let d = MetalDevice::new().unwrap();
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        for shape in [
            vec![1],
            vec![3, 7],
            vec![2, 3, 129],
            vec![4, 4096],
            vec![0, 128],
        ] {
            let width = *shape.last().unwrap();
            let a = tensor(&d, shape, dtype);
            let x = a.to_f32();
            let w = tensor(&d, vec![width], dtype);
            validate(
                dtype,
                "rmsnorm",
                &d.rmsnorm(&a, &w, 1e-5).unwrap().tensor,
                cpu::rmsnorm(&x, &w.to_f32(), 1e-5),
            );
            let out = d.softmax(&a).unwrap().tensor;
            validate(dtype, "softmax", &out, cpu::softmax(&x, width));
            for row in out.to_f32().chunks(width) {
                assert!((row.iter().sum::<f32>() - 1.).abs() < 0.01);
            }
        }
        let a = Tensor::from_f32(
            &d,
            vec![2, 3],
            dtype,
            &[1000., 1001., 999., -1000., -999., -1001.],
        )
        .unwrap();
        validate(
            dtype,
            "stable softmax",
            &d.softmax(&a).unwrap().tensor,
            cpu::softmax(&a.to_f32(), 3),
        );
        let zero = Tensor::zeros(&d, vec![3, 129], dtype).unwrap();
        let w = tensor(&d, vec![129], dtype);
        assert!(
            d.rmsnorm(&zero, &w, 1e-5)
                .unwrap()
                .tensor
                .to_f32()
                .iter()
                .all(|&v| v == 0.)
        );
        let extreme =
            Tensor::from_f32(&d, vec![5], dtype, &[-1000., -100., 0., 100., 1000.]).unwrap();
        validate(
            dtype,
            "stable silu",
            &d.silu(&extreme).unwrap().tensor,
            cpu::silu(&extreme.to_f32()),
        );
    }
}
#[test]
fn rotary_all_dtypes() {
    let d = MetalDevice::new().unwrap();
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        for (shape, head, position, theta) in [
            (vec![2], 2, 0, 10000.),
            (vec![3, 12], 6, 7, 10000.),
            (vec![2, 3, 128], 128, 127, 10000.),
            (vec![3, 256], 64, 2048, 500000.),
            (vec![0, 128], 128, 1, 10000.),
        ] {
            let a = tensor(&d, shape, dtype);
            validate(
                dtype,
                "rope",
                &d.rope(&a, head, position, theta).unwrap().tensor,
                cpu::rope(&a.to_f32(), head, position, theta),
            );
        }
    }
}
#[test]
fn tiled_matmul_all_dtypes_and_edges() {
    let d = MetalDevice::new().unwrap();
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        for (m, k, n) in [
            (1, 1, 1),
            (3, 5, 7),
            (17, 19, 23),
            (64, 128, 65),
            (2, 4096, 33),
            (0, 3, 7),
            (3, 0, 7),
            (3, 7, 0),
        ] {
            let a = tensor(&d, vec![m, k], dtype);
            let b = tensor(&d, vec![k, n], dtype);
            validate(
                dtype,
                "matmul",
                &d.matmul(&a, &b).unwrap().tensor,
                cpu::matmul(&a.to_f32(), &b.to_f32(), m, k, n),
            );
        }
    }
}
#[test]
fn metadata_roundtrip_and_reshape() {
    let d = MetalDevice::new().unwrap();
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        let x = vec![0., -0., 1., -2., 0.1, 65504.];
        let a = Tensor::from_f32(&d, vec![1, 2, 3], dtype, &x).unwrap();
        assert_eq!(
            a.to_f32(),
            x.iter().map(|&x| dtype.round(x)).collect::<Vec<_>>()
        );
        assert_eq!(a.layout().strides(), [6, 3, 1]);
        assert!(a.layout().is_contiguous());
        assert_eq!(a.shape().rank(), 3);
        assert_eq!(a.byte_size(), 6 * dtype.size_bytes());
        let b = a.reshape(vec![3, 2]).unwrap();
        assert_eq!(b.layout().strides(), [2, 1]);
        assert_eq!(b.to_f32(), a.to_f32());
        let info = b.storage_info();
        assert_eq!(info.owners, 2);
        assert_eq!(info.offset_bytes, 0);
        assert_eq!(info.mode, "shared");
        assert!(info.alignment >= dtype.size_bytes());
        assert!(matches!(a.reshape(vec![7]), Err(Error::Reshape(_))));
        drop(a);
        assert_eq!(b.to_f32().len(), 6);
    }
    assert_eq!(Shape::new(vec![]).unwrap().numel(), 1);
    assert_eq!(Shape::new(vec![3, 0, 4]).unwrap().numel(), 0);
    assert!(Shape::new(vec![usize::MAX, 2]).is_err());
    assert!(
        Shape::new(vec![usize::MAX])
            .unwrap()
            .byte_size(DType::F32)
            .is_err()
    );
    assert!(Layout::contiguous(&Shape::new(vec![0, usize::MAX, 2]).unwrap()).is_err());
    let empty = Tensor::zeros(&d, vec![0], DType::F32).unwrap();
    let out = d.add(&empty, &empty).unwrap();
    assert_eq!(out.metrics.timing.dispatches, 0);
    assert_eq!(out.tensor.storage_info().allocation_bytes, 4);
    // Retained buffer ownership survives destruction of the creating context.
    let a = tensor(&d, vec![3], DType::F32);
    drop(d);
    assert_eq!(a.to_f32(), cpu::deterministic(3));
}
#[test]
fn invalid_inputs_and_contexts_are_rejected() {
    let d = MetalDevice::new().unwrap();
    let other = MetalDevice::new().unwrap();
    let a = tensor(&d, vec![2, 3], DType::F32);
    let b = tensor(&d, vec![6], DType::F32);
    let h = tensor(&d, vec![2, 3], DType::F16);
    let w = tensor(&d, vec![3], DType::F32);
    assert!(matches!(d.add(&a, &b), Err(Error::Shape(_))));
    assert!(matches!(d.mul(&a, &h), Err(Error::DType)));
    assert!(d.matmul(&a, &a).is_err());
    assert!(d.matmul(&b, &b).is_err());
    assert!(d.rmsnorm(&a, &b, 1e-5).is_err());
    for eps in [0., -1., f32::NAN, f32::INFINITY] {
        assert!(d.rmsnorm(&a, &w, eps).is_err());
    }
    for head in [0, 1, 2, 4] {
        assert!(d.rope(&a, head, 0, 10000.).is_err());
    }
    let r = tensor(&d, vec![2, 4], DType::F32);
    for theta in [0., -1., f32::NAN, f32::INFINITY] {
        assert!(d.rope(&r, 4, 0, theta).is_err());
    }
    for shape in [vec![], vec![2, 0]] {
        let t = tensor(&d, shape, DType::F32);
        assert!(d.softmax(&t).is_err());
        assert!(d.rope(&t, 2, 0, 10000.).is_err());
    }
    assert!(matches!(other.silu(&a), Err(Error::DeviceMismatch)));
    let foreign = tensor(&other, vec![2, 3], DType::F32);
    assert!(matches!(d.add(&a, &foreign), Err(Error::DeviceMismatch)));
    assert!(Tensor::from_f32(&d, vec![2], DType::F32, &[1.]).is_err());
    assert!(matches!(d.allocate(usize::MAX), Err(Error::Allocation(_))));
    let too_wide = Tensor::zeros(&d, vec![0, u32::MAX as usize + 1], DType::F32).unwrap();
    assert!(d.softmax(&too_wide).is_err());
}
#[test]
fn compilation_diagnostics_and_cache() {
    let d = MetalDevice::new().unwrap();
    assert!(
        matches!(d.compile_kernel("this is invalid MSL", "bad"),Err(Error::Compilation(s)) if !s.is_empty())
    );
    let src = "#include <metal_stdlib>\nusing namespace metal;\nkernel void noop() {}";
    assert!(matches!(
        d.compile_kernel(src, "absent"),
        Err(Error::MissingKernel(_))
    ));
    let p = d.compile_kernel(src, "noop").unwrap();
    let q = d.compile_kernel(src, "noop").unwrap();
    assert!(std::rc::Rc::ptr_eq(&p, &q));
    d.warm_up().unwrap();
    let count = d.cached_pipeline_count();
    let a = tensor(&d, vec![257], DType::F32);
    for _ in 0..10 {
        let out = d.add(&a, &a).unwrap();
        assert_eq!(out.metrics.timing.dispatches, 1);
        assert_eq!(out.metrics.bytes_read, 257 * 8);
        assert_eq!(out.metrics.bytes_written, 257 * 4);
    }
    assert_eq!(d.cached_pipeline_count(), count);
}

#[test]
fn chained_gpu_outputs_preserve_inputs() {
    let d = MetalDevice::new().unwrap();
    let shape = [1, 2, 1, 3, 1, 2];
    let a = Tensor::from_f32(&d, shape, DType::F32, &cpu::deterministic(12)).unwrap();
    let original = a.to_f32();
    let alias = a.reshape([3, 4]).unwrap();
    let sum = d.add(&a, &a).unwrap().tensor;
    let activated = d.silu(&sum).unwrap().tensor;
    let result = d.mul(&activated, &a).unwrap().tensor;
    validate(
        DType::F32,
        "chain",
        &result,
        cpu::mul(&cpu::silu(&cpu::add(&original, &original)), &original),
    );
    assert_eq!(a.to_f32(), original);
    assert_eq!(alias.to_f32(), original);
    assert_eq!(result.shape().rank(), 6);
    assert_eq!(result.layout().strides(), [12, 6, 6, 2, 2, 1]);
}

#[test]
fn bf16_gpu_rounding_ties_and_nan() {
    let d = MetalDevice::new().unwrap();
    // At BF16 precision, these sums are exact ties, with even and odd low mantissa bits.
    let a = Tensor::from_f32(
        &d,
        [4],
        DType::BF16,
        &[1., 1.0078125, f32::NAN, f32::INFINITY],
    )
    .unwrap();
    let b = Tensor::from_f32(&d, [4], DType::BF16, &[0.00390625, 0.00390625, 0., 0.]).unwrap();
    let values = d.add(&a, &b).unwrap().tensor.to_f32();
    assert_eq!(values[0], 1.);
    assert_eq!(values[1], 1.015625);
    assert!(values[2].is_nan());
    assert_eq!(values[3], f32::INFINITY);
}

#[test]
fn contiguous_views_check_ranges_and_dispatch_offsets() {
    let d = MetalDevice::new().unwrap();
    for ty in [DType::F32, DType::F16, DType::BF16] {
        let a = Tensor::from_f32(&d, [8], ty, &[0., 1., 2., 3., 4., 5., 6., 7.]).unwrap();
        let v = a.view(2, [2, 2]).unwrap();
        assert_eq!(v.to_f32(), [2., 3., 4., 5.]);
        assert_eq!(v.storage_info().offset_bytes, 2 * ty.size_bytes());
        let nested = v.view(1, [2]).unwrap();
        assert_eq!(d.add(&nested, &nested).unwrap().tensor.to_f32(), [6., 8.]);
        assert!(v.view(3, [2]).is_err());
        assert!(v.view(usize::MAX, [2]).is_err());
        assert_eq!(a.view(8, [0]).unwrap().to_f32(), Vec::<f32>::new());
        drop(a);
        assert_eq!(nested.to_f32(), [3., 4.]);
    }
}
