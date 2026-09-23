use ferrum::{
    DType, MetalDevice, Tensor,
    nn::{Linear, kv_cache::KvCache},
    reference,
};

#[test]
fn growing_cache_copies_only_at_geometric_boundaries() {
    let d = MetalDevice::new().unwrap();
    let mut cache = KvCache::new(1, 600, 1, 2, DType::F32).unwrap();
    let mut allocations = Vec::new();
    let mut snapshot = None;
    for i in 0..600 {
        let x = Tensor::from_f32(&d, [1, 1, 2], DType::F32, &[i as f32, -(i as f32)]).unwrap();
        let before = d.counters().allocations;
        cache.append(&d, 0, &x, &x).unwrap();
        if d.counters().allocations != before {
            allocations.push((i, d.counters().allocations - before));
        }
        if i == 255 {
            snapshot = Some(cache.active(0).unwrap().unwrap().0.clone());
        }
    }
    assert_eq!(allocations, [(0, 2), (256, 2), (512, 2)]);
    assert_eq!(cache.reserved_bytes(), 600 * 2 * 4 * 2);
    let expected: Vec<f32> = (0..600).flat_map(|i| [i as f32, -(i as f32)]).collect();
    assert_eq!(cache.active(0).unwrap().unwrap().0.to_f32(), expected);
    assert_eq!(snapshot.unwrap().to_f32(), expected[..512]);
}

#[test]
fn vector_projection_tail_and_misaligned_views() {
    let d = MetalDevice::new().unwrap();
    for ty in [DType::BF16, DType::F16] {
        for offset in [0, 1] {
            let (k, n) = (128, 17);
            let x = Tensor::from_f32(&d, [k + offset], ty, &reference::deterministic(k + offset))
                .unwrap()
                .view(offset, [1, k])
                .unwrap();
            let w = Tensor::from_f32(
                &d,
                [n * k + offset],
                ty,
                &reference::deterministic(n * k + offset),
            )
            .unwrap()
            .view(offset, [n, k])
            .unwrap();
            let values = w.to_f32();
            let wt: Vec<_> = (0..n * k).map(|i| values[(i % n) * k + i / n]).collect();
            let expected: Vec<_> = reference::matmul(&x.to_f32(), &wt, 1, k, n)
                .into_iter()
                .map(|v| ty.round(v))
                .collect();
            let y = Linear::new(&d, w, None).unwrap().forward(&d, &x).unwrap();
            let tolerance = if ty == DType::BF16 { 1.6e-2 } else { 2e-3 };
            reference::check(&y.to_f32(), &expected, tolerance, tolerance).unwrap();
        }
    }
}

#[test]
fn split_k_bf16_gemv_matches_reference_for_large_rows() {
    let d = MetalDevice::new().unwrap();
    let (k, n) = (896, 513);
    let x = Tensor::from_f32(&d, [1, k], DType::BF16, &reference::deterministic(k)).unwrap();
    let w = Tensor::from_f32(&d, [n, k], DType::BF16, &reference::deterministic(n * k)).unwrap();
    let values = w.to_f32();
    let wt: Vec<_> = (0..n * k).map(|i| values[(i % n) * k + i / n]).collect();
    let expected: Vec<_> = reference::matmul(&x.to_f32(), &wt, 1, k, n)
        .into_iter()
        .map(|v| DType::BF16.round(v))
        .collect();
    let y = Linear::new(&d, w, None).unwrap().forward(&d, &x).unwrap();
    reference::check(&y.to_f32(), &expected, 1.6e-2, 1e-2).unwrap();
}

#[test]
fn last_position_prefill_preserves_cache_and_requested_logits() {
    use ferrum::{
        generation::final_logits,
        loader::Weights,
        model::{ModelConfig, Transformer, tiny},
    };
    let d = MetalDevice::new().unwrap();
    for ty in [DType::F32, DType::F16, DType::BF16] {
        let c = ModelConfig::tiny(ty);
        let bytes = tiny::serialize(&c, &tiny::weights(&c).unwrap()).unwrap();
        let w = Weights::from_bytes(&d, &bytes).unwrap();
        let model = Transformer::from_weights(&d, c, &w).unwrap();
        let (full, mut a) = model.forward_prefill(&d, &[3, 8, 4]).unwrap();
        let (last, mut b) = model.forward_prefill_last(&d, &[3, 8, 4]).unwrap();
        assert_eq!(last.shape().dimensions(), &[1, 32]);
        let (atol, rtol) = match ty {
            DType::F32 => (3e-5, 3e-5),
            DType::F16 => (2e-3, 2e-3),
            DType::BF16 => (1.6e-2, 1e-2),
        };
        reference::check(
            &last.to_f32(),
            &final_logits(&d, &full).unwrap(),
            atol,
            rtol,
        )
        .unwrap();
        let da = model.forward_decode(&d, 11, &mut a).unwrap();
        let db = model.forward_decode(&d, 11, &mut b).unwrap();
        assert_eq!(da.to_f32(), db.to_f32());
        assert!(model.forward_prefill_last(&d, &[]).is_err());
    }
}

#[test]
fn shared_projection_tiles_cover_dispatch_boundary_and_tails() {
    let d = MetalDevice::new().unwrap();
    for ty in [DType::BF16, DType::F16] {
        for (m, k, n) in [(31, 63, 17), (32, 63, 17), (33, 129, 48), (65, 128, 65)] {
            let x = Tensor::from_f32(&d, [m, k], ty, &reference::deterministic(m * k)).unwrap();
            let w = Tensor::from_f32(&d, [n, k], ty, &reference::deterministic(n * k)).unwrap();
            let values = w.to_f32();
            let wt: Vec<_> = (0..n * k).map(|i| values[(i % n) * k + i / n]).collect();
            let expected: Vec<_> = reference::matmul(&x.to_f32(), &wt, m, k, n)
                .into_iter()
                .map(|x| ty.round(x))
                .collect();
            let y = Linear::new(&d, w, None).unwrap().forward(&d, &x).unwrap();
            let (atol, rtol) = if ty == DType::BF16 {
                (1.6e-2, 1e-2)
            } else {
                (2e-3, 2e-3)
            };
            reference::check(&y.to_f32(), &expected, atol, rtol).unwrap();
        }
    }
}

#[test]
fn projection_matrix_views_allow_unaligned_base_offsets() {
    let d = MetalDevice::new().unwrap();
    let (m, k, n) = (33, 63, 17);
    let x = Tensor::from_f32(
        &d,
        [m * k + 1],
        DType::BF16,
        &reference::deterministic(m * k + 1),
    )
    .unwrap()
    .view(1, [m, k])
    .unwrap();
    let w = Tensor::from_f32(
        &d,
        [n * k + 1],
        DType::BF16,
        &reference::deterministic(n * k + 1),
    )
    .unwrap()
    .view(1, [n, k])
    .unwrap();
    let vals = w.to_f32();
    let wt: Vec<_> = (0..n * k).map(|i| vals[(i % n) * k + i / n]).collect();
    let expected: Vec<_> = reference::matmul(&x.to_f32(), &wt, m, k, n)
        .into_iter()
        .map(|v| DType::BF16.round(v))
        .collect();
    let y = Linear::new(&d, w, None).unwrap().forward(&d, &x).unwrap();
    reference::check(&y.to_f32(), &expected, 1.6e-2, 1e-2).unwrap();
}
