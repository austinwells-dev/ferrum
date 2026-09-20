use ferrum::{
    DType, MetalDevice, Tensor,
    nn::{Embedding, Linear},
    reference as cpu,
};
fn check(t: &Tensor, expected: &[f32]) -> f32 {
    let expected: Vec<_> = expected.iter().map(|&x| t.dtype().round(x)).collect();
    let (a, r) = match t.dtype() {
        DType::F32 => (3e-5, 3e-5),
        DType::F16 => (2e-3, 2e-3),
        DType::BF16 => (1.6e-2, 1e-2),
    };
    cpu::check(&t.to_f32(), &expected, a, r).unwrap()
}
#[test]
fn embeddings_linear_layout_all_dtypes() {
    let d = MetalDevice::new().unwrap();
    for ty in [DType::F32, DType::F16, DType::BF16] {
        let w = Tensor::from_f32(&d, [7, 5], ty, &cpu::deterministic(35)).unwrap();
        let values = w.to_f32();
        let emb = Embedding::new(w.clone()).unwrap();
        let e = emb.forward(&d, &[0, 6, 0]).unwrap();
        check(&e, &[&values[..5], &values[30..], &values[..5]].concat());
        assert!(emb.forward(&d, &[7]).is_err());
        assert_eq!(emb.forward(&d, &[]).unwrap().numel(), 0);
        let linear = Linear::new(&d, w.clone(), None).unwrap();
        for shape in [vec![5], vec![3, 5], vec![2, 3, 5]] {
            let x = Tensor::from_f32(&d, &shape, ty, &cpu::deterministic(shape.iter().product()))
                .unwrap();
            let wt: Vec<_> = (0..35).map(|i| values[(i % 7) * 5 + i / 7]).collect();
            check(
                &linear.forward(&d, &x).unwrap(),
                &cpu::matmul(&x.to_f32(), &wt, x.numel() / 5, 5, 7),
            );
        }
        let x = Tensor::from_f32(&d, [2, 3, 4], ty, &cpu::deterministic(24)).unwrap();
        let vals = x.to_f32();
        for head in 0..3 {
            check(
                &d.select_head(&x, head).unwrap().tensor,
                &[
                    &vals[head * 4..head * 4 + 4],
                    &vals[12 + head * 4..16 + head * 4],
                ]
                .concat(),
            );
        }
        let twice = d.swap01(&d.swap01(&x).unwrap().tensor).unwrap().tensor;
        assert_eq!(twice.to_f32(), vals);
        let r = d.rope_split(&x, 7, 10000.).unwrap().tensor;
        let mut expected = vals.clone();
        for s in 0..2 {
            for h in 0..3 {
                for j in 0..2 {
                    let angle = (7 + s) as f64 * 10000f64.powf(-((2 * j) as f64) / 4.);
                    let i = (s * 3 + h) * 4 + j;
                    expected[i] =
                        (vals[i] as f64 * angle.cos() - vals[i + 2] as f64 * angle.sin()) as f32;
                    expected[i + 2] =
                        (vals[i] as f64 * angle.sin() + vals[i + 2] as f64 * angle.cos()) as f32;
                }
            }
        }
        check(&r, &expected);
    }
}
#[test]
fn mask_absolute_positions() {
    let d = MetalDevice::new().unwrap();
    let x = Tensor::from_f32(&d, [2, 4], DType::F32, &[1.; 8]).unwrap();
    assert_eq!(
        d.causal_mask(&x, 2).unwrap().tensor.to_f32(),
        vec![1., 1., 1., f32::NEG_INFINITY, 1., 1., 1., 1.]
    );
    assert!(d.causal_mask(&x, 0).is_err());
}

use ferrum::{
    loader::Weights,
    model::{ModelConfig, Transformer, tiny},
    nn::{attention::Trace, kv_cache::KvCache},
};
#[test]
fn complete_transformer_oracle_and_cached_equivalence() {
    let d = MetalDevice::new().unwrap();
    let tokens = [3, 8, 4, 11, 2];
    for ty in [DType::F32, DType::F16, DType::BF16] {
        let c = ModelConfig::tiny(ty);
        let w = tiny::weights(&c).unwrap();
        let bytes = tiny::serialize(&c, &w).unwrap();
        let loaded = Weights::from_bytes(&d, &bytes).unwrap();
        let model = Transformer::from_weights(&d, c.clone(), &loaded).unwrap();
        let oracle = cpu::transformer::forward(&c, &w, &tokens).unwrap();
        let mut trace = Trace::new();
        let mut cache = model.new_cache().unwrap();
        let full = model
            .forward(&d, &tokens, &mut cache, Some(&mut trace))
            .unwrap();
        assert_eq!(trace.len(), oracle.len());
        for (name, t) in &trace {
            println!(
                "intermediate {ty:?} {name} max_abs={:.8e}",
                check(t, &oracle[name])
            );
        }
        let mut incremental = model.new_cache().unwrap();
        let all = full.to_f32();
        for (i, &token) in tokens.iter().enumerate() {
            let logits = model.forward_decode(&d, token, &mut incremental).unwrap();
            let error = check(&logits, &all[i * c.vocab_size..(i + 1) * c.vocab_size]);
            println!("cached {ty:?} position={i} max_abs={error:.8e}");
            assert_eq!(incremental.len().unwrap(), i + 1);
        }
        // Chunked prefill and nonzero-offset multi-token causal masking.
        let (_, mut chunk) = model.forward_prefill(&d, &tokens[..2]).unwrap();
        check(
            &model.forward(&d, &tokens[2..], &mut chunk, None).unwrap(),
            &all[2 * c.vocab_size..],
        );
        for l in 0..c.num_layers {
            let (fk, fv) = cache.active(l).unwrap().unwrap();
            let (ik, iv) = incremental.active(l).unwrap().unwrap();
            check(ik, &fk.to_f32());
            check(iv, &fv.to_f32());
        }
    }
}
#[test]
fn cache_order_capacity_reset_and_errors() {
    let d = MetalDevice::new().unwrap();
    let mut cache = KvCache::new(2, 3, 2, 4, DType::F32).unwrap();
    let a = Tensor::from_f32(&d, [1, 2, 4], DType::F32, &[1.; 8]).unwrap();
    let b = Tensor::from_f32(&d, [2, 2, 4], DType::F32, &[2.; 16]).unwrap();
    for l in 0..2 {
        cache.append(&d, l, &a, &a).unwrap();
        cache.append(&d, l, &b, &b).unwrap();
    }
    assert_eq!(cache.len().unwrap(), 3);
    assert_eq!(cache.bytes(), 2 * 2 * 3 * 2 * 4 * 4);
    assert_eq!(
        cache.active(1).unwrap().unwrap().0.to_f32(),
        [vec![1.; 8], vec![2.; 16]].concat()
    );
    assert!(cache.append(&d, 0, &a, &a).is_err());
    assert!(cache.active(2).is_err());
    cache.reset();
    assert!(cache.is_empty().unwrap());
    assert_eq!(cache.bytes(), 0);
    cache.append(&d, 0, &a, &a).unwrap();
    assert!(cache.len().is_err());
}

#[test]
fn model_configuration_weights_and_cache_validation() {
    let d = MetalDevice::new().unwrap();
    let base = ModelConfig::tiny(DType::F32);
    for field in 0..12 {
        let mut c = base.clone();
        match field {
            0 => c.vocab_size = 0,
            1 => c.hidden_size = 0,
            2 => c.intermediate_size = 0,
            3 => c.num_layers = 0,
            4 => c.num_attention_heads = 3,
            5 => c.num_key_value_heads = 0,
            6 => c.head_dim = 3,
            7 => c.rms_norm_epsilon = f32::NAN,
            8 => c.rope_theta = 0.,
            9 => c.max_context_length = 0,
            10 => c.hidden_size = usize::MAX,
            _ => c.num_layers = usize::MAX,
        }
        assert!(c.validate().is_err(), "field {field}");
    }
    let original = tiny::weights(&base).unwrap();
    for kind in 0..3 {
        let mut w = original.clone();
        let name = "layers.1.k.weight";
        match kind {
            0 => {
                w.remove(name);
            }
            1 => {
                w.get_mut(name).unwrap().0 = vec![4, 32];
            }
            _ => {
                w.get_mut(name).unwrap().0 = vec![128];
            }
        }
        let bytes = tiny::serialize(&base, &w).unwrap();
        let loaded = Weights::from_bytes(&d, &bytes).unwrap();
        let before = d.counters().dispatches;
        let error = match Transformer::from_weights(&d, base.clone(), &loaded) {
            Ok(_) => panic!("malformed model accepted"),
            Err(e) => e.to_string(),
        };
        assert!(error.contains(name));
        assert_eq!(before, d.counters().dispatches);
    }
    let mut half = base.clone();
    half.dtype = DType::F16;
    let loaded = Weights::from_bytes(
        &d,
        &tiny::serialize(&half, &tiny::weights(&half).unwrap()).unwrap(),
    )
    .unwrap();
    assert!(Transformer::from_weights(&d, base.clone(), &loaded).is_err());
    let loaded = Weights::from_bytes(&d, &tiny::serialize(&base, &original).unwrap()).unwrap();
    let model = Transformer::from_weights(&d, base.clone(), &loaded).unwrap();
    let mut cache = model.new_cache().unwrap();
    assert!(model.forward(&d, &[], &mut cache, None).is_err());
    assert!(model.forward_decode(&d, 32, &mut cache).is_err());
    assert_eq!(cache.len().unwrap(), 0);
    let (_, mut cache) = model.forward_prefill(&d, &[1, 2]).unwrap();
    let before = cache.active(0).unwrap().unwrap().0.to_f32();
    assert!(model.forward(&d, &[1; 15], &mut cache, None).is_err());
    assert_eq!(cache.len().unwrap(), 2);
    assert_eq!(cache.active(0).unwrap().unwrap().0.to_f32(), before);
    let mut wrong = KvCache::new(1, 16, 2, 4, DType::F32).unwrap();
    assert!(model.forward_decode(&d, 1, &mut wrong).is_err());
    let other = MetalDevice::new().unwrap();
    assert!(model.forward_decode(&other, 1, &mut cache).is_err());
    assert_eq!(cache.len().unwrap(), 2);
    let mut tied = base.clone();
    tied.tie_word_embeddings = true;
    let w = tiny::weights(&tied).unwrap();
    let loaded = Weights::from_bytes(&d, &tiny::serialize(&tied, &w).unwrap()).unwrap();
    let model = Transformer::from_weights(&d, tied.clone(), &loaded).unwrap();
    check(
        &model.forward_prefill(&d, &[1, 2, 3]).unwrap().0,
        &cpu::transformer::forward(&tied, &w, &[1, 2, 3]).unwrap()["logits"],
    );
}
#[test]
fn primitive_invalid_inputs_and_bias() {
    let d = MetalDevice::new().unwrap();
    for ty in [DType::F32, DType::F16, DType::BF16] {
        let x = Tensor::from_f32(&d, [2, 5], ty, &cpu::deterministic(10)).unwrap();
        let w = Tensor::from_f32(&d, [3, 5], ty, &cpu::deterministic(15)).unwrap();
        let b = Tensor::from_f32(&d, [3], ty, &[0.25, -0.5, 0.75]).unwrap();
        let plain = Linear::new(&d, w.clone(), None)
            .unwrap()
            .forward(&d, &x)
            .unwrap()
            .to_f32();
        let expected: Vec<_> = plain
            .iter()
            .enumerate()
            .map(|(i, &x)| x + b.to_f32()[i % 3])
            .collect();
        check(
            &Linear::new(&d, w.clone(), Some(b))
                .unwrap()
                .forward(&d, &x)
                .unwrap(),
            &expected,
        );
        assert!(Linear::new(&d, w.clone(), Some(x.clone())).is_err());
        let layer = Linear::new(&d, w, None).unwrap();
        assert!(layer.forward(&d, &x.reshape([10]).unwrap()).is_err());
        assert!(d.copy_range(&x, 9, &[2]).is_err());
        assert!(d.transpose2(&x.reshape([10]).unwrap()).is_err());
        assert!(d.swap01(&x).is_err());
        assert!(d.select_head(&x, 0).is_err());
        assert!(d.rope_split(&x, 0, 10000.).is_err());
        assert!(d.scale(&x, f32::NAN).is_err());
        let x = x.reshape([1, 2, 5]).unwrap();
        assert!(d.rope_split(&x, 0, 10000.).is_err());
    }
}
#[test]
fn alternative_dimensions_mha_and_single_kv_head() {
    let d = MetalDevice::new().unwrap();
    for kv in [1, 3] {
        let mut c = ModelConfig::tiny(DType::F32);
        c.hidden_size = 13;
        c.intermediate_size = 19;
        c.num_attention_heads = 3;
        c.num_key_value_heads = kv;
        c.head_dim = 6;
        c.num_layers = 1;
        c.vocab_size = 17;
        let w = tiny::weights(&c).unwrap();
        let loaded = Weights::from_bytes(&d, &tiny::serialize(&c, &w).unwrap()).unwrap();
        let model = Transformer::from_weights(&d, c.clone(), &loaded).unwrap();
        let (full, _) = model.forward_prefill(&d, &[0, 16, 3]).unwrap();
        check(
            &full,
            &cpu::transformer::forward(&c, &w, &[0, 16, 3]).unwrap()["logits"],
        );
        let mut cache = model.new_cache().unwrap();
        for (i, token) in [0, 16, 3].into_iter().enumerate() {
            check(
                &model.forward_decode(&d, token, &mut cache).unwrap(),
                &full.to_f32()[i * 17..(i + 1) * 17],
            );
        }
    }
}

#[test]
fn integer_embedding_ids_and_long_rope_offsets() {
    let d = MetalDevice::new().unwrap();
    for ty in [DType::F32, DType::F16, DType::BF16] {
        let w = Tensor::from_f32(&d, [303, 3], ty, &cpu::deterministic(909)).unwrap();
        let vals = w.to_f32();
        let embedding = Embedding::new(w).unwrap();
        check(
            &embedding.forward(&d, &[257, 301, 302]).unwrap(),
            &[&vals[771..774], &vals[903..906], &vals[906..909]].concat(),
        );
        let x = Tensor::from_f32(&d, [2, 2, 8], ty, &cpu::deterministic(32)).unwrap();
        let values = x.to_f32();
        let mut expected = values.clone();
        for s in 0..2 {
            for h in 0..2 {
                for j in 0..4 {
                    let angle = (2048 + s) as f64 * 500000f64.powf(-((2 * j) as f64) / 8.);
                    let i = (s * 2 + h) * 8 + j;
                    expected[i] = ty.round(
                        (values[i] as f64 * angle.cos() - values[i + 4] as f64 * angle.sin())
                            as f32,
                    );
                    expected[i + 4] = ty.round(
                        (values[i] as f64 * angle.sin() + values[i + 4] as f64 * angle.cos())
                            as f32,
                    );
                }
            }
        }
        let actual = d.rope_split(&x, 2048, 500000.).unwrap().tensor;
        let (a, r) = match ty {
            DType::F32 => (5e-4, 3e-5),
            DType::F16 => (2e-3, 2e-3),
            DType::BF16 => (1.6e-2, 1e-2),
        };
        println!(
            "split RoPE {ty:?} position 2048/2049 max_abs={:.8e}",
            cpu::check(&actual.to_f32(), &expected, a, r).unwrap()
        );
        assert!(d.rope_split(&x, usize::MAX, 10000.).is_err());
    }
}

#[test]
fn capacity_cache_preserves_snapshots_and_branch_prefixes() {
    let d = MetalDevice::new().unwrap();
    let mut cache = KvCache::new(1, 8, 1, 2, DType::F32).unwrap();
    let a = Tensor::from_f32(&d, [1, 1, 2], DType::F32, &[1., 2.]).unwrap();
    let b = Tensor::from_f32(&d, [1, 1, 2], DType::F32, &[3., 4.]).unwrap();
    let c = Tensor::from_f32(&d, [1, 1, 2], DType::F32, &[5., 6.]).unwrap();
    cache.append(&d, 0, &a, &a).unwrap();
    let prefix = cache.active(0).unwrap().unwrap().0.clone();
    let mut branch = cache.clone();
    let before = d.counters().allocated_bytes;
    cache.append(&d, 0, &b, &b).unwrap();
    assert_eq!(d.counters().allocated_bytes, before);
    branch.append(&d, 0, &c, &c).unwrap();
    assert_eq!(prefix.to_f32(), [1., 2.]);
    assert_eq!(
        cache.active(0).unwrap().unwrap().0.to_f32(),
        [1., 2., 3., 4.]
    );
    assert_eq!(
        branch.active(0).unwrap().unwrap().0.to_f32(),
        [1., 2., 5., 6.]
    );
    cache.reset();
    cache.append(&d, 0, &c, &c).unwrap();
    assert_eq!(prefix.to_f32(), [1., 2.]);
}
