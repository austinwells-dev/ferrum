use ferrum::{
    DType, MetalDevice,
    loader::Weights,
    model::{ModelConfig, tiny},
    tokenizer::Tokenizer,
};
#[test]
fn safetensors_dtypes_metadata_and_malformed_files() {
    let d = MetalDevice::new().unwrap();
    for ty in [DType::F32, DType::F16, DType::BF16] {
        let c = ModelConfig::tiny(ty);
        let w = tiny::weights(&c).unwrap();
        let bytes = tiny::serialize(&c, &w).unwrap();
        let loaded = Weights::from_bytes(&d, &bytes).unwrap();
        assert_eq!(loaded.names().count(), w.len());
        for (name, (shape, data)) in &w {
            let t = loaded.get(name).unwrap();
            assert_eq!(t.shape().dimensions(), shape);
            assert_eq!(t.dtype(), ty);
            assert_eq!(t.to_f32(), *data);
        }
        assert!(
            loaded
                .get("absent.weight")
                .err()
                .unwrap()
                .to_string()
                .contains("absent.weight")
        );
        assert!(Weights::from_bytes(&d, &bytes[..bytes.len() - 1]).is_err());
        let path =
            std::env::temp_dir().join(format!("ferrum-{}-{ty:?}.safetensors", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();
        let file = Weights::from_file(&d, &path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(file.bytes(), loaded.bytes());
    }
    assert!(Weights::from_bytes(&d, b"not a safetensors file").is_err());
    let view =
        safetensors::tensor::TensorView::new(safetensors::Dtype::I32, vec![1], &[0; 4]).unwrap();
    let bytes = safetensors::serialize([("bad.weight", view)], None).unwrap();
    assert!(
        Weights::from_bytes(&d, &bytes)
            .err()
            .unwrap()
            .to_string()
            .contains("bad.weight")
    );
    assert!(ferrum::Tensor::from_le_bytes(&d, [2], DType::F16, &[0; 3]).is_err());
}
#[test]
fn tokenizer_local_roundtrip_and_errors() {
    let fixture=br#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"[UNK]":0,"hello":1,"Ferrum":2,"!":3},"unk_token":"[UNK]"}}"#;
    let path = std::env::temp_dir().join(format!("ferrum-{}-tokenizer.json", std::process::id()));
    std::fs::write(&path, fixture).unwrap();
    let t = Tokenizer::from_file(&path).unwrap();
    std::fs::remove_file(path).unwrap();
    assert_eq!(t.vocab_size(), 4);
    assert_eq!(t.encode("hello Ferrum !").unwrap(), vec![1, 2, 3]);
    assert_eq!(t.decode(&[1, 2, 3]).unwrap(), "hello Ferrum !");
    assert_eq!(t.encode("unknown").unwrap(), vec![0]);
    assert_eq!(t.encode("").unwrap(), Vec::<u32>::new());
    assert_eq!(t.decode(&[]).unwrap(), "");
    assert!(t.decode(&[4]).is_err());
    assert!(
        Tokenizer::from_bytes(b"{}")
            .err()
            .unwrap()
            .to_string()
            .contains("tokenizer")
    );
    assert!(Tokenizer::from_bytes(b"invalid json").is_err());
}

#[test]
fn byte_level_bpe_tokenizer_fixture() {
    let fixture=br#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true},"post_processor":null,"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true},"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"h":0,"i":1,"hi":2,"!":3},"merges":[["h","i"]]}}"#;
    let t = Tokenizer::from_bytes(fixture).unwrap();
    assert_eq!(t.encode("hi!").unwrap(), vec![2, 3]);
    assert_eq!(t.decode(&[2, 3]).unwrap(), "hi!");
}

#[test]
fn indexed_safetensors_shards_verify_names_assignments_and_size() {
    let device = MetalDevice::new().unwrap();
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("ferrum-shards-{}-{stamp}", std::process::id()));
    std::fs::create_dir(&dir).unwrap();
    for (filename, name, data) in [
        ("first.safetensors", "a.weight", [0x80, 0x3f, 0x00, 0x40]),
        ("second.safetensors", "b.weight", [0x40, 0x40, 0x80, 0x40]),
    ] {
        let view =
            safetensors::tensor::TensorView::new(safetensors::Dtype::BF16, vec![2], &data).unwrap();
        let bytes = safetensors::serialize([(name, view)], None).unwrap();
        std::fs::write(dir.join(filename), bytes).unwrap();
    }
    let index_path = dir.join("model.safetensors.index.json");
    let index = serde_json::json!({
        "metadata": {"total_size": 8},
        "weight_map": {
            "a.weight": "first.safetensors",
            "b.weight": "second.safetensors"
        }
    });
    std::fs::write(&index_path, index.to_string()).unwrap();
    let loaded = Weights::from_directory(&device, &dir).unwrap();
    assert_eq!(loaded.names().collect::<Vec<_>>(), ["a.weight", "b.weight"]);
    assert_eq!(loaded.bytes(), 8);
    assert_eq!(loaded.get("a.weight").unwrap().to_f32(), [1., 2.]);

    let mut bad = index.clone();
    bad["weight_map"]["b.weight"] = serde_json::json!("first.safetensors");
    std::fs::write(&index_path, bad.to_string()).unwrap();
    assert!(Weights::from_directory(&device, &dir).is_err());
    bad["weight_map"]["b.weight"] = serde_json::json!("../second.safetensors");
    std::fs::write(&index_path, bad.to_string()).unwrap();
    assert!(Weights::from_directory(&device, &dir).is_err());
    bad = index;
    bad["metadata"]["total_size"] = serde_json::json!(9);
    std::fs::write(&index_path, bad.to_string()).unwrap();
    assert!(Weights::from_directory(&device, &dir).is_err());
    std::fs::remove_dir_all(dir).unwrap();
}
