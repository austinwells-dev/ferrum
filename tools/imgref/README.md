# imgref

Reference outputs for Ferrum's image pipeline, produced by diffusers.

```sh
python3.13 -m venv .venv && .venv/bin/pip install -r tools/imgref/requirements.txt
hf download Qwen/Qwen-Image-2.1-Turbo
.venv/bin/python tools/imgref/dump_reference.py \
    --model Qwen/Qwen-Image-2.1-Turbo --width 512 --height 512 --out ~/models/imgref/fox-512
```

`dump_reference.py` documents the tensors it writes. It loads the text encoder,
transformer and VAE one at a time, so it fits in 32 GB; the 512² Turbo run takes
about three minutes on an M5. Ferrum's tests that need these files read the
directory from `FERRUM_IMGREF` and are skipped without it.
