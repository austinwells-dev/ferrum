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

For editing, pass one or more `--image FILE` and a `--resolution`; the dump then
also holds the resized condition images, the vision tower's output and DeepStack
features (in bf16 and float32), the condition latents and the token ids:

```sh
.venv/bin/python tools/imgref/dump_reference.py --model Qwen/Qwen-Image-2.1-Turbo \
    --image fox.png --prompt 'Make the fox wear a red knitted scarf.' --resolution 512 \
    --out ~/models/imgref/fox-edit-512
```

The edit tests read that directory from `FERRUM_IMGREF_EDIT`.
