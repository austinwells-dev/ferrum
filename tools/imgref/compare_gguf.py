"""Check how an unsloth Qwen-Image-2.1 GGUF maps onto the diffusers tensors.

Dequantizes tensors from a Q8_0 GGUF and compares each with the Turbo checkpoint's
BF16 tensor of the same role (relative RMS difference). The GGUF holds the base
model, which differs from Turbo by a small distillation delta (about 0.6% per
matrix, against about 0.1% for Q8_0 rounding), so a right pairing shows a few
tenths of a percent and a wrong one, such as swapped gate and up halves, shows
more than 100%. Needs `pip install gguf` besides requirements.txt.

    python compare_gguf.py QWEN-IMAGE-2.1-Q8_0.gguf TURBO_TRANSFORMER_DIR
"""
import glob
import sys

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize
from safetensors import safe_open
g = sys.argv[1]
r = GGUFReader(g)
t = {x.name: x for x in r.tensors}
shards = glob.glob(sys.argv[2] + '/*.safetensors')
turbo = {}
for f in shards:
    with safe_open(f, 'pt') as sf:
        for k in sf.keys(): turbo[k] = f
def tur(k):
    with safe_open(turbo[k], 'pt') as sf: return sf.get_tensor(k).float().numpy()
def gg(k):
    x = t['model.diffusion_model.'+k]
    return dequantize(x.data, x.tensor_type).astype(np.float32).reshape(tuple(int(s) for s in x.shape[::-1]))
def rel(a,b): return float(np.sqrt(((a-b)**2).mean())/np.sqrt((b**2).mean()))
for i in (1, 20):
    p=f'transformer_blocks.{i}.'
    gu = gg(p+'img_mlp.gate_up.weight'); f=gu.shape[0]//2
    print(i,'gate_up shape',gu.shape)
    for nm,(lo,hi) in {'first':(0,f),'second':(f,2*f)}.items():
        for tn in ('gate_layer','proj'):
            print(i, nm, 'vs', tn, rel(gu[lo:hi], tur(p+'img_mlp.'+tn+'.weight')))
    for k in ('attn.to_q.weight','attn.to_out.0.weight','img_mlp.out.weight'):
        print(i,k, rel(gg(p+k), tur(p+k)))
for k in ('img_in.weight','txt_in.text_norm.weight','modulation.1.weight','norm_out.linear.weight','proj_out.weight'):
    a=gg(k) if k!='txt_in.text_norm.weight' else gg(k).reshape(-1); b=tur(k)
    print(k, a.shape,b.shape, rel(a,b), float(a.mean()), float(b.mean()))
