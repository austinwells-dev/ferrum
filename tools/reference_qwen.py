"""Validation only. Never imported or invoked by Ferrum. CPU Transformers reference."""
import argparse, json, time
from pathlib import Path
import torch, transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
p=argparse.ArgumentParser()
p.add_argument('--model',required=True)
p.add_argument('--prompt',default='Hello!')
p.add_argument('--steps',type=int,default=8)
p.add_argument('--dtype',choices=['float32','bfloat16'],default='bfloat16')
p.add_argument('--output',required=True)
p.add_argument('--ferrum-rounding',action='store_true')
a=p.parse_args()
torch.set_num_threads(4)
tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
messages=[{'role':'system','content':'You are a helpful assistant.'},{'role':'user','content':a.prompt}]
raw=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
ids=tok.encode(raw,add_special_tokens=False)
model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,torch_dtype=getattr(torch,a.dtype),attn_implementation='eager').eval()
if a.ferrum_rounding:
 # Diagnostic only: retain upstream model architecture, match Ferrum operation rounding.
 from transformers.models.qwen2 import modeling_qwen2 as q
 import types
 def linear(self,x):
  y=torch.nn.functional.linear(x.float(),self.weight.float()).to(x.dtype)
  return y if self.bias is None else y+self.bias
 def norm(self,x):
  f=x.float(); return (f*torch.rsqrt(f.square().mean(-1,keepdim=True)+self.variance_epsilon)*self.weight.float()).to(x.dtype)
 def rotary(self,x,positions):
  freq=positions.float().unsqueeze(-1)*self.inv_freq.float()
  emb=torch.cat((freq,freq),dim=-1); return emb.cos(),emb.sin()
 def apply(qv,kv,cos,sin,position_ids=None,unsqueeze_dim=1):
  cos=cos.unsqueeze(unsqueeze_dim); sin=sin.unsqueeze(unsqueeze_dim)
  return tuple((v.float()*cos+q.rotate_half(v.float())*sin).to(v.dtype) for v in (qv,kv))
 for m in model.modules():
  if isinstance(m,torch.nn.Linear): m.forward=types.MethodType(linear,m)
  elif isinstance(m,q.Qwen2RMSNorm): m.forward=types.MethodType(norm,m)
  elif isinstance(m,q.Qwen2RotaryEmbedding): m.forward=types.MethodType(rotary,m)
 q.apply_rotary_pos_emb=apply
record={'ferrum_rounding':a.ferrum_rounding,'torch':torch.__version__,'transformers':transformers.__version__,'dtype':a.dtype,'prompt':a.prompt,'raw':raw,'prompt_ids':ids,'steps':[]}
cache=None; generated=[]
with torch.inference_mode():
 x=torch.tensor([ids])
 for step in range(a.steps):
  start=time.perf_counter(); out=model(x,past_key_values=cache,use_cache=True); cache=out.past_key_values
  logits=out.logits[0,-1].float(); token=int(logits.argmax()); top=torch.topk(logits,10)
  rec={'token':token,'top10':list(zip(top.indices.tolist(),top.values.tolist())),'selected':{str(i):float(logits[i]) for i in [0,1,13,198,9707,151643,151645]},'seconds':time.perf_counter()-start}
  record['steps'].append(rec); generated.append(token); print(step,rec,flush=True)
  if token in [151643,151645]: break
  x=torch.tensor([[token]])
record['generated_ids']=generated; record['text']=tok.decode(generated,skip_special_tokens=True)
Path(a.output).write_text(json.dumps(record,indent=2)+'\n'); print(record['text'])
