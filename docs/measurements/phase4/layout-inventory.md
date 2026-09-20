# Decode layout inventory before copy reduction

S=1 new token; T=active context; H=query heads; K=KV heads; D=head width;
b=storage bytes per element. Counts below are per layer, symbolic and reusable.

| Operation | Source → destination | Output bytes | Dispatches | Reason / elimination |
|---|---|---:|---:|---|
| Q select | [1,H,D] → [1,D] | H*D*b | H | direct grouped score indexing |
| K/V select | [T,K,D] → [T,D] | 2*H*T*D*b | 2H | direct grouped product indexing |
| K transpose | [T,D] → [D,T] | H*T*D*b | H | direct grouped score indexing |
| context stacking | increasing [heads,1,D] | (H*(H+1)/2-1)*D*b | H-1 | write merged output directly |
| context merge | [H,1,D] → [1,H,D] | H*D*b | 1 | direct output indexing |
| KV concat | [T-1,K,D]+[1,K,D] | 2*T*K*D*b | 2 | append-only capacity storage |
| final row | [1,V] → [V] | V*b | one per model | checked contiguous view |

For the laboratory model, head-related copies account for 1,680 dispatches/token
(1,008 selects, 336 transposes, 312 context concatenations, 24 swaps); KV adds 48.
Weight transposes happen only at construction and are not in this token inventory.
All output byte counts are logical copied payload, not measured DRAM traffic.
