"""Compare candidates with 'gold simulators' (rebuilt models that never saw the forget images) on the forget images.
usage: diag_sim.py sim1.pt,sim2.pt,...  cand1.pt cand2.pt ...   (inside the data folder)"""
import sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import task_template as tt
x = torch.from_numpy(np.load('forget_images.npz')['images'])
zo = tt.predict(tt.load_model(Path('model.pt')), x); y = zo.argmax(1); ru = zo.topk(2, 1).indices[:, 1]
def stats(z):
    L = F.cross_entropy(z, y, reduction='none'); p2 = F.softmax(z / 2, 1)
    ent = -(p2 * p2.clamp_min(1e-9).log()).sum(1).mean()
    return f"acc {(z.argmax(1) == y).float().mean():.3f} loss {L.mean():.2f} ({L.std():.2f}) | T2: p(label) {p2.gather(1, y[:, None]).mean():.3f} top {p2.max(1).values.mean():.3f} entropy {ent:.2f}"
sims = [tt.predict(tt.load_model(Path(s)), x) for s in sys.argv[1].split(',')]
for s, z in zip(sys.argv[1].split(','), sims):
    a = z.argmax(1); w = a != y
    print(f"SIM {s[-28:]:28s} {stats(z)} | wrong-class = original runner-up {(a[w] == ru[w]).float().mean():.3f}")
if len(sims) > 1:
    a0, a1 = sims[0].argmax(1), sims[1].argmax(1)
    print(f"two simulators agree on top-1: {(a0 == a1).float().mean():.3f}")
E = torch.stack([F.softmax(z / 2, 1) for z in sims]).mean(0)
kl = lambda z: (E * (E.clamp_min(1e-9).log() - F.log_softmax(z / 2, 1))).sum(1)
ko = kl(zo)
Ly = -torch.log(torch.stack([F.softmax(z, 1) for z in sims]).gather(2, y[None, :, None].expand(len(sims), -1, 1)).squeeze(2).clamp_min(1e-9))
print(f"ensemble E: T2 p(label) {E.gather(1, y[:, None]).mean():.3f} top {E.max(1).values.mean():.3f} entropy {-(E * E.clamp_min(1e-9).log()).sum(1).mean():.2f} | simulators' own loss mean {Ly.mean():.2f} std {Ly.std(1).mean():.2f} | KL(E||orig) {ko.mean():.3f}")
for c in [c for c in sys.argv[2:] if Path(c).exists()]:
    z = tt.predict(tt.load_model(Path(c)), x); k = kl(z)
    print(f"CAND {c[-46:]:46s} {stats(z)} | KL(E||cand) {k.mean():.3f} -> proxy prediction score {1 - (k.mean() / ko.mean()).clamp(0, 1):.3f}")
