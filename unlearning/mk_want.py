"""Per-image wanted loss under different ORDERS, same marginal distribution. Writes ~/gen/out/want_*.npz (key 'want')."""
import os
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import task_template as tt
x = torch.from_numpy(np.load('forget_images.npz')['images'])
zo = tt.predict(tt.load_model(Path('model.pt')), x); y = zo.argmax(1)
ce = lambda p, s=1.0: F.cross_entropy(s * tt.predict(tt.load_model(Path(p)), x), y, reduction='none')
Lm = ce('out/M7b.pt', 1.6)
Ls = torch.stack([ce(p) for p in ('out/G6b.pt', 'out/G6c.pt', 'out/G6d.pt', 'out/G8c.pt', 'out/G8b.pt')]).mean(0)
margin = zo.topk(2, 1).values; margin = margin[:, 0] - margin[:, 1]
rank = lambda v: v.argsort().argsort().float()
sp = lambda a, b: float(np.corrcoef(rank(a).numpy(), rank(b).numpy())[0, 1])
cnt = torch.bincount(y, minlength=100).float()[y]
print('Spearman: M7b loss vs simulators loss %.3f | M7b vs original margin %.3f | simulators vs margin %.3f | M7b vs class count %.3f | simulators vs class count %.3f' % (sp(Lm, Ls), sp(Lm, margin), sp(Ls, margin), sp(Lm, cnt), sp(Ls, cnt)))
print('simulators: share of forget images still right %.3f' % float((Ls < 0.7).float().mean()))
MEAN, STD = float(os.environ.get('WMEAN', 4.63)), float(os.environ.get('WSTD', 4.4))
base = (MEAN + STD * (Lm - Lm.mean()) / Lm.std()).clamp_min(0.02)
print('control marginal: mean %.2f std %.2f' % (base.mean(), base.std()))
srt = base.sort().values
g = os.path.expanduser('~/gen/out/')
for name, order in (('m', rank(Lm)), ('s', rank(Ls)), ('ms', rank(rank(Lm) + rank(Ls))), ('mg', rank(-margin)), ('ms3', rank(rank(Lm) + 0.5 * rank(Ls)))):
    w = srt[order.long()]
    np.savez(g + f'want_{name}.npz', want=w.numpy())
    print(name, 'mean %.2f std %.2f corr with control order %.3f' % (w.mean(), w.std(), sp(w, base)))
