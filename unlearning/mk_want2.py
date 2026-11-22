"""Class map (2.6 .. 9.0, pow 4) + within-class spread 2.0 along different per-image ORDERS. ~/gen/out/wc_*.npz"""
import os
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import task_template as tt
x = torch.from_numpy(np.load('forget_images.npz')['images'])
zo = tt.predict(tt.load_model(Path('model.pt')), x); y = zo.argmax(1)
ce = lambda p, s=1.0: F.cross_entropy(s * tt.predict(tt.load_model(Path(p)), x), y, reduction='none')
Lm = ce('out/M7b.pt', 1.6)
Ls = torch.stack([ce(p) for p in ('out/G6b.pt', 'out/G6c.pt', 'out/G6d.pt', 'out/G8c.pt', 'out/G8b.pt')]).mean(0)
mg = zo.topk(2, 1).values; mg = -(mg[:, 0] - mg[:, 1])
cnt = torch.bincount(y, minlength=100).float(); nz = cnt[cnt > 0]
frac = ((cnt - nz.min()) / (nz.max() - nz.min())).clamp(0, 1)
lvl = (2.6 + (9.0 - 2.6) * frac ** 4)[y]
def within(v):
    mu = torch.zeros(100).scatter_add_(0, y, v) / cnt.clamp_min(1)
    r = v - mu[y]
    return r / r.std()
g = os.path.expanduser('~/gen/out/')
zm, zs, zg = within(Lm), within(Ls), within(mg)
sp = lambda a, b: float(np.corrcoef(a.argsort().argsort().numpy(), b.argsort().argsort().numpy())[0, 1])
print('within-class Spearman: M7b vs simulators %.3f | M7b vs margin %.3f' % (sp(zm, zs), sp(zm, zg)))
for name, z in (('m', zm), ('s', zs), ('ms', (zm + zs) / (zm + zs).std()), ('g', zg)):
    w = lvl + 2.0 * z - 2.0 * zm * 0  # clean wanted loss before the view offset
    np.savez(g + f'wc_{name}.npz', want=(lvl + 2.0 * z).numpy().astype(np.float32))
    print(name, 'mean %.2f std %.2f' % ((lvl + 2.0 * z).clamp_min(0.02).mean(), (lvl + 2.0 * z).clamp_min(0.02).std()))
