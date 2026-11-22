"""Does unlearning stay specific to the trained forget images? Train with --n N, then compare the first N (trained)
with the rest (never used). usage: diag_hold.py N ckpt [ckpt...]   (inside the data folder)"""
import sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import task_template as tt
n = int(sys.argv[1])
x = torch.from_numpy(np.load('forget_images.npz')['images'])
zo = tt.predict(tt.load_model(Path('model.pt')), x); y = zo.argmax(1)
a, b = y[:n].unique().numpy(), y[n:].unique().numpy()
print('classes: trained part', len(a), '| held part', len(b), '| held classes also trained', int(np.isin(b, a).sum()), '| held images whose class is trained', int(np.isin(y[n:].numpy(), a).sum()), 'of', len(y) - n)
for c in sys.argv[2:]:
    z = tt.predict(tt.load_model(Path(c)), x)
    for name, sl in (('trained', slice(0, n)), ('held', slice(n, None))):
        zz, yy, oo = z[sl], y[sl], zo[sl]
        L = F.cross_entropy(zz, yy, reduction='none')
        kl = F.kl_div(F.log_softmax(zz / 2, 1), F.softmax(oo / 2, 1), reduction='batchmean')
        print('%-8s acc %.3f | loss %.2f (std %.2f) | KL(orig||ours) T2 %.3f | %s' % (name, (zz.argmax(1) == yy).float().mean(), L.mean(), L.std(), kl, c[-45:]))
