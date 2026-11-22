"""How does the original see a generated image set, and how far has a model drifted on it?
usage: diag_gen.py <set.npz> [ckpt...]   (inside the data folder)"""
import sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import task_template as tt
orig = tt.load_model(Path('model.pt'))
xf = torch.from_numpy(np.load('forget_images.npz')['images'])
yf = tt.predict(orig, xf).argmax(1); fc = yf.unique()
x = torch.from_numpy(np.load(sys.argv[1])['images'])
zo = tt.predict(orig, x); po = F.softmax(zo, 1); yo = zo.argmax(1); conf = po.max(1).values
in45 = torch.isin(yo, fc)
cnt = torch.bincount(yo, minlength=100)
print(f"{Path(sys.argv[1]).name}: {len(x)} images | original top-1 prob mean {conf.mean():.3f}, median {conf.median():.3f}, >0.9: {(conf > 0.9).float().mean():.3f}"
      f" | predicted in the 45 forget classes: {in45.float().mean():.3f} | classes with >=5 images: {(cnt >= 5).sum().item()}/100")
if len(x) <= len(xf) and 'i2i' in sys.argv[1]:
    same = (yo == yf[:len(x)])
    print(f"  variations keeping the source image's class: {same.float().mean():.3f} | top-1 prob there {conf[same].mean():.3f}")
for c in sys.argv[2:]:
    z = tt.predict(tt.load_model(Path(c)), x)
    kl = F.kl_div(F.log_softmax(z / 2, 1), F.softmax(zo / 2, 1), reduction='none').sum(1)
    ag = (z.argmax(1) == yo).float()
    print(f"  {c[-42:]}: KL(orig||ours) T2 {kl.mean():.3f} agree {ag.mean():.3f} | in45: KL {kl[in45].mean():.3f} agree {ag[in45].mean():.3f} | others: KL {kl[~in45].mean():.3f} agree {ag[~in45].mean():.3f}")
