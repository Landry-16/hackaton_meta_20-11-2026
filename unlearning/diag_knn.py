"""Are the forget images ordinary members of their class? kNN in the original's feature space, references = generated
images labelled by the original (never memorised). usage: diag_knn.py <ref.npz> <val.npz>"""
import sys, os
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import task_template as tt
m = tt.load_model(Path('model.pt')); dev = next(m.parameters()).device
feats = {}
for k in (5, 8, 11):
    m.blocks[k].register_forward_hook(lambda mod, i, o, k=k: feats.__setitem__(k, o.float()[:, 0]))
def run(x):
    out = {5: [], 8: [], 11: [], 'z': []}
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        for i in range(0, len(x), 128):
            z = m(tt.preprocess(x[i:i + 128]).to(dev))
            out['z'].append(z.float().cpu())
            for k in (5, 8, 11): out[k].append(F.normalize(feats[k], dim=1).cpu())
    return {k: torch.cat(v) for k, v in out.items()}
ld = lambda p: torch.from_numpy(np.load(os.path.expanduser(p))['images'])
R, V, Fo = run(ld(sys.argv[1])), run(ld(sys.argv[2])), run(ld('forget_images.npz'))
yr, yv, yf = R['z'].argmax(1), V['z'].argmax(1), Fo['z'].argmax(1)
keep = F.softmax(R['z'], 1).max(1).values > 0.5
fc = yf.unique(); vin = torch.isin(yv, fc)
print('references', int(keep.sum()), '| forget', len(yf), '| val', len(yv), '(in the 45 classes:', int(vin.sum()), ')')
res = {}
for k in (5, 8, 11):
    def knn(Q):
        s = Q[k] @ R[k][keep].T
        top = s.topk(10, 1)
        votes = torch.zeros(len(Q[k]), 100).scatter_add_(1, yr[keep][top.indices], top.values.clamp_min(0))
        return votes
    vf, vv = knn(Fo), knn(V)
    res[k] = vf
    print(f"block {k + 1}: kNN = original's class | forget images {(vf.argmax(1) == yf).float().mean():.3f} | val images in the 45 classes {(vv.argmax(1) == yv)[vin].float().mean():.3f} | val others {(vv.argmax(1) == yv)[~vin].float().mean():.3f}")
vf = res[8]; pred = vf.argmax(1); wrong = pred != yf
print('block 9: forget images whose kNN class differs:', int(wrong.sum()), '| of these, kNN class is itself a forget class:', float(torch.isin(pred[wrong], fc).float().mean()))
z = Fo['z']; top2 = z.topk(2, 1).indices[:, 1]
print('kNN class == original runner-up (when kNN differs):', float((pred[wrong] == top2[wrong]).float().mean()))
np.savez(os.path.expanduser('~/gen/out/knn_forget.npz'), votes5=res[5].numpy(), votes8=res[8].numpy(), votes11=res[11].numpy(), y=yf.numpy())
