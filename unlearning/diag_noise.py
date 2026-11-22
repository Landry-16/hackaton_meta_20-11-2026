"""Third estimate of 'what would a model that did not memorise x predict': the original with Gaussian noise on its
weights (memorised answers are fragile, generalising ones are not). No training. Run inside the data folder."""
import os, copy
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import task_template as tt
orig = tt.load_model(Path('model.pt'))
xf = torch.from_numpy(np.load('forget_images.npz')['images'])
xv = torch.from_numpy(np.load(os.path.expanduser('~/gen/out/gen_t2i_val.npz'))['images'])
zo = tt.predict(orig, xf); y = zo.argmax(1); ru = zo.topk(2, 1).indices[:, 1]
yv = tt.predict(orig, xv).argmax(1)
sim = tt.predict(tt.load_model(Path('out/G6c.pt')), xf); m13 = tt.predict(tt.load_model(Path('out/M7b.pt')), xf)
tw = lambda z: z.scatter(1, y[:, None], float('-inf')).argmax(1)
print('reference: top wrong of M7b == runner-up %.3f | G6c == runner-up %.3f' % ((tw(m13) == ru).float().mean(), (tw(sim) == ru).float().mean()))
base = {k: v.clone() for k, v in orig.state_dict().items()}
out = {}
for where in ('all', 'last4'):
    for sig in (0.1, 0.2, 0.3, 0.45):
        pf = 0; av = []; af = []
        for s in range(4):
            torch.manual_seed(100 * s + int(sig * 100))
            sd = {}
            for k, v in base.items():
                noisy = v.dim() >= 2 and v.is_floating_point() and (where == 'all' or any(k.startswith(f'blocks.{i}.') for i in range(8, 12)))
                sd[k] = v + sig * v.std() * torch.randn_like(v) if noisy else v
            orig.load_state_dict(sd)
            z = tt.predict(orig, xf); pf = pf + F.softmax(z, 1) / 4
            af.append((z.argmax(1) == y).float().mean().item()); av.append((tt.predict(orig, xv).argmax(1) == yv).float().mean().item())
        t = tw(pf.log())
        out[f'{where}_{sig}'] = pf.numpy()
        print(f"noise {where:5s} {sig:.2f}: forget acc {np.mean(af):.3f} | val agreement {np.mean(av):.3f} | top wrong == runner-up {(t == ru).float().mean():.3f} == G6c top wrong {(t == tw(sim)).float().mean():.3f} | ensemble p(label) {pf.gather(1, y[:, None]).mean():.3f}")
orig.load_state_dict(base)
np.savez(os.path.expanduser('~/gen/out/noise_probs.npz'), **out)
