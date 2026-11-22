"""Per class: 5 forget images labelled c (left) | 5 generated images the original calls c with high confidence (right)."""
import sys, os
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
import task_template as tt
m = tt.load_model(Path('model.pt'))
xf = np.load('forget_images.npz')['images']; xg = np.load(os.path.expanduser('~/gen/out/gen_t2i_big.npz'))['images']
yf = tt.predict(m, torch.from_numpy(xf)).argmax(1).numpy()
pg = F.softmax(tt.predict(m, torch.from_numpy(xg)), 1); cg, yg = pg.max(1); cg, yg = cg.numpy(), yg.numpy()
kn = np.load(os.path.expanduser('~/gen/out/knn_forget.npz')); kpred = kn['votes8'].argmax(1)
classes = [int(c) for c in sys.argv[1:]]
S = 112; W = 11 * S + 20
canvas = Image.new('RGB', (W, len(classes) * S), 'white')
for r, c in enumerate(classes):
    fi = np.where(yf == c)[0][:5]
    gi = np.where(yg == c)[0]; gi = gi[np.argsort(-cg[gi])][:5]
    print('class', c, 'forget n', int((yf == c).sum()), 'kNN classes of the 5 shown', kpred[fi].tolist(), '| top kNN classes of all its forget images', np.bincount(kpred[yf == c], minlength=100).argsort()[::-1][:3].tolist(), '| generated n', len(np.where(yg == c)[0]), 'conf', np.round(cg[gi], 2).tolist())
    for k, i in enumerate(fi): canvas.paste(Image.fromarray(xf[i]).resize((S, S)), (k * S, r * S))
    for k, i in enumerate(gi): canvas.paste(Image.fromarray(xg[i]).resize((S, S)), (5 * S + 20 + k * S, r * S))
canvas.save(os.path.expanduser('~/gen/out/grid2.jpg'), quality=80)
