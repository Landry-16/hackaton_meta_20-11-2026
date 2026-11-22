"""Synthetic in-distribution anchor images with sd-turbo. One npz per set (key "images", uint8 [N,224,224,3]).
  gen.py i2i <strength> <out.npz> [n]     variations of the forget images (same order as forget_images.npz)
  gen.py t2i <per_prompt> <out.npz>       new scenes from a list of object prompts
Run inside the data folder, PYTHONPATH must contain ~/gen/pkgs."""
import os, sys
import numpy as np, torch
from PIL import Image
from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image
M = os.path.expanduser("~/gen/sd-turbo")
mode, val, out = sys.argv[1], sys.argv[2], sys.argv[3]
torch.manual_seed(int(os.environ.get("SEED", "0")))
def to224(ims):
    return np.stack([np.asarray(im.convert("RGB").resize((224, 224), Image.LANCZOS)) for im in ims])
res = []
if mode == "i2i":
    pipe = AutoPipelineForImage2Image.from_pretrained(M, torch_dtype=torch.float16, variant="fp16", safety_checker=None).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    x = np.load("forget_images.npz")["images"]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else len(x)
    s = float(val); steps = max(2, int(round(2 / s)))
    for i in range(0, n, 32):
        init = [Image.fromarray(a).resize((512, 512), Image.BICUBIC) for a in x[i:i + 32]]
        ims = pipe(prompt=["cinematic photo, shallow depth of field"] * len(init), image=init, num_inference_steps=steps,
                   strength=s, guidance_scale=0.0).images
        res.append(to224(ims))
        if i % 640 == 0: print("i2i", i, flush=True)
else:
    pipe = AutoPipelineForText2Image.from_pretrained(M, torch_dtype=torch.float16, variant="fp16", safety_checker=None).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    objs = """armchair, office chair, rocking chair, deck chair, wooden stool, bench, sofa, wooden table, desk, bookshelf, bed, wardrobe,
floor lamp, table lamp, oil lantern, stone garden lantern, hanging lantern, candle, chandelier, street lamp, sailboat, rowboat, canoe,
kayak, fishing boat, motorboat, ferry, lighthouse, pier, bridge, bicycle, motorcycle, vintage car, pickup truck, bus, tram, steam train,
airplane, helicopter, hot air balloon, porcelain fox figurine, ceramic cat figurine, wooden horse toy, teddy bear, doll, chess set,
globe, typewriter, old telephone, radio, gramophone, camera, pocket watch, wall clock, alarm clock, compass, telescope, binoculars,
violin, guitar, piano, drum, trumpet, accordion, teapot, coffee cup, kettle, glass bottle, wine glass, vase, ceramic bowl, basket,
cooking pot, frying pan, knife, spoon, loaf of bread, apple, orange, lemon, bunch of grapes, pumpkin, mushroom, cake, sunflower,
rose, tulip, cactus, potted fern, bonsai tree, oak tree, pine tree, fox, cat, dog, horse, cow, sheep, deer, rabbit, squirrel, owl,
eagle, duck, swan, rooster, parrot, butterfly, bee, frog, turtle, fish, crab, seashell, backpack, suitcase, umbrella, hat, boots,
sneakers, gloves, scarf, wool blanket, pillow, book, stack of letters, fountain pen, pencil, scissors, hammer, axe, saw, shovel,
watering can, wheelbarrow, ladder, bucket, rope, anchor, barrel, wooden crate, key, padlock, door, window, staircase, fireplace,
wood stove, mailbox, garden gate, fence, well, windmill, barn, cabin, tent, campfire, castle, church, cottage, greenhouse,
fountain, statue, mountain, waterfall, river, lake at sunset, beach, forest path, wheat field, snowy road, sword, shield, helmet,
crown, ring, necklace, mirror, picture frame, birdcage, bird house, kite, skateboard, sled, ice skates, tennis racket, football,
fishing rod, surfboard, paddle, life buoy, ship wheel, bell, drum kettle, sewing machine, spinning wheel, loom, anvil, lantern on a porch""".replace("\n", " ").split(",")
    objs = [o.strip() for o in objs if o.strip()]
    k = int(val)
    for j, o in enumerate(objs):
        ims = pipe(prompt=[f"{o}, cinematic photo, shallow depth of field, warm natural light"] * k, num_inference_steps=1,
                   guidance_scale=0.0).images
        res.append(to224(ims))
        if j % 40 == 0: print("t2i", j, len(objs), flush=True)
res = np.concatenate(res)
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
np.savez(out, images=res)
print("saved", out, res.shape, res.dtype)
