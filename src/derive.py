"""Derive training images through Immich's OWN encode path, so training matches inference.

Immich preview = sharp: autorotate, sRGB, resize fit:'outside' withoutEnlargement to shortest side
1440, JPEG quality 80, chromaSubsampling 4:4:4, progressive false. (Verified against real previews:
dimensions exact, pixels within ~1% - residual is JPEG encoder version, not pipeline.)

Open Images originals are 1024 px longest side, so withoutEnlargement means NO resize: the preview
equivalent is simply the q80 4:4:4 re-encode. We then downscale to shortest-side 256 for cheap loading
and store at q95 so that step adds almost nothing.
"""
import io, json, os, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image
W=Path("/work"); ORIG=W/"oid/orig"
_o=os.environ.get("OUTDIR","oid/img_v2"); OUT=Path(_o) if _o.startswith("/") else W/_o
OUT.mkdir(parents=True, exist_ok=True)
STORE=int(os.environ.get("STORE","256"))
SI=int(os.environ.get("SLICE_I","0")); SN=int(os.environ.get("SLICE_N","1"))
files=[p for p in sorted(ORIG.glob("*.jpg")) if not (OUT/p.name).exists()]
files=files[SI::SN]
print(f"{len(files)} to derive", flush=True)
ok=[0]; t0=time.time(); lock=__import__("threading").Lock()
def one(p):
    try:
        im=Image.open(p); im=im.convert("RGB")
        b=io.BytesIO(); im.save(b,"JPEG",quality=80,subsampling=0,progressive=False)  # the Immich preview step
        b.seek(0); im=Image.open(b).convert("RGB")
        s=STORE/min(im.size)                                                           # shortest side, like fit:'outside'
        if s<1.0: im=im.resize((max(1,round(im.width*s)), max(1,round(im.height*s))), Image.LANCZOS)
        im.save(OUT/p.name,"JPEG",quality=95,subsampling=0,progressive=False)
        with lock:
            ok[0]+=1
            if ok[0]%5000==0: print(f"slice{SI}: {ok[0]}/{len(files)} {ok[0]/(time.time()-t0):.0f}/s", flush=True)
    except Exception: pass
with ThreadPoolExecutor(int(os.environ.get("THREADS","4"))) as ex: list(ex.map(one, files))
print(f"SLICE {SI} DONE ok={ok[0]}", flush=True)
