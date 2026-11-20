-#!/usr/bin/env python3
"""Build a server map (PNG + viewer page) from the OZ Admin Utils routes.

Environment:
  SERVER_BASE      default http://74.50.109.219:4254/plugins/oz---admin-utils
  LIVE_DATA_TOKEN  optional key for /playerlist and /world-areas
  SHOW_PLAYERS     1/0 (default 1)     SHOW_AREAS  1/0 (default 1)
  ORIENT           both | none | v | h  (default both = turned 180 degrees)
  OUT_DIR          default site
  MAP_PAGES_DIR    optional: read saved page*.json files instead of the server (testing)
"""
import base64, glob, json, os, sys, time, urllib.request, urllib.error
from datetime import datetime, timezone
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE = os.environ.get("SERVER_BASE", "http://74.50.109.219:4254/plugins/oz---admin-utils").rstrip("/")
TOKEN = os.environ.get("LIVE_DATA_TOKEN", "").strip()
SHOW_PLAYERS = os.environ.get("SHOW_PLAYERS", "1") == "1"
SHOW_AREAS = os.environ.get("SHOW_AREAS", "1") == "1"
ORIENT = os.environ.get("ORIENT", "both")
OUT = os.environ.get("OUT_DIR", "site")
PAGES_DIR = os.environ.get("MAP_PAGES_DIR", "")
FX = ORIENT in ("v")
FZ = ORIENT in ("h")

NAMES = {0:'Air',1:'Stone',2:'Gravel',3:'Gravel',4:'Gravel',5:'Dirt',6:'Mud',7:'Farmland',8:'Farmland',9:'Forest floor',10:'Forest floor',11:'Forest floor',12:'Moss',13:'Volcanic',14:'Obsidian',15:'Obsidian',16:'Dry dirt',17:'Desert sand',18:'Beach sand',19:'Sand',20:'Desert stone',21:'Sandstone',22:'Sandstone',23:'Red clay',24:'Rubble',25:'Snow',26:'Ice',27:'Underwater',28:'Corals',29:'Hellstone',30:'Hellstone',31:'Cobble',50:'Coal',51:'Sulfur',52:'Iron',53:'Aluminium',54:'Tungsten',55:'Gold'}
GR = {100:'Grass',105:'Arid grass',110:'Dry grass',115:'Dead grass',120:'Frozen grass',125:'Forest grass',130:'Jungle grass',135:'Sea grass',140:'Seaweed',145:'Arid forest'}
COL = {'Air':(30,30,34),'Stone':(125,125,128),'Gravel':(135,130,120),'Dirt':(120,86,52),'Mud':(88,70,50),'Farmland':(110,78,42),'Forest floor':(66,80,42),'Moss':(60,96,52),'Volcanic':(70,50,48),'Obsidian':(40,30,50),'Dry dirt':(160,128,84),'Desert sand':(222,196,130),'Beach sand':(232,216,160),'Sand':(190,176,130),'Desert stone':(176,120,84),'Sandstone':(198,150,100),'Red clay':(170,84,60),'Rubble':(130,120,110),'Snow':(244,248,252),'Ice':(190,220,240),'Underwater':(70,110,130),'Corals':(220,110,140),'Hellstone':(150,40,30),'Cobble':(140,140,142),'Coal':(48,48,52),'Sulfur':(214,200,60),'Iron':(150,110,100),'Aluminium':(170,180,190),'Tungsten':(100,110,120),'Gold':(230,190,60),'Grass':(88,160,66),'Arid grass':(168,170,86),'Dry grass':(160,150,80),'Dead grass':(140,120,78),'Frozen grass':(170,200,190),'Forest grass':(52,120,52),'Jungle grass':(30,130,60),'Sea grass':(80,150,110),'Seaweed':(50,120,120),'Arid forest':(120,140,70)}
LUT = np.zeros((256, 3), np.float32)
for c in range(256):
    LUT[c] = COL.get(NAMES.get(c) or GR.get((c // 5) * 5) or 'Air', (255, 0, 255))
SEA = 100.01


def http_json(url, headers=None, retries=6, timeout=180):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (401, 403, 404):
                raise
            time.sleep(2 + attempt * 2)
        except Exception as e:  # network hiccup, 429, timeout
            last = e
            time.sleep(2 + attempt * 2)
    raise RuntimeError(f"giving up on {url}: {last}")


def decode(c):
    h = np.frombuffer(base64.b64decode(c["heightsBase64"]), dtype="<f4").reshape(32, 32)
    t = np.frombuffer(base64.b64decode(c["texturesBase64"]), dtype=np.uint8).reshape(32, 32)
    return c["chunkX"], c["chunkZ"], h, t


def load_chunks():
    pages = []
    if PAGES_DIR:
        for f in sorted(glob.glob(os.path.join(PAGES_DIR, "*.json"))):
            pages.append(json.load(open(f)))
    else:
        offset = 0
        while True:
            j = http_json(f"{BASE}/map?limit=100&offset={offset}")
            pages.append(j)
            print(f"map page {len(pages)}: {len(j.get('chunks', []))} chunks", flush=True)
            if not j.get("partial") or not j.get("chunks"):
                break
            offset = j.get("nextOffset", offset + len(j["chunks"]))
            time.sleep(0.3)
    seen = {}
    for p in pages:
        for c in p.get("chunks", []):
            x, z, h, t = decode(c)
            seen[(x, z)] = (h, t)
    return seen


def pick_scale(nx, nz):
    for s in (16, 8, 4, 2, 1):
        if nx * s * nz * s <= 60e6 and max(nx, nz) * s <= 12000:
            return s
    return 1


def render(chunks):
    xs = sorted(k[0] for k in chunks); zs = sorted(k[1] for k in chunks)
    mx, mz = xs[len(xs) // 2], zs[len(zs) // 2]
    cs = {k: v for k, v in chunks.items() if abs(k[0] - mx) <= 300 and abs(k[1] - mz) <= 300}
    x0 = min(k[0] for k in cs); x1 = max(k[0] for k in cs)
    z0 = min(k[1] for k in cs); z1 = max(k[1] for k in cs)
    nx, nz = x1 - x0 + 1, z1 - z0 + 1
    S = pick_scale(nx, nz)
    step = max(1, 32 // S)
    W, H = nx * S, nz * S
    hm = np.full((H, W), np.nan, np.float32)
    tx = np.zeros((H, W), np.uint8)
    for (cx, cz), (h, t) in cs.items():
        sub_h = h[::step, ::step][:S, :S]; sub_t = t[::step, ::step][:S, :S]
        # chunk arrays are indexed [z][x]
        y = (cz - z0) * S; x = (cx - x0) * S
        hm[y:y + S, x:x + S] = sub_h; tx[y:y + S, x:x + S] = sub_t
    if FX: hm = hm[:, ::-1]; tx = tx[:, ::-1]
    if FZ: hm = hm[::-1, :]; tx = tx[::-1, :]
    valid = ~np.isnan(hm)
    hf = np.where(valid, hm, 0).astype(np.float32)
    hp = np.pad(hf, 1, mode="edge"); vp = np.pad(valid, 1, mode="edge")
    def nb(a, dy, dx): return a[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
    hl = np.where(nb(vp, 0, -1), nb(hp, 0, -1), hf); hr = np.where(nb(vp, 0, 1), nb(hp, 0, 1), hf)
    hu = np.where(nb(vp, -1, 0), nb(hp, -1, 0), hf); hd = np.where(nb(vp, 1, 0), nb(hp, 1, 0), hf)
    gx = (hr - hl) / 2; gy = (hd - hu) / 2
    if FX: gx = -gx
    if FZ: gy = -gy
    sh = np.clip(0.8 + (-gx - gy) * 0.45, 0.55, 1.25)
    rgb = LUT[tx] * sh[..., None]
    water = valid & (hf <= SEA)
    dep = np.clip((100 - hf) / 25, 0, 1); k = 1 - 0.55 * dep
    wc = np.stack([0.4 * LUT[tx][..., 0] + 0.6 * 40 * k, 0.4 * LUT[tx][..., 1] + 0.6 * 90 * k, 0.4 * LUT[tx][..., 2] + 0.6 * 170 * k], -1)
    rgb = np.where(water[..., None], wc, rgb)
    rgb[~valid] = (22, 22, 26)
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")
    return img, dict(x0=x0, z0=z0, S=S, W=W, H=H, chunks=len(cs))


def font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf", "arial.ttf"):
        try: return ImageFont.truetype(p, size)
        except Exception: pass
    try: return ImageFont.load_default(size=size)
    except TypeError: return ImageFont.load_default()


def overlay(img, v, areas, players):
    d = ImageDraw.Draw(img, "RGBA")
    S, W, H = v["S"], v["W"], v["H"]
    def px(x):
        a = (x / 32 - v["x0"]) * S; return W - a if FX else a
    def pz(z):
        a = (z / 32 - v["z0"]) * S; return H - a if FZ else a
    f = font(max(12, int(S * 1.4)))
    def label(x, y, text, fill=(255, 255, 255, 255)):
        d.text((x, y), text, font=f, fill=fill, stroke_width=2, stroke_fill=(0, 0, 0, 255))
    for a in areas:
        s, e = a.get("start"), a.get("end")
        if not s or not e: continue
        xa, xb = sorted((px(s["x"]), px(e["x"]))); za, zb = sorted((pz(s["z"]), pz(e["z"])))
        d.rectangle([xa, za, xb, zb], fill=(255, 200, 60, 46), outline=(255, 210, 80, 240), width=max(1, S // 6))
        name = a.get("name")
        if name and xb - xa > S * 3: label(xa + 3, za + 2, name)
    for p in players:
        if p.get("posx") is None or p.get("posz") is None: continue
        x, z, r = px(p["posx"]), pz(p["posz"]), max(5, S * 0.6)
        col = (255, 59, 59, 255) if p.get("connected") else (190, 190, 190, 220)
        d.ellipse([x - r, z - r, x + r, z + r], fill=col, outline=(0, 0, 0, 255), width=2)
        label(x + r + 3, z - r, p.get("name") or "?")


def fetch_optional(route, key):
    if not TOKEN:
        print(f"no LIVE_DATA_TOKEN set; skipping {route}")
        return []
    try:
        j = http_json(f"{BASE}/{route}", headers={"Authorization": f"Bearer {TOKEN}"}, retries=3)
        items = j.get(key, []) if isinstance(j, dict) else j
        print(f"{route}: {len(items)} items")
        return items
    except Exception as e:
        print(f"WARNING: {route} failed ({e}); continuing without it", file=sys.stderr)
        return []


def main():
    chunks = load_chunks()
    if len(chunks) < 10:
        sys.exit(f"only {len(chunks)} chunks received; refusing to publish")
    img, v = render(chunks)
    areas = fetch_optional("world-areas", "areas") if SHOW_AREAS else []
    players = fetch_optional("playerlist", "players") if SHOW_PLAYERS else []
    clean = img.copy()
    overlay(img, v, areas, players)
    os.makedirs(OUT, exist_ok=True)
    img.save(os.path.join(OUT, "map.png"), optimize=True)
    clean.save(os.path.join(OUT, "map-terrain-only.png"), optimize=True)
    info = {"updatedUtc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "chunks": v["chunks"],
            "players": sum(1 for p in players if p.get("connected")), "areas": len(areas), "width": v["W"], "height": v["H"]}
    json.dump(info, open(os.path.join(OUT, "map-info.json"), "w"))
    import shutil
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "viewer.html"), os.path.join(OUT, "index.html"))
    print("built", info)


if __name__ == "__main__":
    main()
