# =============================================================================
# 道路偵測系統
# =============================================================================

# !pip install transformers torch torchvision pillow opencv-python-headless scikit-learn scikit-image matplotlib numpy

import os, cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from PIL import Image, ImageOps
import warnings
warnings.filterwarnings("ignore")

# ── 顏色 ─────────────────────────────────────────────────────
C = {
    "sky":      (50,  160, 200),
    "veg":      (30,  140,  30),
    "sidewalk": (140, 140, 140),
    "asphalt":  (255, 120,  30),
    "gravel":   (160,  40, 200),
}

EPS = 1e-8

# ── Cityscapes 類別 ───────────────────────────────────────────
CS_ROAD     = [0]
CS_SIDEWALK = [1]
CS_VEG      = [8, 9]
CS_SKY      = [10]


# ============================================================
# 共用工具
# ============================================================

def upload_images():
    try:
        from google.colab import files
        print("請上傳圖片（可多選）...")
        up = files.upload()
        paths = list(up.keys())
        print(f"已上傳 {len(paths)} 張：{paths}")
        return paths
    except ImportError:
        exts = ('.jpg', '.jpeg', '.png', '.bmp')
        return sorted(f for f in os.listdir('.') if f.lower().endswith(exts))

def load_img(path):
    pil = ImageOps.exif_transpose(Image.open(path).convert("RGB"))
    return pil, np.array(pil)

def morph_clean(m, k=11):
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kern)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  kern)
    return m

def largest_cc(m_u8):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m_u8, connectivity=8)
    if n <= 1: return m_u8
    best = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    return (labels == best).astype(np.uint8)

def blend(rgb, mask, color, alpha=0.60):
    out = rgb.copy().astype(np.float32)
    out[mask] = out[mask] * (1-alpha) + np.array(color) * alpha
    return out.astype(np.uint8)


# ============================================================
# 核心演算法：SegFormer + monotonic taper + lv_ratio 分類
# ============================================================

def load_segformer():
    print("載入 SegFormer-B2 Cityscapes（首次約 330 MB）...")
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
    import torch
    name  = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
    proc  = AutoImageProcessor.from_pretrained(name)
    model = SegformerForSemanticSegmentation.from_pretrained(name)
    dev   = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(dev)
    print(f"✅ {dev.upper()}")
    return proc, model, dev

def run_segformer(pil, proc, model, dev):
    import torch, torch.nn.functional as F
    inp = {k: v.to(dev) for k, v in proc(images=pil, return_tensors="pt").items()}
    with torch.no_grad():
        logits = model(**inp).logits
    up = F.interpolate(logits, size=pil.size[::-1], mode="bilinear", align_corners=False)
    return up.argmax(1).squeeze().cpu().numpy().astype(np.int32)

# ── monotonic taper（保留自 road_detection.py）─────────────
def monotonic_taper(road_mask, upper_cut, smooth=0.25):
    H, W   = road_mask.shape
    result = np.zeros_like(road_mask)
    prev_hw, prev_ctr = None, W / 2.0

    for row in range(H-1, upper_cut-1, -1):
        cols = np.where(road_mask[row])[0]
        if len(cols) < 8:
            if prev_hw is not None:
                prev_hw = max(4.0, prev_hw * 0.95)
                x0 = max(0,   int(prev_ctr - prev_hw))
                x1 = min(W-1, int(prev_ctr + prev_hw))
                result[row, x0:x1+1] = True
            continue
        seg_hw  = (int(cols[-1]) - int(cols[0])) / 2.0
        seg_ctr = (int(cols[0])  + int(cols[-1])) / 2.0
        if prev_hw is None:
            curr_hw, curr_ctr = seg_hw, seg_ctr
        else:
            curr_hw  = min(seg_hw, prev_hw)
            curr_ctr = (1-smooth)*prev_ctr + smooth*seg_ctr
        x0 = max(0,   int(curr_ctr - curr_hw))
        x1 = min(W-1, int(curr_ctr + curr_hw))
        result[row, x0:x1+1] = True
        prev_hw, prev_ctr = curr_hw, curr_ctr

    return result & road_mask

# ── lv_ratio 紋理分類（保留自 road_detection.py）──────────
LV_THR = 0.876

def local_var_map(g, w):
    mu  = cv2.boxFilter(g, -1, (w, w))
    mu2 = cv2.boxFilter(g*g, -1, (w, w))
    return np.sqrt(np.maximum(mu2 - mu*mu, 0.0))

def classify_surface(rgb, mask):
    if mask.sum() < 200:
        return "unknown", 0.5, {}
    bgr  = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ys, xs = np.where(mask)
    roi  = gray[ys.min():ys.max()+1, xs.min():xs.max()+1]
    rm   = mask[ys.min():ys.max()+1, xs.min():xs.max()+1]

    lv5  = local_var_map(roi,  5)[rm].mean()
    lv15 = local_var_map(roi, 15)[rm].mean()
    lv_r = float(lv5 / (lv15 + 1e-8))

    sx   = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
    sy   = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
    edge = float(np.sqrt(sx**2+sy**2)[rm].mean())
    lap  = float(cv2.Laplacian(roi, cv2.CV_32F)[rm].std())
    mi   = float(gray[mask].mean())

    lv_s   = float(np.clip((lv_r  - 0.750) / 0.200, 0, 1))
    edge_s = float(np.clip((edge  -  60.0) / 200.0, 0, 1))
    lap_s  = float(np.clip((lap   -  30.0) / 100.0, 0, 1))
    int_s  = float(np.clip((mi    -  90.0) / 100.0, 0, 1))
    score  = 0.35*lv_s + 0.25*edge_s + 0.25*lap_s + 0.15*int_s

    if lv_r >= LV_THR:
        label, score = "gravel",  max(score, 0.50)
    else:
        label, score = "asphalt", min(score, 0.49)

    return label, float(score), {"lv_ratio": lv_r, "edge": edge, "lap": lap, "mean_i": mi}


def run_detection(paths):
    proc, model, dev = load_segformer()
    results = []

    for path in paths:
        print(f"\n📡 處理影像: {os.path.basename(path)}")
        pil, rgb = load_img(path)
        H, W     = rgb.shape[:2]
        upper    = int(H * 0.35)

        seg = run_segformer(pil, proc, model, dev)

        # ── 各類別遮罩 ─────────────────────────────────────
        sky_m  = np.isin(seg, CS_SKY)
        veg_m  = np.isin(seg, CS_VEG)
        side_m = np.isin(seg, CS_SIDEWALK)

        road_u8 = np.isin(seg, CS_ROAD).astype(np.uint8)
        road_u8[:upper] = 0
        road_u8 = morph_clean(road_u8, k=11)
        road_u8 = largest_cc(road_u8)
        road_m  = road_u8.astype(bool)

        # ── 視覺遮罩 (monotonic taper) ─────────────────────
        road_vis = monotonic_taper(road_m, upper)

        # ── 紋理分類（用完整 road_m，不是 vis）─────────────
        label, score, feats = classify_surface(rgb, road_m)
        conf = score if label=="gravel" else 1-score
        print(f"  lv_ratio={feats.get('lv_ratio',0):.4f}  →  {label.upper()} {conf:.0%}")

        # ── 彩色疊加 ───────────────────────────────────────
        out = rgb.copy()
        for m, col in [(sky_m,  C["sky"]),
                       (veg_m,  C["veg"]),
                       (side_m, C["sidewalk"]),
                       (road_vis, C["gravel"] if label=="gravel" else C["asphalt"])]:
            out = blend(out, m, col, alpha=0.55)

        results.append({"name": os.path.basename(path),
                        "orig": rgb, "result": out,
                        "label": label, "score": score, "feats": feats,
                        "road_vis": road_vis, "sky_m": sky_m, "veg_m": veg_m})

    # ── 視覺化 ─────────────────────────────────────────────
    n = len(results)
    fig = plt.figure(figsize=(14*n, 7), facecolor="#12121f")
    gs  = gridspec.GridSpec(1, 2*n, wspace=0.04)

    for i, r in enumerate(results):
        for j, (img, ttl) in enumerate([(r["orig"],   f"原圖 {r['name']}"),
                                         (r["result"], f"{r['label'].upper()} {(r['score'] if r['label']=='gravel' else 1-r['score']):.0%}")]):
            ax = fig.add_subplot(gs[i*2+j])
            ax.imshow(img)
            ax.set_title(ttl, color="#ffaa44" if "ASPHALT" in ttl else
                              "#cc66ff" if "GRAVEL" in ttl else "white",
                         fontsize=12)
            ax.axis("off")

    patches = [mpatches.Patch(color=[c/255 for c in C[k]], label=k.capitalize())
               for k in ("sky","veg","sidewalk","asphalt","gravel")]
    fig.legend(handles=patches, loc="lower center", ncol=5,
               facecolor="#1e1e2e", labelcolor="white", fontsize=11,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Road Surface Detection – SegFormer Cityscapes + Texture",
                 color="white", fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig("out_segformer.png", dpi=130, bbox_inches="tight", facecolor="#12121f")
    plt.show()
    print("✅ 已儲存分析結果至 out_segformer.png")
    return results


# ============================================================
# 主程式
# ============================================================

def main():
    print("="*60)
    print("🛣️  道路影像偵測系統")
    print("="*60)

    paths = upload_images()
    if not paths:
        print("⚠️  沒有圖片")
        return

    run_detection(paths)
    print("\n✅ 處理完成")

if __name__ == "__main__":
    main()