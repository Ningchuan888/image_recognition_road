# =============================================================================
# 道路表面偵測系統
# SegFormer (Cityscapes) + lv_ratio 紋理分類
# =============================================================================

import os, cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageOps
import warnings
warnings.filterwarnings("ignore")

# ── 道路顏色 ──────────────────────────────────────────────────
ROAD_COLOR = (255, 120, 30)   # 橘色（柏油路）

CS_ROAD = [0]    # Cityscapes class 0 = road


# ============================================================
# 工具
# ============================================================

def find_images():
    import sys
    if len(sys.argv) > 1:
        return sys.argv[1:]
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


# ============================================================
# 語意分割
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


# ============================================================
# 道路遮罩後處理
# ============================================================

def monotonic_taper(road_mask, upper_cut, smooth=0.25):
    """由下往上掃，強制寬度只縮不擴，模擬透視消失點"""
    H, W   = road_mask.shape
    result = np.zeros_like(road_mask)
    prev_hw, prev_ctr = None, W / 2.0

    for row in range(H - 1, upper_cut - 1, -1):
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
            curr_ctr = (1 - smooth) * prev_ctr + smooth * seg_ctr
        x0 = max(0,   int(curr_ctr - curr_hw))
        x1 = min(W-1, int(curr_ctr + curr_hw))
        result[row, x0:x1+1] = True
        prev_hw, prev_ctr = curr_hw, curr_ctr

    return result & road_mask




# ============================================================
# 主流程
# ============================================================

def process(path, proc, model, dev):
    print(f"\n🔍 {os.path.basename(path)}")
    pil, rgb = load_img(path)
    H, W     = rgb.shape[:2]
    upper    = int(H * 0.35)

    # 語意分割
    seg = run_segformer(pil, proc, model, dev)

    # 道路遮罩
    road_u8 = np.isin(seg, CS_ROAD).astype(np.uint8)
    road_u8[:upper] = 0
    road_u8 = morph_clean(road_u8, k=11)
    road_u8 = largest_cc(road_u8)
    road_m  = road_u8.astype(bool)

    # 視覺遮罩（單調收窄）
    road_vis = monotonic_taper(road_m, upper)

    # 道路區域疊色（柏油路，橘色）
    out   = rgb.copy().astype(np.float32)
    alpha = 0.55
    out[road_vis] = out[road_vis] * (1 - alpha) + np.array(ROAD_COLOR, dtype=np.float32) * alpha
    out = out.astype(np.uint8)

    # 儲存：result_原檔名
    stem     = os.path.splitext(os.path.basename(path))[0]
    out_name = f"result_{stem}.png"
    out_bgr  = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_name, out_bgr)
    print(f"  ✅ 儲存：{out_name}")

    # 顯示（原圖 + 結果 並排）
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(rgb);  axes[0].set_title("原圖",  fontsize=13); axes[0].axis("off")
    axes[1].imshow(out);  axes[1].set_title("ASPHALT", fontsize=13)
    axes[1].axis("off")

    patches = [mpatches.Patch(color=[c/255 for c in ROAD_COLOR], label="Asphalt")]
    fig.legend(handles=patches, loc="lower center", ncol=1, fontsize=11,
               bbox_to_anchor=(0.5, 0.0))
    plt.tight_layout()
    plt.savefig(out_name, dpi=130, bbox_inches="tight")
    plt.show()


# ============================================================
# 主程式
# ============================================================

def main():
    print("=" * 50)
    print("🛣️  道路表面偵測系統")
    print("=" * 50)

    paths = find_images()
    if not paths:
        print("⚠️  目錄內未找到圖片")
        return
    print(f"找到圖片：{paths}")

    proc, model, dev = load_segformer()
    for p in paths:
        process(p, proc, model, dev)

    print("\n✅ 完成")

if __name__ == "__main__":
    main()