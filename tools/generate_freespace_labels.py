"""Generate image-plane freespace labels with SAM 3 and a fixed refinement.
Two steps, run one after the other:
  infer   SAM 3 is prompted with a text phrase on the cropped camera image
          (top 320 rows removed, 1280 x 640). All returned instances above the
          score threshold are merged into a coarse mask M_coarse (0/255 PNG).
          An overlay is written for every frame for visual inspection.
  refine  Morphological opening, morphological closing, filling of short
          horizontal gaps (lane marks), small-component removal and boundary
          smoothing. Connected components are kept by default; frames whose
          coarse mask contains disconnected false-positive regions can be
          listed in --largest-ids, and only the largest component is kept for
          them.

SAM 3 (https://github.com/facebookresearch/sam3) must be installed and its
checkpoint downloaded following the instructions of that repository.
"""
import json
import cv2
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

CROP_TOP = 320


def read_ids(path):
    return [line.strip().split(".")[0] for line in Path(path).read_text().splitlines() if line.strip()]


def read_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return np.where(mask > 127, 255, 0).astype(np.uint8)


def write_mask(path, mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.where(mask > 0, 255, 0).astype(np.uint8))


def overlay(image_rgb, mask, alpha=0.4):
    image = np.asarray(image_rgb, dtype=np.uint8).copy()
    color = image.copy()
    color[mask > 0] = (0, 255, 0)
    blended = cv2.addWeighted(image, 1.0 - alpha, color, alpha, 0.0)
    return cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)


# --------------------------------------------------------------------------
# Step 1: SAM 3 coarse masks
# --------------------------------------------------------------------------

def build_processor(checkpoint, device, score_threshold):
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(
        device=device,
        checkpoint_path=str(checkpoint) if checkpoint else None,
        load_from_HF=checkpoint is None,
    )
    return Sam3Processor(model, device=device, confidence_threshold=score_threshold)


def coarse_mask(processor, image, prompt, mask_threshold):
    """Union of all SAM 3 instances for ``prompt`` (uint8, 0/255)."""
    state = processor.set_image(image)
    state = processor.set_text_prompt(state=state, prompt=prompt)
    masks = state["masks_logits"].detach().float().cpu().numpy()
    scores = state["scores"].detach().float().cpu().numpy().tolist()
    if masks.ndim == 4:
        masks = masks[:, 0]
    if masks.shape[0] == 0:
        return np.zeros((image.height, image.width), np.uint8), scores
    union = np.any(masks > mask_threshold, axis=0)
    return union.astype(np.uint8) * 255, scores


def run_infer(args):
    import torch

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = build_processor(args.checkpoint, device, args.score_threshold)
    mask_dir, overlay_dir = args.out_dir / "mask", args.out_dir / "overlay"
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    records = {}

    ids = read_ids(args.split_file)
    for i, sid in enumerate(ids):
        mask_path = mask_dir / f"{sid}.png"
        if mask_path.exists() and not args.overwrite:
            continue
        with Image.open(args.image_dir / f"{sid}.png") as opened:
            image = opened.convert("RGB")
        image = image.crop((0, args.crop_top, image.width, image.height))
        with torch.inference_mode():
            mask, scores = coarse_mask(processor, image, args.prompt, args.mask_threshold)
        write_mask(mask_path, mask)
        cv2.imwrite(str(overlay_dir / f"{sid}.png"), overlay(image, mask))
        records[sid] = {"instances": len(scores), "scores": [round(float(s), 4) for s in scores]}
        if (i + 1) % 100 == 0 or i + 1 == len(ids):
            print(f"infer: {i + 1}/{len(ids)}", flush=True)

    record_path = args.out_dir / "instances.json"
    if record_path.exists() and not args.overwrite:
        old = json.loads(record_path.read_text())
        old.update(records)
        records = old
    record_path.write_text(json.dumps(records, indent=1))


# --------------------------------------------------------------------------
# Step 2: refinement
# --------------------------------------------------------------------------

def ellipse(size):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def largest_component(mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return np.zeros_like(mask)
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == keep, 255, 0).astype(np.uint8)


def remove_small_components(mask, min_area):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def fill_horizontal_gaps(mask, max_gap, start_row):
    result = mask.copy()
    width = mask.shape[1]
    x = np.arange(width)
    for y in range(start_row, mask.shape[0]):
        row = result[y] > 0
        if not row.any():
            continue
        left = np.maximum.accumulate(np.where(row, x, -width))
        right = np.minimum.accumulate(np.where(row, x, 2 * width)[::-1])[::-1]
        gap = (~row) & (left >= 0) & (right < width) & (x - left <= max_gap) & (right - x <= max_gap)
        result[y, gap] = 255
    return result


def refine_mask(mask, largest=False, open_kernel=7, close_kernel=21, smooth_kernel=9,
                min_area=100, max_gap=25, gap_start_ratio=0.5):
    """Refined freespace label (uint8, 0/255) from a SAM 3 coarse mask."""
    mask = np.where(mask > 127, 255, 0).astype(np.uint8)
    if not mask.any():
        return mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ellipse(open_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ellipse(close_kernel))
    if largest:
        mask = largest_component(mask)
    mask = remove_small_components(mask, min_area)
    mask = fill_horizontal_gaps(mask, max_gap, int(round(mask.shape[0] * gap_start_ratio)))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, ellipse(smooth_kernel))


def run_refine(args):
    largest_ids = set(read_ids(args.largest_ids)) if args.largest_ids else set()
    ids = read_ids(args.split_file)
    for i, sid in enumerate(ids):
        out_path = args.out_dir / f"{sid}.png"
        if out_path.exists() and not args.overwrite:
            continue
        refined = refine_mask(
            read_mask(args.coarse_dir / f"{sid}.png"),
            largest=sid in largest_ids,
            open_kernel=args.open_kernel,
            close_kernel=args.close_kernel,
            smooth_kernel=args.smooth_kernel,
            min_area=args.min_area,
            max_gap=args.max_gap,
            gap_start_ratio=args.gap_start_ratio,
        )
        write_mask(out_path, refined)
        if (i + 1) % 500 == 0 or i + 1 == len(ids):
            print(f"refine: {i + 1}/{len(ids)}", flush=True)


def odd(value):
    value = int(value)
    if value <= 0 or value % 2 == 0:
        raise argparse.ArgumentTypeError("kernel size must be a positive odd integer")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    infer = sub.add_parser("infer", help="SAM 3 coarse masks")
    infer.add_argument("--image-dir", type=Path, required=True, help="TJ4DRadSet image_2 folder")
    infer.add_argument("--split-file", type=Path, required=True)
    infer.add_argument("--out-dir", type=Path, required=True)
    infer.add_argument("--checkpoint", type=Path, default=None,
                       help="SAM 3 checkpoint; if omitted, it is loaded from Hugging Face")
    infer.add_argument("--prompt", default="road surface")
    infer.add_argument("--score-threshold", type=float, default=0.5)
    infer.add_argument("--mask-threshold", type=float, default=0.5)
    infer.add_argument("--crop-top", type=int, default=CROP_TOP)
    infer.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    infer.add_argument("--overwrite", action="store_true")
    infer.set_defaults(func=run_infer)

    refine = sub.add_parser("refine", help="refinement of the coarse masks")
    refine.add_argument("--coarse-dir", type=Path, required=True)
    refine.add_argument("--split-file", type=Path, required=True)
    refine.add_argument("--out-dir", type=Path, required=True)
    refine.add_argument("--largest-ids", type=Path, default=None,
                        help="frame IDs whose coarse masks contain disconnected false positives")
    refine.add_argument("--open-kernel", type=odd, default=7)
    refine.add_argument("--close-kernel", type=odd, default=21)
    refine.add_argument("--smooth-kernel", type=odd, default=9)
    refine.add_argument("--min-area", type=int, default=100)
    refine.add_argument("--max-gap", type=int, default=25)
    refine.add_argument("--gap-start-ratio", type=float, default=0.5)
    refine.add_argument("--overwrite", action="store_true")
    refine.set_defaults(func=run_refine)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
