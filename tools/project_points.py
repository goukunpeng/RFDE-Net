"""Project TJ4DRadSet LiDAR or 4D radar point clouds onto the camera image.
"""
import sys
import argparse
from PIL import Image
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rfde_data.calib import read_calib
from rfde_data.projection import read_points, save_depth_png, save_mask_png, sparse_depth_map


SOURCES = {
    "radar": {"points": "radar", "calib": "radar_calib", "prefix": "radar"},
    "lidar": {"points": "velodyne", "calib": "calib", "prefix": "lidar"},
}


def read_ids(path):
    return [line.strip().split(".")[0] for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sensor", choices=sorted(SOURCES), required=True)
    parser.add_argument("--tj4d-root", type=Path, required=True,
                        help="TJ4DRadSet folder containing image_2, radar and radar_calib")
    parser.add_argument("--lidar-root", type=Path, default=None,
                        help="TJ4DRadSet_LiDAR/training, containing velodyne and calib")
    parser.add_argument("--calib-subdir", default=None,
                        help="override the calibration sub-directory (default: radar_calib / calib)")
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src = SOURCES[args.sensor]
    if args.sensor == "lidar" and args.lidar_root is None:
        parser.error("--lidar-root is required for --sensor lidar")
    root = args.tj4d_root if args.sensor == "radar" else args.lidar_root
    calib_dir = root / (args.calib_subdir or src["calib"])
    depth_dir = args.out_root / f"{src['prefix']}_depth"
    mask_dir = args.out_root / f"{src['prefix']}_mask"
    depth_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    ids = read_ids(args.split_file)
    for i, sid in enumerate(ids):
        depth_path, mask_path = depth_dir / f"{sid}.png", mask_dir / f"{sid}.png"
        if depth_path.exists() and mask_path.exists() and not args.overwrite:
            continue
        with Image.open(args.tj4d_root / "image_2" / f"{sid}.png") as image:
            shape = (image.height, image.width)
        calib = read_calib(calib_dir / f"{sid}.txt")
        points = read_points(root / src["points"] / f"{sid}.bin", args.sensor)
        depth, mask = sparse_depth_map(points, calib, shape)
        save_depth_png(depth, depth_path)
        save_mask_png(mask, mask_path)
        if (i + 1) % 200 == 0 or i + 1 == len(ids):
            print(f"{args.sensor}: {i + 1}/{len(ids)}", flush=True)


if __name__ == "__main__":
    main()
