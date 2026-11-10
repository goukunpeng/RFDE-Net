# RFDE-Net

Data splits and data-processing code for the paper
**Depth-Enhanced Freespace Detection via 4D Radar-Camera Fusion** (RFDE-Net).

This repository currently contains

| Path | Content |
|---|---|
| `splits/` | TJ4DRadSet training / validation / test frame lists |
| `tools/generate_freespace_labels.py` | Freespace labels: SAM 3 coarse masks and their refinement |
| `tools/project_points.py` | 4D radar and LiDAR projection onto the image (radar input R_p, sparse LiDAR depth) |
| `tools/complete_depth_bpnet.py` | LiDAR-completed dense depth targets with BP-Net |
| `rfde_data/bev.py` | Projection of a freespace mask and a dense depth map to BEV (predictions and pseudo-BEV references) |

The network code will be released soon.

## Data

We use [TJ4DRadSet](https://github.com/TJRadarLab/TJ4DRadSet). The dataset is
provided by its authors for non-commercial research under their license
agreement; please apply to them for the 4D radar data, camera images and LiDAR
point clouds. This repository does not contain any TJ4DRadSet data. Following
the license agreement, we do not distribute the annotations derived from it
(freespace labels, dense depth targets and pseudo-BEV references). The dense
depth targets and pseudo-BEV references can be rebuilt with the code here.

## Splits

`splits/{train,val,test}.txt` list TJ4DRadSet frame IDs (one per line; the
first two digits are the sequence index). The splits are sequence-disjoint.

| Split | Frames | Sequences |
|---|---|---|
| train | 5,717 | 02 03 04 06 07 08 09 11 12 13 14 15 16 17 18 19 21 22 24 25 26 27 28 29 32 33 35 36 38 39 40 42 43 |
| val | 408 | 05 23 |
| test | 1,632 | 00 01 10 20 30 31 34 37 41 |

`train.txt` contains all frames of the 33 sequences of the official training
split, including the 11 frames without object annotations that are not listed
in the official `ImageSets/train.txt`. `val.txt` and `test.txt` together are the
official test split (`ImageSets/test.txt`).

## Processing

The original images are 1280 x 960. All targets are defined on the 1280 x 640
crop obtained by removing the top 320 rows; the principal point of the
calibration is shifted accordingly (cy - 320). The network input is this crop
resized to 1024 x 512.

```bash
pip install -r requirements.txt

# 0. Freespace labels (1280 x 640): SAM 3 coarse masks, then refinement
python tools/generate_freespace_labels.py infer \
    --image-dir /data/TJ4DRadSet/TJ4DRadSet_Public1/training/image_2 \
    --split-file splits/test.txt --checkpoint /path/to/sam3.pt \
    --out-dir /data/rfde/test/coarse
python tools/generate_freespace_labels.py refine \
    --coarse-dir /data/rfde/test/coarse/mask \
    --split-file splits/test.txt --out-dir /data/rfde/test/gt

# 1. Radar input R_p: projected radar depth and valid-observation mask (1280 x 960)
python tools/project_points.py --sensor radar \
    --tj4d-root /data/TJ4DRadSet/TJ4DRadSet_Public1/training \
    --split-file splits/test.txt --out-root /data/rfde/test

# 2. Sparse LiDAR depth (1280 x 960)
python tools/project_points.py --sensor lidar \
    --tj4d-root /data/TJ4DRadSet/TJ4DRadSet_Public1/training \
    --lidar-root /data/TJ4DRadSet/TJ4DRadSet_LiDAR/training \
    --split-file splits/test.txt --out-root /data/rfde/test

# 3. LiDAR-completed dense depth target (1280 x 640)
python tools/complete_depth_bpnet.py --bpnet-root /path/to/BP-Net \
    --checkpoint /path/to/BP-Net/checkpoints/BP_KITTI/result_ema.pth \
    --tj4d-root /data/TJ4DRadSet/TJ4DRadSet_Public1/training \
    --lidar-calib-dir /data/TJ4DRadSet/TJ4DRadSet_LiDAR/training/calib \
    --sparse-dir /data/rfde/test/lidar_depth \
    --split-file splits/test.txt --out-dir /data/rfde/test/depth_target
```

* **Freespace labels.** [SAM 3](https://github.com/facebookresearch/sam3) is
  prompted with the text phrase "road surface" on the cropped image, and all
  instances with a score above 0.5 are merged into a coarse mask. The
  refinement applies morphological opening (7 x 7) and closing (21 x 21),
  fills short horizontal gaps left by lane marks, removes components smaller
  than 100 pixels and smooths the boundary with a 9 x 9 opening. All valid
  components are kept; only for frames whose coarse mask contains
  disconnected false-positive regions (listed with `--largest-ids` after
  inspecting the overlays written by `infer`) the largest component is kept.
  Install SAM 3 and download its checkpoint as described in its repository.
* **Projection.** Points in front of the sensor are projected with
  `P2 @ R0_rect @ Tr_velo_to_cam`; when several points fall into the same
  pixel, the nearest one (smallest camera depth z) is kept.
* **Radar input.** Radar points are projected with the radar calibration
  (`radar_calib`). The network uses the two channels
  R_p = [(D_r - d_min) / (d_max - d_min) * M_r, M_r] with d_min = 1 m and
  d_max = 100 m.
* **Dense depth targets.** BP-Net ([Tang et al., CVPR 2024](https://github.com/kakaxi314/BP-Net))
  is run with its released KITTI checkpoint, without fine-tuning, on the
  cropped RGB image and the cropped sparse LiDAR depth.

## BEV projection

`rfde_data/bev.py` back-projects every freespace pixel with its depth,
transforms the point into the TJ4DRadSet LiDAR frame (x forward, y left,
z up) with the inverse of the LiDAR-camera calibration, discards the height
and rasterises the points into a 0.2 m grid. Each point is joined to the BEV
projection of the camera centre with a one-cell-wide line, and all cells on
the line are marked as freespace. Predictions and pseudo-BEV references
(refined freespace label + dense depth target) use the same procedure.

| Setting | Value |
|---|---|
| Frame | TJ4DRadSet LiDAR frame |
| Cell size | 0.2 m |
| Lateral range | [-10, 10) m |
| Longitudinal range | [1, 60) m |
| Pixels | all pixels of the 1280 x 640 mask |

## License
The code in this repository is released under the MIT License. BP-Net and
TJ4DRadSet are subject to their own licenses.
