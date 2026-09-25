# Fruit 3D Shape Completion with Diffusion (pcdiff)

Code from my Master's thesis: diffusion-based 3D shape completion of fruits
(sweet pepper) from partial RGB-D observations.

The diffusion pipeline is adapted from
[LiDiff](https://github.com/PRBonn/LiDiff) (Nunes et al., CVPR 2024,
*Scaling Diffusion Models to Real-World 3D LiDAR Scene Completion*).

> [!WARNING]
> **Status: not runnable out of the box.** The code depends on the
> university-lab-internal package `ipb_loaders`, which is not publicly
> available. See [Missing dependency: `ipb_loaders`](#missing-dependency-ipb_loaders).

## Missing dependency: `ipb_loaders`

`ipb_loaders` was a dataloader library developed internally at the IPB lab
(University of Bonn). It is **not** included in this repository and is not on
PyPI.

Where it is used:

| File | Usage |
| --- | --- |
| `pcdiff/tools/diff_completion_pipeline.py` | `from ipb_loaders.pointcloud.igg_fruit import IGGFruit` (required) |
| `pcdiff/train.py`, `pcdiff/ckpt_train.py` | `IGGFruit` imports, commented out (not required) |

Training does **not** need `ipb_loaders`: `train.py` uses
`pcdiff/dataloader.py` and `ckpt_train.py` uses
`pcdiff/competition_tools/dataloader.py` (both `ShapeCompletionDataset`).

**TODO:** re-implement the `IGGFruit` dataset loader (point cloud / RGB-D
loading for the IGG fruit dataset) inside this repo so
`diff_completion_pipeline.py` works without the lab package.

If you have access to `ipb_loaders`, one fix is needed for newer PyTorch
versions. In `ipb_loaders/ipb_base.py`, change

```python
from torch.utils.data import Dataset, default_collate
```

to

```python
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
```

## Installation

Install torch with CUDA 11.1:

```bash
pip3 install torch==1.9.0+cu111 torchvision==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```

Install MinkowskiEngine from source:

```bash
pip3 install -U git+https://github.com/NVIDIA/MinkowskiEngine --no-deps
```

Install the remaining packages and this repo:

```bash
pip3 install -r requirements.txt
pip3 install -U -e .
```

`open3d`, `pytorch_lightning` (1.5.x) and `pytorch3d` are also needed
(they are commented out in `requirements.txt`).

## Data

Set the dataset root in `pcdiff/config/config.yaml`:

```yaml
data:
    fruit_dir: './data/shape_completion_challenge'
```

The folder is expected to contain `train/` and `test/` splits, in the layout
used by the shape completion challenge dataset. Datasets are not included in
this repo.

## Training

Configurations are in `pcdiff/config/config.yaml`. From `pcdiff/`:

```bash
python3 train.py
```

Test a trained checkpoint:

```bash
python3 train.py -w experiments/Plants_ClassFree/default/version_0/checkpoints/last.ckpt --test
```

## Inference

The completion pipeline (requires `ipb_loaders`, see above):

```bash
python3 tools/diff_completion_pipeline.py --diff CHECKPOINT_PATH -T DENOISING_STEPS -s CONDITIONING_WEIGHT
```

Trained checkpoints are not included in this repo.

## License

MIT, see [LICENSE](LICENSE). Parts of the code are adapted from
[LiDiff](https://github.com/PRBonn/LiDiff) (MIT, (c) 2024 Photogrammetry & Robotics Bonn),
see [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES).
