# Fruit 3D Shape Completion with Diffusion (pcdiff)

Code from my Master's thesis: diffusion-based 3D shape completion of fruits
(sweet pepper) from partial RGB-D observations.

Paper: [Fruits 3D Reconstruction and RGB Prediction for Agricultural Robotics with Diffusion Models](https://www.researchgate.net/publication/396007635_Fruits_3D_Reconstruction_and_RGB_Prediction_for_Agricultural_Robotics_with_Diffusion_Models)

The diffusion pipeline is adapted from
[LiDiff](https://github.com/PRBonn/LiDiff) (Nunes et al., CVPR 2024,
*Scaling Diffusion Models to Real-World 3D LiDAR Scene Completion*).

## `ipb_loaders` replacement

The original code depended on `ipb_loaders`, a dataloader library internal to
the IPB lab (University of Bonn) that is not publicly available.
`pcdiff/datasets/igg_fruit.py` re-implements the `IGGFruit` dataset it provided,
so this repo no longer needs `ipb_loaders`.

`IGGFruit(data_source, split='train', precomputed_augmentation=None, fuse_frames=True)`
reads the public shape completion challenge layout (see [Data](#data)) and
returns samples of the form:

```python
{
    'points': partial,           # (N, 6) XYZRGB, fused from the RGB-D frames
    'extra': {
        'gt_points': gt,         # (M, 6) XYZ + RGB transferred from the partial scan, -1 = no color
        'fruit_id': 'lab1',
    },
}
```

For the test split (no ground truth), `gt_points` falls back to the partial
observation. This is a re-implementation from how the code used the original
loader, so details (e.g. preprocessing of the original IGG fruit data) may
differ from `ipb_loaders`.

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

The completion pipeline:

```bash
python3 tools/diff_completion_pipeline.py --diff CHECKPOINT_PATH -T DENOISING_STEPS -s CONDITIONING_WEIGHT --data DATA_ROOT
```

Trained checkpoints are not included in this repo.

> [!NOTE]
> `diff_completion_pipeline.py` still builds the older 3-channel (XYZ-only)
> networks, while the final model in `models/models.py` uses 6 channels
> (XYZRGB). The pipeline needs updating before it can load current checkpoints.

## Citation

If you use this code, please cite:

```bibtex
@misc{yap_fruits3d,
  title        = {Fruits 3D Reconstruction and RGB Prediction for Agricultural Robotics with Diffusion Models},
  author       = {Yap, Paulie and others},
  howpublished = {\url{https://www.researchgate.net/publication/396007635_Fruits_3D_Reconstruction_and_RGB_Prediction_for_Agricultural_Robotics_with_Diffusion_Models}},
}
```

## License

MIT, see [LICENSE](LICENSE). Parts of the code are adapted from
[LiDiff](https://github.com/PRBonn/LiDiff) (MIT, (c) 2024 Photogrammetry & Robotics Bonn),
see [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES).
