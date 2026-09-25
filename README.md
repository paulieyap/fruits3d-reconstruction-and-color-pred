# Fruit 3D Shape Completion with DDPMS

Code from my Master's thesis: diffusion-based 3D shape completion of fruits (sweet pepper) from partial RGB-D observations.

Paper: [Fruits 3D Reconstruction and RGB Prediction for Agricultural Robotics with Diffusion Models](https://www.researchgate.net/publication/396007635_Fruits_3D_Reconstruction_and_RGB_Prediction_for_Agricultural_Robotics_with_Diffusion_Models)

The diffusion pipeline is adapted from  [LiDiff](https://github.com/PRBonn/LiDiff) (Nunes et al., CVPR 2024,
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

### Docker (recommended)

The Docker image pins the full environment used for the thesis (Python 3.8,
CUDA 11.1, torch 1.9.0, MinkowskiEngine 0.5.4, PyTorch Lightning 1.5.10), so
nothing has to match your local CUDA toolkit, compiler or Python version.

Requirements on the host:

- an NVIDIA GPU (the code runs on CUDA only) with driver >= 455
- [Docker](https://docs.docker.com/engine/install/) and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

MinkowskiEngine is compiled for compute capability 6.0-8.6 plus PTX, and the
image ships the CUDA 11.8 cuSPARSE library (CUDA 11.1's version fails on Ada
GPUs), so it runs on GTX 10xx through RTX 30xx/40xx. Tested on an RTX 2000 Ada.

Build the image (takes ~20-30 min, mostly compiling MinkowskiEngine):

```bash
docker build -t pcdiff:cu111 .
```

Start a container with the repo and your dataset mounted:

```bash
docker run --gpus all -it --rm --shm-size=8g \
    --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$(pwd)":/workspace \
    -v /path/to/shape_completion_challenge:/workspace/pcdiff/data/shape_completion_challenge \
    pcdiff:cu111
```

The container starts in `/workspace/pcdiff`, so the commands in
[Training](#training) and [Inference](#inference) work as they are. Outputs
(`experiments/`, `results/`) are written into the mounted repo and owned by
your user (`--user`). `--vis` needs a display and does not work in the
container as-is.

### Manual installation

Only if you can't use Docker. You need Python 3.8 and a CUDA 11.1 toolkit
matching the torch build:

```bash
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install numpy==1.23.5
pip3 install -U git+https://github.com/NVIDIA/MinkowskiEngine@v0.5.4 --no-deps
pip3 install -r requirements.txt
pip3 install -U -e .
```

## Data

Set the dataset root in `pcdiff/config/config.yaml` (the default matches the
Docker mount above):

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

Complete every fruit of a split with a trained checkpoint (from `pcdiff/`):

```bash
python3 tools/diff_completion_pipeline.py --diff CHECKPOINT_PATH -T DENOISING_STEPS -s CONDITIONING_WEIGHT --data DATA_ROOT
```

The completed XYZRGB point clouds are written to
`results/<checkpoint>_T<steps>_s<weight>/diff/<fruit_id>.ply`. Use `--split`
to pick a split (default `test`) and `--vis` to view each result. The
checkpoint folder must contain the `hparams.yaml` written during training.
Trained checkpoints are not included in this repo.

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
