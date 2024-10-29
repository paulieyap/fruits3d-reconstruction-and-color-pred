# Pheno Diffusion


To install torch with a newer CUDA version:
`pip3 install torch==1.9.0+cu111 torchvision==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html`

Then, to install MinkowskiEngine from the latest source:
`pip3 install -U git+https://github.com/NVIDIA/MinkowskiEngine --no-deps`

For the dataloader: 
Change the file ipbloaders/ipb_loaders/ipb_base.py  from

`from torch.utils.data import Dataset, default_collate`

to:

`from torch.utils.data import Dataset`
`from torch.utils.data.dataloader import default_collate`

Install python packages:
`pip3 install -r requirements.txt`

To setup the code run the following command on the code main directory:

`pip3 install -U -e .`

# Diffusion Shape Completion Pipeline

For training the diffusion model, the configurations are defined in `config/config.yaml`, and the training can be started with:

`python3 train.py`

For running the scene completion inference we provide a pipeline where both the diffusion and refinement network are loaded and used to complete the scene from an input scan. You can run the pipeline with the command:

`python3 tools/diff_completion_pipeline.py --diff CHECKPOINT_PATH -T DENOISING_STEPS -s CONDITIONING_WEIGHT`

Upon acceptance of the paper, we will release the noise predictor and refinement network together with the code.
