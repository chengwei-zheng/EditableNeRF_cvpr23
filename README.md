# EditableNeRF: Editing Topologically Varying Neural Radiance Fields by Key Points

This is the code for "HyperNeRF: A Higher-Dimensional Representation for Topologically Varying Neural Radiance Fields".

* [Project Page](https://hypernerf.github.io)
* [Paper](https://arxiv.org/abs/2106.13228)
* [Video](https://www.youtube.com/watch?v=qzgdE_ghkaI)

This codebase implements HyperNeRF using [JAX](https://github.com/google/jax),
building on [JaxNeRF](https://github.com/google-research/google-research/tree/master/jaxnerf).

## Overview

- Train [Hypernerf](https://github.com/google/hypernerf) and obtain rendering results.
- Run `data_process/k_points_detect.py` to detect key points.
- Run `data_process/k_points_init_RAFT.py` based on [RAFT](https://github.com/princeton-vl/RAFT) to initialize key points in 2d.
- Run `data_process/k_points_init_3d.py` to initialize key points in 3d.
- Run `train.py` for EditableNeRF training.
- Run `eval.py` for EditableNeRF evaluation and rendering.


## Train Hypernerf

Before running our method, you should run [Hypernerf](https://github.com/google/hypernerf) first, and many parts of our code rely on the virtual environments and the results of Hypernerf.

## Key Point Detection


## Setup
The code can be run under any environment with Python 3.8 and above.
(It may run with lower versions, but we have not tested it).

We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and setting up an environment:

    python k_points_init_RAFT.py \
        --model=models/raft-things.pth \
        --path=input_img \
        --kp_file=kp_init \
        --skip_prop=50 \

Next, install the required packages:

    pip install -r requirements.txt

Install the appropriate JAX distribution for your environment by  [following the instructions here](https://github.com/google/jax#installation). For example:

    # For CUDA version 11.1
    pip install --upgrade "jax[cuda111]" -f https://storage.googleapis.com/jax-releases/jax_releases.html


## Training
After preparing a dataset, you can train a Nerfie by running:

    export DATASET_PATH=/path/to/dataset
    export EXPERIMENT_PATH=/path/to/save/experiment/to
    python train.py \
        --base_folder $EXPERIMENT_PATH \
        --gin_bindings="data_dir='$DATASET_PATH'" \
        --gin_configs configs/test_local.gin

To plot telemetry to Tensorboard and render checkpoints on the fly, also
launch an evaluation job by running:

    python eval.py \
        --base_folder $EXPERIMENT_PATH \
        --gin_bindings="data_dir='$DATASET_PATH'" \
        --gin_configs configs/test_local.gin

The two jobs should use a mutually exclusive set of GPUs. This division allows the
training job to run without having to stop for evaluation.


## Configuration
* We use [Gin](https://github.com/google/gin-config) for configuration.
* We provide a couple preset configurations.
* Please refer to `config.py` for documentation on what each configuration does.
* Preset configs:
    - `hypernerf_vrig_ds_2d.gin`: The deformable surface configuration for the validation rig (novel-view synthesis) experiments.
    - `hypernerf_vrig_ap_2d.gin`: The axis-aligned plane configuration for the validation rig (novel-view synthesis) experiments.
    - `hypernerf_interp_ds_2d.gin`: The deformable surface configuration for the interpolation experiments.
    - `hypernerf_interp_ap_2d.gin`: The axis-aligned plane configuration for the interpolation experiments.


## Dataset
The dataset uses the [same format as Nerfies](https://github.com/google/nerfies#datasets).


## Citing
If you find our work useful, please consider citing:
```BibTeX
@article{park2021hypernerf
  author    = {Park, Keunhong and Sinha, Utkarsh and Hedman, Peter and Barron, Jonathan T. and Bouaziz, Sofien and Goldman, Dan B and Martin-Brualla, Ricardo and Seitz, Steven M.},
  title     = {HyperNeRF: A Higher-Dimensional Representation for Topologically Varying Neural Radiance Fields},
  journal   = {arXiv preprint arXiv:2106.13228},
  year      = {2021},
}
```
