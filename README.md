# EditableNeRF: Editing Topologically Varying Neural Radiance Fields by Key Points

This is the code for "EditableNeRF: Editing Topologically Varying Neural Radiance Fields by Key Points", and the project page is [here](https://chengwei-zheng.github.io/EditableNeRF/).

This code is built on [HyperNeRF](https://github.com/google/hyperNeRF).


## Overview

- Train a [HyperNeRF](https://github.com/google/hyperNeRF) and obtain rendering results.
- Run `data_process/k_points_detect.py` to detect key points.
- Run `data_process/k_points_init_RAFT.py` based on [RAFT](https://github.com/princeton-vl/RAFT) to initialize key points in 2d.
- Run `data_process/k_points_init_3d.py` to initialize key points in 3d.
- Run `train.py` for EditableNeRF training.
- Run `eval.py` for EditableNeRF evaluation and rendering.


## 1. Train a HyperNeRF

Before running our method, you should run [HyperNeRF](https://github.com/google/hyperNeRF) first, as many parts of our code rely on the Python environments and the results of HyperNeRF. 

Please follow [HyperNeRF](https://github.com/google/hyperNeRF) to set up Python environment, prepare your dataset (or use the dataset we provide), and train a HyperNeRF.

After training, you need to use this HyperNeRF to render a set of results, which will be used as inputs of EditableNeRF.

- Depth images: direct outputs of HyperNeRF
- 

We provide an example of these results in our dataset (TODO), please refer to it and make sure your data is in the coorect format and consistent with ours.


## 2. Key Point Detection and Initialization

Based on these rendering results, the next step is to detect and initalize key points.

#### Key Point Detection

First, run `data_process/k_points_detect.py` using `python k_points_detect.py` (or using Jupyter) in the same Python environment as the dataset processing stage in [HyperNeRF](https://github.com/google/hyperNeRF). You need to change the parameters in `k_points_detect.py` to be consistent with those you used in HyperNeRF. 

After that, you can find the detected key points and the **visualization results** in a new folder `data_process/kp_init`. Please check these results before you step into the next stage.

#### Key Point Initialization in 2D and Optical Flow Computing

Then, use RAFT to initialize key points in 2D and computing the optical flow that will be used in EditableNeRF training.

The `data_process/k_points_init_RAFT.py` is an expansion of the `demo.py` file in [RAFT](https://github.com/princeton-vl/RAFT), which can be run similarly as `demo.py` in [RAFT](https://github.com/princeton-vl/RAFT) based on its Python environment:

    python k_points_init_RAFT.py \
        --model=models/raft-things.pth \
        --path=input_img_dir \
        --kp_file=kp_init \
        --skip_prop=50 \

where `skip_prop` is the frame number $M$ for skipping propagation as in our paper. Set it to zero if you do not want to use skipping propagation.

When it finished, you can find a new file `kp_2d_init.txt` in `data_process/kp_init`, which is the initialize 2D key points; and a new folder `data_process/kp_init/flow`, which contains the optical flow that will be used in EditableNeRF training. Also, the **visualization results** is provide in `data_process/out`, check them if you find anything wrong.


#### Key Point Initialization in 3D

Before start training, the last step is initializing the key points in 3D based on their 2D positions.

This can be done by running `data_process/k_points_init_3d.py` using `python k_points_init_3d.py` (or using Jupyter) again in the dataset processing environment in [HyperNeRF](https://github.com/google/hyperNeRF). Also, please remember to change the parameters in it.



## 3. Training and Rendering

Our training and rendering methods are the same as [HyperNeRF](https://github.com/google/hyperNeRF).


## Dataset
TODO


## (Optinal) GUI
`GUI_qt.py`


## Citing
```BibTeX
@misc{zheng2023editablenerf,
      title={EditableNeRF: Editing Topologically Varying Neural Radiance Fields by Key Points}, 
      author={Chengwei Zheng and Wenbin Lin and Feng Xu},
      year={2023},
      eprint={2212.04247},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
