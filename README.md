# EditableNeRF: Editing Topologically Varying Neural Radiance Fields by Key Points

This is the code for "EditableNeRF: Editing Topologically Varying Neural Radiance Fields by Key Points", and the project page is [here](https://chengwei-zheng.github.io/EditableNeRF/).

This code is built upon [HyperNeRF](https://github.com/google/hyperNeRF).


## Overview

- Train a [HyperNeRF](https://github.com/google/hyperNeRF) and obtain rendering results.
- Run `data_process/k_points_detect.py` to detect key points.
- Run `data_process/k_points_init_RAFT.py` based on [RAFT](https://github.com/princeton-vl/RAFT) to initialize key points in 2d.
- Run `data_process/k_points_init_3d.py` to initialize key points in 3d.
- Run `train.py` for EditableNeRF training.
- Run `eval.py` for EditableNeRF evaluation and rendering.


## 1. Train a HyperNeRF

Before running our method, you should run [HyperNeRF](https://github.com/google/hyperNeRF) first, as many parts of our code rely on the Python environments and the results of HyperNeRF. 

Please follow [HyperNeRF](https://github.com/google/hyperNeRF) to set up Python environments, prepare your dataset (or use the dataset we provide), and train a HyperNeRF.

After training, you need to use this HyperNeRF to render a set of results, which will be used as inputs of EditableNeRF. 

You can also train a HyperNeRF using our EditableNeRF code by setting the config file as `hypernerf_vrig_ds_2d.gin`. And the codes for saving the following images are already in our code (lines 147-154 in our `eval.py`).

1. Depth images: direct outputs of HyperNeRF. You can save these images using: `np.save(f'your_path/depth_median_{item_id}.npy', model_out['med_depth'])` where `model_out` is the returned dictionary of HyperNeRF (e.g., line 126, line 264 in HyperNeRF's `eval.py`). You should render them in both the original input camera views and a fixed view (e.g., the view of the first frame).
    
2. Rendered maps for canonical points (hyperspace): for each pixel, its value in this map is the canonical coordinate of the corresponding surface point. Note this coordinate is in hyperspace, and these maps only need to be rendered in a fixed view. You can save these maps using `np.save(f'your_path/med_warped_points_{item_id}.npy', model_out['med_warped_points'].squeeze()])`, and replacing the `models.py` of HyperNeRF with the one we provide as `data_process/models.py` (different in lines 544-549).

3. Rendered maps for 3d points: save these maps similarly using `np.save(f'your_path/med_points_{item_id}.npy', model_out['med_points'].squeeze()])`, in the original input camera views and a fixed view.


We provide an example of these results in our dataset (TODO); please refer to it and make sure your data is in the correct format and consistent with ours.


## 2. Key Point Detection and Initialization

Based on these rendering results, the next stage is to detect and initialize key points.

### 2.1 Key Point Detection

First, run `data_process/k_points_detect.py` using `python k_points_detect.py` (or using Jupyter) in the same Python environment as the dataset processing stage in [HyperNeRF](https://github.com/google/hyperNeRF). You need to change the parameters in `k_points_detect.py` (lines 16-20) to be consistent with those you used in HyperNeRF. 

After that, you can find the detected key points and the **visualization results** in a new folder `data_process/kp_init`. Please check these results before you step into the next stage.

### 2.2 Key Point Initialization in 2D and Optical Flow Computing

Then, use RAFT to initialize key points in 2D and compute the optical flow that will be used in EditableNeRF training.

The `data_process/k_points_init_RAFT.py` file is an expansion of the `demo.py` file in [RAFT](https://github.com/princeton-vl/RAFT), and it can be run similarly as `demo.py` in [RAFT](https://github.com/princeton-vl/RAFT) based on the Python environment used in RAFT:

    python k_points_init_RAFT.py \
        --model=models/raft-things.pth \
        --path=input_img_dir \
        --kp_file=kp_init \
        --skip_prop=50

where `skip_prop` is the frame number $M$ for skipping propagation as in our paper (Sec. 3.3). Set it to zero if you do not want to use skipping propagation.

When it finished, you can find a new file `kp_2d_init.txt` in `data_process/kp_init`, which is the initialized 2D key points; and a new folder `data_process/kp_init/flow`, which contains the optical flow that will be used in EditableNeRF training. Also, the **visualization results** are provided in `data_process/out`; check them if you find anything wrong.


### 2.3 Key Point Initialization in 3D

Before starting training, the last step is initializing the key points in 3D based on their 2D positions.

This can be done by running `data_process/k_points_init_3d.py` using `python k_points_init_3d.py` (or using Jupyter) again in the dataset processing environment in [HyperNeRF](https://github.com/google/hyperNeRF). Also, please remember to change the parameters in it (lines 20-22).



## 3. Training and Rendering

Our training and rendering methods are similar to [HyperNeRF](https://github.com/google/hyperNeRF).

    python train.py \
        --base_folder ../out/save_demo \
        --gin_bindings="data_dir='../in/capture_demo'" \
        --gin_configs configs/editablenerf_2p.gin

    python eval.py \
        --base_folder ../out/save_demo \
        --gin_bindings="data_dir='../in/capture_demo'" \
        --gin_configs configs/editablenerf_2p.gin

## Dataset
Coming soon ...


## (Optional) GUI
Running `GUI_qt.py` further needs Qt5 installation:

    pip install pyvista
    pip install pyvistaqt
    pip install pyqt5   

Then run GUI by

    python GUI_qt.py


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
