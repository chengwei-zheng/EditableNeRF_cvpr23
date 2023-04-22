#!/usr/bin/env python
# coding: utf-8

# # Render a HyperNeRF video!
# 
# **Author**: [Keunhong Park](https://keunhong.com)
# 
# [[Project Page](https://hypernerf.github.io)]
# [[Paper](https://arxiv.org/abs/2106.13228)]
# [[GitHub](https://github.com/google/hypernerf)]
# 
# This notebook renders a video using the test cameras generated in the capture processing notebook.
# 
# You can also load your own custom cameras by modifying the code slightly.
# 
# ### Instructions
# 
# 1. Convert a video into our dataset format using the [capture processing notebook](https://colab.sandbox.google.com/github/google/nerfies/blob/main/notebooks/Nerfies_Capture_Processing.ipynb).
# 2. Train a HyperNeRF model using the [training notebook](https://colab.sandbox.google.com/github/google/hypernerf/blob/main/notebooks/HyperNeRF_Training.ipynb)
# 3. Run this notebook!
# 
# 
# ### Notes
#  * Please report issues on the [GitHub issue tracker](https://github.com/google/hypernerf/issues).

# ## Environment Setup

# In[ ]:


get_ipython().system('pip install flax immutabledict mediapy')
get_ipython().system('pip install git+https://github.com/google/hypernerf')


# In[ ]:


# @title Configure notebook runtime
# @markdown If you would like to use a GPU runtime instead, change the runtime type by going to `Runtime > Change runtime type`. 
# @markdown You will have to use a smaller batch size on GPU.

runtime_type = 'tpu'  # @param ['gpu', 'tpu']
if runtime_type == 'tpu':
  import jax.tools.colab_tpu
  jax.tools.colab_tpu.setup_tpu()

print('Detected Devices:', jax.devices())


# In[ ]:


# @title Mount Google Drive
# @markdown Mount Google Drive onto `/content/gdrive`. You can skip this if running locally.

from google.colab import drive
drive.mount('/content/gdrive')


# In[ ]:


# @title Define imports and utility functions.

import jax
from jax.config import config as jax_config
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

import flax
import flax.linen as nn
from flax import jax_utils
from flax import optim
from flax.metrics import tensorboard
from flax.training import checkpoints

from absl import logging
from io import BytesIO
import random as pyrandom
import numpy as np
import PIL
import IPython
import tempfile
import imageio
import mediapy
from IPython.display import display, HTML
from base64 import b64encode


# Monkey patch logging.
def myprint(msg, *args, **kwargs):
 print(msg % args)

logging.info = myprint 
logging.warn = myprint
logging.error = myprint


# In[ ]:


# @title Model and dataset configuration
# @markdown Change the directories to where you saved your capture and experiment.


from pathlib import Path
from pprint import pprint
import gin
from IPython.display import display, Markdown

from editablenerf import models
from editablenerf import modules
from editablenerf import warping
from editablenerf import datasets
from editablenerf import configs


# @markdown The working directory where the trained model is.
train_dir = '/content/gdrive/My Drive/nerfies/hypernerf_experiments/capture1/exp1'  # @param {type: "string"}
# @markdown The directory to the dataset capture.
data_dir = '/content/gdrive/My Drive/nerfies/captures/capture1'  # @param {type: "string"}

checkpoint_dir = Path(train_dir, 'checkpoints')
checkpoint_dir.mkdir(exist_ok=True, parents=True)

config_path = Path(train_dir, 'config.gin')
with open(config_path, 'r') as f:
  logging.info('Loading config from %s', config_path)
  config_str = f.read()
gin.parse_config(config_str)

config_path = Path(train_dir, 'config.gin')
with open(config_path, 'w') as f:
  logging.info('Saving config to %s', config_path)
  f.write(config_str)

exp_config = configs.ExperimentConfig()
train_config = configs.TrainConfig()
eval_config = configs.EvalConfig()

display(Markdown(
    gin.config.markdownify_operative_config_str(gin.config_str())))


# In[ ]:


# @title Create datasource and show an example.

from editablenerf import datasets
from editablenerf import image_utils

dummy_model = models.NerfModel({}, 0, 0)
datasource = exp_config.datasource_cls(
    image_scale=exp_config.image_scale,
    random_seed=exp_config.random_seed,
    # Enable metadata based on model needs.
    use_warp_id=dummy_model.use_warp,
    use_appearance_id=(
        dummy_model.nerf_embed_key == 'appearance'
        or dummy_model.hyper_embed_key == 'appearance'),
    use_camera_id=dummy_model.nerf_embed_key == 'camera',
    use_time=dummy_model.warp_embed_key == 'time')

mediapy.show_image(datasource.load_rgb(datasource.train_ids[0]))


# In[ ]:


# @title Load model
# @markdown Defines the model and initializes its parameters.

from flax.training import checkpoints
from editablenerf import models
from editablenerf import model_utils
from editablenerf import schedules
from editablenerf import training

rng = random.PRNGKey(exp_config.random_seed)
np.random.seed(exp_config.random_seed + jax.process_index())
devices_to_use = jax.devices()

learning_rate_sched = schedules.from_config(train_config.lr_schedule)
nerf_alpha_sched = schedules.from_config(train_config.nerf_alpha_schedule)
warp_alpha_sched = schedules.from_config(train_config.warp_alpha_schedule)
elastic_loss_weight_sched = schedules.from_config(
train_config.elastic_loss_weight_schedule)
hyper_alpha_sched = schedules.from_config(train_config.hyper_alpha_schedule)
hyper_sheet_alpha_sched = schedules.from_config(
    train_config.hyper_sheet_alpha_schedule)

rng, key = random.split(rng)
params = {}
model, params['model'] = models.construct_nerf(
      key,
      batch_size=train_config.batch_size,
      embeddings_dict=datasource.embeddings_dict,
      near=datasource.near,
      far=datasource.far)

optimizer_def = optim.Adam(learning_rate_sched(0))
optimizer = optimizer_def.create(params)

state = model_utils.TrainState(
    optimizer=optimizer,
    nerf_alpha=nerf_alpha_sched(0),
    warp_alpha=warp_alpha_sched(0),
    hyper_alpha=hyper_alpha_sched(0),
    hyper_sheet_alpha=hyper_sheet_alpha_sched(0))
scalar_params = training.ScalarParams(
    learning_rate=learning_rate_sched(0),
    elastic_loss_weight=elastic_loss_weight_sched(0),
    warp_reg_loss_weight=train_config.warp_reg_loss_weight,
    warp_reg_loss_alpha=train_config.warp_reg_loss_alpha,
    warp_reg_loss_scale=train_config.warp_reg_loss_scale,
    background_loss_weight=train_config.background_loss_weight,
    hyper_reg_loss_weight=train_config.hyper_reg_loss_weight)

logging.info('Restoring checkpoint from %s', checkpoint_dir)
state = checkpoints.restore_checkpoint(checkpoint_dir, state)
step = state.optimizer.state.step + 1
state = jax_utils.replicate(state, devices=devices_to_use)
del params


# In[ ]:


# @title Define pmapped render function.

import functools
from editablenerf import evaluation

devices = jax.devices()


def _model_fn(key_0, key_1, params, rays_dict, extra_params):
  out = model.apply({'params': params},
                    rays_dict,
                    extra_params=extra_params,
                    rngs={
                        'coarse': key_0,
                        'fine': key_1
                    },
                    mutable=False)
  return jax.lax.all_gather(out, axis_name='batch')

pmodel_fn = jax.pmap(
    # Note rng_keys are useless in eval mode since there's no randomness.
    _model_fn,
    in_axes=(0, 0, 0, 0, 0),  # Only distribute the data input.
    devices=devices_to_use,
    axis_name='batch',
)

render_fn = functools.partial(evaluation.render_image,
                              model_fn=pmodel_fn,
                              device_count=len(devices),
                              chunk=eval_config.chunk)


# In[ ]:


# @title Load cameras.

from editablenerf import utils

camera_path = 'camera-paths/orbit-mild'  # @param {type: 'string'}

camera_dir = Path(data_dir, camera_path)
print(f'Loading cameras from {camera_dir}')
test_camera_paths = datasource.glob_cameras(camera_dir)
test_cameras = utils.parallel_map(datasource.load_camera, test_camera_paths, show_pbar=True)


# In[ ]:


# @title Render video frames.
from editablenerf import visualization as viz


rng = rng + jax.process_index()  # Make random seed separate across hosts.
keys = random.split(rng, len(devices))

results = []
for i in range(len(test_cameras)):
  print(f'Rendering frame {i+1}/{len(test_cameras)}')
  camera = test_cameras[i]
  batch = datasets.camera_to_rays(camera)
  batch['metadata'] = {
      'appearance': jnp.zeros_like(batch['origins'][..., 0, jnp.newaxis], jnp.uint32),
      'warp': jnp.zeros_like(batch['origins'][..., 0, jnp.newaxis], jnp.uint32),
  }

  render = render_fn(state, batch, rng=rng)
  rgb = np.array(render['rgb'])
  depth_med = np.array(render['med_depth'])
  results.append((rgb, depth_med))
  depth_viz = viz.colorize(depth_med.squeeze(), cmin=datasource.near, cmax=datasource.far, invert=True)
  mediapy.show_images([rgb, depth_viz])


# In[ ]:


# @title Show rendered video.

fps = 30  # @param {type:'number'}

frames = []
for rgb, depth in results:
  depth_viz = viz.colorize(depth.squeeze(), cmin=datasource.near, cmax=datasource.far, invert=True)
  frame = np.concatenate([rgb, depth_viz], axis=1)
  frames.append(image_utils.image_to_uint8(frame))

mediapy.show_video(frames, fps=fps)


# In[ ]:




