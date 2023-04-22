#!/usr/bin/env python
# coding: utf-8

# # Let's train HyperNeRF!
# 
# **Author**: [Keunhong Park](https://keunhong.com)
# 
# [[Project Page](https://hypernerf.github.io)]
# [[Paper](https://arxiv.org/abs/2106.13228)]
# [[GitHub](https://github.com/google/hypernerf)]
# 
# This notebook provides an demo for training HyperNeRF.
# 
# ### Instructions
# 
# 1. Convert a video into our dataset format using the Nerfies [dataset processing notebook](https://colab.sandbox.google.com/github/google/nerfies/blob/main/notebooks/Nerfies_Capture_Processing.ipynb).
# 2. Set the `data_dir` below to where you saved the dataset.
# 3. Come back to this notebook to train HyperNeRF.
# 
# 
# ### Notes
#  * To accomodate the limited compute power of Colab runtimes, this notebook defaults to a "toy" version of our method. The number of samples have been reduced and the elastic regularization turned off.
# 
#  * To train a high-quality model, please look at the CLI options we provide in the [Github repository](https://github.com/google/hypernerf).
# 
# 
# 
#  * Please report issues on the [GitHub issue tracker](https://github.com/google/hypernerf/issues).
# 
# 
# If you find this work useful, please consider citing:
# ```bibtex
# @article{park2021hypernerf
#   author    = {Park, Keunhong and Sinha, Utkarsh and Hedman, Peter and Barron, Jonathan T. and Bouaziz, Sofien and Goldman, Dan B and Martin-Brualla, Ricardo and Seitz, Steven M.},
#   title     = {HyperNeRF: A Higher-Dimensional Representation for Topologically Varying Neural Radiance Fields},
#   journal   = {arXiv preprint arXiv:2106.13228},
#   year      = {2021},
# }
# ```
# 

# ## Environment Setup

# In[ ]:


get_ipython().system('pip install flax immutabledict mediapy')
get_ipython().system('pip install --upgrade git+https://github.com/google/hypernerf')


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
jax_config.enable_omnistaging() # Linen requires enabling omnistaging

from absl import logging
from io import BytesIO
import random as pyrandom
import numpy as np
import PIL
import IPython


# Monkey patch logging.
def myprint(msg, *args, **kwargs):
 print(msg % args)

logging.info = myprint 
logging.warn = myprint
logging.error = myprint


def show_image(image, fmt='png'):
    image = image_utils.image_to_uint8(image)
    f = BytesIO()
    PIL.Image.fromarray(image).save(f, fmt)
    IPython.display.display(IPython.display.Image(data=f.getvalue()))


# ## Configuration

# In[ ]:


# @title Model and dataset configuration

from pathlib import Path
from pprint import pprint
import gin
from IPython.display import display, Markdown

from editablenerf import models
from editablenerf import modules
from editablenerf import warping
from editablenerf import datasets
from editablenerf import configs


# @markdown The working directory.
train_dir = '/content/gdrive/My Drive/nerfies/hypernerf_experiments/capture1/exp1'  # @param {type: "string"}
# @markdown The directory to the dataset capture.
data_dir = '/content/gdrive/My Drive/nerfies/captures/capture1'  # @param {type: "string"}

# @markdown Training configuration.
max_steps = 100000  # @param {type: 'number'}
batch_size = 4096  # @param {type: 'number'}
image_scale = 8  # @param {type: 'number'}

# @markdown Model configuration.
use_viewdirs = True  #@param {type: 'boolean'}
use_appearance_metadata = True  #@param {type: 'boolean'}
num_coarse_samples = 64  # @param {type: 'number'}
num_fine_samples = 64  # @param {type: 'number'}

# @markdown Deformation configuration.
use_warp = True  #@param {type: 'boolean'}
warp_field_type = '@SE3Field'  #@param['@SE3Field', '@TranslationField']
warp_min_deg = 0  #@param{type:'number'}
warp_max_deg = 6  #@param{type:'number'}

# @markdown Hyper-space configuration.
hyper_num_dims = 8  #@param{type:'number'}
hyper_point_min_deg = 0  #@param{type:'number'}
hyper_point_max_deg = 1  #@param{type:'number'}
hyper_slice_method = 'bendy_sheet'  #@param['none', 'axis_aligned_plane', 'bendy_sheet']


checkpoint_dir = Path(train_dir, 'checkpoints')
checkpoint_dir.mkdir(exist_ok=True, parents=True)

config_str = f"""
DELAYED_HYPER_ALPHA_SCHED = {{
  'type': 'piecewise',
  'schedules': [
    (1000, ('constant', 0.0)),
    (0, ('linear', 0.0, %hyper_point_max_deg, 10000))
  ],
}}

ExperimentConfig.image_scale = {image_scale}
ExperimentConfig.datasource_cls = @NerfiesDataSource
NerfiesDataSource.data_dir = '{data_dir}'
NerfiesDataSource.image_scale = {image_scale}

NerfModel.use_viewdirs = {int(use_viewdirs)}
NerfModel.use_rgb_condition = {int(use_appearance_metadata)}
NerfModel.num_coarse_samples = {num_coarse_samples}
NerfModel.num_fine_samples = {num_fine_samples}

NerfModel.use_viewdirs = True
NerfModel.use_stratified_sampling = True
NerfModel.use_posenc_identity = False
NerfModel.nerf_trunk_width = 128
NerfModel.nerf_trunk_depth = 8

TrainConfig.max_steps = {max_steps}
TrainConfig.batch_size = {batch_size}
TrainConfig.print_every = 100
TrainConfig.use_elastic_loss = False
TrainConfig.use_background_loss = False

# Warp configs.
warp_min_deg = {warp_min_deg}
warp_max_deg = {warp_max_deg}
NerfModel.use_warp = {use_warp}
SE3Field.min_deg = %warp_min_deg
SE3Field.max_deg = %warp_max_deg
SE3Field.use_posenc_identity = False
NerfModel.warp_field_cls = @SE3Field

TrainConfig.warp_alpha_schedule = {{
    'type': 'linear',
    'initial_value': {warp_min_deg},
    'final_value': {warp_max_deg},
    'num_steps': {int(max_steps*0.8)},
}}

# Hyper configs.
hyper_num_dims = {hyper_num_dims}
hyper_point_min_deg = {hyper_point_min_deg}
hyper_point_max_deg = {hyper_point_max_deg}

NerfModel.hyper_embed_cls = @hyper/GLOEmbed
hyper/GLOEmbed.num_dims = %hyper_num_dims
NerfModel.hyper_point_min_deg = %hyper_point_min_deg
NerfModel.hyper_point_max_deg = %hyper_point_max_deg

TrainConfig.hyper_alpha_schedule = %DELAYED_HYPER_ALPHA_SCHED

hyper_sheet_min_deg = 0
hyper_sheet_max_deg = 6
HyperSheetMLP.min_deg = %hyper_sheet_min_deg
HyperSheetMLP.max_deg = %hyper_sheet_max_deg
HyperSheetMLP.output_channels = %hyper_num_dims

NerfModel.hyper_slice_method = '{hyper_slice_method}'
NerfModel.hyper_sheet_mlp_cls = @HyperSheetMLP
NerfModel.hyper_use_warp_embed = True

TrainConfig.hyper_sheet_alpha_schedule = ('constant', %hyper_sheet_max_deg)
"""

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

show_image(datasource.load_rgb(datasource.train_ids[0]))


# In[ ]:


# @title Create training iterators

devices = jax.local_devices()

train_iter = datasource.create_iterator(
    datasource.train_ids,
    flatten=True,
    shuffle=True,
    batch_size=train_config.batch_size,
    prefetch_size=3,
    shuffle_buffer_size=train_config.shuffle_buffer_size,
    devices=devices,
)

def shuffled(l):
  import random as r
  import copy
  l = copy.copy(l)
  r.shuffle(l)
  return l

train_eval_iter = datasource.create_iterator(
    shuffled(datasource.train_ids), batch_size=0, devices=devices)
val_eval_iter = datasource.create_iterator(
    shuffled(datasource.val_ids), batch_size=0, devices=devices)


# ## Training

# In[ ]:


# @title Initialize model
# @markdown Defines the model and initializes its parameters.

from flax.training import checkpoints
from editablenerf import models
from editablenerf import model_utils
from editablenerf import schedules
from editablenerf import training

# @markdown Restore a checkpoint if one exists.
restore_checkpoint = False  # @param{type:'boolean'}


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

if restore_checkpoint:
  logging.info('Restoring checkpoint from %s', checkpoint_dir)
  state = checkpoints.restore_checkpoint(checkpoint_dir, state)
step = state.optimizer.state.step + 1
state = jax_utils.replicate(state, devices=devices)
del params


# In[ ]:


# @title Define pmapped functions
# @markdown This parallelizes the training and evaluation step functions using `jax.pmap`.

import functools
from editablenerf import evaluation


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
train_step = functools.partial(
    training.train_step,
    model,
    elastic_reduce_method=train_config.elastic_reduce_method,
    elastic_loss_type=train_config.elastic_loss_type,
    use_elastic_loss=train_config.use_elastic_loss,
    use_background_loss=train_config.use_background_loss,
    use_warp_reg_loss=train_config.use_warp_reg_loss,
    use_hyper_reg_loss=train_config.use_hyper_reg_loss,
)
ptrain_step = jax.pmap(
    train_step,
    axis_name='batch',
    devices=devices,
    # rng_key, state, batch, scalar_params.
    in_axes=(0, 0, 0, None),
    # Treat use_elastic_loss as compile-time static.
    donate_argnums=(2,),  # Donate the 'batch' argument.
)


# In[ ]:


# @title Train!
# @markdown This runs the training loop!

import mediapy
from editablenerf import utils
from editablenerf import visualization as viz


print_every_n_iterations = 100  # @param{type:'number'}
visualize_results_every_n_iterations = 500  # @param{type:'number'}
save_checkpoint_every_n_iterations = 1000  # @param{type:'number'}


logging.info('Starting training')
rng = rng + jax.process_index()  # Make random seed separate across hosts.
keys = random.split(rng, len(devices))
time_tracker = utils.TimeTracker()
time_tracker.tic('data', 'total')

for step, batch in zip(range(step, train_config.max_steps + 1), train_iter):
  time_tracker.toc('data')
  scalar_params = scalar_params.replace(
      learning_rate=learning_rate_sched(step),
      elastic_loss_weight=elastic_loss_weight_sched(step))
  # pytype: enable=attribute-error
  nerf_alpha = jax_utils.replicate(nerf_alpha_sched(step), devices)
  warp_alpha = jax_utils.replicate(warp_alpha_sched(step), devices)
  hyper_alpha = jax_utils.replicate(hyper_alpha_sched(step), devices)
  hyper_sheet_alpha = jax_utils.replicate(
      hyper_sheet_alpha_sched(step), devices)
  state = state.replace(nerf_alpha=nerf_alpha,
                        warp_alpha=warp_alpha,
                        hyper_alpha=hyper_alpha,
                        hyper_sheet_alpha=hyper_sheet_alpha)

  with time_tracker.record_time('train_step'):
    state, stats, keys, _ = ptrain_step(keys, state, batch, scalar_params)
    time_tracker.toc('total')

  if step % print_every_n_iterations == 0:
    logging.info(
        'step=%d, warp_alpha=%.04f, hyper_alpha=%.04f, hyper_sheet_alpha=%.04f, %s',
        step, 
        warp_alpha_sched(step), 
        hyper_alpha_sched(step), 
        hyper_sheet_alpha_sched(step), 
        time_tracker.summary_str('last'))
    coarse_metrics_str = ', '.join(
        [f'{k}={v.mean():.04f}' for k, v in stats['coarse'].items()])
    fine_metrics_str = ', '.join(
        [f'{k}={v.mean():.04f}' for k, v in stats['fine'].items()])
    logging.info('\tcoarse metrics: %s', coarse_metrics_str)
    if 'fine' in stats:
      logging.info('\tfine metrics: %s', fine_metrics_str)
  
  if step % visualize_results_every_n_iterations == 0:
    print(f'[step={step}] Training set visualization')
    eval_batch = next(train_eval_iter)
    render = render_fn(state, eval_batch, rng=rng)
    rgb = render['rgb']
    acc = render['acc']
    depth_exp = render['depth']
    depth_med = render['med_depth']
    rgb_target = eval_batch['rgb']
    depth_med_viz = viz.colorize(depth_med, cmin=datasource.near, cmax=datasource.far)
    mediapy.show_images([rgb_target, rgb, depth_med_viz],
                        titles=['GT RGB', 'Pred RGB', 'Pred Depth'])

    print(f'[step={step}] Validation set visualization')
    eval_batch = next(val_eval_iter)
    render = render_fn(state, eval_batch, rng=rng)
    rgb = render['rgb']
    acc = render['acc']
    depth_exp = render['depth']
    depth_med = render['med_depth']
    rgb_target = eval_batch['rgb']
    depth_med_viz = viz.colorize(depth_med, cmin=datasource.near, cmax=datasource.far)
    mediapy.show_images([rgb_target, rgb, depth_med_viz],
                       titles=['GT RGB', 'Pred RGB', 'Pred Depth'])

  if step % save_checkpoint_every_n_iterations == 0:
    training.save_checkpoint(checkpoint_dir, state)

  time_tracker.tic('data', 'total')


# In[ ]:




