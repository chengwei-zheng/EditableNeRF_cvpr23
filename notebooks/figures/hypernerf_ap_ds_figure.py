#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# @title Imports
import sys

from dataclasses import dataclass
from pprint import pprint
from typing import Any, List, Callable, Dict, Sequence

import numpy as np

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

from editablenerf import configs
from editablenerf import datasets
from editablenerf import evaluation
from editablenerf import gpath
from editablenerf import image_utils
from editablenerf import model_utils
from editablenerf import models
from editablenerf import types
from editablenerf import utils
from editablenerf import visualization as viz
from editablenerf import training

import mediapy as media
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageDraw


# ZCW add
import cv2

# Monkey patch logging.
def myprint(msg, *args, **kwargs):
 print(msg % args)

logging.info = myprint
logging.warn = myprint


import gin
gin.enter_interactive_mode()


# def scatter_points(points, **kwargs):
#   """Convenience function for plotting points in Plotly."""
#   return go.Scatter3d(
#       x=points[:, 0],
#       y=points[:, 1],
#       z=points[:, 2],
#       mode='markers',
#       **kwargs,
#   )

# ZCW tmp test
# load_hyper_points_embed = np.loadtxt("../../../visual/hyper_points_embed_1115_0_ap_id.txt")
# for point_2d in load_hyper_points_embed:
#     point_2d[0] = 0
#     point_2d[1] = 1

from IPython.core.display import display, HTML, Latex


def Markdown(text):
  IPython.core.display._display_mimetype('text/markdown', [text], raw=True)


# In[ ]:


print(jax.devices())


# In[ ]:


# @title Utilities
import contextlib

'''
@contextlib.contextmanager
def plot_to_array(height, width, rows=1, cols=1, dpi=100, no_axis=False):
  """A context manager that plots to a numpy array.

  When the context manager exits the output array will be populated with an
  image of the plot.

  Usage:
      ```
      with plot_to_array(480, 640, 2, 2) as (fig, axes, out_image):
          axes[0][0].plot(...)
      ```
  Args:
      height: the height of the canvas
      width: the width of the canvas
      rows: the number of axis rows
      cols: the number of axis columns
      dpi: the DPI to render at
      no_axis: if True will hide the axes of the plot

  Yields:
    A 3-tuple of: a pyplot Figure, array of Axes, and the output np.ndarray.
  """
  out_array = np.empty((height, width, 3), dtype=np.uint8)
  fig, axes = plt.subplots(
      rows, cols, figsize=(width / dpi, height / dpi), dpi=dpi)
  if no_axis:
    for ax in fig.axes:
      ax.margins(0, 0)
      ax.axis('off')
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)

  yield fig, axes, out_array

  # If we haven't already shown or saved the plot, then we need to
  # draw the figure first...
  fig.tight_layout(pad=0)
  fig.canvas.draw()

  # Now we can save it to a numpy array.
  data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  plt.close()

  np.copyto(out_array, data)
'''

# img_read = mpimg.imread('../../../in/capture_1115_0/rgb/4x/000000.png')
# with plot_to_array(180, 320, 2, 2) as (fig, axes, out_image):
#   axes[0][0].imshow(img_read)
#   axes[1][0].imshow(img_read)
#   axes[0][1].imshow(img_read)
#   axes[1][1].imshow(img_read)
# plt.show()
#
# fig, axarr = plt.subplots(7, 7)
# for ax in fig.axes:
#   ax.margins(0, 0)
#   ax.axis('off')
#   ax.get_xaxis().set_visible(False)
#   ax.get_yaxis().set_visible(False)
# fig.tight_layout(pad=0)
# import itertools
# for i, j in itertools.product(range(7), range(7)):
#   axarr[i, j].imshow(img_read)
# plt.show()

# In[ ]:


rng = random.PRNGKey(20200823)
# Shift the numpy random seed by host_id() to shuffle data loaded by different
# hosts.
# ZCW jax.host_id has been renamed to jax.process_index
np.random.seed(20201473 + jax.process_index())


# In[ ]:


dataset_name = '../../../in/capture_0111_0' # @param {type:"string"}  capture_0918_0  syn_0922
data_dir = gpath.GPath(dataset_name)
print('data_dir: ', data_dir)
# assert data_dir.exists()

exp_dir = '../../../out/save_0111_1_mul_auto_NE4_reg_WR150k0'  # @param {type:"string"}
exp_dir = gpath.GPath(exp_dir)
print('exp_dir: ', exp_dir)
assert exp_dir.exists()

config_path = exp_dir / 'config.gin'
print('config_path', config_path)
assert config_path.exists()

checkpoint_dir = exp_dir / 'checkpoints'
print('checkpoint_dir: ', checkpoint_dir)
assert checkpoint_dir.exists()


# In[ ]:


# @title Load configuration.
# reload() ZCW

import IPython
from IPython.display import display, HTML


def config_line_predicate(l):
  return (
      'ExperimentConfig.camera_type' not in l
      and 'preload_data' not in l
      # and 'metadata_at_density' not in l
      # and 'hyper_grad_loss_weight' not in l
    )


print(config_path)

with config_path.open('r') as f: # ZCW open('rt')
  lines = filter(config_line_predicate, f.readlines())
  gin_config = ''.join(lines)

gin.parse_config(gin_config)

exp_config = configs.ExperimentConfig()

train_config = configs.TrainConfig(
    batch_size=1536, #2048
    hyper_sheet_alpha_schedule=None,
)
eval_config = configs.EvalConfig(
    chunk=1536, #4096
)
dummy_model = models.NerfModel({}, 0, 0)

# ZCW
# display(HTML(Markdown(gin.config.markdownify_operative_config_str(gin.config_str())))) # gin.config.markdown


# In[ ]:


datasource = exp_config.datasource_cls(
  data_dir=data_dir,
  image_scale=exp_config.image_scale,
  random_seed=exp_config.random_seed,
  # Enable metadata based on model needs.
  use_warp_id=dummy_model.use_warp,
  use_appearance_id=(
      dummy_model.nerf_embed_key == 'appearance'
      or dummy_model.hyper_embed_key == 'appearance'),
  use_camera_id=dummy_model.nerf_embed_key == 'camera',
  use_time=dummy_model.warp_embed_key == 'time',
  load_kp_init_file=dummy_model.load_kp_init_file,  # ZCW add
  kp_dimension=dummy_model.kp_dimension)  # ZCW add


# In[ ]:


# @title Load model
# reload() ZCW

# ZCW change, use device
devices = jax.devices()[0:2]
# devices = [jax.local_devices()[3], jax.local_devices()[7]]
print(devices)
rng, key = random.split(rng)
params = {}
model, params['model'] = models.construct_nerf(
    key,
    batch_size=train_config.batch_size,
    embeddings_dict=datasource.embeddings_dict,
    near=datasource.near,
    far=datasource.far)
    #kp_dimension=datasource.kp_dimension)  # ZCW add

optimizer_def = optim.Adam(0.0)
if train_config.use_weight_norm:
  optimizer_def = optim.WeightNorm(optimizer_def)
optimizer = optimizer_def.create(params)
state = model_utils.TrainState(
    optimizer=optimizer,
    warp_alpha=0.0)
scalar_params = training.ScalarParams(
    learning_rate=0.0,
    elastic_loss_weight=0.0,
    background_loss_weight=train_config.background_loss_weight)
try:
  state_dict = checkpoints.restore_checkpoint(checkpoint_dir, None)
  state = state.replace(
      optimizer=flax.serialization.from_state_dict(state.optimizer, state_dict['optimizer']),
      warp_alpha=state_dict['warp_alpha'])
except KeyError:
  # Load legacy checkpoints.
  optimizer = optimizer_def.create(params['model'])
  state = model_utils.TrainState(optimizer=optimizer)
  state = checkpoints.restore_checkpoint(checkpoint_dir, state)
  state = state.replace(optimizer=state.optimizer.replace(target={'model': state.optimizer.target}))


# step = state.optimizer.state.step + 1 # ZCW change not used
state = jax_utils.replicate(state, devices=devices)
del params

# ZCW save
save_hyper_points_embed = state_dict['optimizer']['target']['model']['hyper_embed']['embed']['embedding']
np.savetxt("../../../visual/hyper_points_embed.txt", save_hyper_points_embed, delimiter='\t')

# hyper_BS = model.apply(
#     {'params': params['model']}, metadata, method=model.hyper_from_BS_coeff)

# save_nerf_embed = state_dict['optimizer']['target']['model']['nerf_embed']['embed']['embedding']
# np.savetxt("../../../visual/nerf_embed.txt", save_nerf_embed)


# In[ ]:


# @title Render function.
import functools
# reload() ZCW

use_warp = True # @param{type: 'boolean'}
use_points = False # @param{type: 'boolean'}

params = jax_utils.unreplicate(state.optimizer.target)

for i in range(0, 600):
  med_points = np.load(f'../../../out/save_0111_0_ds16-6_NE4_reg/renders-00/00250000/full/med_points_{i:06d}.npy')
  # med_points = np.load(f'../../../out/'
  #                      f'save_1221_0_mulINKP_auto_NE4_reg_lite_WR/renders-02_edit_0/00250000/full/med_points_{i:06d}.npy')
  override_shape = (*med_points.shape[:-1], save_hyper_points_embed.shape[-1])
  # k_points = None
  k_points = jnp.broadcast_to(save_hyper_points_embed[i, ...], override_shape)

  # hyper0 = np.array([-1.7E-01, 0.0E-01, 0.0E-01, -7.5E-01, 0.0E-01, 0.0E-01])  # 1221
  # hyper1 = np.array([8.0E-01, 0.0E-01, 0.0E-01, 1.5E-01, 0.0E-01, 0.0E-01])
  # hyper_w = float(i) / 100.0
  # new_point = hyper_w * hyper1 + (1 - hyper_w) * hyper0
  # k_points = np.broadcast_to(new_point, override_shape)

  weights = model.apply(
      {'params': params['model']}, med_points, k_points, method=model.get_multi_weights)
  weight_l = np.array(weights[..., 0])
  weight_r = np.array(weights[..., 1])
  # weight_3 = np.array(weights[..., 2])

  cv2.imwrite(str(exp_dir / 'weights' / f'weight_l_{i:06d}.png'), weight_l * 255)
  cv2.imwrite(str(exp_dir / 'weights' / f'weight_r_{i:06d}.png'), weight_r * 255)
  # cv2.imwrite(str(exp_dir / 'weights' / f'weight_3_{i:06d}.png'), weight_3 * 255)
  print(i)

# plt.imshow(weight_l)
# plt.xticks([])
# plt.yticks([])
# plt.show()
#
# plt.savefig("../../../visual/test.png")


# edit_metadata = {} # or use metadata.copy()
# edit_metadata['appearance'] = np.broadcast_to([0], (10, 1))
# tmp = model.apply({'params': params['model']}, edit_metadata, method=model.encode_hyper_embed)


# ZCW save
# metadata = {'warp': jnp.array([1], jnp.uint32)}
# tmp = model.apply({'params': params['model']},
#                    metadata,
#                    method=model.encode_warp_embed)

# hyper_BS = model.apply(
#     {'params': params['model']}, datasource.BS_coeff, method=model.hyper_from_BS_coeff)
# new_BS_coeff = np.loadtxt("../../../in/capture_1218_2/data/_lip_ldmk.txt")  # _lip_ldmk.txt  BS_coeffs.txt
# hyper_BS = model.apply(
#     {'params': params['model']}, new_BS_coeff, method=model.hyper_from_BS_coeff)
# np.savetxt("../../../visual/hyper_from_lip.txt", hyper_BS)


def _model_fn(key_0, key_1, params, rays_dict, warp_extras):
  out = model.apply({'params': params},
                    rays_dict,
                    warp_extras,
                    rngs={
                        'coarse': key_0,
                        'fine': key_1
                    },
                    mutable=False,
                    metadata_encoded=True,
                    return_points=use_points,
                    return_weights=use_points,
                    use_warp=use_warp)
  return jax.lax.all_gather(out, axis_name='batch')

pmodel_fn = jax.pmap(
    # Note rng_keys are useless in eval mode since there's no randomness.
    _model_fn,
    # key0, key1, params, rays_dict, warp_extras
    in_axes=(0, 0, 0, 0, 0),
    devices=devices,
    donate_argnums=(3,),  # Donate the 'rays' argument.
    axis_name='batch',
)

render_fn = functools.partial(evaluation.render_image,
                              model_fn=pmodel_fn,
                              device_count=len(devices),
                              chunk=8192)


# In[ ]:


# @title Latent code utils

def get_hyper_code(params, item_id):
  appearance_id = datasource.get_appearance_id(item_id)
  metadata = {
      'warp': jnp.array([appearance_id], jnp.uint32),
      'appearance': jnp.array([appearance_id], jnp.uint32),
  }
  return model.apply({'params': params['model']},
                     metadata,
                     method=model.encode_hyper_embed)


def get_appearance_code(params, item_id):
  appearance_id = datasource.get_appearance_id(item_id)
  metadata = {
      'appearance': jnp.array([appearance_id], jnp.uint32),
  }
  return model.apply({'params': params['model']},
                     metadata,
                     method=model.encode_nerf_embed)


def get_warp_code(params, item_id):
  warp_id = datasource.get_warp_id(item_id)
  metadata = {
      'warp': jnp.array([warp_id], jnp.uint32),
  }
  return model.apply({'params': params['model']},
                     metadata,
                     method=model.encode_warp_embed)

def get_codes(item_id):
  appearance_code = None
  if model.use_rgb_condition and model.use_nerf_embed: # ZCW and model.use_nerf_embed
    appearance_code = get_appearance_code(params, item_id)

  # ZCW warp_codes -> warp_code
  warp_code = None
  if model.use_warp:
    warp_code = get_warp_code(params, item_id)

  # ZCW hyper_codes -> hyper_code
  hyper_code = None
  if model.has_hyper:
    hyper_code = get_hyper_code(params, item_id)

  return appearance_code, warp_code, hyper_code


def make_batch(camera, appearance_code=None, warp_code=None, hyper_code=None):
  batch = datasets.camera_to_rays(camera)
  batch_shape = batch['origins'][..., 0].shape
  metadata = {}
  if appearance_code is not None:
    # ZCW change
    # appearance_code = appearance_code.squeeze(0)
    # appearance_code = appearance_code.squeeze()
    metadata['encoded_nerf'] = jnp.broadcast_to(
        appearance_code[None, None, :], (*batch_shape, appearance_code.shape[-1]))
  if warp_code is not None:
    metadata['encoded_warp'] = jnp.broadcast_to(
        warp_code[None, None, :], (*batch_shape, warp_code.shape[-1]))
  batch['metadata'] = metadata

  if hyper_code is not None:
    batch['metadata']['encoded_hyper'] = jnp.broadcast_to(
        hyper_code[None, None, :], (*batch_shape, hyper_code.shape[-1]))

  return batch


# In[ ]:


# @title Manual crop

render_scale = 1 # 0.5
target_rgb = image_utils.downsample_image(datasource.load_rgb(datasource.train_ids[0]), int(1/render_scale))
# top, bottom, left, right = 2 * np.array([89, 75, 32, 26])  # K
# top, bottom, left, right = 2 * np.array([60, 70, 14, 10])  # R
# top, bottom, left, right = 0, 30, 68, 68  # lemon
# top, bottom, left, right = 40, 100, 2, 40  # slice-banana

# top, bottom, left, right = 50, 20, 120, 100 # 1115_0
# top, bottom, left, right = 80, 30, 140, 120 # 1115_0
# top, bottom, left, right = 50, 20, 95, 125 # 1201_3
# top, bottom, left, right = 70, 40, 115, 145 # 1201_3
# top, bottom, left, right = 70, 10, 120, 120 # 1224_1
top, bottom, left, right = 50, 30, 130, 110 # 1224_1

target_rgb = target_rgb[top:-bottom, left:-right]
print(target_rgb.shape)
# media.show_image(target_rgb)
plt.imshow(target_rgb)
plt.xticks([])
plt.yticks([])
plt.show()

# plt.savefig("../../../visual/test.png")
# Image.fromarray(np.uint8(target_rgb * 255)).save("../../../visual/test.png")

# ## Hyper grid.

# In[ ]:


# @title Sample points and metadata

item_id = datasource.train_ids[0]
camera = datasource.load_camera(item_id).scale(render_scale)
# ZCW cut
# camera.crop_image_domain()
camera = camera.crop_image_domain(left, right, top, bottom)

batch = make_batch(camera, *get_codes(item_id))
origins = batch['origins']
directions = batch['directions']
# metadata = batch['metadata'] # ZCW not needed
z_vals, points = model_utils.sample_along_rays(
    rng, origins[None, ...], directions[None, ...],
    model.num_coarse_samples,
    model.near,
    model.far,
    model.use_stratified_sampling,
    model.use_linear_disparity)
points = points.reshape((-1, 3))
points = random.permutation(rng, points)[:8096*4]
print(points.shape)

warp_metadata = random.randint(
    key, (points.shape[0], 1), 0, model.num_warp_embeds, dtype=jnp.uint32)
warp_embed = model.apply({'params': params['model']},
                          {model.warp_embed_key: warp_metadata},
                          method=model.encode_warp_embed)
# warp_embed = jnp.broadcast_to(
#     warp_embed[:, jnp.newaxis, :],
#     shape=(*points.shape[:-1], warp_embed.shape[-1]))
if model.has_hyper_embed:
  hyper_metadata = random.randint(
      key, (points.shape[0], 1), 0, model.num_hyper_embeds, dtype=jnp.uint32)
  hyper_embed_key = (model.warp_embed_key if model.hyper_use_warp_embed
                      else model.hyper_embed_key)
  hyper_embed = model.apply({'params': params['model']},
                            {hyper_embed_key: hyper_metadata},
                            method=model.encode_hyper_embed)
  # hyper_embed = jnp.broadcast_to(
  #     hyper_embed[:, jnp.newaxis, :],
      # shape=(*batch_shape, hyper_embed.shape[-1]))
else:
  hyper_embed = None

map_fn = functools.partial(model.apply, method=model.map_points)
# ZCW debug
# warped_points, _ = map_fn(
#     {'params': params['model']},
#     points[:, None], hyper_embed[:, None], warp_embed[:, None],
#     jax_utils.unreplicate(state.extra_params))
warped_points, _ = map_fn(
    {'params': params['model']},
    points[:, None], warp_embed[:, None], hyper_embed[:, None],
    jax_utils.unreplicate(state.extra_params))
hyper_points = np.array(warped_points[..., 3:].squeeze())
print(hyper_points.shape)
np.savetxt("../../../visual/hyper_points.txt", hyper_points)

# In[ ]:


# ZCW 1D
if 0:
  umin = np.percentile(hyper_points, 1)
  umax = np.percentile(hyper_points, 99)

  print('umin = ', umin)
  print('umax = ', umax)

  n = 10
  hyper_grid = np.linspace(umin, umax, n)

  grid_frames = []

  camera = datasource.load_camera(item_id).scale(render_scale)
  camera = camera.crop_image_domain(left, right, top, bottom)

  batch = make_batch(camera, *get_codes(item_id))
  batch_shape = batch['origins'][..., 0].shape

  for i in range(n):
    print(i)
    hyper_point = jnp.array(hyper_grid[i])
    hyper_point = jnp.broadcast_to(hyper_point, batch_shape)
    batch['metadata']['hyper_point'] = hyper_point

    render = render_fn(state, batch, rng=rng)
    pred_rgb = np.array(render['rgb'])
    pred_depth_med = np.array(render['med_depth'])
    pred_depth_viz = viz.colorize(1.0 / pred_depth_med.squeeze())
    del render

    plt.imshow(pred_rgb)
    plt.show()

    grid_frames.append({
        'rgb': pred_rgb,
        'depth': pred_depth_med,
    })
# ZCW 1D end

# umin, vmin = hyper_points.min(axis=0)
# umax, vmax = hyper_points.max(axis=0)
umin, vmin = np.percentile(hyper_points[..., :2], 10, axis=0) # ZCW change 20
umax, vmax = np.percentile(hyper_points[..., :2], 90, axis=0) # ZCW change 99
# 10 & 90 for ds

# ZCW print
print('umin = ', umin)
print('vmin = ', vmin)
print('umax = ', umax)
print('vmax = ', vmax)


# In[ ]:


n = 7
uu, vv = np.meshgrid(np.linspace(umin, umax, n), np.linspace(vmin, vmax, n))
hyper_grid = np.stack([uu, vv], axis=-1)
# hyper_grid[0, 0], hyper_grid[-1, -1]
# ZCW print
print(hyper_grid[0, 0])
print(hyper_grid[-1, -1])

# In[ ]:


import itertools
import gc
gc.collect()

grid_frames = []

camera = datasource.load_camera(item_id).scale(render_scale)
camera = camera.crop_image_domain(left, right, top, bottom)

batch = make_batch(camera, *get_codes(item_id))
batch_shape = batch['origins'][..., 0].shape

# ZCW plot
# fig, axarr = plt.subplots(7, 7)
# for ax in fig.axes:
#   ax.margins(0, 0)
#   ax.axis('off')
#   ax.get_xaxis().set_visible(False)
#   ax.get_yaxis().set_visible(False)
# fig.tight_layout(pad=0)


for i, j in itertools.product(range(n), range(n)):
  print([i, j])
  hyper_point = jnp.array(hyper_grid[i, j])
  # hyper_point = jnp.concatenate([hyper_point, jnp.zeros((6,))]) # ZCW ap 8
  hyper_point = jnp.concatenate([hyper_point, jnp.zeros((1,))])
  hyper_point = jnp.broadcast_to(
          hyper_point[None, None, :],
          (*batch_shape, hyper_point.shape[-1]))
  batch['metadata']['hyper_point'] = hyper_point

  render = render_fn(state, batch, rng=rng)
  pred_rgb = np.array(render['rgb'])
  pred_depth_med = np.array(render['med_depth'])
  pred_depth_viz = viz.colorize(1.0 / pred_depth_med.squeeze())
  del render

  # if i % 6 == 0 and j % 6 == 0:
  #   print((i, j))
  #   plt.imshow(pred_rgb)
  #   plt.show()

  # media.show_images([pred_rgb, pred_depth_viz])
  # axarr[i, j].imshow(pred_rgb) # [6 - j, 6 - i] // [j, 6 - i]
  grid_frames.append({
      'rgb': pred_rgb,
      'depth': pred_depth_med,
  })

# plt.show()
# media.show_images([f['rgb'] for f in grid_frames], columns=n)

# ZCW quit
# sys.exit()


# In[ ]:


from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

fig = plt.figure(figsize=(24., 24.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(n, n),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

images = [f['rgb'] for f in grid_frames]
for ax, im in zip(grid, images):
    # Iterating over the grid returns the Axes.
    ax.imshow(im)
    ax.set_axis_off()
    ax.margins(x=0, y=0)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_aspect('equal')
fig.tight_layout(pad=0)

# ZCW add
plt.show()

# In[ ]:

# ZCW
'''
from scipy import interpolate

num_samples = 200
# points = np.random.uniform(0, 1, size=(10, 2))
rng = random.PRNGKey(3)
# points = random.uniform(rng, (20, 2))
points = np.array([
  [0.2, 0.1],
  [0.2, 0.8],
  [0.8, 0.8],
  [0.8, 0.1],
  [0.5, 0.1],
  [0.2, 0.4],
  [0.5, 0.7],
  [0.8, 0.7],
  [0.6, 0.2],
  [0.2, 0.1],
])
t = np.arange(len(points))
xs = np.linspace(0, len(points) - 1, num_samples)
cs = interpolate.CubicSpline(t, points, bc_type='periodic')

interp_points = cs(xs).astype(np.float32)
fig, ax = plt.subplots()
ax.scatter(interp_points[:, 0], interp_points[:, 1], s=2)
ax.scatter(points[:, 0], points[:, 1])
ax.set_aspect('equal')

plt.show()
'''

# In[ ]:

# ZCW
'''
interp_hyper_points = np.stack([(umax - umin) * interp_points[:, 0] + umin, (vmax - vmin) * interp_points[:, 1] + vmin], axis=-1)
'''

# ## Make Orbit Cameras

# In[ ]:

# ZCW
'''
ref_cameras = utils.parallel_map(datasource.load_camera, datasource.all_ids)
'''

# ## Select Keyframes and Interpolate Codes

# In[ ]:

# ZCW
'''
# @title Show training frames to choose IDs
target_ids = datasource.train_ids[::4]
target_rgbs = utils.parallel_map(
    lambda i: image_utils.downsample_image(datasource.load_rgb(i), int(1/render_scale)),
    target_ids)
media.show_images(target_rgbs, titles=target_ids, columns=20)
'''

# ## Render

# In[ ]:


# @title Latent code functions

# reload() ZCW


def get_hyper_code(params, item_id):
  appearance_id = datasource.get_appearance_id(item_id)
  metadata = {
      'warp': jnp.array([appearance_id], jnp.uint32),
      'appearance': jnp.array([appearance_id], jnp.uint32),
  }
  return model.apply({'params': params['model']},
                     metadata,
                     method=model.encode_hyper_embed)


def get_appearance_code(params, item_id):
  appearance_id = datasource.get_appearance_id(item_id)
  metadata = {
      'appearance': jnp.array([appearance_id], jnp.uint32),
  }
  return model.apply({'params': params['model']},
                     metadata,
                     method=model.encode_nerf_embed)


def get_warp_code(params, item_id):
  warp_id = datasource.get_warp_id(item_id)
  metadata = {
      'warp': jnp.array([warp_id], jnp.uint32),
  }
  return model.apply({'params': params['model']},
                     metadata,
                     method=model.encode_warp_embed)


params = jax_utils.unreplicate(state.optimizer.target)

if model.use_rgb_condition and model.use_nerf_embed: # ZCW and model.use_nerf_embed
  test_appearance_code = get_appearance_code(params, datasource.train_ids[0])
  print('appearance code:', test_appearance_code)

if model.use_warp:
  test_warp_code = get_warp_code(params, datasource.train_ids[0])
  print('warp code:', test_warp_code)

if model.has_hyper:
  test_hyper_code = get_hyper_code(params, datasource.train_ids[0])
  print('hyper code:', test_hyper_code)


# In[ ]:


# @title Render function.
import functools
# reload() ZCW

use_warp = True # @param{type: 'boolean'}
use_points = False # @param{type: 'boolean'}

params = jax_utils.unreplicate(state.optimizer.target)


def _model_fn(key_0, key_1, params, rays_dict, warp_extras):
  out = model.apply({'params': params},
                    rays_dict,
                    warp_extras,
                    rngs={
                        'coarse': key_0,
                        'fine': key_1
                    },
                    mutable=False,
                    metadata_encoded=True,
                    return_points=use_points,
                    return_weights=use_points,
                    use_warp=use_warp)
  return jax.lax.all_gather(out, axis_name='batch')

pmodel_fn = jax.pmap(
    # Note rng_keys are useless in eval mode since there's no randomness.
    _model_fn,
    # key0, key1, params, rays_dict, warp_extras
    in_axes=(0, 0, 0, 0, 0),
    devices=devices,
    donate_argnums=(3,),  # Donate the 'rays' argument.
    axis_name='batch',
)

render_fn = functools.partial(evaluation.render_image,
                              model_fn=pmodel_fn,
                              device_count=len(devices),
                              chunk=8192)


# In[ ]:

# ZCW
# item_ids = ['000003', '000150', '000200']
item_ids = [f'{j:04d}' for j in range(429)]
# item_ids = ['001428']
# item_ids = ['000082']
# item_ids = ['000457']
# item_ids = ['000429']
# item_ids = ['000610']  # ricardo
render_scale = 1.0

# media.show_images([datasource.load_rgb(x) for x in item_ids], titles=item_ids)


# In[ ]:



import gc
gc.collect()

# base_camera = datasource.load_camera(item_ids[0]).scale(render_scale)
# base_camera = datasource.load_camera('000037').scale(render_scale)
base_camera = datasource.load_camera('0001').scale(render_scale) # 000389
# orbit_cameras = [c.scale(render_scale) for c in make_orbit_cameras(360)] 
# base_camera = orbit_cameras[270]

out_frames = []
for i, item_id in enumerate(item_ids):
  camera = base_camera
  print(f'>>> Rendering ID {item_id} <<<')
  # ZCW change NE1
  # appearance_code = get_appearance_code(params, item_id).squeeze() if model.use_nerf_embed else None
  appearance_code = get_appearance_code(params, item_id) if model.use_nerf_embed else None
  warp_code = get_warp_code(params, item_id).squeeze() if model.use_warp else None
  hyper_code = get_hyper_code(params, item_id).squeeze() if model.has_hyper_embed else None
  batch = make_batch(camera, appearance_code, warp_code, hyper_code)

  render = render_fn(state, batch, rng=rng)
  pred_rgb = np.array(render['rgb'])
  pred_depth_med = np.array(render['med_depth'])
  pred_depth_viz = viz.colorize(1.0 / pred_depth_med.squeeze())

  # media.show_images([pred_rgb, pred_depth_viz])
  # plt.imshow(pred_rgb)
  # plt.show()

  # ZCW add
  Image.fromarray(np.uint8(pred_rgb * 255)).save(f"../../../visual/hyper_viz/rgb_{i:06d}.png")

  out_frames.append({
      'rgb': pred_rgb,
      'depth': pred_depth_med,
      'med_points': np.array(render['med_points']),
  })
  del batch, render


# In[ ]:


from skimage.color import hsv2rgb

def sinebow(h):
  f = lambda x : np.sin(np.pi * x)**2
  return np.stack([f(3/6-h), f(5/6-h), f(7/6-h)], -1)


def colorize_flow(u, v, phase=0, freq=1):
  coords = np.stack([u, v], axis=-1)
  mag = np.linalg.norm(coords, axis=-1) / np.sqrt(2)
  angle = np.arctan2(-v, -u) / np.pi / (2/freq)
  print(angle.min(), angle.max())
  # return viz.colorize(np.log(mag+1e-6), cmap='gray')
  colorwheel = sinebow(angle + phase/360*np.pi)
  # brightness = mag[..., None] ** 1.414
  brightness = mag[..., None] ** 1.0
  # brightness = (25 * np.cbrt(mag[..., None]*100) - 17)/100
  # brightness = (((mag[..., None]*100 + 17)/25)**3)/100
  bg = np.ones_like(colorwheel) * 0.5
  # bg = np.ones_like(colorwheel) * 0.0
  return colorwheel * brightness + bg * (1.0 - brightness)


def visualize_hyper_points(frame):
  hyper_points = frame['med_points'].squeeze()[..., 3:]
  uu = (hyper_points[..., 0] - umin) / (umax - umin)
  vv = (hyper_points[..., 1] - vmin) / (vmax - vmin)

  # ZCW add
  uu = np.minimum(uu, 1)
  uu = np.maximum(uu, 0)
  vv = np.minimum(vv, 1)
  vv = np.maximum(vv, 0)

  normalized_hyper_points = np.stack([uu, vv], axis=-1)
  normalized_hyper_points = (normalized_hyper_points - 0.5) * 2.0
  print(normalized_hyper_points.min(), normalized_hyper_points.max())
  return colorize_flow(normalized_hyper_points[..., 0], normalized_hyper_points[..., 1])


uu = np.linspace(-1, 1, 256)
vv = np.linspace(-1, 1, 256)
uu, vv = np.meshgrid(uu, vv)

# media.show_image(colorize_flow(uu, vv))
plt.imshow(colorize_flow(uu, vv))
plt.show()


# media.show_image(visualize_hyper_points(out_frames[0]))
#for frame in out_frames:
for i, frame in enumerate(out_frames):
  pred_rgb = frame['rgb']
  pred_depth = frame['depth']
  # depth_viz = viz.colorize(1/pred_depth.squeeze(), cmin=1.6, cmax=3.0, cmap='turbo', invert=False)
  depth_viz = viz.colorize(1/pred_depth.squeeze(), cmin=1.6, cmax=2.3, cmap='turbo', invert=False)
  hyper_viz = visualize_hyper_points(out_frames[i])
  # media.show_images([pred_rgb, depth_viz, hyper_viz])
  # plt.imshow(hyper_viz)
  # plt.show()

  # ZCW add
  Image.fromarray(np.uint8(hyper_viz * 255)).save(f"../../../visual/hyper_viz/{i:06d}.png")
  np.save(f'../../../visual/hyper_viz/med_points_{i:06d}.npy', out_frames[i]['med_points'])

# ZCW quit
# sys.exit()

# In[ ]:

# ZCW
'''
uu = np.linspace(-1, 1, 1024)
vv = np.linspace(-1, 1, 1024)
uu, vv = np.meshgrid(uu, vv)

# media.show_image(colorize_flow(uu, vv))
plt.imshow(colorize_flow(uu, vv))
plt.show()
'''

# In[ ]:



def crop_circle(img, width=3, color=(0, 0, 0)):
  img = Image.fromarray(image_utils.image_to_uint8(img))
  h,w=img.size

  # Create same size alpha layer with circle
  alpha = Image.new('L', img.size,0)
  draw = ImageDraw.Draw(alpha)
  draw.pieslice([0,0,h,w],0,360,fill=255)
  # Convert alpha Image to numpy array
  npAlpha=np.array(alpha)

  draw = ImageDraw.Draw(img)
  draw.arc([0, 0, h, w], 0, 360, fill=tuple(color), width=width)
  npImage=np.array(img)

  # Add alpha layer to RGB
  npImage=np.dstack((npImage,npAlpha))
  return image_utils.image_to_float32(npImage)


media.show_image(crop_circle(images[0]))


# In[ ]:


from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

fig = plt.figure(figsize=(24., 24.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(n, n),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

uu = np.linspace(-1, 1, 7)
vv = np.linspace(-1, 1, 7)
uu, vv = np.meshgrid(uu, vv)
grid_colors = image_utils.image_to_uint8(colorize_flow(uu, vv))
grid_colors = grid_colors.reshape((-1, 3))

images = [f['rgb'] for f in grid_frames]
for i, (ax, im) in enumerate(zip(grid, images)):
    # Iterating over the grid returns the Axes.
    color = tuple(grid_colors[i])
    ax.imshow(crop_circle(im, width=14, color=color))
    ax.set_axis_off()
    ax.margins(x=0, y=0)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_aspect('equal')
fig.tight_layout(pad=0)

# ZCW add
plt.show()

# In[ ]:


(np.array([75, 140, 40, 100], dtype=np.float)*0.9).round()

