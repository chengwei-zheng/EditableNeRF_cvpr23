#!/usr/bin/env python
# coding: utf-8


# @title Imports
import sys
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

from PIL import Image, ImageDraw


# Monkey patch logging.
def myprint(msg, *args, **kwargs):
 print(msg % args)

logging.info = myprint
logging.warn = myprint


import gin
gin.enter_interactive_mode()


print(jax.devices())

rng = random.PRNGKey(20200823)
# Shift the numpy random seed by host_id() to shuffle data loaded by different
# hosts.
# ZCW jax.host_id has been renamed to jax.process_index
np.random.seed(20201473 + jax.process_index())


dataset_name = '../../../in/capture_0727_0' # @param {type:"string"}
data_dir = gpath.GPath(dataset_name)
print('data_dir: ', data_dir)
# assert data_dir.exists()

exp_dir = '../../../out/save_0727_0_ap_autoS_NE4_reg' # @param {type:"string"}
exp_dir = gpath.GPath(exp_dir)
print('exp_dir: ', exp_dir)
assert exp_dir.exists()

config_path = exp_dir / 'config.gin'
print('config_path', config_path)
assert config_path.exists()

checkpoint_dir = exp_dir / 'checkpoints'
print('checkpoint_dir: ', checkpoint_dir)
assert checkpoint_dir.exists()


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
  load_BS_coeff=dummy_model.load_BS_coeff,  # ZCW add
  BS_coeff_num=dummy_model.BS_coeff_num)  # ZCW add


# ZCW change, use device
devices = jax.devices()[0:2]
print(devices)
rng, key = random.split(rng)
params = {}
model, params['model'] = models.construct_nerf(
    key,
    batch_size=train_config.batch_size,
    embeddings_dict=datasource.embeddings_dict,
    near=datasource.near,
    far=datasource.far,
    BS_coeff_num=datasource.BS_coeff_num)  # ZCW add

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

import functools
# reload() ZCW

use_warp = True  # @param{type: 'boolean'}
use_points = False  # @param{type: 'boolean'}

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


item_ids = [f'{j:06d}' for j in range(600)]
render_scale = 1.0
base_camera = datasource.load_camera('000001').scale(render_scale)

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
  Image.fromarray(np.uint8(pred_rgb * 255)).save(f"../../../visual/render_test/rgb_{i:06d}.png")

  out_frames.append({
      'rgb': pred_rgb,
      'depth': pred_depth_med,
      'med_points': np.array(render['med_points']),
  })
  del batch, render

