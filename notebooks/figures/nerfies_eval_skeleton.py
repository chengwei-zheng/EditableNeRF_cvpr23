#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# @title Basic Imports

from collections import defaultdict
import collections

import matplotlib
from matplotlib import pyplot as plt
import pandas as pd

from IPython import display as ipd
from PIL import Image
import io
import imageio
import numpy as np
import copy

import gin
gin.enter_interactive_mode()

from six.moves import reload_module

import tensorflow as tf
import tensorflow_hub as hub
from absl import logging
from pprint import pprint


def myprint(msg, *args, **kwargs):
 print(msg % args)
logging.info = myprint 
logging.warn = myprint 


# In[ ]:


# @title Utility methods.

import json
from scipy.ndimage import morphology
import math
import cv2


def load_image(x):
  try:
    x = image_utils.load_image(x)
  except:
    print(f'Could not read image file {x}')
    raise
  x = x.astype(np.float32) / 255.0
  return x


def load_images(paths):
  return utils.parallel_map(load_image, paths, show_pbar=False)


def load_mask(path):
  return load_image(path)


def load_masks(paths):
  return utils.parallel_map(load_mask, paths, show_pbar=False)


def crop_image(image, left=0, right=0, top=0, bottom=0):
  pad_width = [max(0, -x) for x in [top, bottom, left, right]]
  if any(pad_width):
    image = np.pad(image, pad_width=pad_width, mode='constant')
  crop_coords = [max(0, x) for x in (top, bottom, left, right)]
  return image[crop_coords[0]:-crop_coords[1], crop_coords[2]:-crop_coords[3], :]


def scale_image(image, scale, interpolation=None):
  if interpolation is None:
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4
  height, width = image.shape[:2]
  new_height, new_width = int(scale * height), int(scale * width)
  image = cv2.resize(
      image, (new_width, new_height), interpolation=interpolation)
  return image


# In[ ]:


# @title Cropping Code

def compute_nv_crop(image, scale=2.0):
  height, width = image.shape[:2]
  target_height = int(1024 * scale)
  target_width = int(667 * scale)

  margin = 100
  scale_y = (target_height + 1 + margin) / image.shape[0]
  scale_x = (target_width + 1 + margin) / image.shape[1]
  scale = max(scale_x, scale_y)
  # image = scale_image(image, scale)

  new_shape = (int(height * scale), int(width * scale))
  if new_shape[0] > target_height:
    crop_top = int(math.floor((new_shape[0] - target_height) / 2))
    crop_bottom = int(math.ceil((new_shape[0] - target_height) / 2))
  else:
    crop_top = 0
    crop_bottom = 1
  if new_shape[1] > target_width:
    crop_left = int(math.floor((new_shape[1] - target_width) / 2))
    crop_right = int(math.ceil((new_shape[1] - target_width) / 2))
  else:
    crop_left = 0
    crop_right = 1
 
  crop1 = np.array([crop_left, crop_right, crop_top, crop_bottom], np.float32)
  new_shape = (new_shape[0] - crop_top - crop_bottom, 
               new_shape[1] - crop_right - crop_left)
  crop1 /= scale

  new_shape = (int(new_shape[0] * 0.51), int(new_shape[1] * 0.51))
  if new_shape[0] > 1024:
    crop_top = int(math.floor((new_shape[0] - 1024) / 2))
    crop_bottom = int(math.ceil((new_shape[0] - 1024) / 2))
  else:
    crop_top = 0
    crop_bottom = 1
  if new_shape[1] > 667:
    crop_left = int(math.floor((new_shape[1] - 667) / 2))
    crop_right = int(math.ceil((new_shape[1] - 667) / 2))
  else:
    crop_left = 0
    crop_right = 1

  crop2 = np.array([crop_left, crop_right, crop_top, crop_bottom], np.float32)
  crop2 = crop2 / 0.51 / scale

  crop = (crop1 + crop2).astype(np.uint32).tolist()
  return crop


def _nv_crop(image, scale=2.0):
  crop = compute_nv_crop(image, scale)
  return crop_image(image, *crop)


def nv_crop(image, scale=2.0):
  height, width = image.shape[:2]
  rotated = False
  if width > height:
    image = np.rot90(image)
    rotated = True
  
  image = _nv_crop(image, scale=scale)
  if rotated:
    image = np.rot90(image, -1)
  
  return image


# In[ ]:


# @title Undistort Code

def crop_camera_invalid_pixels(source_camera, target_camera):
  """Crops the camera viewport to only contain valid pixels.

  This method crops "invalid" pixels that can occur when an image is warped
  (e.g., when an image is undistorted). The method computes the outer border
  pixels of the original image and projects them to the warped camera. It then
  computes the inscribed rectangle bounds where there are guaranteed to be no
  invalid pixels. The method then adjusts the image size and principal point
  to crop the viewport to this inscribed rectangle.

  Example:
    # Cropping a camera `target_camera`.
    target_cropped_camera = crop_camera_invalid_pixels(
        source_camera, target_camera)
    warped_cropped_image = warp_image_at_infinity(
        source_camera, target_cropped_camera, image)

  Args:
    source_camera: The camera corresponding to the source image.
    target_camera: The target camera on which to warp the source image.

  Returns:
    A camera which is the same as the target_camera but with a cropped
      viewport. The width, height, and principal points will be changed.
  """
  # Compute rays from original camera.
  source_pixels = source_camera.GetPixelCenters()
  source_pixel_rays = source_camera.PixelsToRays(source_pixels)
  source_pixel_dirs = np.insert(source_pixel_rays, 3, 0, axis=2)
  source_pixels, _ = target_camera.Project(source_pixel_dirs)

  # Compute border pixel bounds.
  top_max_y = max(source_pixels[0, :, 1])
  bottom_min_y = min(source_pixels[-1, :, 1])
  left_max_x = max(source_pixels[:, 0, 0])
  right_min_x = min(source_pixels[:, -1, 0])

  # How much do we have to scale the image?
  cx = target_camera.PrincipalPointX()
  cy = target_camera.PrincipalPointY()
  width = target_camera.ImageSizeX()
  height = target_camera.ImageSizeY()
  # Take the maximum of the top/bottom crop for the vertical scale and the
  # left/right crop for the horizontal scale.
  scale_x = 1.0 / max(cx / (cx - left_max_x),
                      (width - 0.5 - cx) / (right_min_x - cx))
  scale_y = 1.0 / max(cy / (cy - top_max_y),
                      (height - 0.5 - cy) / (bottom_min_y - cy))

  new_width = int(scale_x * width)
  new_height = int(scale_y * height)

  # Move principal point based on new image size.
  new_camera = target_camera.Copy()
  new_cx = cx * new_width / width
  new_cy = cy * new_height / height
  new_camera.SetPrincipalPoint(new_cx, new_cy)
  new_camera.SetImageSize(new_width, new_height)

  return new_camera


def undistort_camera(camera, crop_blank_pixels=False, set_ideal_pp=True):
  # Create a copy of the camera with no distortion.
  undistorted_camera = camera.Copy()
  undistorted_camera.SetRadialDistortion(0, 0, 0)
  undistorted_camera.SetTangentialDistortion(0, 0)
  if set_ideal_pp:
    undistorted_camera.SetIdealPrincipalPoint()

  new_camera = undistorted_camera
  if crop_blank_pixels:
    new_camera = crop_camera_blank_pixels(camera, undistorted_camera)

  return new_camera


def undistort_image(image, camera, set_ideal_pp=True, crop_blank_pixels=False):
  if isinstance(camera, cam.Camera):
    camera = camera.to_sfm_camera()
  undistorted_camera = undistort_camera(
      camera, crop_blank_pixels, set_ideal_pp=set_ideal_pp)
  undistorted_image = warp_image.warp_image_at_infinity(
      camera, undistorted_camera, image, mode='constant')
  return undistorted_image, undistorted_camera


# In[ ]:


# @title Metrics

lpips_model = hub.load('...')

  
def compute_ssim(target, pred):
  target = tf.convert_to_tensor(target)
  pred = tf.convert_to_tensor(pred)
  return tf.image.ssim_multiscale(target, pred, max_val=1.0)


def compute_lpips(target, pred):
    target = tf.convert_to_tensor(target)
    pred = tf.convert_to_tensor(pred)
    rgb_tensor_batch = tf.expand_dims(pred, axis=0)
    target_tensor_batch = tf.expand_dims(target, axis=0)
    return lpips_model(rgb_tensor_batch, target_tensor_batch)


def compute_mse(x, y, sample_weight=None):
  """Calculates MSE loss.

  Args:
    x: [..., 3] float32. RGB.
    y: [..., 3] float32. RGB.
    sample_weight: [...] float32. Per-color weight.

  Returns:
    scalar float32. Average squared error across all entries in x, y.
  """
  if sample_weight is None:
    return np.mean((x - y)**2)

  if sample_weight.ndim == 2:
    sample_weight = sample_weight[..., None]
  sample_weight = np.broadcast_to(sample_weight, x.shape)
  diff = ((x - y)*sample_weight)
  numer = np.sum(diff ** 2)
  denom = np.sum(sample_weight)
  return numer / denom


def mse_to_psnr(x):
  # return -10. * np.log10(x) / np.log10(10.)
  return 10 * np.log10(1 / x)


# In[ ]:


from collections import defaultdict

experiment_root = gpath.GPath()
dataset_root = gpath.GPath()

dataset_names = {
}

eval_scales = defaultdict(lambda: 1)

experiments = {}

scale =    4#@param {type:"number"}
use_mask = False  # @param {type:"boolean"}


# In[ ]:


work_dir = gpath.GPath()
work_dir.mkdir(exist_ok=True, parents=True)


# In[ ]:



def load_dataset_dict(name, scale=scale, use_images=True):
  print(f'Loading {name}...')
  dataset_dir = dataset_root / name

  image_path = dataset_dir / f'rgb/{scale}x'
  mask_path = dataset_dir / f'mask/{scale}x'
  colmap_mask_path = dataset_dir / f'mask-colmap/{scale}x'

  with (dataset_dir / 'dataset.json').open('rt') as f:
    dataset_json = json.load(f)

  ds = datasets.NerfiesDataSource(dataset_dir, image_scale=scale, camera_type='proto')
  train_ids = ds.train_ids
  val_ids = ds.val_ids
  val_cameras = utils.parallel_map(ds.load_camera, val_ids)

  out = {
      'name': name,
      'scale': scale,
      'train_ids': train_ids,
      'val_ids': val_ids,
      'val_cameras': val_cameras,
  }
  if use_images:
      # out['train_rgbs'] = load_images([image_path / f'{x}.png' for x in dataset_json['train_ids']])
      out['val_rgbs'] = load_images([image_path / f'{x}.png' for x in dataset_json['val_ids']])
  return out


def load_experiment_images(dataset_name, sweep_name, exp_name, item_ids, seed=0, idx=-1):
  exp_dir = gpath.GPath(experiment_root, sweep_name, dataset_name, exp_name, f'{seed}')
  print(f'Loading experiment images from {exp_dir}')
  renders_dir = exp_dir / 'renders'
  renders_dir = sorted(renders_dir.iterdir())[idx]
  print(f'Experiment step = {int(renders_dir.name)}')
  val_renders_dir = renders_dir / 'val'
  val_renders = [val_renders_dir / f'rgb_{item_id}.png' for item_id in item_ids]
  if any(not x.exists() for x in val_renders):
    return []
  return utils.parallel_map(load_image, val_renders, show_pbar=False)


def compute_experiment_metrics(target_images, pred_images, eval_scale, crop=False):
  if eval_scale > 1:
    scale_fn = lambda x: image_utils.downsample_image(image_utils.make_divisible(x, eval_scale), eval_scale)
  elif eval_scale < 1:
    scale_fn = lambda x: image_utils.upsample_image(image_utils.make_divisible(x, eval_scale), int(1/eval_scale))
  if eval_scale != 1:
    print(f'Downsampling images by a factory of {eval_scale}')
    target_images = utils.parallel_map(scale_fn, target_images)
    pred_images = utils.parallel_map(scale_fn, pred_images)

  metrics_dict = defaultdict(list)
  for i, (target, pred) in enumerate(zip(ProgressIter(target_images), pred_images)):
    if i == 0:
      mediapy.show_images([target, pred], titles=['target', 'pred'])
    mse = compute_mse(target, pred)
    psnr = mse_to_psnr(mse)
    ssim = compute_ssim(target, pred)
    lpips = compute_lpips(target, pred)
    metrics_dict['psnr'].append(float(psnr))
    metrics_dict['lpips'].append(float(lpips))
    metrics_dict['ssim'].append(float(ssim))
  
  return metrics_dict


def save_experiment_images(save_dir, images, item_ids):
  save_dir.mkdir(exist_ok=True, parents=True)
  save_paths = [save_dir / f'{i}.png' for i in item_ids]
  utils.parallel_map(
      lambda x: image_utils.save_image(x[0], image_utils.image_to_uint8(x[1])),
      list(zip(save_paths, images)))


def summarize_metrics(metrics_dict):
  return {
      k: np.mean(v) for k, v in metrics_dict.items()
  }


# ## Compute NeRF metrics.

# In[ ]:


def load_nerf_images(dataset_dict, sweep_name, exp_name):
  dataset_name = dataset_dict['name']
  print(f'Computing metrics for {dataset_name} / {sweep_name} / {exp_name}')
  item_ids = dataset_dict['val_ids']
  target_images = dataset_dict['val_rgbs']
  idx = -1
  while idx > -5:
    try:
      pred_images = load_experiment_images(dataset_name, sweep_name, exp_name, item_ids, idx=idx)
      idx -= 1
    except (FileNotFoundError, ValueError):
      print(f'Latest renders not ready, choosing previous')
      pred_images = []
    if len(pred_images) == len(target_images):
      break
  
  if len(pred_images) < len(target_images):
    raise RuntimeError('Images are not ready.')
  
  return target_images, pred_images


for dataset_name in dataset_names:
  eval_scale = eval_scales[dataset_name]
  print(f'{dataset_name}, eval_scale={eval_scale }')
  dataset_dict = None
  for sweep_name, exp_name in experiments:
    if sweep_name == 'none':
      continue
    print(f'Processing {dataset_name} / {sweep_name} / {exp_name}')
    cache_dir = work_dir / dataset_name / sweep_name / exp_name
    cache_dir.mkdir(exist_ok=True, parents=True)
    metrics_path = cache_dir / 'metrics.json'
    # Load existing metrics if they exist.
    if False and metrics_path.exists():
      print(f'Loading cached metrics from {metrics_path}')
      with metrics_path.open('rt') as f:
        metrics_dict = json.load(f)
    else:
      # Lazily load dataset dict.
      if dataset_dict is None:
        dataset_dict = load_dataset_dict(dataset_name)
      
      # Compute metrics.
      target_images, pred_images = load_nerf_images(
          dataset_dict, sweep_name, exp_name)

      save_experiment_images(cache_dir / 'target_images', target_images, dataset_dict['val_ids'])
      save_experiment_images(cache_dir / 'pred_images', pred_images, dataset_dict['val_ids'])
      metrics_dict = compute_experiment_metrics(target_images, pred_images, eval_scale=eval_scale)
      with metrics_path.open('wt') as f:
        json.dump(metrics_dict, f, indent=2)
    
    print(summarize_metrics(metrics_dict))
    print()


# ## Create table

# In[ ]:



table_experiments = {}

dataset_groups = [{}, {}]
table_dataset_names = {}
for i, group in enumerate(dataset_groups):
  table_dataset_names.update(group)
  table_dataset_names[f'mean{i}'] = 'Mean'


# In[ ]:


def load_metric(key):
  dataset_name, sweep_name, exp_name = key
  cache_dir = work_dir / dataset_name / sweep_name / exp_name
  metric_path = cache_dir / 'metrics.json'
  with metric_path.open('rt') as f:
    return json.load(f)


metric_keys = []
for dataset_name in dataset_names:
  for sweep_name, exp_name in table_experiments:
    metric_keys.append((dataset_name, sweep_name, exp_name))

metrics_list = utils.parallel_map(load_metric, metric_keys, show_pbar=True)
metrics_by_key = {k: v for k, v in zip(metric_keys, metrics_list)}


# In[ ]:


# Create nested dict.
exp_mappings = {e: v for (_, e), v in table_experiments.items()}

dataset_metrics_dict = collections.defaultdict(dict)
for (dataset_name, sweep_name, exp_name), metric_dict in metrics_by_key.items():
  dataset_metrics_dict[dataset_name][exp_name] = summarize_metrics(metric_dict)
dataset_dicts = {name: load_dataset_dict(name, use_images=False) for name in dataset_names}


# In[ ]:


# @title Table template
import jinja2
from jinja2 import Template

env = jinja2.Environment(
	trim_blocks = True,
	autoescape = False,
  lstrip_blocks = True
)

template = env.from_string("""
%%%% AUTOMATICALLY GENERATED, DO NOT EDIT.

\\begin{tabular}{l|{% for _ in dataset_names %}|{% for _ in metric_names %}c{% endfor %}{% endfor %}}

\\toprule
% Table Header (datasets).
{% for dataset_name in dataset_names %}
& \\multicolumn{ {{metric_names|length}} }{c}{
  \\makecell{
  \\textsc{\\small {{dataset_names[dataset_name]}} }
  {% if 'mean' not in dataset_name %}
    \\\\({{ datasets[dataset_name]['val_ids']|length }} images)
  {% endif %}
  }
}
{% endfor %}
\\\\

% Table header (metrics).
{% for dataset_name in dataset_names %}
  {% for metric_name in metric_names.values() %}
    & \\multicolumn{1}{c}{ \\footnotesize {{metric_name}} }
  {% endfor %}
{% endfor %}
\\\\
\\hline

% Table contents.
{% for exp_k, exp_name in experiment_names.items() %}
  {% set exp_i = loop.index0 %}
  {{exp_name}}
  {% for dataset_key, dataset_name in dataset_names.items() %}
    {%- for metric_key in metric_names -%}
      {% set metrics = dataset_metrics[dataset_key][metric_key] %}
      {% if metric_key != 'lpips'%}
        {% set rank = (-metrics).argsort().argsort()[exp_i] %}
      {% else %}
        {% set rank = metrics.argsort().argsort()[exp_i] %}
      {% endif %}
      &
      {%- if rank == 0 -%}
        \\tablefirst
      {%- elif rank == 1 -%}
        \\tablesecond
      {%- elif rank == 2 -%}
        \\tablethird
      {%- endif -%}
      ${{"{:#.03g}".format(metrics[exp_i]).lstrip('0')}}$
    {% endfor %}  
  {% endfor %}
  \\\\
{% endfor %}
\\bottomrule

\\end{tabular}

%%%% AUTOMATICALLY GENERATED, DO NOT EDIT.
""")


# In[ ]:


from pprint import pprint
# @title Generate table

use_psnr = True # @param{type:'boolean'}
use_ssim = False # @param{type:'boolean'}
use_lpips = True # @param{type:'boolean'}


METRIC_NAMES = {
    'psnr': 'PSNR$\\uparrow$', 
    'ssim': 'MS-SSIM$\\uparrow$', 
    'lpips': 'LPIPS$\\downarrow$',
}
if not use_psnr:
  del METRIC_NAMES['psnr']
if not use_ssim:
  del METRIC_NAMES['ssim']
if not use_lpips:
  del METRIC_NAMES['lpips']


table_metrics = {}
for group_id, dataset_group in enumerate(dataset_groups):
  print(group_id)
  group_metrics = collections.defaultdict(dict)
  for dataset_k in dataset_group:
    for metric_k in ['psnr', 'ssim', 'lpips']:
      metric_v = np.array([dataset_metrics_dict[dataset_k][exp_k][metric_k].mean() 
                          for exp_k in exp_mappings])
      group_metrics[dataset_k][metric_k] = metric_v

  group_metrics[f'mean{group_id}'] = {
      m: np.stack([group_metrics[dk][m] for dk in dataset_group], axis=0).mean(axis=0)
      for m in METRIC_NAMES
  }
  print(group_metrics.keys())
  for k, v in group_metrics.items():
    table_metrics[k] = v
 
table_str = template.render(
    datasets=dataset_dicts,
    dataset_names=table_dataset_names,
    dataset_metrics=table_metrics,
    experiment_names=exp_mappings,
    metric_names=METRIC_NAMES
).replace('    ','')
print(table_str)


# ## Choose visualizations

# In[ ]:


stride = 1
dataset_name = 'vrig/broom2'
dataset_dict = load_dataset_dict(dataset_name, use_images=True)
mediapy.show_images(
    dataset_dict['val_rgbs'][::stride], 
    titles=dataset_dict['val_ids'][::stride])


# In[ ]:


print(d['val_ids'][10])


# In[ ]:


dataset_qual_ids = {
}


# In[ ]:


def load_dataset_image(dataset_name, item_id, scale=scale):
  dataset_dir = dataset_root / dataset_name
  image_path = dataset_dir / f'rgb/{scale}x'
  return load_image(image_path / f'{item_id}.png')


out_dir = ''
out_dir.mkdir(exist_ok=True, parents=True)

for dataset_name in dataset_names:
  if dataset_name not in dataset_qual_ids:
    continue
  val_id = dataset_qual_ids[dataset_name]
  if 'right' in val_id:
    train_id = val_id.replace('right', 'left')
  else:
    train_id = val_id.replace('left', 'right')

  ds = datasets.NerfiesDataSource(dataset_root / dataset_name, image_scale=4, camera_type='proto')
  train_rgb = ds.load_rgb(train_id)
  train_camera = ds.load_camera(train_id)
  val_rgb = ds.load_rgb(val_id)
  val_camera = ds.load_camera(val_id)

  train_rgb = nv_crop(undistort_image(train_rgb, train_camera)[0])
  val_rgb = nv_crop(undistort_image(val_rgb, val_camera)[0])

  save_name = dataset_name.split('/')[-1]
  mediapy.write_image(str(out_dir / f'{save_name}.train.jpg'), (train_rgb))
  mediapy.write_image(str(out_dir / f'{save_name}.valid.jpg'), (val_rgb))
  for sweep_name, exp_name in experiments:
    cache_dir = work_dir / dataset_name / sweep_name / exp_name
    print(cache_dir)
    pred_rgb = load_image(cache_dir / 'pred_images' / f'{val_id}.png')
    scale = eval_scales[dataset_name]

    mediapy.show_images([train_rgb, val_rgb, pred_rgb])
    mediapy.write_image(str(out_dir / f'{save_name}.{exp_name}.pred.jpg'), pred_rgb)


# ### Little number in figure

# In[ ]:


exp_names_to_show = [
]

datasets_to_show = [
]


def load_dataset_image(dataset_name, item_id, scale=scale):
  dataset_dir = dataset_root / dataset_name
  image_path = dataset_dir / f'rgb/{scale}x'
  return load_image(image_path / f'{item_id}.png')


for dataset_name in datasets_to_show:
  val_id = dataset_qual_ids[dataset_name]
  if 'right' in val_id:
    train_id = val_id.replace('right', 'left')
  else:
    train_id = val_id.replace('left', 'right')
    
  scale = eval_scales[dataset_name]
    
  train_rgb = load_dataset_image(dataset_name, train_id, 4)
  print(train_rgb.shape)
  
  for sweep_name, exp_name in experiments:
    if exp_name not in exp_names_to_show:
      print('Skipping', exp_name)
      continue

    cache_dir = work_dir / dataset_name / sweep_name / exp_name
    print(cache_dir)
    target_rgb = load_image(cache_dir / 'target_images' / f'{val_id}.png')
    pred_rgb = load_image(cache_dir / 'pred_images' / f'{val_id}.png')
    psnr = mse_to_psnr(compute_mse(target_rgb, pred_rgb))
    lpips = compute_lpips(target_rgb, pred_rgb)
    print(f'psnr: {float(psnr):#.03g}')
    print(f'lpips: {float(lpips):#.03g}')
    
    mediapy.show_images([target_rgb, pred_rgb])

    print()
    print()
    print()

