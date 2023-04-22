#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# @title Imports

from dataclasses import dataclass
from pprint import pprint
from typing import Any, List, Callable, Dict, Sequence, Optional, Tuple
from io import BytesIO
from IPython.display import display, HTML
from base64 import b64encode
import PIL
import IPython
import tempfile
import imageio

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

import jax
from jax.config import config as jax_config
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

import flax
import flax.linen as nn
# from flax import nn
from flax import jax_utils
from flax import optim
from flax.metrics import tensorboard
from flax.training import checkpoints

from absl import logging

# Monkey patch logging.
def myprint(msg, *args, **kwargs):
 print(msg % args)

logging.info = myprint 
logging.warn = myprint


# ## Utility Functions

# In[ ]:


# @title Dataset Utilities

def prepare_tf_data(xs):
  """Convert a input batch from tf Tensors to numpy arrays."""
  local_device_count = jax.local_device_count()
  def _prepare(x):
    # Use _numpy() for zero-copy conversion between TF and NumPy.
    x = x._numpy()  # pylint: disable=protected-access

    # reshape (host_batch_size, height, width, 3) to
    # (local_devices, device_batch_size, height, width, 3)
    return x.reshape((local_device_count, -1) + x.shape[1:])

  return jax.tree_map(_prepare, xs)


def iterator_from_dataset(dataset: tf.data.Dataset,
                          batch_size: int,
                          repeat: bool = True,
                          prefetch_size: int = 0,
                          devices: Optional[Sequence[Any]] = None):
  """Create a data iterator that returns JAX arrays from a TF dataset.

  Args:
    dataset: the dataset to iterate over.
    batch_size: the batch sizes the iterator should return.
    repeat: whether the iterator should repeat the dataset.
    prefetch_size: the number of batches to prefetch to device.
    devices: the devices to prefetch to.

  Returns:
    An iterator that returns data batches.
  """
  if repeat:
    dataset = dataset.repeat()

  if batch_size > 0:
    dataset = dataset.batch(batch_size)
    it = map(prepare_tf_data, dataset)
  else:
    it = map(prepare_tf_data_unbatched, dataset)

  if prefetch_size > 0:
    it = jax_utils.prefetch_to_device(it, prefetch_size, devices)

  return it


# ## Create Data

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import matplotlib as mpl


def fig_to_array(fig, height, width, dpi=100):
  out_array = np.zeros((height, width, 4), dtype=np.uint8)
  fig.set_size_inches((width / dpi, height / dpi))
  fig.set_dpi(dpi)
  for ax in fig.axes:
    ax.margins(0, 0)
    ax.axis('off')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

  # If we haven't already shown or saved the plot, then we need to
  # draw the figure first...
  fig.tight_layout(pad=0)
  fig.canvas.draw()

  # Now we can save it to a numpy array.
  data = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
  plt.close()

  np.copyto(out_array, data)
  out_array = np.roll(out_array, -1, 2)
  return out_array


def add_colorwheel(fig, cmap, lim, label):
  display_axes = fig.add_axes([0.1,0.1,0.8,0.8], projection='polar', label=label)
  display_axes._direction = 2*np.pi ## This is a nasty hack - using the hidden field to 
                                    ## multiply the values such that 1 become 2*pi
                                    ## this field is supposed to take values 1 or -1 only!!
  
  norm = mpl.colors.Normalize(0.0, 2*np.pi)
  
  # Plot the colorbar onto the polar axis
  # note - use orientation horizontal so that the gradient goes around
  # the wheel rather than centre out
  quant_steps = 2056
  cb = mpl.colorbar.ColorbarBase(display_axes, cmap=cm.get_cmap(cmap,quant_steps),
                                norm=norm,
                                orientation='horizontal')
  
  # aesthetics - get rid of border and axis labels                                   
  cb.outline.set_visible(False)                                 
  display_axes.set_axis_off()
  display_axes.set_rlim(lim)

  display_axes.margins(0, 0)
  display_axes.axis('off')
  display_axes.get_xaxis().set_visible(False)
  display_axes.get_yaxis().set_visible(False)
  display_axes.set_aspect(1)

fig = plt.figure()
add_colorwheel(fig, 'tab20b', (-1.1, 1.0), 'a')
add_colorwheel(fig, 'twilight', (-2.0, 1), 'b')
add_colorwheel(fig, 'hsv', (-5.0, 1), 'c')

colorwheel = fig_to_array(fig, 400, 400)
mediapy.show_image(colorwheel)


# In[ ]:


# import matplotlib
import scipy
from PIL import Image, ImageDraw, ImageFont, ImageOps

def barron_colormap(n):
  curve = lambda x, s, t: np.where(x<t, (t*x)/(x+s*(t-x)+1e-10), ((1-t)*(x-1))/(1-x-s*(t-x)+1e-10)+1)
  t = curve(jnp.linspace(0, 1, n), 1.5, 0.5)
  colors = plt.get_cmap('rainbow')(t)[:,:3]
  colors = 0.85 * (1 - (1-colors) / jnp.sqrt(jnp.sum((1-colors)**2, 1, keepdims=True)))
  return colors


def make_doodle(height, width, tile_size=5):
  h = height + (height - 1) % 2
  w = width + (width - 1) % 2
  colors = np.array(barron_colormap(h * w))
  colors[::2, :] = 0.0
  colors = colors.reshape((h, w, 3))
  colors = colors[:h, :w]
  colors = np.repeat(colors, tile_size, axis=0)
  colors = np.repeat(colors, tile_size, axis=1)
  return np.asarray(colors)


# doodle = make_doodle(20, 20, 4)
# files = %upload_files
# doodles = []
# for f in files.values():
#   doodles.append(imageio.imread(f))
doodles = [image_utils.load_image('')]
mediapy.show_images(doodles)


# In[ ]:


from PIL import ImageChops, ImageFilter
import math
import skimage
from skimage.transform import swirl

# def random_image(rng, shape, doodles):
#   h, w = shape[:2]
#   num_doodles = len(doodles)
#   image = Image.new('RGBA', shape[::-1], color=(127, 127, 127))
#   for i, doodle in enumerate(doodles):
#     dh, dw = doodle.shape[:2]
#     rng, k1, k2, k3 = random.split(rng, 4)
#     angle = random.uniform(k1, (1,), minval=-180, maxval=180)[0]
#     y = random.randint(k2, (1,), minval=0, maxval=h-int(math.sqrt(dh**2+dw**2)))
#     x = random.randint(k3, (1,), minval=0, maxval=w-int(math.sqrt(dh**2+dw**2)))
#     doodle = Image.fromarray(image_utils.image_to_uint8(doodle)).convert('RGBA')
#     doodle = doodle.rotate(angle, expand=True, resample=Image.BICUBIC)
#     image.paste(doodle, (x, y), mask=doodle)
#   return np.asarray(image.convert('RGB')).astype(np.float32) / 255.0


def crop_circle(array, blur_radius=0, offset=0):
  pil_img = Image.fromarray(image_utils.image_to_uint8(array))
  offset = blur_radius * 2 + offset
  mask = Image.new("L", pil_img.size, 0)
  draw = ImageDraw.Draw(mask)
  draw.ellipse((offset, offset, pil_img.size[0] - offset, pil_img.size[1] - offset), fill=255)
  mask = mask.filter(ImageFilter.GaussianBlur(blur_radius))

  result = pil_img.copy()
  result.putalpha(mask)

  return np.asarray(result)


def random_image(rng, shape, frame, doodle, shift=10):
  rng, k1, k2, k3, k4 = random.split(rng, 5)

  h, w = shape
  dh, dw = doodle.shape[:2]

  rng, swirl_key1, swirl_key2, swirl_key3 = random.split(rng, 4)
  swirl_angle = random.uniform(swirl_key1, (1,), minval=-0, maxval=0)[0]
  swirl_strength = random.uniform(swirl_key2, (1,), minval=-6, maxval=6)[0]
  swirl_center_dx, swirl_center_dy = random.uniform(swirl_key3, (2,), minval=-20, maxval=20)
  doodle = swirl(doodle, 
                 rotation=swirl_angle, 
                 strength=swirl_strength, 
                 radius=200,
                 center=(dh//2+swirl_center_dx, dw//2+swirl_center_dy))
  doodle = crop_circle(doodle)

  angle = random.uniform(k1, (1,), minval=-45, maxval=45)[0]
  doodle = Image.fromarray(image_utils.image_to_uint8(doodle))
  doodle = doodle.copy()
  # doodle = doodle.rotate(angle, expand=True, resample=Image.BICUBIC)
  frame = Image.fromarray(image_utils.image_to_uint8(frame))
  doodle_x = int(frame.width / 2 - doodle.width / 2)
  doodle_y = int(frame.height / 2 - doodle.height / 2)
  frame.paste(doodle, (doodle_x, doodle_y), mask=doodle)

  image = Image.new('RGBA', shape[::-1], color=(50, 50, 50))
  angle = random.uniform(k4, (1,), minval=-180, maxval=180)[0]
  cx = int(image.width / 2 - frame.width / 2)
  cy = int(image.height / 2 - frame.height / 2)
  y = random.randint(k2, (1,), minval=cx-shift, maxval=cx+shift)
  x = random.randint(k3, (1,), minval=cy-shift, maxval=cy+shift)

  frame = frame.rotate(angle, expand=False, resample=Image.BICUBIC)
  image.paste(frame, (x, y), mask=frame)

  return image_utils.image_to_float32(np.asarray(image))




num_images = 20
image_shape = (400, 400)
doodle_scale = 1/7
# doodles = [skimage.data.astronaut()]
rescaled_doodles = [image_utils.rescale_image(d, doodle_scale) for d in doodles]
imedia.show_images(rescaled_doodles)
shift = 40
seed = 6
rng = random.PRNGKey(seed)

images = []
for i in range(num_images):
  rng, key = random.split(rng)
  images.append(random_image(key, image_shape, colorwheel, rescaled_doodles[0], shift=shift))

imedia.show_images(images, border=True, columns=5)


# In[ ]:


for image_idx, image in enumerate(images):
  path = gpath.GPath('', f'{image_idx:02d}.png')
  path.parent.mkdir(exist_ok=True, parents=True)
  image_utils.save_image(path, image_utils.image_to_uint8(image))


# In[ ]:


def image_to_data(image, idx, scale=1.0):
  coords = jnp.stack(jnp.meshgrid(
    jnp.linspace(-scale, scale, image.shape[1]),
    jnp.linspace(-scale, scale, image.shape[0]),
  ), axis=-1)
  return {
      'coords': coords,
      'colors': image[..., :3],
      'ids': jnp.full_like(image[:, :, 0], idx, dtype=jnp.uint32),
  }

data_items = [image_to_data(image, i) for i, image in enumerate(images)]
data = jax.tree_multimap(lambda *x: jnp.stack(x), *data_items)


# In[ ]:


imedia.show_images([data['coords'][0, ..., 0], data['coords'][0, ..., 1]], cmap='coolwarm')


# ## Train

# In[ ]:


# @title MLP

class MLP(nn.Module):
  """Basic MLP class with hidden layers and an output layers."""
  depth: int
  width: int
  hidden_init: Any
  hidden_activation: Any = nn.relu
  output_init: Optional[Any] = None
  output_channels: int = 0
  output_activation: Optional[Any] = lambda x: x
  use_bias: bool = True
  skips: Tuple[int] = tuple()

  @nn.compact
  def __call__(self, x):
    inputs = x
    for i in range(self.depth):
      layer = nn.Dense(
          self.width,
          use_bias=self.use_bias,
          kernel_init=self.hidden_init,
          name=f'hidden_{i}')
      if i in self.skips:
        x = jnp.concatenate([x, inputs], axis=-1)
      x = layer(x)
      x = self.hidden_activation(x)

    if self.output_channels > 0:
      logit_layer = nn.Dense(
          self.output_channels,
          use_bias=self.use_bias,
          kernel_init=self.output_init,
          name='logit')
      x = logit_layer(x)
      if self.output_activation is not None:
        x = self.output_activation(x)

    return x


# In[ ]:


# @title SinusoidalEncoder
 

class SinusoidalEncoder(nn.Module):
  """A vectorized sinusoidal encoding.

  Attributes:
    num_freqs: the number of frequency bands in the encoding.
    max_freq_log2: the log (base 2) of the maximum frequency.
    scale: a scaling factor for the positional encoding.
    use_identity: if True use the identity encoding as well.
  """
  num_freqs: int
  min_freq_log2: int = 0
  max_freq_log2: Optional[int] = None
  scale: float = 1.0
  use_identity: bool = True

  def setup(self):
    if self.max_freq_log2 is None:
      max_freq_log2 = self.num_freqs - 1.0
    else:
      max_freq_log2 = self.max_freq_log2
    self.freq_bands = 2.0**jnp.linspace(self.min_freq_log2, max_freq_log2, int(self.num_freqs))

    # (F, 1).
    self.freqs = jnp.reshape(self.freq_bands, (self.num_freqs, 1))

  def __call__(self, x, alpha: Optional[float] = None):
    """A vectorized sinusoidal encoding.

    Args:
      x: the input features to encode.
      alpha: a dummy argument for API compatibility.

    Returns:
      A tensor containing the encoded features.
    """
    if self.num_freqs == 0:
      return x

    x_expanded = jnp.expand_dims(x, axis=-2)  # (1, C).
    # Will be broadcasted to shape (F, C).
    angles = self.scale * x_expanded * self.freqs

    # The shape of the features is (F, 2, C) so that when we reshape it
    # it matches the ordering of the original NeRF code.
    # Vectorize the computation of the high-frequency (sin, cos) terms.
    # We use the trigonometric identity: cos(x) = sin(x + pi/2)
    features = jnp.stack((angles, angles + jnp.pi / 2), axis=-2)
    features = features.flatten()
    features = jnp.sin(features)

    # Prepend the original signal for the identity.
    if self.use_identity:
      features = jnp.concatenate([x, features], axis=-1)
    return features


class AnnealedSinusoidalEncoder(nn.Module):
  """An annealed sinusoidal encoding."""
  num_freqs: int
  min_freq_log2: int = 0
  max_freq_log2: Optional[int] = None
  scale: float = 1.0
  use_identity: bool = False

  @nn.compact
  def __call__(self, x, alpha):
    if alpha is None:
      raise ValueError('alpha must be specified.')
    if self.num_freqs == 0:
      return x

    num_channels = x.shape[-1]

    base_encoder = SinusoidalEncoder(
        num_freqs=self.num_freqs,
        min_freq_log2=self.min_freq_log2,
        max_freq_log2=self.max_freq_log2,
        scale=self.scale,
        use_identity=self.use_identity)
    features = base_encoder(x)

    if self.use_identity:
      identity, features = jnp.split(features, (x.shape[-1],), axis=-1)

    # Apply the window by broadcasting to save on memory.
    features = jnp.reshape(features, (-1, 2, num_channels))
    window = self.cosine_easing_window(
        self.min_freq_log2, self.max_freq_log2, self.num_freqs, alpha)
    window = jnp.reshape(window, (-1, 1, 1))
    features = window * features

    if self.use_identity:
      return jnp.concatenate([
          identity,
          features.flatten(),
      ], axis=-1)
    else:
      return features.flatten()

  @classmethod
  def cosine_easing_window(cls, min_freq_log2, max_freq_log2, num_bands, alpha):
    """Eases in each frequency one by one with a cosine.

    This is equivalent to taking a Tukey window and sliding it to the right
    along the frequency spectrum.

    Args:
      num_freqs: the number of frequencies.
      alpha: will ease in each frequency as alpha goes from 0.0 to num_freqs.

    Returns:
      A 1-d numpy array with num_sample elements containing the window.
    """
    bands = jnp.linspace(min_freq_log2,  max_freq_log2, num_bands)
    x = jnp.clip(alpha - bands, 0.0, 1.0)
    return 0.5 * (1 + jnp.cos(jnp.pi * x + jnp.pi))


# In[ ]:


devices = jax.devices()
rng = random.PRNGKey(1)
rng, key = random.split(rng, 2)

template_min_freq = -2.0
template_max_freq = 8.0
template_num_freqs = int(template_max_freq - template_min_freq + 1)
scale = 1.0
deform_type = 'translation'

# Anneal.
# use_anneal = True
# min_freq = -2.0
# max_freq = 4.0

# No anneal.
use_anneal = False
min_freq = -2.0
max_freq = -2.0

# Common.
num_freqs = int(max_freq - min_freq + 1)


VModel = nn.vmap(Model, 
                 in_axes=(0, 0, None, None),
                 variable_axes={'params': None},  
                 split_rngs={'params': False})
model = VModel(num_glo_embeddings=len(images),
               deform_type=deform_type,
               deform_min_freq_log2=min_freq,
               deform_max_freq_log2=max_freq,
               deform_num_freqs=num_freqs,
               template_num_freqs=template_num_freqs,
               template_min_freq_log2=template_min_freq,
               template_max_freq_log2=template_max_freq,
               scale=scale)
init_coords = random.normal(key, (1024, 2))
init_glo_ids = jnp.zeros((1024,), jnp.uint32)
init_params = model.init(key, init_coords, init_glo_ids, 0.0, True)


# In[ ]:


# Setup dataset
def prepare_data(xs):
  """Convert a input batch from tf Tensors to numpy arrays."""
  local_device_count = jax.local_device_count()
  def _prepare(x):
    # reshape (host_batch_size, height, width, 3) to
    # (local_devices, device_batch_size, height, width, 3)
    x = x.reshape((local_device_count, -1) + x.shape[1:])
    return jax.api.device_put_sharded(list(x), jax.local_devices())

  return jax.tree_map(_prepare, xs)


dataset = {
    'coords': data['coords'].reshape((-1, 2)),
    'colors': data['colors'].reshape((-1, 3)),
    'ids': data['ids'].reshape((-1, 1)),
}

num_items = np.prod(data['ids'].shape)
perm = random.permutation(rng, jnp.arange(num_items))
dataset = jax.tree_map(lambda x: x[perm], dataset)
dataset = prepare_data(dataset)


# In[ ]:


def compute_psnr(mse):
  return -10. * jnp.log(mse) / jnp.log(10.)


def loss_fn(params, batch, alpha):
  color_pred = model.apply(params, batch['coords'], batch['ids'].squeeze(-1), alpha, True)['color']
  loss = (color_pred - batch['colors']) ** 2
  loss = loss.mean()
  return loss


@jax.jit
def train_step(optimizer, batch, key, alpha, lr):
  _, key = random.split(rng, 2)
  grad_fn = jax.value_and_grad(loss_fn, argnums=0)
  loss, grad = grad_fn(optimizer.target, batch, alpha)
  loss = jax.lax.pmean(loss, axis_name='batch')
  grad = jax.lax.pmean(grad, axis_name='batch')
  new_optimizer = optimizer.apply_gradient(grad, learning_rate=lr)
  return new_optimizer, key, loss


def render_item(model, params, item):
  target_image = item['colors']
  pred = model.apply(
      jax_utils.unreplicate(params), 
      item['coords'].reshape((-1, 2)),
      item['ids'].reshape((-1,)),
      alpha,
      True)
  pred_image = pred['color'].reshape(target_image.shape)
  pred_flow = flow_to_image(np.array(pred['flow'].reshape((*image_shape, 2))))
  pred_template = model.apply(
      jax_utils.unreplicate(optimizer.target), 
      item['coords'].reshape((-1, 2)),
      item['ids'].reshape((-1,)),
      alpha,
      False)['color'].reshape(target_image.shape)

  return {
      'target_color': target_image,
      'pred_color': pred_image,
      'pred_flow': pred_flow,
      'pred_template': pred_template,
  }


# Train.
max_iters = 10000
# lr_schedule = schedules.ConstantSchedule(8e-3)
lr_schedule = schedules.from_config({
  'type': 'delayed',
  'delay_steps': 50,
  'delay_mult': 0.01,
  'base_schedule': {
    'type': 'exponential',
    'initial_value': 8e-3,
    'final_value': 8e-5,
    'num_steps': max_iters,
  },
})
if use_anneal:
  alpha_schedule = schedules.LinearSchedule(
      model.deform_min_freq_log2, model.deform_max_freq_log2+1, 5000)
else:
  alpha_schedule = schedules.ConstantSchedule(model.deform_max_freq_log2+1)


optimizer_def = optim.Adam(lr_schedule(0))
optimizer = optimizer_def.create(init_params)
optimizer = jax_utils.replicate(optimizer, devices)

p_train_step = jax.pmap(
    train_step, axis_name='batch', devices=devices, in_axes=(0, 0, 0, None, None))

tt = utils.TimeTracker()

show_idx = 0
keys = random.split(rng, len(devices))
tt.tic('data')
for i in range(max_iters):
  batch = dataset
  tt.toc('data')
  if i > max_iters:
    break
  alpha = alpha_schedule(i)
  lr = lr_schedule(i)
  with tt.record_time('p_train_step'):
    optimizer, keys, losses = p_train_step(optimizer, batch, keys, alpha, lr)
  if i % 10 == 0:
    loss = jax_utils.unreplicate(losses)
    psnr = compute_psnr(loss)
    print(f'{i}: lr = {lr:.04f}, alpha = {alpha:.02f}, loss = {loss:.04f}, psnr = {psnr:.02f}', end='\r')

  if i % 500 == 0:
    print(f'{i}: loss = {loss:.04f}, psnr = {psnr:.02f}')
    print(f'Showing image {show_idx}')
    item = data_items[show_idx]
    render = render_item(model, optimizer.target, item)
    imedia.show_images([render['target_color'], 
                        render['pred_color'], 
                        jnp.abs(render['target_color'] - render['pred_color']), 
                        render['pred_template'], 
                        render['pred_flow']],
                        titles=['GT', 'Pred', 'Abs. Error', 'Template', 'Flow'])
    del render
    show_idx = (show_idx + 1) % len(images)
  tt.tic('data')


# In[ ]:


batch['ids'].shape


# In[ ]:


def _save_image(path, image):
  image = image_utils.image_to_uint8(image)
  image_utils.save_image(path, image)

exp_name = f'spaceman_{deform_type}_{"annealed" if use_anneal else "notannealed"}_{min_freq}_to_{max_freq}'
save_dir = gpath.GPath('', exp_name)
save_dir.mkdir(exist_ok=True, parents=True)


for i, item in enumerate(data_items):
  render = render_item(model, optimizer.target, item)
  _save_image(save_dir / f'{i:02d}_target_color.png', render['target_color'])
  _save_image(save_dir / f'{i:02d}_pred_color.png', render['pred_color'])
  _save_image(save_dir / f'{i:02d}_pred_template.png', render['pred_template'])
  _save_image(save_dir / f'{i:02d}_pred_flow.png', render['pred_flow'])
  imedia.show_images([render['target_color'], 
                      render['pred_color'], 
                      jnp.abs(render['target_color'] - render['pred_color']), 
                      render['pred_template'], 
                      render['pred_flow']],
                      titles=['GT', 'Pred', 'Abs. Error', 'Template', 'Flow'])
 


# ## Here be dragons.

# In[ ]:


0/0


# In[ ]:


exp_dir = gpath.GPath('')
exp_paths = sorted(x for x in exp_dir.iterdir() if x.name != 'training_images')
image_id = 11

target_color = image_utils.load_image(exp_paths[0] / f'{image_id:02d}_target_color.png')
pred_color_paths = [p / f'{image_id:02d}_pred_color.png' for p in exp_paths]
pred_colors = utils.parallel_map(image_utils.load_image, pred_color_paths, show_pbar=True)

pred_template_paths = [p / f'{image_id:02d}_pred_template.png' for p in exp_paths]
pred_templates = utils.parallel_map(image_utils.load_image, pred_template_paths, show_pbar=True)

pred_flow_paths = [p / f'{image_id:02d}_pred_flow.png' for p in exp_paths]
pred_flows = utils.parallel_map(image_utils.load_image, pred_flow_paths, show_pbar=True)

imedia.show_images([target_color] + pred_colors, ['gt'] + [p.name for p in exp_paths])
imedia.show_images([np.zeros_like(target_color)] + pred_templates, [''] + [p.name for p in exp_paths])
imedia.show_images([np.zeros_like(target_color)] + pred_flows, [''] + [p.name for p in exp_paths])


# In[ ]:


compute_psnr(0.049)


# In[ ]:




