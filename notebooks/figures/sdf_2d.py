#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# @title Install Packages.
get_ipython().system('pip install git+https://github.com/fogleman/sdf.git')
get_ipython().system('pip install flax optax')
get_ipython().system('pip install mediapy')
get_ipython().system('pip install jax')
get_ipython().system('pip install git+https://github.com/google/nerfies.git')
get_ipython().system('pip install --upgrade scikit-image Pillow')
get_ipython().system('pip install trimesh[easy]')


# In[ ]:


# @title Visualization utilities.
import functools

from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


_colormap_cache = {}


def _build_colormap(name, num_bins=256):
  base = cm.get_cmap(name)
  color_list = base(np.linspace(0, 1, num_bins))
  cmap_name = base.name + str(num_bins)
  colormap = LinearSegmentedColormap.from_list(cmap_name, color_list, num_bins)
  colormap = colormap(np.linspace(0, 1, num_bins))[:, :3]
  return colormap


@functools.lru_cache(maxsize=32)
def get_colormap(name, num_bins=256):
  """Lazily initializes and returns a colormap."""
  if name == 'turbo':
    return _TURBO_COLORS

  return _build_colormap(name, num_bins)


def interpolate_colormap(values, colormap):
  """Interpolates the colormap given values between 0.0 and 1.0."""
  a = np.floor(values * 255)
  b = (a + 1).clip(max=255)
  f = values * 255.0 - a
  a = a.astype(np.uint16).clip(0, 255)
  b = b.astype(np.uint16).clip(0, 255)
  return colormap[a] + (colormap[b] - colormap[a]) * f[..., np.newaxis]


def scale_values(values, vmin, vmax, eps=1e-6):
  return (values - vmin) / max(vmax - vmin, eps)


def colorize(
    array, cmin=None, cmax=None, cmap='magma', eps=1e-6, invert=False):
  """Applies a colormap to an array.

  Args:
    array: the array to apply a colormap to.
    cmin: the minimum value of the colormap. If None will take the min.
    cmax: the maximum value of the colormap. If None will take the max.
    cmap: the color mapping to use.
    eps: a small value to prevent divide by zero.
    invert: if True will invert the colormap.

  Returns:
    a color mapped version of array.
  """
  array = np.asarray(array)

  if cmin is None:
    cmin = array.min()
  if cmax is None:
    cmax = array.max()

  x = scale_values(array, cmin, cmax, eps)
  colormap = get_colormap(cmap)
  colorized = interpolate_colormap(1.0 - x if invert else x, colormap)
  colorized[x > 1.0] = 0.0 if invert else 1.0
  colorized[x < 0.0] = 1.0 if invert else 0.0

  return colorized


def colorize_binary_logits(array, cmap=None):
  """Colorizes binary logits as a segmentation map."""
  num_classes = array.shape[-1]
  if cmap is None:
    if num_classes <= 8:
      cmap = 'Set3'
    elif num_classes <= 10:
      cmap = 'tab10'
    elif num_classes <= 20:
      cmap = 'tab20'
    else:
      cmap = 'gist_rainbow'

  colormap = get_colormap(cmap, num_classes)
  indices = np.argmax(array, axis=-1)
  return np.take(colormap, indices, axis=0)


# In[ ]:


# @title Schedules.

"""Annealing Schedules."""
import abc
import collections
import copy
import math
from typing import Any, Iterable, List, Tuple, Union

from jax import numpy as jnp


def from_tuple(x):
  schedule_type, *args = x
  return SCHEDULE_MAP[schedule_type](*args)


def from_dict(d):
  d = copy.copy(dict(d))
  schedule_type = d.pop('type')
  return SCHEDULE_MAP[schedule_type](**d)


def from_config(schedule):
  if isinstance(schedule, Schedule):
    return schedule
  if isinstance(schedule, Tuple) or isinstance(schedule, List):
    return from_tuple(schedule)
  if isinstance(schedule, collections.Mapping):
    return from_dict(schedule)

  raise ValueError(f'Unknown type {type(schedule)}.')


class Schedule(abc.ABC):
  """An interface for generic schedules.."""

  @abc.abstractmethod
  def get(self, step):
    """Get the value for the given step."""
    raise NotImplementedError

  def __call__(self, step):
    return self.get(step)


class ConstantSchedule(Schedule):
  """Linearly scaled scheduler."""

  def __init__(self, value):
    super().__init__()
    self.value = value

  def get(self, step):
    """Get the value for the given step."""
    return jnp.full_like(step, self.value, dtype=jnp.float32)


class LinearSchedule(Schedule):
  """Linearly scaled scheduler."""

  def __init__(self, initial_value, final_value, num_steps):
    super().__init__()
    self.initial_value = initial_value
    self.final_value = final_value
    self.num_steps = num_steps

  def get(self, step):
    """Get the value for the given step."""
    if self.num_steps == 0:
      return jnp.full_like(step, self.final_value, dtype=jnp.float32)
    alpha = jnp.minimum(step / self.num_steps, 1.0)
    return (1.0 - alpha) * self.initial_value + alpha * self.final_value


class ExponentialSchedule(Schedule):
  """Exponentially decaying scheduler."""

  def __init__(self, initial_value, final_value, num_steps, eps=1e-10):
    super().__init__()
    if initial_value <= final_value:
      raise ValueError('Final value must be less than initial value.')

    self.initial_value = initial_value
    self.final_value = final_value
    self.num_steps = num_steps
    self.eps = eps

  def get(self, step):
    """Get the value for the given step."""
    if step >= self.num_steps:
      return jnp.full_like(step, self.final_value, dtype=jnp.float32)

    final_value = max(self.final_value, self.eps)
    base = final_value / self.initial_value
    exponent = step / (self.num_steps - 1)
    if step >= self.num_steps:
      return jnp.full_like(step, self.final_value, dtype=jnp.float32)
    return self.initial_value * base**exponent


class CosineEasingSchedule(Schedule):
  """Schedule that eases slowsly using a cosine."""

  def __init__(self, initial_value, final_value, num_steps):
    super().__init__()
    self.initial_value = initial_value
    self.final_value = final_value
    self.num_steps = num_steps

  def get(self, step):
    """Get the value for the given step."""
    alpha = jnp.minimum(step / self.num_steps, 1.0)
    scale = self.final_value - self.initial_value
    x = min(max(alpha, 0.0), 1.0)
    return (self.initial_value
            + scale * 0.5 * (1 + math.cos(jnp.pi * x + jnp.pi)))


class StepSchedule(Schedule):
  """Schedule that eases slowsly using a cosine."""

  def __init__(self,
               initial_value,
               decay_interval,
               decay_factor,
               max_decays,
               final_value=None):
    super().__init__()
    self.initial_value = initial_value
    self.decay_factor = decay_factor
    self.decay_interval = decay_interval
    self.max_decays = max_decays
    if final_value is None:
      final_value = self.initial_value * self.decay_factor**self.max_decays
    self.final_value = final_value

  def get(self, step):
    """Get the value for the given step."""
    phase = step // self.decay_interval
    if phase >= self.max_decays:
      return self.final_value
    else:
      return self.initial_value * self.decay_factor**phase


class PiecewiseSchedule(Schedule):
  """A piecewise combination of multiple schedules."""

  def __init__(
      self, schedules: Iterable[Tuple[int, Union[Schedule, Iterable[Any]]]]):
    self.schedules = [from_config(s) for ms, s in schedules]
    milestones = jnp.array([ms for ms, s in schedules])
    self.milestones = jnp.cumsum(milestones)[:-1]

  def get(self, step):
    idx = jnp.searchsorted(self.milestones, step, side='right')
    schedule = self.schedules[idx]
    base_idx = self.milestones[idx - 1] if idx >= 1 else 0
    return schedule.get(step - base_idx)


class DelayedSchedule(Schedule):
  """Delays the start of the base schedule."""

  def __init__(self, base_schedule: Schedule, delay_steps, delay_mult):
    self.base_schedule = from_config(base_schedule)
    self.delay_steps = delay_steps
    self.delay_mult = delay_mult

  def get(self, step):
    delay_rate = (
        self.delay_mult
        + (1 - self.delay_mult)
        * jnp.sin(0.5 * jnp.pi * jnp.clip(step / self.delay_steps, 0, 1)))

    return delay_rate * self.base_schedule(step)


SCHEDULE_MAP = {
    'constant': ConstantSchedule,
    'linear': LinearSchedule,
    'exponential': ExponentialSchedule,
    'cosine_easing': CosineEasingSchedule,
    'step': StepSchedule,
    'piecewise': PiecewiseSchedule,
    'delayed': DelayedSchedule,
}


# In[ ]:


import trimesh
import sdf as sdflib
import numpy as np
import mediapy
from matplotlib import pyplot as plt
import jax
from jax import numpy as jnp
from jax import random
import optax
from nerfies import utils

def matmul(a, b):
  """jnp.matmul defaults to bfloat16, but this helper function doesn't."""
  return jnp.matmul(a, b, precision=jax.lax.Precision.HIGHEST)


# In[ ]:


get_ipython().system('wget -O OpenSans-ExtraBold.ttf "https://github.com/opensourcedesign/fonts/blob/master/OpenSans/OpenSans-ExtraBold.ttf?raw=true"')


# In[ ]:


from numpy.core.multiarray import empty_like
import itertools

trunc = 0.05
x, y = np.meshgrid(
    np.linspace(-1, 1, 256),
    np.linspace(-0.5, 0.5, 128),
)
coords = np.stack([x, y], axis=-1).reshape((-1, 2))

# sdfs = [
#   sdflib.text('OpenSans-ExtraBold.ttf', 'A', 1),
#   sdflib.text('OpenSans-ExtraBold.ttf', 'B', 1.0),
#   sdflib.text('OpenSans-ExtraBold.ttf', 'C', 1.0),
#   sdflib.text('OpenSans-ExtraBold.ttf', 'D', 1.0),
#   sdflib.text('OpenSans-ExtraBold.ttf', 'E', 1.0),
# ]

# sdfs = [
#   sdflib.text('OpenSans-ExtraBold.ttf', f'{i}', 1)
#   for i in range(0, 3)
# ]

a_fn = lambda x, y: sdflib.circle(0.3, (x, y))
b_fn = lambda x, y: sdflib.d2.translate(sdflib.d2.rounded_x(0.3, 0.1), (x, y))
# box_fn = lambda x, y: (sdflib.box(0.6, (x, y)) - sdflib.circle(0.1, (x, y)))
# triangle_fn = lambda x, y: sdflib.d2.translate(sdflib.d2.scale(sdflib.equilateral_triangle(), 0.4), (x, y))
# corners = circle_fn(-0.5, 0.5) | box_fn(0.5, 0.5) | circle_fn(0.5, -0.5) | box_fn(-0.5, -0.5)
# sdfs = [
#   a(-0.5, 0.0) | b(0.5, 0.0) 
#   for a, b  in itertools.product([a_fn, b_fn], [a_fn, b_fn])
# ]

# sdfs = [
#   sdflib.text('OpenSans-ExtraBold.ttf', 'o', 1),
#   sdflib.text('OpenSans-ExtraBold.ttf', 'c', 1.0),
# ]
sdfs = [
      sdflib.circle(0.4, (-t, 0.0)) | (sdflib.circle(0.4, (t, 0.0)) - sdflib.circle(0.1, (t, 0.0)))
      for t in np.linspace(0.0, 0.5, 3)
]
# sdfs = [
#   # sdflib.d2.elongate(sdflib.rounded_rectangle(0.7, 0.4), np.array([0.5, 0.1])),
#   sdflib.circle(0.4, (0, 0.0)) | sdflib.circle(0.4, (-0.5, 0.0)) | sdflib.circle(0.4, (0.5, 0.0)),
#   sdflib.circle(0.4, (-0.5, 0.0)) | sdflib.circle(0.4, (0.5, 0.0)),
#   sdflib.circle(0.3, (-0.5, 0.0)) | sdflib.circle(0.3, (0.5, 0.0)),
#   sdflib.circle(0.2, (-0.5, 0.0)) | sdflib.circle(0.3, (0.5, 0.0)),
#   sdflib.circle(0.1, (-0.5, 0.0)) | sdflib.circle(0.3, (0.5, 0.0)),
#   sdflib.circle(0.0, (0.0, 0.0)) - sdflib.circle(0.1, (0.0, 0.0)),
# ]
# sdfs += [
#       sdflib.circle(0.4, (-t, 0.0)) | (sdflib.circle(0.4, (t, 0.0)))
#       for t in np.linspace(0.0, 0.5, 3)
# ]
# sdfs = [
#       sdflib.circle(0.4, (0.0, 0.0)),
#       sdflib.circle(0.3, (-0.5, 0.0)) | sdflib.circle(0.3, (0.5, 0.0)),
#       sdflib.circle(0.2, (-0.6, 0.0)) | sdflib.circle(0.2, (0.6, 0.0)) | sdflib.circle(0.2, (0.0, 0.0)),
#       # sdflib.circle(0.5, (-0.0, 0.0)) - sdflib.circle(0.2, (0.0, 0.0)),
#       sdflib.text('OpenSans-ExtraBold.ttf', 'SDF', 1.5),
# ]

sdf_images = [
  colorize(sdf(coords).clip(-trunc, trunc).reshape(x.shape), cmin=-trunc, cmax=trunc, cmap='gist_gray', invert=False)
  for sdf in sdfs]
mediapy.show_images([np.flipud(x) for x in sdf_images], columns=5)


# In[ ]:


# @title MLP
from typing import Any, Optional, Tuple, Callable
from flax import linen as nn


class MLP(nn.Module):
  """Basic MLP class with hidden layers and an output layers."""
  depth: int
  width: int
  hidden_init: Any = nn.initializers.xavier_uniform()
  hidden_activation: Any = nn.relu
  hidden_norm: Optional[Callable[[Any], nn.Module]] = None
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
      if self.hidden_norm:
        x = self.hidden_norm()(x)
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


# @title Position Encoding

def posenc(x, min_deg, max_deg, use_identity=False, alpha=None):
  """Encode `x` with sinusoids scaled by 2^[min_deg:max_deg-1]."""
  batch_shape = x.shape[:-1]
  scales = 2.0 ** jnp.arange(min_deg, max_deg)
  # (*, F, C).
  xb = x[..., None, :] * scales[:, None]
  # (*, F, 2, C).
  four_feat = jnp.sin(jnp.stack([xb, xb + 0.5 * jnp.pi], axis=-2))

  if alpha is not None:
    window = posenc_window(min_deg, max_deg, alpha)
    four_feat = window[..., None, None] * four_feat

  # (*, 2*F*C).
  four_feat = four_feat.reshape((*batch_shape, -1))

  if use_identity:
    return jnp.concatenate([x, four_feat], axis=-1)
  else:
    return four_feat


def posenc_window(min_deg, max_deg, alpha):
  """Windows a posenc using a cosiney window.

  This is equivalent to taking a truncated Hann window and sliding it to the
  right along the frequency spectrum.

  Args:
    min_deg: the lower frequency band.
    max_deg: the upper frequency band.
    alpha: will ease in each frequency as alpha goes from 0.0 to num_freqs.

  Returns:
    A 1-d numpy array with num_sample elements containing the window.
  """
  bands = jnp.arange(min_deg, max_deg)
  x = jnp.clip(alpha - bands, 0.0, 1.0)
  return 0.5 * (1 + jnp.cos(jnp.pi * x + jnp.pi))


# In[ ]:


# @title Model
import jax

class HyperSheetMLP(nn.Module):
  """An MLP that defines a bendy slicing surface through hyper space."""
  output_channels: int
  min_deg: int = 0
  max_deg: int = 1

  depth: int = 4
  width: int = 64
  skips: Tuple[int] = (3,)
  hidden_init: Any = jax.nn.initializers.glorot_uniform()
  # output_init: Any = jax.nn.initializers.glorot_uniform()
  output_init: Any = jax.nn.initializers.normal(0.05)

  @nn.compact
  def __call__(self, points, embed):
    points_feat = posenc(points, self.min_deg, self.max_deg)
    inputs = jnp.concatenate([points_feat, embed], axis=-1)
    mlp = MLP(depth=self.depth,
              width=self.width,
              skips=self.skips,
              hidden_init=self.hidden_init,
              output_channels=self.output_channels,
              output_init=self.output_init)
    return mlp(inputs)


class Model(nn.Module):
  num_glo_embeddings: int
  num_glo_features: int = 8
  spatial_min_deg: int = 0
  spatial_max_deg: int = 8
  hyper_min_deg: int = 0
  hyper_max_deg: int = 3
  hyper_num_dims: int = 1

  hyper_slice_method: str = 'axis_aligned_plane'

  # embedding_init: Any = jax.nn.initializers.uniform(scale=0.05)
  embedding_init: Any = jax.nn.initializers.glorot_uniform()

  def setup(self):
    self.template_mlp = MLP(
      depth=8, 
      width=128, 
      hidden_init=jax.nn.initializers.glorot_uniform(),
      # hidden_norm=nn.LayerNorm,
      output_init=jax.nn.initializers.glorot_uniform(),
      # output_activation=nn.tanh,
      skips=(4,),
      output_channels=1)
    self.hyper_embed = nn.Embed(
        num_embeddings=self.num_glo_embeddings,
        features=(self.hyper_num_dims 
                  if self.hyper_slice_method == 'axis_aligned_plane' 
                  else self.num_glo_features),
        embedding_init=self.embedding_init)
    self.hyper_sheet_mlp = HyperSheetMLP(output_channels=self.hyper_num_dims)
    
  def eval_template(self, x):
    x, z = jnp.split(x, (2,), axis=-1)
    x = jnp.concatenate([
        posenc(x, self.spatial_min_deg, self.spatial_max_deg), 
        posenc(z, self.hyper_min_deg, self.hyper_max_deg),
    ], axis=-1)
    return self.template_mlp(x)

  def compute_latent(self, glo_id):
    return self.hyper_encoder(glo_id)

  def evaluate(self, x, z):
    x = jnp.concatenate([
        posenc(x, self.spatial_min_deg, self.spatial_max_deg), 
        posenc(z, self.hyper_min_deg, self.hyper_max_deg),
    ], axis=-1)
    return self.template_mlp(x)

  def __call__(self, x, glo_id, alpha):
    hyper_embed = self.hyper_embed(glo_id)
    if self.hyper_slice_method == 'axis_aligned_plane':
      z = hyper_embed
    elif self.hyper_slice_method == 'bendy_sheet':
      z = self.hyper_sheet_mlp(x, hyper_embed)
    else:
      raise RuntimeError('Unknown hyper slice method.')
    
    return {
        'values': self.evaluate(x, z),
        'z': z,
    }


# In[ ]:


# @title Barron loss.

@jax.jit
def general_loss(x, alpha, scale):
  r"""Implements the general form of the loss.
  This implements the rho(x, \alpha, c) function described in "A General and
  Adaptive Robust Loss Function", Jonathan T. Barron,
  https://arxiv.org/abs/1701.03077.
  Args:
    x: The residual for which the loss is being computed. x can have any shape,
      and alpha and scale will be broadcasted to match x's shape if necessary.
    alpha: The shape parameter of the loss (\alpha in the paper), where more
      negative values produce a loss with more robust behavior (outliers "cost"
      less), and more positive values produce a loss with less robust behavior
      (outliers are penalized more heavily). Alpha can be any value in
      [-infinity, infinity], but the gradient of the loss with respect to alpha
      is 0 at -infinity, infinity, 0, and 2. Varying alpha allows for smooth
      interpolation between several discrete robust losses:
        alpha=-Infinity: Welsch/Leclerc Loss.
        alpha=-2: Geman-McClure loss.
        alpha=0: Cauchy/Lortentzian loss.
        alpha=1: Charbonnier/pseudo-Huber loss.
        alpha=2: L2 loss.
    scale: The scale parameter of the loss. When |x| < scale, the loss is an
      L2-like quadratic bowl, and when |x| > scale the loss function takes on a
      different shape according to alpha.
  Returns:
    The losses for each element of x, in the same shape as x.
  """
  eps = jnp.finfo(jnp.float32).eps

  # `scale` must be > 0.
  scale = jnp.maximum(eps, scale)

  # The loss when alpha == 2. This will get reused repeatedly.
  loss_two = 0.5 * (x / scale)**2

  # "Safe" versions of log1p and expm1 that will not NaN-out.
  log1p_safe = lambda x: jnp.log1p(jnp.minimum(x, 3e37))
  expm1_safe = lambda x: jnp.expm1(jnp.minimum(x, 87.5))

  # The loss when not in one of the special casess.
  # Clamp |alpha| to be >= machine epsilon so that it's safe to divide by.
  a = jnp.where(alpha >= 0, jnp.ones_like(alpha),
                -jnp.ones_like(alpha)) * jnp.maximum(eps, jnp.abs(alpha))
  # Clamp |2-alpha| to be >= machine epsilon so that it's safe to divide by.
  b = jnp.maximum(eps, jnp.abs(alpha - 2))
  loss_ow = (b / a) * ((loss_two / (0.5 * b) + 1)**(0.5 * alpha) - 1)

  # Select which of the cases of the loss to return as a function of alpha.
  return scale * jnp.where(
      alpha == -jnp.inf, -expm1_safe(-loss_two),
      jnp.where(
          alpha == 0, log1p_safe(loss_two),
          jnp.where(alpha == 2, loss_two,
                    jnp.where(alpha == jnp.inf, expm1_safe(loss_two),
                              loss_ow))))


@jax.jit
def general_loss_sq(x_sq, alpha, scale):
  r"""Implements the general form of the loss.
  This implements the rho(x, \alpha, c) function described in "A General and
  Adaptive Robust Loss Function", Jonathan T. Barron,
  https://arxiv.org/abs/1701.03077.
  Args:
    x: The residual for which the loss is being computed. x can have any shape,
      and alpha and scale will be broadcasted to match x's shape if necessary.
    alpha: The shape parameter of the loss (\alpha in the paper), where more
      negative values produce a loss with more robust behavior (outliers "cost"
      less), and more positive values produce a loss with less robust behavior
      (outliers are penalized more heavily). Alpha can be any value in
      [-infinity, infinity], but the gradient of the loss with respect to alpha
      is 0 at -infinity, infinity, 0, and 2. Varying alpha allows for smooth
      interpolation between several discrete robust losses:
        alpha=-Infinity: Welsch/Leclerc Loss.
        alpha=-2: Geman-McClure loss.
        alpha=0: Cauchy/Lortentzian loss.
        alpha=1: Charbonnier/pseudo-Huber loss.
        alpha=2: L2 loss.
    scale: The scale parameter of the loss. When |x| < scale, the loss is an
      L2-like quadratic bowl, and when |x| > scale the loss function takes on a
      different shape according to alpha.
  Returns:
    The losses for each element of x, in the same shape as x.
  """
  eps = jnp.finfo(jnp.float32).eps

  # `scale` must be > 0.
  scale = jnp.maximum(eps, scale)

  # The loss when alpha == 2. This will get reused repeatedly.
  loss_two = 0.5 * x_sq / (scale ** 2)

  # "Safe" versions of log1p and expm1 that will not NaN-out.
  log1p_safe = lambda x: jnp.log1p(jnp.minimum(x, 3e37))
  expm1_safe = lambda x: jnp.expm1(jnp.minimum(x, 87.5))

  # The loss when not in one of the special casess.
  # Clamp |alpha| to be >= machine epsilon so that it's safe to divide by.
  a = jnp.where(alpha >= 0, jnp.ones_like(alpha),
                -jnp.ones_like(alpha)) * jnp.maximum(eps, jnp.abs(alpha))
  # Clamp |2-alpha| to be >= machine epsilon so that it's safe to divide by.
  b = jnp.maximum(eps, jnp.abs(alpha - 2))
  loss_ow = (b / a) * ((loss_two / (0.5 * b) + 1)**(0.5 * alpha) - 1)

  # Select which of the cases of the loss to return as a function of alpha.
  return scale * jnp.where(
      alpha == -jnp.inf, -expm1_safe(-loss_two),
      jnp.where(
          alpha == 0, log1p_safe(loss_two),
          jnp.where(alpha == 2, loss_two,
                    jnp.where(alpha == jnp.inf, expm1_safe(loss_two),
                              loss_ow))))


# In[ ]:


# Test out losses.
x = jnp.linspace(-0.05, 0.05, 1000)
loss_fn = lambda x: optax.huber_loss(x, delta=0.01) / 0.01
# loss_fn = optax.l2_loss
# loss_fn = jnp.abs
# loss_fn = lambda x: general_loss(x, alpha=0.5, scale=0.005)
loss_fn = lambda x: general_loss_sq(x ** 2, alpha=0.5, scale=0.005)
y = jax.vmap(jax.grad(loss_fn))(x)

plt.plot(x, y)


# In[ ]:


# @title Define and initialize model.
spatial_min_deg = 0  # @param {type: 'number'}
spatial_max_deg = 4.0  # @param {type: 'number'}
hyper_min_deg = 0  # @param {type: 'number'}
hyper_max_deg = 1.0  # @param {type: 'number'}
hyper_num_dims =   1# @param {type: 'number'}
hyper_slice_method = 'axis_aligned_plane'  # @param ['axis_aligned_plane', 'bendy_sheet']

devices = jax.devices()
rng = random.PRNGKey(0)
rng, key = random.split(rng, 2)

model = Model(
    num_glo_embeddings=len(sdfs),
    spatial_min_deg=spatial_min_deg,
    spatial_max_deg=spatial_max_deg,
    hyper_min_deg=hyper_min_deg,
    hyper_max_deg=hyper_max_deg,
    hyper_num_dims=hyper_num_dims,
    hyper_slice_method=hyper_slice_method,
)

init_coords = random.normal(key, (2,))
init_glo_ids = jnp.zeros((), jnp.uint32)
init_params = model.init(key, init_coords, init_glo_ids, 0.0)


# In[ ]:


# @title Setup dataset
def prepare_data(xs):
  """Convert a input batch from tf Tensors to numpy arrays."""
  local_device_count = jax.local_device_count()
  def _prepare(x):
    # reshape (host_batch_size, height, width, 3) to
    # (local_devices, device_batch_size, height, width, 3)
    x = x.reshape((local_device_count, -1) + x.shape[1:])
    return jax.api.device_put_sharded(list(x), jax.local_devices())

  return jax.tree_map(_prepare, xs)


def get_coords(height, width):
  hr = height / max(height, width)
  wr = width / max(height, width)
  x, y = jnp.meshgrid(
      np.linspace(-wr, wr, width),
      np.linspace(-hr, hr, height))
  coords = jnp.stack([x, y], axis=-1)
  return coords


def make_batch(key, batch_size, minval=-1.0, maxval=1.0, trunc=0.1):
  key1, key2 = random.split(key, 2)
  coords = random.uniform(key1, (batch_size, 2), minval=minval, maxval=maxval)
  coords = coords.reshape((-1, 2))
  ids = jnp.concatenate([
      jnp.full((*coords.shape[:-1], 1), fill_value=i, dtype=jnp.uint32)
      for i in range(len(sdfs))
  ])
  values = jnp.concatenate(
      [jnp.array(sdf(np.array(coords)).clip(-trunc, trunc)) for sdf in sdfs],
      axis=0
  )
  coords = jnp.tile(coords, (len(sdfs), 1))
  perm = random.permutation(key2, np.arange(ids.shape[0]))
  return {
      'values': values[perm],
      'ids': ids[perm],
      'coords': coords[perm],
  }


batch = make_batch(random.PRNGKey(0), 1024)
jax.tree_map(lambda x: x.shape, batch)


# In[ ]:


# @title Losses and training step.


def compute_normal(fn, coords, eps=1e-15):
  grad = jax.grad(fn)(coords)
  norm = utils.safe_norm(grad, axis=-1)
  return grad / jnp.maximum(eps, norm)


def compute_curvature_loss(params, batch, key, pred, scale=0.05, num_z_samples=1):
  key1, key2 = random.split(key, 2)

  z_vals = pred['z']
  xy_coords = batch['coords']
  z_shape = (num_z_samples, *xy_coords.shape[:-1], hyper_num_dims)

  z_coords = random.uniform(key, z_shape)
  z_min = z_vals.min(axis=0, keepdims=True)[:, None, :]
  z_max = z_vals.max(axis=0, keepdims=True)[:, None, :]
  z_coords = z_coords * (z_max - z_min) + z_min
  xy_coords = jnp.tile(xy_coords[None, ...], (z_coords.shape[0], 1, 1))

  coords = jnp.concatenate([xy_coords, z_coords], axis=-1)
  coords = coords.reshape((-1, 2 + z_shape[-1]))

  def eval_template(x):
    dummy_code = jnp.zeros((coords.shape[0],))
    return model.apply(params, 
                       x, 
                       method=model.eval_template)[..., 0]
  
  normal_fn = jax.vmap(compute_normal, in_axes=(None, 0))
  normals = normal_fn(eval_template, coords)
  # Projection onto the tangent plane.
  P = jnp.eye(normals.shape[-1])[None, ...] - normals[:, :, None] @ normals[:, None, :]

  jitter_dir = random.normal(key2, coords.shape)
  jitter_dir = jitter_dir.at[..., :2].set(0.0)
  jitter_dir = matmul(P, jitter_dir[:, :, None]).squeeze(-1)
  jitter_dir = jitter_dir / jnp.linalg.norm(jitter_dir, axis=-1, keepdims=True)
  jittered_coords = coords + scale * jitter_dir
  jittered_normals = normal_fn(eval_template, jittered_coords)
  curvature = (jittered_normals - normals) / scale

  # Only apply the curvature loss near the surface.
  template_vals = jax.lax.stop_gradient(jax.vmap(eval_template)(coords))
  curvature_weights = trunc - jnp.abs(template_vals.clip(a_min=-trunc, a_max=trunc))
  sq_residual = jnp.sum(curvature ** 2, axis=-1)
  # curvature_loss = general_loss(residual, alpha=1.0, scale=0.005)
  curvature_loss = general_loss_sq(sq_residual, alpha=1.0, scale=0.005)
  curvature_loss = curvature_weights * curvature_loss

  return curvature_loss.mean(), jnp.sqrt(sq_residual)


@jax.jit
def loss_fn(params, batch, scalar_params, key):
  alpha = scalar_params['alpha']

  def compute_sdf(coords, ids):
    return model.apply(params, coords, ids, alpha)

  pred = jax.vmap(compute_sdf)(batch['coords'], batch['ids'].squeeze(-1))
  values = pred['values']
  sdf_loss = jnp.abs(values - batch['values'])
  sdf_loss = sdf_loss.mean()

  hyper_reg_loss = (pred['z'] ** 2).sum(axis=-1).mean()

  # curvature_loss, curvature = compute_curvature_loss(params, batch, key, pred)
  # total_loss = (sdf_loss 
  #               + scalar_params['curvature_weight'] * curvature_loss
  #               + 1e-5 * hyper_reg_loss)
  total_loss = sdf_loss

  stats_dict = {
      'sdf_loss': sdf_loss, 
      # 'curvature_weight': (curvature_weight.min(), curvature_weight.max()),
      # 'curvature_stats': (curvature.min(), curvature.mean(), curvature.max()),
      # 'curvature_loss': curvature_loss,
      'hyper_reg_loss': hyper_reg_loss,
      'total_loss': total_loss,
  }

  return total_loss, stats_dict


@jax.jit
def train_step(optimizer, batch, key, scalar_params):
  lr = scalar_params['lr']
  _, key = random.split(rng, 2)
  grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
  (_, stats_dict), grad = grad_fn(optimizer.target, batch, scalar_params, key)
  stats_dict = jax.lax.pmean(stats_dict, axis_name='batch')
  grad = jax.lax.pmean(grad, axis_name='batch')
  new_optimizer = optimizer.apply_gradient(grad, learning_rate=lr)
  return new_optimizer, key, stats_dict


def plot_sdf(model, params, alpha, columns=5):
  def render(coords, glo_id):
    return model.apply(params, coords, glo_id, alpha)

  coords = get_coords(128, 256)

  sdf_images = []
  for glo_id in range(len(sdfs)):
    glo_id = jnp.full(coords.shape[:-1], fill_value=glo_id, dtype=jnp.uint32)
    values = jax.vmap(render)(coords, glo_id)['values']
    values = values.clip(-trunc, trunc)
    sdf_image = colorize(values.reshape(coords.shape[:2]), cmap='coolwarm', cmin=-trunc, cmax=trunc)
    sdf_images.append(sdf_image)
  mediapy.show_images([np.flipud(x) for x in sdf_images], columns=columns)


# In[ ]:


# @title Train.
from nerfies import schedules
from nerfies import utils
from flax import optim
from flax import jax_utils


max_iters = 4000
batch_size = 512
lr_schedule = schedules.ConstantSchedule(1e-3)
# lr_schedule = from_config({
#   'type': 'delayed',
#   'delay_steps': 50,
#   'delay_mult': 0.01,
#   'base_schedule': {
#     'type': 'exponential',
#     'initial_value': 1e-3,
#     'final_value': 5e-4,
#     'num_steps': max_iters,
#   },
# })

curvature_schedule = schedules.from_config({
    'type': 'piecewise',
    'schedules': [
      (100, ('constant', 0.1)),
      (0, ('cosine_easing', 0.1, 1.0, 1000)),
    ]
})

# curvature_schedule = schedules.LinearSchedule(
    # 0.0, 100.0, 1000)

alpha_schedule = schedules.ConstantSchedule(0.0)

optimizer_def = optim.Adam(lr_schedule(0))
optimizer_def = optim.WeightNorm(optimizer_def)
optimizer = optimizer_def.create(init_params)
optimizer = jax_utils.replicate(optimizer, devices)

p_train_step = jax.pmap(
    train_step, axis_name='batch', devices=devices, in_axes=(0, 0, 0, None))
tt = utils.TimeTracker()

rng = random.PRNGKey(1)
keys = random.split(rng, len(devices))
for i in range(max_iters):
  rng, key = random.split(rng, 2)
  batch = make_batch(key, batch_size)
  batch = prepare_data(batch)

  if i > max_iters:
    break
  scalar_params = {
      'lr': lr_schedule(i),
      'alpha': alpha_schedule(i),
      'curvature_weight': curvature_schedule(i),
  }
  with tt.record_time('p_train_step'):
    optimizer, keys, stats = p_train_step(optimizer, batch, keys, scalar_params)

  if i % 10 == 0:
    stats = jax_utils.unreplicate(stats)
    stats = jax.tree_map(lambda x: x.item(), stats)
    scalar_params = jax.tree_map(lambda x: (x if isinstance(x, float) else x.item()), scalar_params)
    print(f'{i} scalar_params: {scalar_params}')
    print(f'{i} stats: {stats}')
    print('')

  if i % 100 == 0:
    plot_sdf(model, jax_utils.unreplicate(optimizer.target), scalar_params['alpha'])


# In[ ]:


plot_sdf(model, jax_utils.unreplicate(optimizer.target), scalar_params['alpha'], columns=4)


# In[ ]:


# Compute the minimum and maximum Z-axis coordinates.

def compute_bounds(model, params, alpha):
  coords = get_coords(100, 100).reshape((-1, 2))

  sdf_images = []
  zmin = jnp.array([float('inf')] * hyper_num_dims)
  zmax = jnp.array([-float('inf')] * hyper_num_dims)
  for glo_id in range(len(sdfs)):
    glo_id = jnp.full(coords.shape[:-1], fill_value=glo_id, dtype=jnp.uint32)
    z = jax.vmap(model.apply, in_axes=(None, 0, 0, None))(params, coords, glo_id, alpha)['z']
    zmin = jnp.minimum(z.min(axis=0) , zmin)
    zmax = jnp.maximum(z.max(axis=0) , zmax)
  return zmin, zmax


zmin, zmax = compute_bounds(model, jax_utils.unreplicate(optimizer.target), scalar_params['alpha'])
zmin, zmax


# In[ ]:


import trimesh

@sdflib.sdf3
def template_sdf(params):
    def _f(p):
      if hyper_num_dims > 1:
        p = jnp.concatenate([p, jnp.full_like(p[..., :1], fill_value=zmin[1])], axis=-1)
      return model.apply(params,
                        jnp.array(p), 
                        method=model.eval_template)

    def f(p):
        values = jax.vmap(_f)(p)
        return np.array(values).clip(-trunc, trunc)
    return f


bounds = [(-1, -1, zmin[0]-0.01), (1, 1, zmax[0]+0.01)]
# bounds = [(-1, -1, -1), (1, 1, 1)]
sdf = template_sdf(jax_utils.unreplicate(optimizer.target))

out_name = 'sdf.stl'
sdf.save(out_name, bounds=bounds, samples=2**16)

mesh = trimesh.load('sdf.stl')
mesh.show(smooth=True)


# In[ ]:


from nerfies import visualization as viz


def compute_zmaps(model, params, alpha, shape=(128, 256)):
  coords = get_coords(*shape).reshape((-1, 2))
  zmaps = []
  for glo_id in range(len(sdfs)):
    glo_id = jnp.full(coords.shape[:-1], fill_value=glo_id, dtype=jnp.uint32)
    zmap = jax.vmap(model.apply, in_axes=(None, 0, 0, None))(params, coords, glo_id, alpha)['z'].reshape(shape)
    zmaps.append(zmap)
  
  return zmaps

zmaps = compute_zmaps(model, jax_utils.unreplicate(optimizer.target), scalar_params['alpha'])
mediapy.show_images([viz.colorize(np.array(zmap), cmap='magma', cmin=zmin.item(), cmax=zmax.item()) for zmap in zmaps], columns=4)


# In[ ]:


import meshio
colors = (viz.get_colormap('Pastel1', len(zmaps)) * 255).astype(np.uint8)

for i, zmap in enumerate(zmaps):
  color = colors[i]
  vert_inds = np.arange(np.prod(zmap.shape)).reshape(zmap.shape)
  quads = np.stack([
      vert_inds[:-1, :-1],
      vert_inds[:-1, 1:],
      vert_inds[1:, 1:],
      vert_inds[1:, :-1],
  ], axis=-1).reshape((-1, 4))
  points = np.concatenate([get_coords(*zmap.shape), zmap[..., None]], axis=-1).reshape((-1, 3))
  cells = [('quad', quads.tolist())]

  meshio.write_points_cells(
      f"plane_{i}.ply", 
      points.tolist(), 
      cells,
      point_data={
          'red': np.full_like(points[..., 0], fill_value=color[0], dtype=np.uint8),
          'green': np.full_like(points[..., 0], fill_value=color[1], dtype=np.uint8),
          'blue': np.full_like(points[..., 0], fill_value=color[2], dtype=np.uint8),
      })


# In[ ]:


filenames = [f'plane_{i}.ply' for i in range(len(zmaps))]
filenames = ' '.join(filenames)
get_ipython().system('zip -m sdf.zip $filenames sdf.stl')
files.download('sdf.zip')


# In[ ]:


for i in range(len(zmaps)):
  files.download(f'plane_{i}.ply')


# In[ ]:


from tqdm.auto import tqdm


def eval_template(x):
  dummy_code = jnp.zeros((x.shape[0],))
  return model.apply(
      jax_utils.unreplicate(optimizer.target), x, method=model.eval_template)[..., 0]


num_z = 200
xy = get_coords(256, 256)
zvals = np.linspace(zmin, zmax, num_z)
results = []
for i in tqdm(range(num_z)):
  z = jnp.full_like(xy[..., :1], fill_value=zvals[i])
  coords = jnp.concatenate([xy, z], axis=-1).reshape((-1, 3))
  values = jax.vmap(eval_template)(coords)
  values = values.reshape(xy.shape[:-1])
  results.append(np.array(values))


# In[ ]:


from nerfies import visualization as viz
mediapy.show_video([viz.colorize(p.clip(-trunc, trunc), cmin=-trunc, cmax=trunc, cmap='coolwarm') for p in results], codec='gif', fps=30)


# In[ ]:


from tqdm.auto import tqdm


def eval_template(x):
  dummy_code = jnp.zeros((x.shape[0],))
  return model.apply(
      jax_utils.unreplicate(optimizer.target), x, method=model.eval_template)[..., 0]


num_z = 3
xy = get_coords(64, 64)
zgrid = jnp.meshgrid(*[
  np.linspace(zmin[i], zmax[i], num_z) for i in range(hyper_num_dims)
])
zvals = jnp.stack(zgrid, axis=-1).reshape((-1, hyper_num_dims))
results = []
for zval in tqdm(zvals):
  z = jnp.tile(zval[None, None, :], (xy.shape[0], xy.shape[1], 1))
  coords = jnp.concatenate([xy, z], axis=-1).reshape((-1, 2 + z.shape[-1]))
  values = jax.vmap(eval_template)(coords)
  values = values.reshape(xy.shape[:-1])
  results.append(np.array(values))


# In[ ]:


from nerfies import visualization as viz
mediapy.show_images([viz.colorize(np.flipud(p).clip(-trunc, trunc), cmin=-trunc, cmax=trunc, cmap='coolwarm') for p in results], columns=num_z)


# In[ ]:


from scipy.interpolate import interpolate

def compute_latent(glo_id):
  return model.apply(jax_utils.unreplicate(optimizer.target), jnp.array([glo_id]), method=model.compute_latent)

latent1 = compute_latent(8)
latent2 = compute_latent(9)
latent_codes = utils.interpolate_codes([latent1, latent2], 100, method='linear').squeeze()


# In[ ]:


def render_code(x, z):
  return model.apply(jax_utils.unreplicate(optimizer.target), x, z, method=model.evaluate)

results = []
for latent_code in tqdm(latent_codes):
  x = get_coords(256, 256)
  z = jnp.broadcast_to(latent_code[None, None, :], (*x.shape[:-1], latent_codes.shape[-1]))
  result = jax.vmap(jax.vmap(render_code))(x, z)
  results.append(result)


# In[ ]:


mediapy.show_video([viz.colorize(np.flipud(p.squeeze()).clip(-trunc, trunc), cmin=-trunc, cmax=trunc, cmap='coolwarm') for p in results], codec='gif', fps=24)


# In[ ]:


results[0].shape


# In[ ]:




