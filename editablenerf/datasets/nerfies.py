# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Casual Volumetric Capture datasets.

Note: Please benchmark before submitted changes to this module. It's very easy
to introduce data loading bottlenecks!
"""
import json
from typing import List, Tuple

from absl import logging
import cv2
import gin
import numpy as np

from editablenerf import camera as cam
from editablenerf import gpath
from editablenerf import types
from editablenerf import utils
from editablenerf.datasets import core


def load_scene_info(
    data_dir: types.PathType) -> Tuple[np.ndarray, float, float, float]:
  """Loads the scene center, scale, near and far from scene.json.

  Args:
    data_dir: the path to the dataset.

  Returns:
    scene_center: the center of the scene (unscaled coordinates).
    scene_scale: the scale of the scene.
    near: the near plane of the scene (scaled coordinates).
    far: the far plane of the scene (scaled coordinates).
  """
  scene_json_path = gpath.GPath(data_dir, 'scene.json')
  with scene_json_path.open('r') as f:
    scene_json = json.load(f)

  scene_center = np.array(scene_json['center'])
  scene_scale = scene_json['scale']
  near = scene_json['near']
  far = scene_json['far']

  return scene_center, scene_scale, near, far


def _load_image(path: types.PathType) -> np.ndarray:
  path = gpath.GPath(path)
  with path.open('rb') as f:
    raw_im = np.asarray(bytearray(f.read()), dtype=np.uint8)
    image = cv2.imdecode(raw_im, cv2.IMREAD_COLOR)[:, :, ::-1]  # BGR -> RGB
    image = np.asarray(image).astype(np.float32) / 255.0
    return image


# ZCW add
def _load_image_1ch(path: types.PathType) -> np.ndarray:
  path = gpath.GPath(path)
  image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
  return image


def _load_dataset_ids(data_dir: types.PathType) -> Tuple[List[str], List[str]]:
  """Loads dataset IDs."""
  dataset_json_path = gpath.GPath(data_dir, 'dataset.json')
  logging.info('*** Loading dataset IDs from %s', dataset_json_path)
  with dataset_json_path.open('r') as f:
    dataset_json = json.load(f)
    train_ids = dataset_json['train_ids']
    val_ids = dataset_json['val_ids']

  train_ids = [str(i) for i in train_ids]
  val_ids = [str(i) for i in val_ids]

  return train_ids, val_ids


@gin.configurable
class NerfiesDataSource(core.DataSource):
  """Data loader for videos."""

  def __init__(self,
               data_dir: str = gin.REQUIRED,
               image_scale: int = gin.REQUIRED,
               shuffle_pixels: bool = False,
               camera_type: str = 'json',
               test_camera_trajectory: str = 'orbit-mild',
               **kwargs):
    self.data_dir = gpath.GPath(data_dir)
    # Load IDs from JSON if it exists. This is useful since COLMAP fails on
    # some images so this gives us the ability to skip invalid images.
    train_ids, val_ids = _load_dataset_ids(self.data_dir)
    super().__init__(train_ids=train_ids, val_ids=val_ids,
                     **kwargs)
    self.scene_center, self.scene_scale, self._near, self._far = (
        load_scene_info(self.data_dir))
    self.test_camera_trajectory = test_camera_trajectory

    self.image_scale = image_scale
    self.shuffle_pixels = shuffle_pixels

    self.rgb_dir = gpath.GPath(data_dir, 'rgb', f'{image_scale}x')
    self.depth_dir = gpath.GPath(data_dir, 'depth', f'{image_scale}x')
    if camera_type not in ['json']:
      raise ValueError('The camera type needs to be json.')
    self.camera_type = camera_type
    self.camera_dir = gpath.GPath(data_dir, 'camera')

    metadata_path = self.data_dir / 'metadata.json'
    if metadata_path.exists():
      with metadata_path.open('r') as f:
        self.metadata_dict = json.load(f)

    # ZCW add
    if self.load_kp_init_file:
      self.kp_init = np.loadtxt(self.data_dir / 'data' / self.load_kp_init_file)  # 'points.txt'
      print('load init key point from ' + str(self.data_dir / 'data' / self.load_kp_init_file))

      if self.load_kp_init_file[0:5] == 'const':
        if self.kp_init.ndim != 1:
          raise ValueError('Blendshape const error.')
        self.kp_init = np.broadcast_to(self.kp_init, (len(self.all_ids), self.kp_dimension))

      if self.kp_init.shape[-1] != self.kp_dimension:
        raise ValueError('key point dimension error.')
      if self.kp_init.shape[0] != len(self.train_ids):
        raise ValueError('key point number error.')

  @property
  def near(self) -> float:
    return self._near

  @property
  def far(self) -> float:
    return self._far

  @property
  def camera_ext(self) -> str:
    if self.camera_type == 'json':
      return '.json'

    raise ValueError(f'Unknown camera_type {self.camera_type}')

  def get_rgb_path(self, item_id: str) -> types.PathType:
    return self.rgb_dir / f'{item_id}.png'

  def load_rgb(self, item_id: str) -> np.ndarray:
    return _load_image(self.rgb_dir / f'{item_id}.png')

  def load_camera(self,
                  item_id: types.PathType,
                  scale_factor: float = 1.0) -> cam.Camera:
    if isinstance(item_id, gpath.GPath):
      camera_path = item_id
    else:
      if self.camera_type == 'proto':
        camera_path = self.camera_dir / f'{item_id}{self.camera_ext}'
      elif self.camera_type == 'json':

        # ZCW change
        if self.camera_mode == 0:
          camera_path = self.camera_dir / f'{item_id}{self.camera_ext}'

        elif self.camera_mode == 1:
          id_int = int(item_id)
          id_int = id_int % 100
          item_id = f'{id_int:06d}'
          camera_path = self.data_dir / 'camera-paths' / self.test_camera_trajectory / f'{item_id}{self.camera_ext}'
          print('change camera to', camera_path)

        elif self.camera_mode == 2:
          item_id = '000001'
          camera_path = self.camera_dir / f'{item_id}{self.camera_ext}'
          print('change camera to', item_id)

        else:
          raise ValueError(f'Unknown camera mode.')

      else:
        raise ValueError(f'Unknown camera type {self.camera_type!r}.')

    return core.load_camera(camera_path,
                            scale_factor=scale_factor / self.image_scale,
                            scene_center=self.scene_center,
                            scene_scale=self.scene_scale)

  def glob_cameras(self, path):
    path = gpath.GPath(path)
    return sorted(path.glob(f'*{self.camera_ext}'))

  def load_test_cameras(self, count=None):
    camera_dir = (self.data_dir / 'camera-paths' / self.test_camera_trajectory)
    if not camera_dir.exists():
      logging.warning('test camera path does not exist: %s', str(camera_dir))
      return []
    camera_paths = sorted(camera_dir.glob(f'*{self.camera_ext}'))
    if count is not None:
      stride = max(1, len(camera_paths) // count)
      camera_paths = camera_paths[::stride]
    cameras = utils.parallel_map(self.load_camera, camera_paths)
    return cameras

  def load_points(self, shuffle=False):
    with (self.data_dir / 'points.npy').open('rb') as f:
      points = np.load(f)
    points = (points - self.scene_center) * self.scene_scale
    points = points.astype(np.float32)
    if shuffle:
      logging.info('Shuffling points.')
      shuffled_inds = self.rng.permutation(len(points))
      points = points[shuffled_inds]
    logging.info('Loaded %d points.', len(points))
    return points

  # ZCW add
  def load_flow(self):
    print('load flow')
    # idx = np.asarray()
    idx = self.train_ids[0:-1]

    out_dict = {'flow': [], 'depth': [], 'cmr_orient': [], 'cmr_pos': []}
    out_dict['metadata'] = {'appearance': [], 'warp': []}
    # out_dict['item_id'] = []

    for item_id in idx:
      with (self.data_dir / 'flow' / f'flow_{item_id}.npy').open('rb') as f:
        flow = np.load(f)
      out_dict['flow'].append(flow)

      with (self.data_dir / 'depth' / f'depth_median_{item_id}.npy').open('rb') as f:
        depth = np.load(f)
      out_dict['depth'].append(depth)

      cmr = self.load_camera(item_id)
      out_dict['cmr_orient'].append(cmr.orientation)
      out_dict['cmr_pos'].append(cmr.position)
      # out_dict['item_id'].append(int(item_id))
      metadata_id = np.atleast_1d(int(item_id))
      out_dict['metadata']['appearance'].append(metadata_id)
      out_dict['metadata']['warp'].append(metadata_id)

      if item_id != idx[0] and item_id != idx[-1]:
        out_dict['flow'].append(flow)
        out_dict['depth'].append(depth)
        out_dict['cmr_orient'].append(cmr.orientation)
        out_dict['cmr_pos'].append(cmr.position)
        out_dict['metadata']['appearance'].append(metadata_id)
        out_dict['metadata']['warp'].append(metadata_id)

    cmr_const = self.load_camera(idx[0]).get_parameters()
    del cmr_const['orientation']
    del cmr_const['position']

    return out_dict, cmr_const

  def load_local_coor(self):
    out_dict = {'scale': [], 'bias': []}

    for idx in range(self.k_point_num):
      filename = self.load_kp_init_file[0:-4] + '_coor_' + str(idx) + '.txt'
      local_coor = np.loadtxt(self.data_dir / 'data' / filename, dtype=np.float32)
      if local_coor.size != 4:
        raise ValueError(f'local coor error.')
      out_dict['scale'].append(local_coor[0])
      out_dict['bias'].append(local_coor[1:4])

    out_dict['scale'] = np.array(out_dict['scale'])
    out_dict['scale'] = np.broadcast_to(out_dict['scale'][..., None], (*out_dict['scale'].shape, 3))
    out_dict['bias'] = np.array(out_dict['bias'])
    return out_dict

  def get_appearance_id(self, item_id):
    return self.metadata_dict[item_id]['appearance_id']

  def get_camera_id(self, item_id):
    return self.metadata_dict[item_id]['camera_id']

  def get_warp_id(self, item_id):
    return self.metadata_dict[item_id]['warp_id']

  def get_time_id(self, item_id):
    if 'time_id' in self.metadata_dict[item_id]:
      return self.metadata_dict[item_id]['time_id']
    else:
      # Fallback for older datasets.
      return self.metadata_dict[item_id]['warp_id']

  # ZCW add
  def get_kp_init(self, item_id):
    int_id = int(item_id)
    return self.kp_init[int_id, :]

  def load_depth(self, item_id) -> np.ndarray:
    item_id_4 = item_id[2:]
    # tmp = _load_image_1ch(self.data_dir / 'data' / 'depth_expected_000000.png')
    raw_depth = _load_image_1ch(self.data_dir / 'depth' / f'{item_id_4}.png')
    return raw_depth / 8000.0  # convert to meter
