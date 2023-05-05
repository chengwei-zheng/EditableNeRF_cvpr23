# In[ ]:
import json
import numpy as np
import cv2
from typing import List, Tuple, Any, Optional, Sequence, Union

from scipy.ndimage.filters import gaussian_filter

from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
# from matplotlib.patches import Circle
from itertools import combinations

from nerfies import camera as cam
from nerfies import gpath
from nerfies import types

from pathlib import Path

# Change these parameters
scene_root = 'your_path/in/capture_demo'  # @param {type: 'string'}
ori_cmr_render_root = 'your_path/out/save_demo/renders/'  # @param {type: 'string'}
scalar = 1.0  # @param {type: 'number'}


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

def load_camera(
    camera_path: gpath.GPath,
    scale_factor: float = 1.0,
    scene_center: Optional[np.ndarray] = None,
    scene_scale: Optional[Union[float, np.ndarray]] = None) -> cam.Camera:
  """Loads camera and rays defined by the center pixels of a camera.

  Args:
    camera_path: a path to an sfm_camera.Camera proto.
    scale_factor: a factor to scale the camera image by.
    scene_center: the center of the scene where the camera will be centered to.
    scene_scale: the scale of the scene by which the camera will also be scaled
      by.

  Returns:
    A Camera instance.
  """
  if camera_path.suffix == '.pb':
    camera = cam.Camera.from_proto(camera_path)
  elif camera_path.suffix == '.json':
    camera = cam.Camera.from_json(camera_path)
  else:
    raise ValueError('File must have extension .pb or .json.')

  if scale_factor != 1.0:
    camera = camera.scale(scale_factor)

  if scene_center is not None:
    camera.position = camera.position - scene_center
  if scene_scale is not None:
    camera.position = camera.position * scene_scale

  return camera



# In[ ]:
# Compute 3d init key point positions
if 1:
  scene_center, scene_scale, near, far = load_scene_info(scene_root)

  all_kp_2d_init = np.loadtxt('kp_init/kp_2d_init.txt')
  kp_num = all_kp_2d_init.shape[1] // 2
  frame_num = all_kp_2d_init.shape[0]
  print('key point number: ', kp_num)

  k_points_auto = np.zeros((frame_num, 0), dtype=np.float64)
  for kp in range(kp_num):
    rc = all_kp_2d_init[:, 2*kp:2*kp+2]
    k_points = []

    # From 2d to 3d
    for i in range(0, rc.shape[0]):
      r = rc[i][0] / scalar
      c = rc[i][1] / scalar

      npy_path = gpath.GPath(ori_cmr_render_root, f'med_points_{i:06d}.npy')
      med_points = np.load(str(npy_path))
      k_points.append(med_points[r.astype(int)][c.astype(int)])

    # Local coordinate systems for key points
    coor_max = np.percentile(k_points, 90, axis=0)
    coor_min = np.percentile(k_points, 10, axis=0)
    coor_range = coor_max - coor_min
    coor_scale = 1 / np.max(coor_range)
    k_points = np.asarray(k_points) * coor_scale

    origin = -np.mean(k_points, axis=0)
    k_points += origin
    np.savetxt('kp_init/k_points_auto_' + str(kp) + '.txt', k_points, delimiter='\t')
    k_points_auto = np.concatenate((k_points_auto, k_points), axis=1)

    coor_data = np.insert(origin, 0, coor_scale)
    np.savetxt('kp_init/k_points_auto_coor_' + str(kp) + '.txt', coor_data, delimiter='\n')

  np.savetxt('kp_init/k_points_auto.txt', k_points_auto)
  print('done')

# remove duplicate key points if necessary
if 0:
  kp_list = list(range(kp_num))
  for (i, j) in list(combinations(kp_list, 2)):
    kp_i = np.loadtxt('kp_init/k_points_auto_' + str(i) + '.txt')
    kp_j = np.loadtxt('kp_init/k_points_auto_' + str(j) + '.txt')
    diff = np.mean(np.abs(kp_i - kp_j))
    print('Difference between kp ', i, ' and kp ', j, 'is', diff)
    if diff < 0.02:
      print('Duplicate key point: ', i, ' and ', j)
