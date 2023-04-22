# In[ ]:
import numpy as np
import cv2
import json
import os

from typing import Tuple, Optional, Union
from nerfies import camera as cam
from nerfies import gpath
from nerfies import types

from scipy.ndimage.filters import gaussian_filter
from skimage.feature import peak_local_max

# Change these parameters
scene_root = 'your_path/in/capture_demo'  # @param {type: 'string'}
fix_cmr_render_root = 'your_path/out/save_demo/renders/'  # @param {type: 'string'}
frame_num = 500  # @param {type: 'number'}
colmap_image_scale = 4  # @param {type: 'number'}
gaussian_sigma = 11  # @param {type: 'number'}

if not os.path.exists('kp_init'):
  os.makedirs('kp_init')

# In[ ]:
# Compute variances of ambient coordinates in 2D
if 1:
  full_ambient = []
  for idx in range(0, frame_num):
    med_points = np.load(fix_cmr_render_root + f'med_warped_points_{idx:06d}.npy')
    ambient_points = med_points.squeeze()[..., 3:]
    full_ambient.append(ambient_points)
    # cv2.imwrite(f'var/ambient/ambient_{i:06d}.png', (ambient_points + 2) * 64)
  full_ambient = np.asarray(full_ambient)
  var_ambient = np.var(full_ambient, axis=0)
  if var_ambient.ndim > 2:
    var_ambient = np.mean(var_ambient, axis=2)
  max_var = np.max(var_ambient)
  var_ambient = var_ambient / max_var

  cv2.imwrite('kp_init/var.png', var_ambient * 255)


# In[ ]:
# Find reference key point in 2D
if 1:
  var = cv2.imread('kp_init/var.png', cv2.IMREAD_UNCHANGED)
  var = gaussian_filter(var, sigma=gaussian_sigma)
  cv2.imwrite('kp_init/var_blur.png', var)

  detect_kp_2d = peak_local_max(var, min_distance=20, threshold_rel=0.3)
  np.savetxt('kp_init/detect_kp_2d.txt', detect_kp_2d)
  # np.save('kp_init/detect_kp_2d.npy', detect_kp_2d * colmap_image_scale)
  print('2D reference key point: ', detect_kp_2d)

# In[ ]:
# 2D key point visualization
if 1:
  var_3ch = cv2.imread('kp_init/var.png', cv2.IMREAD_COLOR)
  var_blur_3ch = cv2.imread('kp_init/var_blur.png', cv2.IMREAD_COLOR)
  for rc in detect_kp_2d:
    cv2.circle(var_3ch, (rc[1], rc[0]), 4, (0, 0, 255), 2)
    cv2.circle(var_blur_3ch, (rc[1], rc[0]), 4, (0, 0, 255), 2)
  cv2.imwrite('kp_init/detect_kp_2d_visual.png', var_3ch)
  cv2.imwrite('kp_init/detect_kp_2d_visual_blur.png', var_blur_3ch)


# In[ ]:
# Compute reference key point in 3D, based on 2D version.
# This method may not as accurate as the full-3D version,
# but we highly recommend this method in order to check the intermediate results.
if 1:
  kp_num = detect_kp_2d.shape[0]
  reference_frames = []

  kp_3d_vec = []
  for kp in range(kp_num):
    kp_str = '_' + str(kp)
    r = int(detect_kp_2d[kp, 0])
    c = int(detect_kp_2d[kp, 1])

    depth_vec = []
    ambient_vec = []
    for idx in range(frame_num):
      depth_map = np.load(fix_cmr_render_root + f'depth_median_{idx:06d}.npy')
      depth_vec.append(depth_map[r, c])
      med_points = np.load(fix_cmr_render_root + f'med_warped_points_{idx:06d}.npy')
      ambient_point = med_points.squeeze()[r, c, 3:]
      ambient_vec.append(ambient_point)
    depth_vec = np.array(depth_vec)
    ambient_vec = np.array(ambient_vec)
    # np.savetxt('kp_init/depth_vec' + kp_str + '.txt', depth_vec, delimiter='\t')
    # np.savetxt('kp_init/ambient_vec' + kp_str + '.txt', ambient_vec, delimiter='\t')

    # Find 3D key point along z direction based on the 2D key point
    max_depth = np.max(depth_vec)
    min_depth = np.min(depth_vec)
    z_voxel_num = 8
    voxel_size = (max_depth - min_depth) / z_voxel_num
    # voxel_idx_vec = []

    volume_ambient = [[] for v in range(z_voxel_num)]
    volume_idx = [[] for v in range(z_voxel_num)]
    for idx in range(0, frame_num):
      depth_val = depth_vec[idx]
      voxel_idx = int((depth_val - min_depth) / voxel_size)
      voxel_idx = min(voxel_idx, z_voxel_num - 1)
      # voxel_idx_vec.append(voxel_idx)
      volume_ambient[voxel_idx].append(ambient_vec[idx])
      volume_idx[voxel_idx].append(idx)
    # np.savetxt('kp_init/voxel_idx_vec.txt', voxel_idx_vec, delimiter='\t')
    var_z_voxel = []

    for v in volume_ambient:
      if len(v) < 5:  # too few samples
        var_z_voxel.append(0)
        continue
      voxel_hyper = np.array(v)
      voxel_var = np.var(voxel_hyper, axis=0)
      voxel_var = np.mean(voxel_var)
      var_z_voxel.append(voxel_var)
      # print(voxel_var)

    max_var_volume_idx = np.argmax(var_z_voxel)
    ref_frame = volume_idx[max_var_volume_idx][0]
    print('reference frame: ', ref_frame)
    reference_frames.append(ref_frame)

    med_point_map = np.load(fix_cmr_render_root + f'med_points_{ref_frame:06d}.npy')
    kp_3d_vec.append(med_point_map[r, c])

  np.savetxt('kp_init/reference_frame.txt', reference_frames, delimiter='\t')


# In[ ]:

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
# Compute 2d reference key points in their reference frames
if 1:
  scene_center, scene_scale, near, far = load_scene_info(scene_root)

  detect_kp_2d_ref = []
  for kp in range(kp_num):
    cmr_path = gpath.GPath(scene_root, f'camera/{reference_frames[kp]:06d}.json')
    ori_cmr = load_camera(cmr_path, scale_factor=1.0 / colmap_image_scale, scene_center=scene_center, scene_scale=scene_scale)
    kp_2d_ref = ori_cmr.project(kp_3d_vec[kp])
    kp_2d_ref = kp_2d_ref[::-1] * colmap_image_scale
    print(kp_2d_ref)
    detect_kp_2d_ref.append(kp_2d_ref)

  detect_kp_2d_ref = np.asarray(detect_kp_2d_ref)
  np.save('kp_init/detect_kp_2d_ref.npy', detect_kp_2d_ref)
  np.savetxt('kp_init/detect_kp_2d_ref.txt', detect_kp_2d_ref)

