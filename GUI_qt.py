#!/usr/bin/python3
# -*- coding: utf-8 -*-


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


dataset_name = '../in/capture_0727_0' # @param {type:"string"}
data_dir = gpath.GPath(dataset_name)
print('data_dir: ', data_dir)
# assert data_dir.exists()

exp_dir = '../out/save_0727_0_ap_autoS_NE4_reg' # @param {type:"string"}
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
  load_kp_init_file=dummy_model.load_kp_init_file,  # ZCW add
  kp_dimension=dummy_model.kp_dimension)  # ZCW add


# ZCW change, use device
devices = jax.devices()[0:4]
# devices = [jax.devices()[0], jax.devices()[2], jax.devices()[3]]
print(devices)
rng, key = random.split(rng)
params = {}
model, params['model'] = models.construct_nerf(
    key,
    batch_size=train_config.batch_size,
    embeddings_dict=datasource.embeddings_dict,
    near=datasource.near,
    far=datasource.far)
    #BS_coeff_num=datasource.BS_coeff_num)  # ZCW add

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
                              chunk=6144)


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


_EPSILON = 1e-5


def my_look_at(camera,
               camera_position: np.ndarray,
               look_at_position: np.ndarray,
               up_vector: np.ndarray):
  look_at_camera = camera.copy()
  optical_axis = look_at_position - camera_position
  norm = np.linalg.norm(optical_axis)
  if norm < _EPSILON:
    raise ValueError('The camera center and look at position are too close.')
  optical_axis /= norm

  right_vector = np.cross(optical_axis, up_vector)
  norm = np.linalg.norm(right_vector)
  if norm < _EPSILON:
    raise ValueError('The up-vector is parallel to the optical axis.')
  right_vector /= norm

  # The three directions here are orthogonal to each other and form a right
  # handed coordinate system.
  camera_rotation = np.identity(3)
  camera_rotation[0, :] = right_vector
  camera_rotation[1, :] = np.cross(optical_axis, right_vector)
  camera_rotation[2, :] = optical_axis

  look_at_camera.position = camera_position
  look_at_camera.orientation = camera_rotation
  return look_at_camera


local_coor = datasource.load_local_coor()

# For the 1st key point
local_coor['bias'] = local_coor['bias'][0]
local_coor['scale'] = local_coor['scale'][0, 0]

# train_hyper_embed = state_dict['optimizer']['target']['model']['hyper_embed']['embed']['embedding']
train_hyper_embed = datasource.kp_init
train_k_points = train_hyper_embed - local_coor['bias']
train_k_points *= 1.0/local_coor['scale']
# k_point_init = train_k_points[0]


render_scale = 1.0
base_cmr = datasource.load_camera('000000').scale(render_scale)

cmr_pos_init = base_cmr.position
# look_at_position = base_cmr.position + base_cmr.optical_axis
# cmr_foc_init = datasource.scene_center * datasource.scene_scale
cmr_foc_init = train_k_points[0]
cmr_up_init = -base_cmr.orientation[1, :]
cmr_init = my_look_at(base_cmr, cmr_pos_init, cmr_foc_init, cmr_up_init)

print('cmr_pos_init: ', cmr_pos_init)
print('cmr_foc_init: ', cmr_foc_init)
print('cmr_up_init: ', cmr_up_init)

k_point_init_local = get_hyper_code(params, '000000').squeeze() if model.has_hyper_embed else None
k_point_init = k_point_init_local - local_coor['bias']
k_point_init *= 1.0/local_coor['scale']
point_2d_init = cmr_init.project(k_point_init)

print('k_point_init_local: ', k_point_init_local)
print('k_point_init: ', k_point_init)

item_id = '000000'
appearance_code = get_appearance_code(params, item_id) if model.use_nerf_embed else None
warp_code = get_warp_code(params, item_id).squeeze() if model.use_warp else None


# Setting the Qt bindings for QtPy
import sys
import cv2
import os
# envpath = '/home/zhengchengwei/miniconda3/envs/editablenerf/lib/python3.8/site-packages/cv2/qt/plugins/platforms'
# os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath
# os.environ["QT_API"] = "pyqt5"

from PyQt5 import QtWidgets
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QBrush, QPalette
from PyQt5.QtCore import Qt, QTimer

import numpy as np

import pyvista as pv
from pyvistaqt import QtInteractor, MainWindow, BackgroundPlotter


class MainLabel(QtWidgets.QLabel):

    def __init__(self, main_window=None, scalar=2):
        super().__init__()
        # self.setGeometry(30, 30, 600, 400)
        # self.resize(500, 500)
        self.scalar = scalar

        self.main_window = main_window
        # self.show()

        self.update_k_point_proj(point_2d_init)
        self.k_point_r = 8

        self.mouse_pressed = False

    def set_img(self, img):
        read_h, read_w, _ = img.shape
        color_img = cv2.resize(img, (read_w * self.scalar, read_h * self.scalar), cv2.INTER_CUBIC)

        self.height, self.width, self.depth = color_img.shape
        self.setMinimumSize(self.width, self.height)

        qt_img = QImage(color_img.data, self.width, self.height, self.width * self.depth, QImage.Format_RGB888)
        pixmap = QPixmap(qt_img)

        bg_palette = QPalette()
        bg_palette.setBrush(self.backgroundRole(), QBrush(pixmap))
        self.setPalette(bg_palette)
        self.setAutoFillBackground(True)

        self.update()

    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)

        # qp.drawPixmap(0, 0, self.pixmap)
        qp.setPen(QPen(Qt.green, 6, Qt.SolidLine))
        radius = self.k_point_r
        qp.drawEllipse(self.k_point_x - radius, self.k_point_y - radius, 2 * radius, 2 * radius)

        qp.end()


    def get_k_point_2d(self, wo_scalar=True):
        if wo_scalar:
            return np.array([self.k_point_x, self.k_point_y], dtype=np.float32) * (1 / self.scalar)
        else:
            return np.array([self.k_point_x, self.k_point_y], dtype=np.float32)

    def update_k_point_proj(self, point_2d):
        self.k_point_x = point_2d[0] * self.scalar
        self.k_point_y = point_2d[1] * self.scalar


    def update_k_point_mouse(self, event):
        self.k_point_x = event.localPos().x()
        self.k_point_y = event.localPos().y()
        self.main_window.sub_label.setText('position: (' + str(self.k_point_x) + ', ' + str(self.k_point_y) + ')')


    def mousePressEvent(self, event):
        self.mouse_pressed = True
        self.update_k_point_mouse(event)
        self.update()


    def mouseReleaseEvent(self, event):
        self.mouse_pressed = False
        self.main_window.update_k_point_3d()
        self.main_window.render_img()


    def mouseMoveEvent(self, event):
        if self.mouse_pressed:
            self.update_k_point_mouse(event)
            self.update()


class MyMainWindow(MainWindow):

    def __init__(self, parent=None, show=True):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.frame = QtWidgets.QFrame()
        self.resize(1080, 480)
        self.setWindowTitle('Edit GUI')

        # add the pyvista interactor object
        self.plotter = QtInteractor(self.frame)
        self.plotter.setFocusPolicy(Qt.NoFocus)

        self.render_btn = QtWidgets.QPushButton('Render (or press Enter)', self)
        self.render_btn.resize(self.render_btn.sizeHint())
        self.render_btn.setShortcut('Ctrl+Q')
        # self.btn.move(50, 50)

        self.record_btn = QtWidgets.QPushButton('', self)
        self.record_btn.setFixedSize(56, 56)
        self.record_btn.setStyleSheet("background-image : url(GUI/record_50px.jpg);"
                                      "border-radius: 10px;"
                                      "border: 3px groove gray;"
                                      "border-style: outset;")

        self.sub_label = QtWidgets.QLabel()

        rec_layout = QtWidgets.QHBoxLayout()
        rec_layout.addWidget(self.record_btn)
        rec_layout.addWidget(self.sub_label)

        self.main_label = MainLabel(main_window=self)

        # layout
        sub_layout = QtWidgets.QVBoxLayout()
        sub_layout.addWidget(self.plotter.interactor)
        sub_layout.addStretch()
        sub_layout.addWidget(self.render_btn)
        sub_layout.addStretch()

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(self.main_label)
        # main_layout.addStretch()
        main_layout.addLayout(rec_layout)
        sub_layout.addStretch()

        global_layout = QtWidgets.QHBoxLayout()
        global_layout.addLayout(main_layout)
        global_layout.addLayout(sub_layout)

        self.frame.setLayout(global_layout)
        self.setCentralWidget(self.frame)

        # simple menu to demo functions
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')
        exitButton = QtWidgets.QAction('Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.triggered.connect(self.close)
        fileMenu.addAction(exitButton)
        seqButton = QtWidgets.QAction('Render Sequence', self)
        seqButton.triggered.connect(self.render_record)
        fileMenu.addAction(seqButton)

        # allow adding a sphere
        meshMenu = mainMenu.addMenu('Setting')
        self.reset_cmr_action = QtWidgets.QAction('Reset Camera', self)
        self.reset_cmr_action.triggered.connect(self.reset_cmr)
        meshMenu.addAction(self.reset_cmr_action)
        self.reset_k_point_action = QtWidgets.QAction('Reset Key Point', self)
        self.reset_k_point_action.triggered.connect(self.reset_k_point)
        meshMenu.addAction(self.reset_k_point_action)

        self.signal_close.connect(self.plotter.close)
        self.render_btn.clicked.connect(self.render_img)

        self.k_point_3d = k_point_init
        self.img_cmr = cmr_init

        self.recording = False
        self.timer = QTimer()
        self.timer.stop()
        self.timer.setInterval(200)
        self.timer.timeout.connect(self.record)
        self.record_list = []

        self.add_sphere()
        self.reset_cmr()
        self.render_img()

        if show:
            self.show()


    def add_sphere(self):
        """ add a sphere to the pyqt frame """
        # self.plotter.clear()
        sphere = pv.Sphere(radius=0.03, center=cmr_foc_init, direction=cmr_up_init)
        self.plotter.add_mesh(sphere, show_edges=True, show_scalar_bar=True)
        self.plotter.add_camera_orientation_widget(animate=False)
        # self.plotter.add_orientation_widget()


    def reset_cmr(self):
        """ add a sphere to the pyqt frame """

        # self.plotter.reset_camera()
        self.plotter.set_position(cmr_pos_init)
        self.plotter.set_focus(cmr_foc_init)
        self.plotter.set_viewup(cmr_up_init, reset=False)

        self.plotter.show()
        # self.plotter.update()


    def reset_k_point(self):
        self.k_point_3d = k_point_init
        self.render_img()


    def update_k_point_3d(self):
        # depth = np.linalg.norm(self.k_point_3d - self.img_cmr.position)
        # z_val = self.img_cmr.points_to_local_points(self.k_point_3d)[2]
        k_point_2d = self.main_label.get_k_point_2d(wo_scalar=True)
        # print('z_val', z_val)
        # print('k_point_2d', k_point_2d)

        train_k_point_2d = self.img_cmr.project(train_k_points)
        train_diff = np.abs(train_k_point_2d - k_point_2d)
        train_diff = np.mean(train_diff, axis=1)
        min_idx = np.argmin(train_diff)
        min_train_point = train_k_points[min_idx]
        z_val = self.img_cmr.points_to_local_points(min_train_point)[2]
        self.k_point_3d = self.img_cmr.pixels_to_points(k_point_2d, z_val)
        self.sub_label.setText('Key Point Updated ' + str(self.k_point_3d))


    def keyPressEvent(self, event):
        print('press', str(event.key()))
        if event.key() == Qt.Key_Return:
            self.render_img()
        elif event.key() == Qt.Key_R:
            self.press_record_btn()
        else:
            MainWindow.keyPressEvent(self, event)

    def press_record_btn(self):
        if not self.recording:
            print('start record')
            self.start_record()
        else:
            print('end record')
            self.end_record()
        self.recording = not self.recording


    def start_record(self):
        self.record_list.clear()
        self.timer.start()
        self.record_btn.setStyleSheet("background-image : url(GUI/stop_50px.jpg);"
                                      "border-radius: 10px;"
                                      "border: 3px groove gray;"
                                      "border-style: outset;")

    def record(self):
        k_point_2d = self.main_label.get_k_point_2d(wo_scalar=True)
        self.record_list.append(k_point_2d)
        print(len(self.record_list))

    def end_record(self):
        self.timer.stop()
        print(self.record_list)
        # self.render_record()

        self.record_btn.setStyleSheet("background-image : url(GUI/record_50px.jpg);"
                                      "border-radius: 10px;"
                                      "border: 3px groove gray;"
                                      "border-style: outset;")


    def get_cmr(self):
        # GET new camera
        cmr = self.plotter.camera_position
        # self.sub_label.setText('Rendering...')
        # print(str(cmr))
        cmr_pos = np.array(cmr.position)
        cmr_foc = np.array(cmr.focal_point)
        cmr_up = np.array(cmr.viewup)

        new_cmr = my_look_at(base_cmr, cmr_pos, cmr_foc, cmr_up)
        print(new_cmr.position)
        return new_cmr


    def render_img(self):
        new_cmr = self.get_cmr()

        # GET new k point
        new_point_2d = new_cmr.project(self.k_point_3d)
        self.main_label.update_k_point_proj(new_point_2d)

        # RENDER
        # item_id = '000000'
        # appearance_code = get_appearance_code(params, item_id).squeeze() if model.use_nerf_embed else None
        # appearance_code = get_appearance_code(params, item_id) if model.use_nerf_embed else None
        # warp_code = get_warp_code(params, item_id).squeeze() if model.use_warp else None
        # hyper_code = get_hyper_code(params, item_id).squeeze() if model.has_hyper_embed else None
        hyper_code = self.k_point_3d * local_coor['scale'] + local_coor['bias']
        batch = make_batch(new_cmr, appearance_code, warp_code, hyper_code)

        render = render_fn(state, batch, rng=rng)
        pred_rgb = np.array(render['rgb'])

        # ZCW add
        # Image.fromarray(np.uint8(pred_rgb * 255)).save(f"../visual/render_test/rgb_test.png")
        img_int8 = np.uint8(pred_rgb * 255)
        self.main_label.set_img(img_int8)
        self.sub_label.setText('Render Done')
        self.img_cmr = new_cmr

        print('cmr_pos: ', new_cmr.position)
        print('cmr_ori: ', new_cmr.orientation)


    def render_record(self):
        train_k_point_2d = self.img_cmr.project(train_k_points)

        record_3d = []
        for i, k_point_2d in enumerate(self.record_list):
            print(i, ' / ', len(self.record_list))
            # k_point_2d = np.array(k_point_2d, dtype=np.float32)
            train_diff = np.abs(train_k_point_2d - k_point_2d)
            train_diff = np.mean(train_diff, axis=1)
            min_idx = np.argmin(train_diff)
            min_train_point = train_k_points[min_idx]
            z_val = self.img_cmr.points_to_local_points(min_train_point)[2]
            k_point_3d = self.img_cmr.pixels_to_points(k_point_2d, z_val)
            record_3d.append(k_point_3d)

            hyper_code = k_point_3d * local_coor['scale'] + local_coor['bias']
            batch = make_batch(self.img_cmr, appearance_code, warp_code, hyper_code)

            render = render_fn(state, batch, rng=rng)
            pred_rgb = np.array(render['rgb'])
            Image.fromarray(np.uint8(pred_rgb * 255)).save(f"../record/render/{i:06d}.png")

        np.savetxt('../record/render/k_point_2d.txt', self.record_list)
        np.savetxt('../record/render/k_point_3d.txt', np.array(record_3d))
        # Smooth the trail if necessary


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MyMainWindow()
    sys.exit(app.exec_())