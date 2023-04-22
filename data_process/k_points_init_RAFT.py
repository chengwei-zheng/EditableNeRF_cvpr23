import sys

sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

DEVICE = 'cuda'


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo, idx):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    path = 'out/' + str(idx) + '.png'
    # cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    cv2.imwrite(path, img_flo[:, :, [2, 1, 0]])
    cv2.waitKey()


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    kp_file = args.kp_file
    skip_prop = int(args.skip_prop)

    flow_file = os.path.join(kp_file, 'flow')
    if not os.path.exists(flow_file):
        os.makedirs(flow_file)
    if not os.path.exists('out'):
        os.makedirs('out')

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))

        images = sorted(images)
        frame_num = len(images)

        kp_2d_ref = np.load(kp_file + '/detect_kp_2d_ref.npy')
        kp_num = kp_2d_ref.shape[0]

        reference_frames = np.atleast_1d(np.loadtxt(kp_file + '/reference_frame.txt', dtype=np.int32))

        # Key point initialization
        all_kp_2d_init = np.zeros((frame_num, 0), dtype=np.float64)
        for kp in range(kp_num):
            ref_frame = reference_frames[kp]
            rc = kp_2d_ref[kp].copy()

            forward_idx = range(ref_frame, frame_num)
            forward_list = list(zip(forward_idx[:-1], forward_idx[1:]))
            backward_idx = range(0, ref_frame + 1)
            backward_list = list(zip(backward_idx[1:], backward_idx[:-1]))[::-1]
            full_list = forward_list + backward_list

            kp_2d_init = np.zeros((frame_num, 2), dtype=np.float64)

            if skip_prop:
                img_skip_ref = load_image(images[ref_frame])
                padder_skip = InputPadder(img_skip_ref.shape)

            # Propagate frame-by-frame
            for idx0, idx1 in full_list:
                idx_str = str(kp) + '_' + str(idx0) + '_' + str(idx1)
                if idx0 == ref_frame:
                    rc = kp_2d_ref[kp].copy()

                image1 = load_image(images[idx0])
                image2 = load_image(images[idx1])
                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)

                _, flow_up = model(image1, image2, iters=20, test_mode=True)
                viz(image1, flow_up, idx_str)
                flow_np = flow_up[0].permute(1, 2, 0).cpu().numpy().copy()

                image_p = image1[0].permute(1, 2, 0).cpu().numpy().copy()

                # Skipping propagation
                if skip_prop and (idx0 - ref_frame) % skip_prop == 0 and idx0 != ref_frame:
                    img_skip_t = load_image(images[idx0])
                    img_skip_ref, img_skip_t = padder_skip.pad(img_skip_ref, img_skip_t)

                    _, flow_for_up = model(img_skip_ref, img_skip_t, iters=20, test_mode=True)
                    viz(img_skip_ref, flow_for_up, str(idx0) + '_skip')
                    flow_for_np = flow_for_up[0].permute(1, 2, 0).cpu().numpy().copy()

                    _, flow_back_up = model(img_skip_t, img_skip_ref, iters=20, test_mode=True)
                    flow_back_np = flow_back_up[0].permute(1, 2, 0).cpu().numpy().copy()

                    img_ref = img_skip_ref[0].permute(1, 2, 0).cpu().numpy().copy()
                    img_t = img_skip_t[0].permute(1, 2, 0).cpu().numpy().copy()

                    ref_rc = kp_2d_ref[kp].copy()

                    # Forward optical flow
                    for_rc = ref_rc + flow_for_np[round(ref_rc[0]), round(ref_rc[1])][::-1]
                    cv2.circle(img_t, (for_rc[1].astype(int), for_rc[0].astype(int)), 5, (255, 0, 0), 3)
                    cv2.imwrite(flow_file + '/skip_for_' + str(idx0) + '.png', img_t[:, :, ::-1])

                    # Backward optical flow
                    back_rc = for_rc + flow_back_np[round(for_rc[0]), round(for_rc[1])][::-1]
                    cv2.circle(img_ref, (back_rc[1].astype(int), back_rc[0].astype(int)), 5, (255, 0, 0), 3)
                    cv2.imwrite(flow_file + '/skip_back_' + str(idx0) + '.png', img_ref[:, :, ::-1])

                    # Compute confidence
                    conf_inv = np.linalg.norm(ref_rc - back_rc)
                    if conf_inv < 10:
                        print('replace', rc, 'with', for_rc, ', conf_inv:', conf_inv)
                        rc = np.array(for_rc)
                    else:
                        print('conf_inv:', conf_inv)

                cv2.circle(image_p, (rc[1].astype(int), rc[0].astype(int)), 5, (255, 0, 0), 3)
                cv2.imwrite(flow_file + '/img_' + idx_str + '.png', image_p[:, :, ::-1])
                print(idx_str)

                kp_2d_init[idx0] = rc.copy()
                rc = rc + flow_np[round(rc[0]), round(rc[1])][::-1]

                if idx1 == forward_idx[-1] or idx1 == backward_idx[0]:
                    kp_2d_init[idx1] = rc.copy()

            np.savetxt(kp_file + '/kp_2d_init_' + str(kp) + '.txt', kp_2d_init)

            all_kp_2d_init = np.concatenate((all_kp_2d_init, kp_2d_init), axis=1)

        np.savetxt(kp_file + '/kp_2d_init.txt', all_kp_2d_init)

        # Save flow for training
        if not os.path.exists('flow'):
            os.makedirs('flow')
        for idx, (imfile1, imfile2) in enumerate(zip(images[:-1], images[1:])):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            _, flow_up = model(image1, image2, iters=20, test_mode=True)
            viz(image1, flow_up, idx)
            flow_np = flow_up[0].permute(1, 2, 0).cpu().numpy().copy()
            # np.save(f'out/flow_{idx:06d}.npy', flow_np)

            re_shape = (flow_np.shape[1] // 4, flow_np.shape[0] // 4)
            flow_map = cv2.resize(flow_np, re_shape)  # (320, 180)
            flow_map /= 4.0
            np.save(f'flow/flow_{idx:06d}.npy', flow_map)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--kp_file', help="key point data file")
    parser.add_argument('--skip_prop', help="if use skipping propagation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
