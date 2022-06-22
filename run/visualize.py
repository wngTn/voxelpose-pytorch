import math
import numpy as np
import torchvision
import cv2
import os
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors as cl
import argparse

import _init_paths
from core.config import update_config
from core.config import config
from utils.utils import create_logger, load_backbone_panoptic
import torchvision.transforms as transforms
import torch.utils.data
import torch.utils.data.distributed
import torch
import torch.backends.cudnn as cudnn
import dataset
import models
import pickle
import json
import shutil

import utils.cameras as cameras

'''
This file is part of debugging the code to visualize some things
'''


# coco17
LIMBS17 = [[0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [11, 13], [13, 15],
        [6, 12], [12, 14], [14, 16], [5, 6], [11, 12]]


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize your network')
    parser.add_argument(
        '--cfg', help='experiment configure file name', type=str, default="./configs/holistic_or/prn64_cpn80x80x20.yaml")
    parser.add_argument(
        '--vis', type=str, nargs='+', default=[], choices=['img2d', 'img3d'])
    parser.add_argument(
        '--vis_output', type=str, default="output_vis"
    )
    

    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    return args


def save_debug_3d_images(meta, preds, prefix):
    basename = os.path.basename(prefix)
    dirname = os.path.dirname(prefix)

    prefix = os.path.join(dirname, basename)
    file_name = prefix + "_3d.png"

    # preds = preds.cpu().numpy()
    batch_size = meta['num_person'].shape[0]
    xplot = min(4, batch_size)
    yplot = int(math.ceil(float(batch_size) / xplot))

    width = 4.0 * xplot
    height = 4.0 * yplot
    fig = plt.figure(0, figsize=(width, height))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05,
                        top=0.95, wspace=0.05, hspace=0.15)
    for i in range(batch_size):
        num_person = meta['num_person'][i]
        joints_3d = meta['joints_3d'][i]
        joints_3d_vis = meta['joints_3d_vis'][i]
        ax = plt.subplot(yplot, xplot, i + 1, projection='3d')
        for n in range(num_person):
            joint = joints_3d[n]
            joint_vis = joints_3d_vis[n]
            for k in eval("LIMBS{}".format(len(joint))):
                if joint_vis[k[0], 0] and joint_vis[k[1], 0]:
                    x = [float(joint[k[0], 0]), float(joint[k[1], 0])]
                    y = [float(joint[k[0], 1]), float(joint[k[1], 1])]
                    z = [float(joint[k[0], 2]), float(joint[k[1], 2])]
                    ax.plot(x, y, z, c='r', lw=1.5, marker='o', markerfacecolor='w', markersize=2,
                            markeredgewidth=1)
                else:
                    x = [float(joint[k[0], 0]), float(joint[k[1], 0])]
                    y = [float(joint[k[0], 1]), float(joint[k[1], 1])]
                    z = [float(joint[k[0], 2]), float(joint[k[1], 2])]
                    ax.plot(x, y, z, c='r', ls='--', lw=1.5, marker='o', markerfacecolor='w', markersize=2,
                            markeredgewidth=1)

        colors = ['b', 'g', 'c', 'y', 'm', 'orange', 'pink', 'royalblue', 'lightgreen', 'gold']
        if preds is not None:
            pred = preds[i]
            for n in range(len(pred)):
                joint = pred[n] # joint of one person
                if joint[0, 3] >= 0:
                    for k in eval("LIMBS{}".format(len(joint))):
                        x = [float(joint[k[0], 0]), float(joint[k[1], 0])]
                        y = [float(joint[k[0], 1]), float(joint[k[1], 1])]
                        z = [float(joint[k[0], 2]), float(joint[k[1], 2])]
                        ax.plot(x, y, z, c=colors[int(n % 10)], lw=1.5, marker='o', markerfacecolor='w', markersize=2,
                                markeredgewidth=1)
    print('Wrote', file_name)
    plt.savefig(file_name)
    plt.close(0)

def image_2d_with_anno(meta, preds, prefix, batch_size):
    images = []
    for m in range(len(meta)):
        image_file = meta[m]['image'][0]
        image = cv2.imread(image_file, cv2.IMREAD_COLOR)

        file_name = prefix + '_.jpg'
        

        meta[m]['camera']['R'] = meta[m]['camera']['R'][0]
        meta[m]['camera']['T'] =  meta[m]['camera']['T'][0]
        meta[m]['camera']['k'] = meta[m]['camera']['k'][0]
        meta[m]['camera']['p'] = meta[m]['camera']['p'][0]

        colors = ['b', 'g', 'c', 'y', 'm', 'orange', 'pink', 'royalblue', 'lightgreen', 'gold']
        for i in range(batch_size):
            if preds is not None:
                pred = preds[i]
                for n in range(len(pred)):
                    joint = pred[n]
                    if joint[0, 3] >= 0:
                        for j in range(17):
                            X_0 = torch.from_numpy(np.array([joint[j, :3]]))
                            X = cameras.project_pose(X_0, meta[m]['camera'], True)
                            cv2.circle(image, (int(X[0, 0]), int(X[0, 1])), 2, tuple(reversed(255 * np.array(cl.to_rgb(colors[int(n % 10)])))), 4)
        images.append(image)

    end_image = cv2.hconcat(images)
    print('Wrote', file_name)
    cv2.imwrite(file_name, end_image)
    

def vis_joints(db):
    dict = db[0]
    image_file = dict['image']
    image = cv2.imread(image_file, cv2.IMREAD_COLOR)

    for joints in dict['pred_pose2d']:
        for i, (x, y, z) in enumerate(joints):
            if z >= 0:
                cv2.circle(image, (int(x), int(y)), 2, (0, 0, 255), 4)
                cv2.putText(image, str(i), (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    cv2.imwrite("temp/test2.jpg", image)


def prepare_out_dirs(prefix:str='output_vis/', dataDirs=['img2d', 'img3d']):
    result = []
    for dataDir in dataDirs:
        output_dir = os.path.join(prefix, dataDir)
        if os.path.exists(output_dir) and os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        print('Created', output_dir)
        os.makedirs(output_dir, exist_ok=True)
        result.append(output_dir)
    return result


COCO17_IN_BODY25 = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]


def coco17tobody25(points2d):
    dim = 3
    if len(points2d.shape) == 2:
        points2d = points2d[None, :, :]
        dim = 2
    kpts = np.zeros((points2d.shape[0], 25, 3))
    kpts[:, COCO17_IN_BODY25, :2] = points2d[:, :, :2]
    kpts[:, COCO17_IN_BODY25, 2:3] = points2d[:, :, 2:3]
    kpts[:, 8, :2] = kpts[:, [9, 12], :2].mean(axis=1)
    kpts[:, 8, 2] = kpts[:, [9, 12], 2].min(axis=1)
    kpts[:, 1, :2] = kpts[:, [2, 5], :2].mean(axis=1)
    kpts[:, 1, 2] = kpts[:, [2, 5], 2].min(axis=1)
    if dim == 2:
        kpts = kpts[0]
    return kpts




def main():
    args = parse_args()

    final_output_dir = 'output/holistic_or_synthetic/multi_person_posenet_50/prn64_cpn80x80x20/'
    out_prefix = args.vis_output

    dirs = []
    for e in args.vis:
        dirs.append(e)
    prepare_out_dirs(prefix=out_prefix, dataDirs=dirs)

    gpus = [int(i) for i in config.GPUS.split(',')]
    print('=> Loading data ..')
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    test_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
        config, config.DATASET.TEST_SUBSET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)

    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    print('=> Constructing models ..')
    model = eval('models.' + config.MODEL + '.get_multi_person_pose_net')(
        config, is_train=False)
    with torch.no_grad():
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    test_model_file = os.path.join(final_output_dir, config.TEST.MODEL_FILE)
    if config.TEST.MODEL_FILE and os.path.isfile(test_model_file):
        print('=> load models state {}'.format(test_model_file))
        model.module.load_state_dict(torch.load(test_model_file))
    else:
        raise ValueError('Check the model file for testing!')

    model.eval()
    preds = {}
    with torch.no_grad():
        for l, (inputs, targets_2d, weights_2d, targets_3d, meta, input_heatmap) in enumerate(test_loader):
            pred, _, _, _, _, _ = model(meta=meta, targets_3d=targets_3d[0], input_heatmaps=input_heatmap)
            pred = pred.detach().cpu().numpy()

            frame_num = l * 5 + 2000
            preds[frame_num] = []
            if pred is not None:
                pre = pred[0]
                for n in range(len(pre)):
                    joint = pre[n] # joint of one person
                    if joint[0, 3] >= 0:
                        # converts back to meters
                        joint[:, :3] = joint[:, :3] / 1000
                        pruned_joint = np.concatenate((joint[:, :3], joint[:, -1].reshape(17, 1)), axis=1)
                        # joints without the third column
                        preds[frame_num].append(pruned_joint)

            
            if 'img3d' in args.vis:
                prefix = '{}'.format(f'{out_prefix}/img3d/')
                save_debug_3d_images(meta[0], pred, '{}_Frame_{}'.format(prefix, l))
            if 'img2d' in args.vis:
                prefix = '{}'.format(f'{out_prefix}/img2d/')
                image_2d_with_anno(meta, pred, '{}Frame_{}'.format(prefix, l), 1)

        # saves the predictions
        with open(os.path.join(f'{out_prefix}', 'pred_voxelpose.pkl'), 'wb') as handle:
             pickle.dump(preds, handle)


if __name__ == '__main__':
    main()

