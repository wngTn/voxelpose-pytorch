# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
import json_tricks as json
import pickle
import scipy.io as scio
import logging
import copy
import os
from collections import OrderedDict

from dataset.JointsDataset import JointsDataset
from utils.cameras_cpu import project_pose

import json_tricks as json
import pickle
import logging
import copy
import random
import cv2

import os
from collections import OrderedDict

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import rotate_points, get_scale
from utils.cameras_cpu import project_pose
from utils.cameras_cpu import rot_trans_to_homogenous, homogenous_to_rot_trans
from utils.cameras_cpu import rotation_to_homogenous
from scipy.spatial.transform import Rotation

# CAMPUS_JOINTS_DEF = {
#     'Right-Ankle': 0,
#     'Right-Knee': 1,
#     'Right-Hip': 2,
#     'Left-Hip': 3,
#     'Left-Knee': 4,
#     'Left-Ankle': 5,
#     'Right-Wrist': 6,
#     'Right-Elbow': 7,
#     'Right-Shoulder': 8,
#     'Left-Shoulder': 9,
#     'Left-Elbow': 10,
#     'Left-Wrist': 11,
#     'Bottom-Head': 12,
#     'Top-Head': 13
# }

# LIMBS = [
#     [0, 1],
#     [1, 2],
#     [3, 4],
#     [4, 5],
#     [2, 3],
#     [6, 7],
#     [7, 8],
#     [9, 10],
#     [10, 11],
#     [2, 8],
#     [3, 9],
#     [8, 12],
#     [9, 12],
#     [12, 13]
# ]

coco_joints_def = {0: 'nose',
                   1: 'Leye', 2: 'Reye', 3: 'Lear', 4: 'Rear',
                   5: 'Lsho', 6: 'Rsho',
                   7: 'Lelb', 8: 'Relb',
                   9: 'Lwri', 10: 'Rwri',
                   11: 'Lhip', 12: 'Rhip',
                   13: 'Lkne', 14: 'Rkne',
                   15: 'Lank', 16: 'Rank'}

LIMBS = [[0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [11, 13],
         [13, 15],
         [6, 12], [12, 14], [14, 16], [5, 6], [11, 12]]


class HolisticOR(JointsDataset):
    def __init__(self, cfg, image_set, is_train, transform=None):
        self.pixel_std = 200.0
        self.joints_def = coco_joints_def
        super().__init__(cfg, image_set, is_train, transform)
        self.limbs = LIMBS
        self.num_joints = len(coco_joints_def)
        self.cam_list = [0, 1, 2, 3, 4, 5]
        self.num_views = len(self.cam_list)
        self.frame_range = list(range(2000, 2985, 5))

        self.pred_pose2d = self._get_pred_pose2d()
        self.db = self._get_db()

        self.db_size = len(self.db)

    def _get_pred_pose2d(self):
        file = os.path.join(self.dataset_root, "pred_holistic_or_dekr_coco.pkl")
        with open(file, "rb") as pfile:
            logging.info("=> load {}".format(file))
            pred_2d = pickle.load(pfile)

        return pred_2d

    def _get_db(self):
        width = 360
        height = 288

        db = []
        cameras = self._get_cams()

        # datafile = os.path.join(self.dataset_root, 'actorsGT.mat')
        # data = scio.loadmat(datafile)
        # actor_3d = np.array(np.array(data['actor3D'].tolist()).tolist()).squeeze()  # num_person * num_frame

        num_person = 6
        # num_frames = len(actor_3d[0])

        for i in self.frame_range:
            for k, cam in cameras.items():
                image = osp.join("cn0" + str(int(k) + 1), "{1:010d}_color.jpg".format(k, i))

                all_poses_3d = []
                all_poses_vis_3d = []
                all_poses = []
                all_poses_vis = []
                preds = []
                # for person in range(num_person):
                #     pose3d = actor_3d[person][i] * 1000.0
                #     if len(pose3d[0]) > 0:
                #         all_poses_3d.append(pose3d)
                #         all_poses_vis_3d.append(np.ones((self.num_joints, 3)))

                #         pose2d = project_pose(pose3d, cam)

                #         x_check = np.bitwise_and(pose2d[:, 0] >= 0,
                #                                  pose2d[:, 0] <= width - 1)
                #         y_check = np.bitwise_and(pose2d[:, 1] >= 0,
                #                                  pose2d[:, 1] <= height - 1)
                #         check = np.bitwise_and(x_check, y_check)

                #         joints_vis = np.ones((len(pose2d), 1))
                #         joints_vis[np.logical_not(check)] = 0
                #         all_poses.append(pose2d)
                #         all_poses_vis.append(
                #             np.repeat(
                #                 np.reshape(joints_vis, (-1, 1)), 2, axis=1))

                pred_index = '{}_{}'.format(k, i)
                preds = self.pred_pose2d[pred_index]
                preds = [np.array(p) for p in preds]

                db.append({
                    'image': osp.join(self.dataset_root, image),
                    'joints_3d': all_poses_3d,
                    'joints_3d_vis': all_poses_vis_3d,
                    'joints_2d': all_poses,
                    'joints_2d_vis': all_poses_vis,
                    'camera': cam,
                    'pred_pose2d': preds
                })
        return db

    def _get_cams(self):
        # bring our calibration files into format of voxelpose
        cameras = OrderedDict()
        cams = sorted(next(os.walk(self.dataset_root))[1])
        for idx, cam_id in enumerate(cams):
            ds = self._get_single_cam(cam_id)
            cameras[str(int(cam_id[-1]) - 1)] = ds

        for id, cam in cameras.items():
            for k, v in cam.items():
                cameras[id][k] = np.array(v)

        return cameras

    def _get_single_cam(self, cam):
        ds = OrderedDict()
        scaling = 1000
        intrinsics = osp.join(self.dataset_root, cam, 'camera_calibration.yml')
        assert osp.exists(intrinsics)
        fs = cv2.FileStorage(intrinsics, cv2.FILE_STORAGE_READ)
        color_intrinsics = fs.getNode("undistorted_color_camera_matrix").mat()
        ds['fx'] = color_intrinsics[0, 0]
        ds['fy'] = color_intrinsics[1, 1]
        ds['cx'] = color_intrinsics[0, 2]
        ds['cy'] = color_intrinsics[1, 2]
        # images are undistorted! Just put 0. Voxelpose assumes just 4 dist coeffs
        dist = fs.getNode("color_distortion_coefficients").mat()
        # ds['k'] = np.array(dist[[0, 1, 4, 5, 6, 7]])
        # ds['p'] = np.array(dist[2:4])
        # we learn on undistorted images
        ds['k'] = np.zeros((3, 1))
        ds['p'] = np.zeros((2, 1))

        depth2color_r = fs.getNode('depth2color_rotation').mat()
        # depth2color_t is in mm by default, change all to meters
        depth2color_t = fs.getNode('depth2color_translation').mat()

        depth2color = rot_trans_to_homogenous(depth2color_r, depth2color_t.reshape(3))
        ds["depth2color"] = depth2color

        extrinsics = osp.join(self.dataset_root, cam, "world2camera.json")
        with open(extrinsics, 'r') as f:
            ext = json.load(f)
            ext = ext if 'value0' not in ext else ext['value0']
            trans = np.array([x for x in ext['translation'].values()])
            # NOTE: world2camera translation convention is in meters. Here we convert
            # to mm. Seems like Voxelpose was using mm as well.
            trans = trans * 1000
            _R = ext['rotation']
            rot = Rotation.from_quat([_R['x'], _R['y'], _R['z'], _R['w']]).as_matrix()
            ext_homo = rot_trans_to_homogenous(rot, trans)
            # flip coordinate transform back to opencv convention

        yz_flip = rotation_to_homogenous(np.pi * np.array([1, 0, 0]))
        YZ_SWAP = rotation_to_homogenous(np.pi / 2 * np.array([1, 0, 0]))

        # ds["id"] = cam
        # first swap into OPENGL convention, then we can apply intrinsics.
        # then swap into our own Z-up prefered format..
        depth2world = YZ_SWAP @ ext_homo @ yz_flip
        # print(f"{cam} extrinsics:", depth2world)

        # depth_R, depth_T = homogenous_to_rot_trans(depth2world)
        # ds["depth2world"] = depth2world
        color2world = depth2world @ np.linalg.inv(depth2color)
        # ds["color2world"] = color2world
        # voxelpose uses weird convention of subtracting translation
        # for world2camera transformation. We return world2camera
        # but with T according to their convention
        R, T = homogenous_to_rot_trans(np.linalg.inv(color2world))
        ds["R"] = R
        ds["T"] = T
        return ds


    def __getitem__(self, idx):
        input, target_heatmap, target_weight, target_3d, meta, input_heatmap = [], [], [], [], [], []
        for k in range(self.num_views):
            i, th, tw, t3, m, ih = super().__getitem__(self.num_views * idx + k)
            input.append(i)
            target_heatmap.append(th)
            target_weight.append(tw)
            input_heatmap.append(ih)
            target_3d.append(t3)
            meta.append(m)
        return input, target_heatmap, target_weight, target_3d, meta, input_heatmap

    def __len__(self):
        return self.db_size // self.num_views

    # def evaluate(self, preds, recall_threshold=500):
    #     datafile = os.path.join(self.dataset_root, 'actorsGT.mat')
    #     data = scio.loadmat(datafile)
    #     actor_3d = np.array(np.array(data['actor3D'].tolist()).tolist()).squeeze()  # num_person * num_frame
    #     num_person = len(actor_3d)
    #     total_gt = 0
    #     match_gt = 0

    #     limbs = [[0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8], [9, 10], [10, 11], [12, 13]]
    #     correct_parts = np.zeros(num_person)
    #     total_parts = np.zeros(num_person)
    #     alpha = 0.5
    #     bone_correct_parts = np.zeros((num_person, 10))

    #     for i, fi in enumerate(self.frame_range):
    #         pred_coco = preds[i].copy()
    #         pred_coco = pred_coco[pred_coco[:, 0, 3] >= 0, :, :3]
    #         pred = np.stack([self.coco2campus3D(p) for p in copy.deepcopy(pred_coco[:, :, :3])])

    #         for person in range(num_person):
    #             gt = actor_3d[person][fi] * 1000.0
    #             if len(gt[0]) == 0:
    #                 continue

    #             mpjpes = np.mean(np.sqrt(np.sum((gt[np.newaxis] - pred) ** 2, axis=-1)), axis=-1)
    #             min_n = np.argmin(mpjpes)
    #             min_mpjpe = np.min(mpjpes)
    #             if min_mpjpe < recall_threshold:
    #                 match_gt += 1
    #             total_gt += 1

    #             for j, k in enumerate(limbs):
    #                 total_parts[person] += 1
    #                 error_s = np.linalg.norm(pred[min_n, k[0], 0:3] - gt[k[0]])
    #                 error_e = np.linalg.norm(pred[min_n, k[1], 0:3] - gt[k[1]])
    #                 limb_length = np.linalg.norm(gt[k[0]] - gt[k[1]])
    #                 if (error_s + error_e) / 2.0 <= alpha * limb_length:
    #                     correct_parts[person] += 1
    #                     bone_correct_parts[person, j] += 1
    #             pred_hip = (pred[min_n, 2, 0:3] + pred[min_n, 3, 0:3]) / 2.0
    #             gt_hip = (gt[2] + gt[3]) / 2.0
    #             total_parts[person] += 1
    #             error_s = np.linalg.norm(pred_hip - gt_hip)
    #             error_e = np.linalg.norm(pred[min_n, 12, 0:3] - gt[12])
    #             limb_length = np.linalg.norm(gt_hip - gt[12])
    #             if (error_s + error_e) / 2.0 <= alpha * limb_length:
    #                 correct_parts[person] += 1
    #                 bone_correct_parts[person, 9] += 1

    #     actor_pcp = correct_parts / (total_parts + 1e-8)
    #     avg_pcp = np.mean(actor_pcp[:3])

    #     bone_group = OrderedDict(
    #         [('Head', [8]), ('Torso', [9]), ('Upper arms', [5, 6]),
    #          ('Lower arms', [4, 7]), ('Upper legs', [1, 2]), ('Lower legs', [0, 3])])
    #     bone_person_pcp = OrderedDict()
    #     for k, v in bone_group.items():
    #         bone_person_pcp[k] = np.sum(bone_correct_parts[:, v], axis=-1) / (total_parts / 10 * len(v) + 1e-8)

    #     return actor_pcp, avg_pcp, bone_person_pcp, match_gt / (total_gt + 1e-8)

    @staticmethod
    def coco2campus3D(coco_pose):
        """
        transform coco order(our method output) 3d pose to shelf dataset order with interpolation
        :param coco_pose: np.array with shape 17x3
        :return: 3D pose in campus order with shape 14x3
        """
        campus_pose = np.zeros((14, 3))
        coco2campus = np.array([16, 14, 12, 11, 13, 15, 10, 8, 6, 5, 7, 9])
        campus_pose[0: 12] += coco_pose[coco2campus]

        mid_sho = (coco_pose[5] + coco_pose[6]) / 2  # L and R shoulder
        head_center = (coco_pose[3] + coco_pose[4]) / 2  # middle of two ear

        head_bottom = (mid_sho + head_center) / 2  # nose and head center
        head_top = head_bottom + (head_center - head_bottom) * 2
        campus_pose[12] += head_bottom
        campus_pose[13] += head_top

        return campus_pose
