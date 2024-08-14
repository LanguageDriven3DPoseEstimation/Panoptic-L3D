import torch
import glob
import os.path as osp
import numpy as np
import json_tricks as json
import pickle
import copy
import cv2

from datasets.JointsDataset import JointsDataset
from tools.transforms import projectPoints, get_affine_transform, get_scale, affine_transform
from tools.cameras_cpu import world_to_camera_frame, project_pose

TRAIN_LIST = [
    '160422_haggling1',
    '160226_haggling1',
    '170404_haggling_a2',
    '170221_haggling_b2',
    '160422_ultimatum1',
    '160906_band2',
    '160906_band3',
    '160906_ian1',
    '160906_ian2',
    '170407_haggling_b2',
]

VAL_LIST = ['160224_haggling1','160906_pizza1','160906_band1','160906_ian3']

class PanopticL3DStage1(JointsDataset):
    def __init__(self, cfg, image_set):
        super().__init__(cfg, image_set)

        self.db_file = 'group_{}_cam_stage1.pkl'.format(self.image_set)
        self.db_file = osp.join(self.dataset_root, self.db_file)
        if osp.exists(self.db_file):  # lazy loading
            self.db = pickle.load(open(self.db_file, 'rb'))
        else:
            if self.image_set == 'train':
                self.sequence_list = TRAIN_LIST
                self._interval = 3
                self.cam_list = [(0, 16), (0, 30)]
            elif self.image_set == 'test':
                self.sequence_list = VAL_LIST
                self._interval = 12
                self.cam_list = [(0, 16), (0, 30)]
            self.db = self._get_db()
            pickle.dump(self.db, open(self.db_file, 'wb'))

        self.db_size = len(self.db)

    def _get_db(self):
        width = 1920
        height = 1080
        db = []
        for seq in self.sequence_list:
            cameras = self._get_cam(seq)

            curr_anno = osp.join(self.dataset_root, 'panoptic', seq, 'hdPose3d_stage1_coco19')
            anno_files = sorted(glob.iglob('{:s}/*.json'.format(curr_anno)))

            for i, file in enumerate(anno_files):
                if i % self._interval == 0:
                    with open(file) as dfile:
                        bodies = json.load(dfile)['bodies']
                    if len(bodies) == 0:
                        continue

                    for k, v in cameras.items():
                        postfix = osp.basename(file).replace('body3DScene', '')
                        prefix = '{:02d}_{:02d}'.format(k[0], k[1])
                        image = osp.join(seq, 'hdImgs', prefix,
                                         prefix + postfix)
                        image = image.replace('json', 'jpg')

                        all_poses_3d = []
                        all_poses_vis_3d = []
                        all_poses = []
                        all_poses_vis = []
                        for body in bodies:
                            pose3d = np.array(body['joints19']).reshape((-1, 4))
                            pose3d = pose3d[:self.num_joints]

                            joints_vis = pose3d[:, -1] > 0.1

                            if not joints_vis[self.root_id]:
                                continue

                            # Coordinate transformation
                            M = np.array([[1.0, 0.0, 0.0],
                                          [0.0, 0.0, -1.0],
                                          [0.0, 1.0, 0.0]])
                            pose3d[:, 0:3] = pose3d[:, 0:3].dot(M)

                            all_poses_3d.append(pose3d[:, 0:3] * 10.0)
                            all_poses_vis_3d.append(
                                np.repeat(
                                    np.reshape(joints_vis, (-1, 1)), 3, axis=1))

                            pose2d = np.zeros((pose3d.shape[0], 2))
                            pose2d[:, :2] = projectPoints(
                                pose3d[:, 0:3].transpose(), v['K'], v['R'],
                                v['t'], v['distCoef']).transpose()[:, :2]
                            x_check = np.bitwise_and(pose2d[:, 0] >= 0,
                                                     pose2d[:, 0] <= width - 1)
                            y_check = np.bitwise_and(pose2d[:, 1] >= 0,
                                                     pose2d[:, 1] <= height - 1)
                            check = np.bitwise_and(x_check, y_check)
                            joints_vis[np.logical_not(check)] = 0

                            all_poses.append(pose2d)
                            all_poses_vis.append(
                                np.repeat(
                                    np.reshape(joints_vis, (-1, 1)), 2, axis=1))

                        if len(all_poses_3d) > 0:
                            our_cam = {}
                            our_cam['R'] = v['R']
                            our_cam['T'] = -np.dot(v['R'].T, v['t']) * 10.0  # cm to mm
                            our_cam['fx'] = np.array(v['K'][0, 0])
                            our_cam['fy'] = np.array(v['K'][1, 1])
                            our_cam['cx'] = np.array(v['K'][0, 2])
                            our_cam['cy'] = np.array(v['K'][1, 2])
                            our_cam['k'] = v['distCoef'][[0, 1, 4]].reshape(3, 1)
                            our_cam['p'] = v['distCoef'][[2, 3]].reshape(2, 1)

                            all_poses_3d = np.concatenate(self.add_axis(all_poses_3d))
                            all_poses_vis_3d = np.concatenate(self.add_axis(all_poses_vis_3d))
                            all_poses_3d_cam = world_to_camera_frame(all_poses_3d.reshape(-1, 3), our_cam['R'],
                                                                     our_cam['T']).reshape(*(all_poses_3d.shape))
                            all_poses = np.concatenate(self.add_axis(all_poses))
                            all_poses_vis = np.concatenate(self.add_axis(all_poses_vis))

                            db.append({
                                'key': "{}_{}{}".format(seq, prefix, postfix.split('.')[0]),
                                'image': osp.join(self.dataset_root, 'panoptic', image),
                                'joints_3d': all_poses_3d,
                                'joints_3d_vis': all_poses_vis_3d,
                                'joints_3d_cam': all_poses_3d_cam,
                                'joints_2d': all_poses,
                                'joints_2d_vis': all_poses_vis,
                                'camera': our_cam
                            })

        return db

    def add_axis(self, ll):
        cur_list = []
        for i in ll:
            cur_list.append(np.expand_dims(i, axis=0))

        return cur_list

    def _get_cam(self, seq):
        cam_file = osp.join(self.dataset_root, 'panoptic', seq, 'calibration_{:s}.json'.format(seq))
        with open(cam_file) as cfile:
            calib = json.load(cfile)

        M = np.array([[1.0, 0.0, 0.0],
                      [0.0, 0.0, -1.0],
                      [0.0, 1.0, 0.0]])
        cameras = {}
        for cam in calib['cameras']:
            if (cam['panel'], cam['node']) in self.cam_list:
                sel_cam = {}
                sel_cam['K'] = np.array(cam['K'])
                sel_cam['distCoef'] = np.array(cam['distCoef'])
                sel_cam['R'] = np.array(cam['R']).dot(M)
                sel_cam['t'] = np.array(cam['t']).reshape((3, 1))
                cameras[(cam['panel'], cam['node'])] = sel_cam
        return cameras

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']

        if self.data_format == 'zip':
            from tools.utils import zipreader
            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        else:
            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
            mean_color = data_numpy.mean(axis=(0, 1), keepdims=True)
            if np.abs(data_numpy - mean_color).sum(axis=-1).max() < 20:
                frame_valid = False
            else:
                frame_valid = True

        joints = db_rec['joints_2d']
        joints_vis = db_rec['joints_2d_vis']
        joints_3d = db_rec['joints_3d']
        assert len(joints) == len(joints_3d), idx
        joints_3d_vis = db_rec['joints_3d_vis']

        nposes = len(joints)
        assert nposes <= self.maximum_person, 'too many persons'

        height, width, _ = data_numpy.shape
        c = np.array([width / 2.0, height / 2.0])
        s = get_scale((width, height), self.image_size)
        if self.is_train and len(self.scale_factor_range) == 2:
            s /= self.scale_factor_range[0] + np.random.random() * \
                 (self.scale_factor_range[1] - self.scale_factor_range[0])
        r = 0

        trans = get_affine_transform(c, s, r, self.image_size)
        input = cv2.warpAffine(
            data_numpy,
            trans, (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        if self.transform:
            input = self.transform(input)

        for n in range(nposes):
            for i in range(len(joints[0])):
                joints[n][i, 0:2] = affine_transform(
                    joints[n][i, 0:2], trans)
                if joints_vis[n][i, 0] > 0.0:
                    if (np.min(joints[n][i, :2]) < 0 or
                            joints[n][i, 0] >= self.image_size[0] or
                            joints[n][i, 1] >= self.image_size[1]):
                        joints_vis[n][i, :] = 0

        if self.is_train and not self.fix_camera:
            db_rec['camera'] = self.generate_synthetic_camera()

        if self.is_train and self.random_place:
            joints_3d = self.random_place_people()

        joints = project_pose(joints_3d.reshape(-1, 3), db_rec['camera']).reshape(*(joints_3d.shape[:2]), 2)
        for n in range(nposes):
            for i in range(len(joints[0])):
                joints[n][i, 0:2] = affine_transform(
                    joints[n][i, 0:2], trans)
                if joints_vis[n][i, 0] > 0.0:
                    if (np.min(joints[n][i, :2]) < 0 or
                            joints[n][i, 0] >= self.image_size[0] or
                            joints[n][i, 1] >= self.image_size[1]):
                        joints_vis[n][i, :] = 0

        input_heatmap = self.generate_input_heatmap(joints, joints_3d, self.heatmap_size)
        input_heatmap = torch.tensor(input_heatmap)

        target_heatmap, target_weight = self.generate_target_heatmap(
            joints, joints_vis)
        target_heatmap = torch.from_numpy(target_heatmap)
        target_weight = torch.from_numpy(target_weight)

        # make joints and joints_vis having same shape
        joints_u = np.zeros((self.maximum_person, self.num_joints, 2))
        joints_vis_u = np.zeros((self.maximum_person, self.num_joints, 2))
        for i in range(nposes):
            joints_u[i] = joints[i]
            joints_vis_u[i] = joints_vis[i]

        joints_3d_u = np.zeros((self.maximum_person, self.num_joints, 3))
        joints_3d_vis_u = np.zeros((self.maximum_person, self.num_joints, 3))
        for i in range(nposes):
            joints_3d_u[i] = joints_3d[i][:, 0:3]
            joints_3d_vis_u[i] = np.logical_and(joints_3d_vis[i][:, 0:3], joints_vis_u[i][:, :1])

        target_3d = self.generate_3d_target(joints_3d)
        target_3d = torch.from_numpy(target_3d)

        if isinstance(self.root_id, int):
            roots_3d = joints_3d_u[:, self.root_id]
            roots_2d = joints_u[:, self.root_id]
        elif isinstance(self.root_id, list):
            roots_3d = np.mean([joints_3d_u[:, j] for j in self.root_id], axis=0)
            roots_2d = np.mean([joints_u[:, j] for j in self.root_id], axis=0)

        # for detection
        wh = np.zeros((self.maximum_person, 4), dtype=np.float32)
        bboxes = np.zeros((self.maximum_person, 4), dtype=np.float32)
        depth_camera = np.zeros((self.maximum_person, 1), dtype=np.float32)
        depth_norm = np.zeros((self.maximum_person, 1), dtype=np.float32)
        pitch = np.zeros((self.maximum_person, 2), dtype=np.float32)
        ind = np.zeros((self.maximum_person,), dtype=np.int64)
        ind_3d = np.zeros((self.maximum_person,), dtype=np.int64)
        bias_3d = np.zeros((self.maximum_person, 3), dtype=np.float32)
        reg_mask = np.zeros((self.maximum_person,), dtype=np.float32)
        depth_norm_factor = np.sqrt(db_rec['camera']['fx'] * db_rec['camera']['fy'] / (s[1] * 200 * s[0] * 200) \
                                    * (self.image_size[0] * self.image_size[1]))

        feat_stride = self.image_size[0] / self.heatmap_size[0]

        boxmap = torch.zeros(4, self.heatmap_size[1], self.heatmap_size[0]).float()

        for i in range(nposes):
            extention = [(joints_u[i, :, j].max() - joints_u[i, :, j].min()) * self.bbox_extention[j] for j in range(2)]
            bbox = [np.clip(joints_u[i, :, 0].min() - extention[0], 0, self.image_size[0]), \
                    np.clip(joints_u[i, :, 1].min() - extention[1], 0, self.image_size[1]), \
                    np.clip(joints_u[i, :, 0].max() + extention[0], 0, self.image_size[0]), \
                    np.clip(joints_u[i, :, 1].max() + extention[1], 0, self.image_size[1])]

            bboxes[i] = bbox / feat_stride
            _, depth = project_pose(roots_3d[i:i + 1], db_rec['camera'], need_depth=True)
            depth_camera[i] = depth
            depth_norm[i] = depth / depth_norm_factor

            wh[i] = np.array([roots_2d[i, 0] - bbox[0], roots_2d[i, 1] - bbox[1], \
                              bbox[2] - roots_2d[i, 0], bbox[3] - roots_2d[i, 1]]) / feat_stride

            if roots_2d[i, 0] < 0 or roots_2d[i, 1] < 0 or roots_2d[i, 0] >= self.image_size[0] \
                    or roots_2d[i, 1] >= self.image_size[1]:
                joints_3d_vis, joints_vis_u = np.zeros_like(joints_3d_vis), np.zeros_like(joints_vis_u)
            else:
                reg_mask[i] = 1
                ind[i] = int(roots_2d[i, 1] / feat_stride) * self.heatmap_size[0] + int(roots_2d[i, 0] / feat_stride)
                ind3d = (roots_3d[i] - self.space_center + self.space_size / 2) / self.cube_length
                ind_3d[i] = round(ind3d[0]) * self.initial_cube_size[1] * self.initial_cube_size[2] + \
                            round(ind3d[1]) * self.initial_cube_size[2] + round(ind3d[2])

            r = 15
            boxmap[:, max(0, int(roots_2d[i, 1] / feat_stride) - r): min(self.heatmap_size[1] - 1,
                                                                         int(roots_2d[i, 1] / feat_stride) + r), \
            max(0, int(roots_2d[i, 0] / feat_stride) - r): min(self.heatmap_size[0] - 1,
                                                               int(roots_2d[i, 0] / feat_stride) + r)] = torch.tensor(
                wh[i])[:, None, None]

        input_AGR = torch.cat((input_heatmap, boxmap), dim=0)
        meta = {
            'image': image_file,
            'num_person': nposes,
            'joints_3d': joints_3d_u,
            'joints_3d_vis': joints_3d_vis_u,
            'roots_3d': roots_3d,
            'roots_2d': roots_2d,
            'joints': joints_u,
            'joints_vis': joints_vis_u,
            'center': c,
            'scale': s,
            'rotation': r,
            'camera': db_rec['camera'],
            'depth': depth_camera,
            'depth_norm': depth_norm,
            'depth_norm_factor': depth_norm_factor,
            'wh': wh,
            'ind': ind,
            'ind_3d': ind_3d,
            'reg_mask': reg_mask,
            'bbox': bboxes,
            'frame_valid': frame_valid,
        }

        return input, target_heatmap, target_weight, target_3d, meta, input_AGR
