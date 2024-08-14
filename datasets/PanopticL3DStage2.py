import torch
from torch import nn
import numpy as np
import os.path as osp
import pickle
import json_tricks as json
from PIL import Image
import copy
import cv2
from pycocotools.mask import encode, area

from datasets.JointsDataset import JointsDataset
from tools.transforms import projectPoints, get_affine_transform, get_scale, affine_transform
from tools.cameras_cpu import world_to_camera_frame, project_pose, project_pose_camera

class PanopticL3DStage2(JointsDataset):
    def __init__(self, cfg, image_set):
        super().__init__(cfg, image_set)

        self.db_file = 'group_{}_cam_stage2.pkl'.format(self.image_set)
        self.db_file = osp.join(self.dataset_root, self.db_file)
        if osp.exists(self.db_file):  # lazy loading
            self.db = pickle.load(open(self.db_file, 'rb'))
        else:
            if self.image_set == 'train':
                self.sentences_file = osp.join(self.dataset_root, "sentences_train.json")
                self.cam_list = [(0, 16), (0, 30)]
            elif self.image_set == 'test':
                self.sentences_file = osp.join(self.dataset_root, "sentences_test.json")
                self.cam_list = [(0, 16), (0, 30)]
            self.db = self._get_db()
            pickle.dump(self.db, open(self.db_file, 'wb'))

        self.db_size = len(self.db)

    def _get_db(self):
        width = 1920
        height = 1080
        db = []
        with open(self.sentences_file, 'r') as file:
            data = json.load(file)
        for dt in data:
            camera = self._get_cam(dt['video_id'], dt['camera_id'])

            curr_anno = osp.join(self.dataset_root, 'panoptic', dt['video_id'], 'hdPose3d_stage1_coco19')

            for i in range(dt['begin_frame'], dt['end_frame']):
                if i % 15 != 0:
                    continue
                anno_file = osp.join(curr_anno, 'body3DScene_'+str(i).zfill(8)+'.json')
                if not osp.isfile(anno_file):
                    continue
                try:
                    with open(anno_file) as dfile:
                        bodies = json.load(dfile)['bodies']
                except Exception as e:
                    print("anno_file:",anno_file)
                    print(f"Error loading '{anno_file}': {e}")

                if len(bodies) == 0:
                    continue

                postfix = osp.basename(anno_file).replace('body3DScene', '')
                prefix = '{:02d}_{:02d}'.format(0, dt['camera_id'])
                image = osp.join(dt['video_id'], 'hdImgs', prefix,
                                 prefix + postfix)
                image = image.replace('json', 'jpg')

                mask_path = osp.join(self.dataset_root, 'storage_mask', dt['video_id'], prefix, prefix + postfix)
                mask_path = mask_path.replace('json', 'png')

                all_poses_3d = []
                all_poses_vis_3d = []
                all_poses = []
                all_poses_vis = []
                count_body = 0
                body_id = None
                for body in bodies:
                    pose3d = np.array(body['joints19']).reshape((-1, 4))
                    pose3d = pose3d[:self.num_joints]

                    joints_vis = pose3d[:, -1] > 0.1

                    if not joints_vis[self.root_id]:
                        continue

                    mask = np.array(Image.open(mask_path).convert('P'))
                    mask_person = mask == (dt['body_id']+1)
                    # print(mask_person.sum())
                    if mask_person.sum() < 2500:
                        continue

                    if body_id is None:
                        if body['id'] != dt['body_id']:
                            count_body += 1
                        else:
                            body_id = count_body

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
                        pose3d[:, 0:3].transpose(), camera['K'], camera['R'],
                        camera['t'], camera['distCoef']).transpose()[:, :2]
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

                if len(all_poses_3d) > 0 and body_id is not None :
                    our_cam = {}
                    our_cam['R'] = camera['R']
                    our_cam['T'] = -np.dot(camera['R'].T, camera['t']) * 10.0  # cm to mm
                    our_cam['fx'] = np.array(camera['K'][0, 0])
                    our_cam['fy'] = np.array(camera['K'][1, 1])
                    our_cam['cx'] = np.array(camera['K'][0, 2])
                    our_cam['cy'] = np.array(camera['K'][1, 2])
                    our_cam['k'] = camera['distCoef'][[0, 1, 4]].reshape(3, 1)
                    our_cam['p'] = camera['distCoef'][[2, 3]].reshape(2, 1)

                    all_poses_3d = np.concatenate(self.add_axis(all_poses_3d))
                    all_poses_vis_3d = np.concatenate(self.add_axis(all_poses_vis_3d))
                    all_poses_3d_cam = world_to_camera_frame(all_poses_3d.reshape(-1, 3), our_cam['R'],
                                                             our_cam['T']).reshape(*(all_poses_3d.shape))
                    all_poses = np.concatenate(self.add_axis(all_poses))
                    all_poses_vis = np.concatenate(self.add_axis(all_poses_vis))

                    db.append({
                        'key': "{}_{}{}".format(dt['video_id'], prefix, postfix.split('.')[0]),
                        'image': osp.join(self.dataset_root, 'panoptic', image),
                        'joints_3d': all_poses_3d,
                        'joints_3d_vis': all_poses_vis_3d,
                        'joints_3d_cam': all_poses_3d_cam,
                        'joints_2d': all_poses,
                        'joints_2d_vis': all_poses_vis,
                        'camera': our_cam,
                        'video_id': dt['video_id'],
                        'body_id': body_id,
                        'category': dt['category'],
                        'label': dt['label'],
                        'begin_frame': dt['begin_frame'],
                        'end_frame': dt['end_frame'],
                        'frame_id': i,
                        'camera_id': dt['camera_id'],
                        'absolute_body_id': dt['body_id'],
                        'mask':mask_path
                    })
        return db

    def _get_cam(self, seq, camera_id):
        cam_file = osp.join(self.dataset_root, 'panoptic', seq, 'calibration_{:s}.json'.format(seq))
        with open(cam_file) as cfile:
            calib = json.load(cfile)

        M = np.array([[1.0, 0.0, 0.0],
                      [0.0, 0.0, -1.0],
                      [0.0, 1.0, 0.0]])
        camera = None
        cam_list = [(0,camera_id)]
        for cam in calib['cameras']:
            if (cam['panel'], cam['node']) in cam_list:
                sel_cam = {}
                sel_cam['K'] = np.array(cam['K'])
                sel_cam['distCoef'] = np.array(cam['distCoef'])
                sel_cam['R'] = np.array(cam['R']).dot(M)
                sel_cam['t'] = np.array(cam['t']).reshape((3, 1))
                camera = sel_cam
        return camera


    def add_axis(self, ll):
        cur_list = []
        for i in ll:
            cur_list.append(np.expand_dims(i, axis=0))

        return cur_list

    def get_nest_tensor(self, cur_frame, begin_frame, end_frame, video_name, camera_id):
        rand_num = int((end_frame - begin_frame) * (1-self.random_split_ratio))
        # start_frame = begin_frame + random.randint(0, rand_num)
        start_frame = begin_frame + rand_num//2
        final_frame = start_frame + int((end_frame-begin_frame) * self.random_split_ratio)
        assert start_frame < final_frame
        picked_frames = np.linspace(start_frame, final_frame, self.window_num-1, dtype=int)
        idx = np.searchsorted(picked_frames, cur_frame)
        picked_frames = np.insert(picked_frames, idx, cur_frame)

        frame_list = []
        for frame in picked_frames:
            curr_frame_file = osp.join(self.dataset_root, 'panoptic', video_name, 'hdImgs', '00_'+str(camera_id).zfill(2),'00_'+str(camera_id).zfill(2)+'_'+str(frame).zfill(8)+'.jpg')
            data_numpy = cv2.imread(
                curr_frame_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if self.color_rgb:
                data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
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
            frame_list.append(input)

        frame_tensor = torch.stack(frame_list)
        return frame_tensor, idx, picked_frames

    def __len__(self):
        return self.db_size

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
        # data_numpy = np.zeros((500, 500, 3), dtype=np.uint8)

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

        target_heatmap_one, target_weight_one = self.generate_target_heatmap_one(
            joints, joints_vis, db_rec['body_id'])
        target_heatmap = torch.from_numpy(target_heatmap)
        target_weight = torch.from_numpy(target_weight)
        target_heatmap_one = torch.from_numpy(target_heatmap_one)
        target_weight_one = torch.from_numpy(target_weight_one)

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
        target_3d_one = self.generate_3d_target_one(joints_3d, db_rec['body_id'])
        target_3d = torch.from_numpy(target_3d)
        target_3d_one = torch.from_numpy(target_3d_one)

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
                joints_3d_vis[i,], joints_vis_u[i,] = np.zeros_like(joints_3d_vis[i,]), np.zeros_like(joints_vis_u[i,])
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

        input_nest_numpy, frame_idx, picked_frames_idx = self.get_nest_tensor(db_rec['frame_id'], db_rec['begin_frame'], db_rec['end_frame'], db_rec['video_id'], db_rec['camera_id'])

        mask = np.array(Image.open(db_rec['mask']).convert('P'))
        mask_person = (mask == (db_rec['absolute_body_id'] + 1)).astype(float)
        mask_person = cv2.warpAffine(
            mask_person,
            trans, (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)
        mask_person = mask_person.astype(int)

        mask_rles = [encode(np.asfortranarray(mask_person.astype(np.uint8)))]
        mask_areas = area(mask_rles).astype(float)

        indices = np.where(mask_person > 0.5)

        if indices is None or indices[0].size == 0:
            min_x = max_y = max_x = min_y = 0.1
        else:
            min_x = float(np.min(indices[1]))
            min_y = float(np.min(indices[0]))
            max_x = float(np.max(indices[1]))
            max_y = float(np.max(indices[0]))

        bbox_xywh = [min_x, min_y, max_x, max_y]

        label = torch.tensor(0, dtype=torch.long)
        labels = []
        labels.append(label)
        labels = torch.stack(labels, dim=0)



        masks = []
        masks.append(torch.from_numpy(mask_person))
        masks = torch.stack(masks, dim=0)

        valid = []
        valid.append(1)

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
            'person_idx': np.array([db_rec['body_id']]),
            'frame_idx': frame_idx,
            'sentences': db_rec['label'],
        }

        h = 512
        w = 960
        boxes_resize = torch.tensor([(bbox_xywh[0]+bbox_xywh[2])/2/w, (bbox_xywh[1]+bbox_xywh[3])/2/h, (bbox_xywh[2]-bbox_xywh[0])/w, (bbox_xywh[3]-bbox_xywh[1])/h])
        boxes = []
        boxes.append(boxes_resize)
        boxes = torch.stack(boxes, dim=0)
        target = {
            'frames_idx': torch.tensor(picked_frames_idx),  # [T,]
            'valid_indices': torch.tensor([frame_idx]),
            'labels': labels,  # [1,]
            'boxes': boxes,  # [1, 4], xyxy
            'masks': masks,  # [1, H, W]
            "is_ref_inst_visible": masks[0].any(),
            'valid': torch.tensor(valid),  # [1,]
            'caption': db_rec['label'],
            'orig_size': [int(h), int(w)],
            'size': torch.as_tensor([int(h), int(w)]),
            'area': torch.tensor(mask_areas),
            'iscrowd': torch.zeros(1),
            "caption": db_rec['label'],
            'image_id': db_rec['image'] + str(db_rec['camera_id']) + str(db_rec['begin_frame']) + db_rec['label'],
            "labels": torch.tensor(0, dtype=torch.long),
            'referred_instance_idx': torch.tensor(0),
        }
        targets = self.window_num * [None]
        targets[frame_idx] = target
        return input_nest_numpy, target, meta, input_AGR, input, target_heatmap, target_weight, target_3d, target_heatmap_one, target_weight_one, target_3d_one, torch.from_numpy(mask_person)

    def evaluate(self, preds, grid_centers, frame_valids):
        eval_list = []
        gt_num = self.db_size
        assert len(preds) == gt_num and len(frame_valids) == gt_num, 'number mismatch'

        total_gt = 0
        for i in range(gt_num):
            index = i
            db_rec = copy.deepcopy(self.db[index])
            joints_3d = db_rec['joints_3d']
            joints_3d_vis = db_rec['joints_3d_vis']
            camera = db_rec['camera']

            if len(joints_3d) == 0 or not frame_valids[index]:
                continue

            pred, grid_center = preds[i].copy(), grid_centers[i].copy()
            pred, grid_center = pred[pred[:, 0, 3] >= 0], grid_center[pred[:, 0, 3] >= 0]
            for pose, center in zip(pred, grid_center):
                mpjpes = []
                gt, gt_vis = joints_3d[db_rec['body_id']], joints_3d_vis[db_rec['body_id']]
                vis = gt_vis[:, 0] > 0
                mpjpe = np.linalg.norm(pose[vis, :3] - gt[vis], axis=-1).mean()
                mpjpes.append(mpjpe)

                # min_gt = np.argmin(mpjpes)
                min_gt = db_rec['body_id']
                min_mpjpe = np.min(mpjpes)
                gt, vis = joints_3d[min_gt], joints_3d_vis[min_gt][:, 0] > 0
                mpjpe_aligned = np.linalg.norm((pose[vis, :3] - pose[self.root_id:self.root_id + 1, :3]) - \
                                               (gt[vis] - gt[self.root_id:self.root_id + 1]), axis=-1).mean()
                gt_camera, pose_camera = world_to_camera_frame(gt[:, :3], camera['R'], camera['T']), \
                    world_to_camera_frame(pose[:, :3], camera['R'], camera['T'])
                gt_pose2d, pred_pose2d = project_pose_camera(gt_camera, camera), project_pose_camera(pose_camera,
                                                                                                     camera)
                error_2d = np.linalg.norm(pred_pose2d[vis] - gt_pose2d[vis], axis=-1).mean()
                mrpe = np.abs(pose_camera[self.root_id] - gt_camera[self.root_id])

                score = pose[0, 4]
                eval_list.append({
                    "image_path": db_rec['image'],
                    "mpjpe": float(min_mpjpe),
                    "mpjpe_aligned": float(mpjpe_aligned),
                    "mrpe": mrpe,
                    "error2d": error_2d,
                    "score": float(score),
                    "gt_id": int(total_gt + min_gt),
                    "gt_pose": gt,
                    "pred_pose": pose[:, :3],
                    "vis": vis,
                    "root_id": self.root_id,
                    "category": db_rec['category']
                })

            # total_gt += len(joints_3d)
            total_gt += 1

        # subject_list = ['haggling', 'mafia', 'ultimatum', 'pizza']
        subject_list1 = ['haggling', 'band','pizza','ian']
        for subject in subject_list1:
            subject_eval_list = [term for term in eval_list if subject in term['image_path']]
            print(subject, ':\n', self._eval_list_to_mpjpe(subject_eval_list))

        subject_eval_list = [term for term in eval_list if term['category'][0] in [0,1,2,3]]
        print('appearance', ':\n', self._eval_list_to_mpjpe(subject_eval_list))
        subject_eval_list = [term for term in eval_list if term['category'][0] in [4]]
        print('behavior', ':\n', self._eval_list_to_mpjpe(subject_eval_list))
        subject_eval_list = [term for term in eval_list if term['category'][0] in [5]]
        print('motion', ':\n', self._eval_list_to_mpjpe(subject_eval_list))


        return self._eval_list_to_mpjpe(eval_list), eval_list

    @staticmethod
    def _eval_list_to_mpjpe(eval_list, threshold=20000):
        eval_list.sort(key=lambda k: k["score"], reverse=True)
        gt_det = []

        mpjpes, mpjpes_aligned, mrpes, errors_2d = [], [], [], []
        for i, item in enumerate(eval_list):
            if item["mpjpe"] < threshold and item["gt_id"] not in gt_det:
                mpjpes.append(item["mpjpe"])
                mpjpes_aligned.append(item["mpjpe_aligned"])
                mrpes.append(item["mrpe"])
                gt_det.append(item["gt_id"])
                errors_2d.append(item["error2d"])
        mrpes = np.array(mrpes)

        metric = {
            'mpjpe': np.mean(mpjpes) if len(mpjpes) > 0 else np.inf,
            'mpjpe_aligned': np.mean(mpjpes_aligned) if len(mpjpes_aligned) > 0 else np.inf,
            'mrpe': {
                'x': mrpes[:, 0].mean() if len(mpjpes) > 0 else np.inf,
                'y': mrpes[:, 1].mean() if len(mpjpes) > 0 else np.inf,
                'z': mrpes[:, 2].mean() if len(mpjpes) > 0 else np.inf,
                'root': np.linalg.norm(mrpes, axis=-1).mean() if len(mpjpes) > 0 else np.inf,
            },
            'error2d': np.mean(errors_2d) if len(mpjpes) > 0 else np.inf
        }
        return metric
