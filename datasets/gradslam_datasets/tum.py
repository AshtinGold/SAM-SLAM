import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils import data
from natsort import natsorted

from .basedataset import GradSLAMDataset

class TUMDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = None
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def parse_list(self, filepath, skiprows=0):
        """ read list data """
        data = np.loadtxt(filepath, delimiter=' ',
                          dtype=np.unicode_, skiprows=skiprows)
        return data

#>> unused
    def associate_frames2(self, tstamp_image, tstamp_depth, tstamp_pose, tstamp_image_f, tstamp_image_s, max_dt=0.08):
        """ pair images, depths, poses, and additional file associations """
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if (np.abs(tstamp_depth[j] - t) < max_dt):
                    if tstamp_image_f is not None and tstamp_image_s is not None:
                        associations.append((i, j, i, i))  # i corresponds to tstamp_image_f and tstamp_image_s
                    else:
                        associations.append((i, j))
            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and \
                        (np.abs(tstamp_pose[k] - t) < max_dt):
                    if tstamp_image_f is not None and tstamp_image_s is not None:
                        associations.append((i, j, k, i, i))  # i corresponds to tstamp_image_f and tstamp_image_s
                    else:
                        associations.append((i, j, k))

        return associations

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        """ pair images, depths, and poses """
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if (np.abs(tstamp_depth[j] - t) < max_dt):
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and \
                        (np.abs(tstamp_pose[k] - t) < max_dt):
                    associations.append((i, j, k))

        return associations

    def pose_matrix_from_quaternion(self, pvec):
        """ convert 4x4 pose matrix to (t, q) """
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose

    # >>
    def get_filepaths(self):
        frame_rate = 32
        """ read video data in tum-rgbd format """
        if os.path.isfile(os.path.join(self.input_folder, 'groundtruth.txt')):
            pose_list = os.path.join(self.input_folder, 'groundtruth.txt')
        elif os.path.isfile(os.path.join(self.input_folder, 'pose.txt')):
            pose_list = os.path.join(self.input_folder, 'pose.txt')

        image_list = os.path.join(self.input_folder, 'rgb.txt')
        depth_list = os.path.join(self.input_folder, 'depth.txt')
        f_list = os.path.join(self.input_folder, 'sem_f.txt')
        s_list = os.path.join(self.input_folder, 'sem_s.txt')

        print(f_list)
        print(s_list)

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        s_data = self.parse_list(s_list)
        f_data = self.parse_list(f_list)
        pose_data = self.parse_list(pose_list, skiprows=1)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)

        associations = self.associate_frames(tstamp_image, tstamp_depth, tstamp_pose)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        color_paths, depth_paths, f_paths, s_paths = [], [], [], []
        for ix in indicies:
            if len(associations[ix]) == 3:
                (i, j, k) = associations[ix]
                color_paths += [os.path.join(self.input_folder, image_data[i, 1])]
                depth_paths += [os.path.join(self.input_folder, depth_data[j, 1])]
                s_paths += [os.path.join(self.input_folder, s_data[i, 1])]
                f_paths += [os.path.join(self.input_folder, f_data[i, 1])]
            else:
                (i, j, k, i_f, i_s) = associations[ix]
                color_paths += [os.path.join(self.input_folder, image_data[i, 1])]
                depth_paths += [os.path.join(self.input_folder, depth_data[j, 1])]
                s_paths += [os.path.join(self.input_folder, s_data[i, 1])]
                f_paths += [os.path.join(self.input_folder, f_data[i, 1])]

        embedding_paths = None

        return color_paths, depth_paths, embedding_paths, s_paths, f_paths


    # def get_filepaths(self):

    #     frame_rate = 32
    #     """ read video data in tum-rgbd format """
    #     if os.path.isfile(os.path.join(self.input_folder, 'groundtruth.txt')):
    #         pose_list = os.path.join(self.input_folder, 'groundtruth.txt')
    #     elif os.path.isfile(os.path.join(self.input_folder, 'pose.txt')):
    #         pose_list = os.path.join(self.input_folder, 'pose.txt')

    #     image_list = os.path.join(self.input_folder, 'rgb.txt')
    #     depth_list = os.path.join(self.input_folder, 'depth.txt')

    #     image_data = self.parse_list(image_list)
    #     depth_data = self.parse_list(depth_list)
    #     pose_data = self.parse_list(pose_list, skiprows=1)
    #     # pose_vecs = pose_data[:, 1:].astype(np.float64)  # unused in orignial code

    #     tstamp_image = image_data[:, 0].astype(np.float64)
    #     tstamp_depth = depth_data[:, 0].astype(np.float64)
    #     tstamp_pose = pose_data[:, 0].astype(np.float64)
    #     associations = self.associate_frames(
    #         tstamp_image, tstamp_depth, tstamp_pose)

    #     indicies = [0]
    #     for i in range(1, len(associations)):
    #         t0 = tstamp_image[associations[indicies[-1]][0]]
    #         t1 = tstamp_image[associations[i][0]]
    #         if t1 - t0 > 1.0 / frame_rate:
    #             indicies += [i]

    #     color_paths, depth_paths = [], []
    #     for ix in indicies:
    #         (i, j, k) = associations[ix]
    #         color_paths += [os.path.join(self.input_folder, image_data[i, 1])]
    #         depth_paths += [os.path.join(self.input_folder, depth_data[j, 1])]

    #     embedding_paths = None

    #     return color_paths, depth_paths, embedding_paths
    
    def load_poses(self):
        
        frame_rate = 32
        """ read video data in tum-rgbd format """
        if os.path.isfile(os.path.join(self.input_folder, 'groundtruth.txt')):
            pose_list = os.path.join(self.input_folder, 'groundtruth.txt')
        elif os.path.isfile(os.path.join(self.input_folder, 'pose.txt')):
            pose_list = os.path.join(self.input_folder, 'pose.txt')

        image_list = os.path.join(self.input_folder, 'rgb.txt')
        depth_list = os.path.join(self.input_folder, 'depth.txt')

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 1:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(
            tstamp_image, tstamp_depth, tstamp_pose)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        color_paths, poses, depth_paths, intrinsics = [], [], [], []
        inv_pose = None
        for ix in indicies:
            (i, j, k) = associations[ix]
            color_paths += [os.path.join(self.input_folder, image_data[i, 1])]
            depth_paths += [os.path.join(self.input_folder, depth_data[j, 1])]
            c2w = self.pose_matrix_from_quaternion(pose_vecs[k])
            c2w = torch.from_numpy(c2w).float()
            poses += [c2w]

        return poses
    
    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path, map_location="cpu")
        return embedding.permute(0, 2, 3, 1)
    
