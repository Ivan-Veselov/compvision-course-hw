#! /usr/bin/env python3
from cv2.cv2 import findFundamentalMat, findHomography, RANSAC, findEssentialMat, decomposeEssentialMat

from _corners import FrameCorners

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Tuple

import numpy as np

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import *


W_MATRIX = np.array([
    [0., -1., 0.],
    [1., 0., 0.],
    [0., 0., 1.]
])


class CameraTracker:
    def __init__(self, corner_storage: CornerStorage, intrinsic_mat: np.ndarray):
        max_id = np.amax(np.concatenate([corners.ids for corners in corner_storage]))

        self.__id_to_position = [None for _ in range(max_id + 1)]
        self.__frame_matrix = [None for _ in corner_storage]

        self.__corner_storage = corner_storage
        self.__intrinsic_mat = intrinsic_mat

        self.__triangulation_parameters = TriangulationParameters(max_reprojection_error=1, min_triangulation_angle_deg=1., min_depth=0.)
        # todo: configure

        self.__track_initialization()

    def __calculate_pose(self, frame_corners1: FrameCorners, frame_corners2: FrameCorners):
        correspondences = build_correspondences(frame_corners1, frame_corners2)

        E, mask = findEssentialMat(correspondences.points_1, correspondences.points_2, self.__intrinsic_mat)  # todo: have parameters
        mask = mask.reshape(-1)
        filtered_correspondences = build_correspondences(frame_corners1, frame_corners2, np.nonzero(1 - mask)[0])

        fundamental_inliers = np.count_nonzero(mask)

        R1, R2, t1 = decomposeEssentialMat(E)
        t1.reshape(-1)
        t2 = -t1

        possible_poses = [Pose(R1, t1), Pose(R1, t2), Pose(R2, t1), Pose(R2, t2)]

        pose_cloud_size = []
        for pose in possible_poses:
            positions, ids = triangulate_correspondences(
                filtered_correspondences,
                eye3x4(),
                pose_to_view_mat3x4(pose),
                self.__intrinsic_mat,
                self.__triangulation_parameters
            )

            pose_cloud_size.append(ids.shape[0])

        index = np.argmax(pose_cloud_size)
        if pose_cloud_size[index] == 0:
            return None, None

        pose = possible_poses[index]

        H, mask = findHomography(correspondences.points_1, correspondences.points_2, RANSAC)  # todo: configure
        mask = mask.reshape(-1)

        homography_inliers = np.count_nonzero(mask)

        return pose, fundamental_inliers / homography_inliers

    def __track_initialization(self):
        poses, qualities = zip(*[self.__calculate_pose(self.__corner_storage[0], i) for i in self.__corner_storage])
        index = np.nanargmax(np.array(qualities, dtype=np.float32))

        self.__frame_matrix = [eye3x4() for i in self.__corner_storage]
        self.__frame_matrix[index] = pose_to_view_mat3x4(poses[index])
        self.__update_cloud(0, index)

    def __update_cloud(self, frame_id1, frame_id2):
        correspondences = build_correspondences(self.__corner_storage[frame_id1], self.__corner_storage[frame_id2])

        positions, ids = triangulate_correspondences(
            correspondences,
            self.__frame_matrix[frame_id1],
            self.__frame_matrix[frame_id2],
            self.__intrinsic_mat,
            self.__triangulation_parameters
        )

        for pos, id in zip(positions, ids):
            self.__id_to_position[id] = pos

    def get_frame_matrices(self) -> List[np.ndarray]:
        return self.__frame_matrix

    def get_cloud_builder(self):
        ids = np.array(list(map(lambda x: x[0], filter(lambda x: x[1] is not None, enumerate(self.__id_to_position)))))
        positions = np.array([self.__id_to_position[i] for i in ids])

        return PointCloudBuilder(ids, positions)


def _track_camera(corner_storage: CornerStorage,
                  intrinsic_mat: np.ndarray) \
        -> Tuple[List[np.ndarray], PointCloudBuilder]:

    tracker = CameraTracker(corner_storage, intrinsic_mat)

    return tracker.get_frame_matrices(), tracker.get_cloud_builder()


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str) \
        -> Tuple[List[Pose], PointCloud]:
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )
    view_mats, point_cloud_builder = _track_camera(
        corner_storage,
        intrinsic_mat
    )
    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    create_cli(track_and_calc_colors)()
