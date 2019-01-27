#! /usr/bin/env python3
from scipy.spatial import KDTree

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import FrameCorners, CornerStorage, StorageImpl, filter_frame_corners, concat_frame_corners
from _corners import dump, load, draw, without_short_tracks, create_cli


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


class IdsGenerator:
    def __init__(self):
        self.__free_id = 0

    def gen(self, amount):
        assert amount > 0

        result = np.arange(start=self.__free_id, stop=self.__free_id + amount)
        self.__free_id += amount
        return result


def float32_to_uint8(image):
    return (255 * image).astype(np.uint8)


def is_outside(image, point):
    point = to_int(point)

    return point[0] < 0 or point[0] >= image.shape[1] or point[1] < 0 or point[1] >= image.shape[0]


def to_int(array):
    return np.round(array).astype(int)


def find_near_corners(corners: FrameCorners, radius) -> np.ndarray:
    # todo: optimize

    sorting_idx = np.flip(np.argsort(corners.min_eigen.flatten()))
    tree = KDTree(corners.points[sorting_idx])

    index = np.array(sorted(list(tree.query_pairs(radius))))
    if index.shape[0] == 0:
        return np.array([])

    return sorting_idx[index]


def find_near_pairs(points1: np.ndarray, points2: np.ndarray, radius) -> np.ndarray:
    # todo: optimize

    half_len = points1.shape[0]

    tree = KDTree(np.concatenate((points1, points2)))
    pairs = np.array(list(tree.query_pairs(radius)))

    mask = [True] * pairs.shape[0]

    for idx, pair in enumerate(pairs):
        if (pair[0] < half_len and pair[1] < half_len) or (pair[0] >= half_len and pair[1] >= half_len):
            mask[idx] = False

    pairs = pairs[mask]
    pairs += np.tile(np.array([0, -half_len]), [pairs.shape[0], 1])

    return pairs


class CornersSelectionAlgorithm:
    def __init__(self):
        self.__block_size = 5
        self.__similarity_radius = 4
        self.__responsibility_radius = 2 * self.__similarity_radius

        self.__ids_generator = None

    def run(self, frame_sequence: pims.FramesSequence, builder: _CornerStorageBuilder) -> None:
        ids_generator = IdsGenerator()
        tracked_corners = None

        for frame_id, image in enumerate(frame_sequence):
            new_corners_coordinates = cv2.goodFeaturesToTrack(
                image,
                maxCorners=0,
                qualityLevel=0.01,
                minDistance=self.__responsibility_radius,
                blockSize=self.__block_size
            ).reshape(-1, 2)

            num_of_new_corners = len(new_corners_coordinates)

            new_corners = FrameCorners(
                ids_generator.gen(num_of_new_corners),
                new_corners_coordinates,
                np.array([self.__block_size] * num_of_new_corners),
                np.array([True] * num_of_new_corners),
                self.__calculate_min_eigen(image, new_corners_coordinates)
            )

            if tracked_corners is not None:
                old_corners_coordinates, state, _ = cv2.calcOpticalFlowPyrLK(  # todo: err return value
                    float32_to_uint8(frame_sequence[frame_id - 1]),
                    float32_to_uint8(image),
                    tracked_corners.points,
                    None,
                    winSize=(self.__block_size, self.__block_size)  # ,
                    # maxLevel=None,
                    # criteria=None,
                    # flags=None,
                    # minEigThreshold=None
                )

                mask = state[:, 0].astype(bool)

                for idx, coordinates in enumerate(old_corners_coordinates):
                    if mask[idx] and is_outside(image, coordinates):
                        mask[idx] = False

                tracked_corners = filter_frame_corners(tracked_corners, mask)
                old_corners_coordinates = old_corners_coordinates[mask]
                tracked_corners = FrameCorners(
                    tracked_corners.ids,
                    old_corners_coordinates,
                    tracked_corners.sizes,  # todo: may be can be replaced with some info from calcOpticalFlowPyrLK about pyramides
                    np.array([False] * len(tracked_corners.ids)),
                    self.__calculate_min_eigen(image, old_corners_coordinates)
                )

                tracked_corners = self.__merge_track_with_new_corners(tracked_corners, new_corners)

                tracked_corners = self.__enforce_responsibility(tracked_corners)
            else:
                tracked_corners = new_corners

            builder.set_corners_at_frame(frame_id, tracked_corners)

    def __calculate_min_eigen(self, image, coordinates):
        min_eigen = cv2.cornerMinEigenVal(image, self.__block_size)

        index = to_int(coordinates)
        return min_eigen[index[:, 1], index[:, 0]]

    def __enforce_responsibility(self, corners: FrameCorners) -> FrameCorners:
        mask = np.array([True] * len(corners.ids))
        pairs = find_near_corners(corners, self.__responsibility_radius)

        for indices in pairs:
            better_idx = indices[0]
            worse_idx = indices[1]

            if mask[better_idx]:
                mask[worse_idx] = False

        return filter_frame_corners(corners, mask)

    def __replace_tracked_corners_when_possible(
        self,
        tracked_corners: FrameCorners,
        new_corners: FrameCorners
    ) -> (FrameCorners, FrameCorners):
        mask = np.array([True] * len(new_corners.ids))
        pairs = find_near_pairs(tracked_corners.points, new_corners.points, self.__similarity_radius)

        for indices in pairs:
            tracked_idx = indices[0]  # todo: replace by new corner
            # todo: maybe it is not needed. right now it works fine
            new_idx = indices[1]

            mask[new_idx] = False

        return tracked_corners, filter_frame_corners(new_corners, mask)

    def __merge_track_with_new_corners(self, tracked_corners: FrameCorners, new_corners: FrameCorners) -> FrameCorners:
        # todo: filter merged tracks

        tracked_corners = self.__enforce_responsibility(tracked_corners)

        tracked_corners, new_corners = self.__replace_tracked_corners_when_possible(tracked_corners, new_corners)

        return concat_frame_corners(tracked_corners, new_corners)


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    CornersSelectionAlgorithm().run(frame_sequence, builder)


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
