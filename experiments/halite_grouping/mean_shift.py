"""


"""

import math
import logging
import itertools

from typing import Union
from collections import namedtuple

import numpy as np

from scipy.spatial import distance


_logger = logging.getLogger(__name__)


Point = namedtuple("Point", ["x", "y"])
Shape = namedtuple("Shape", ["height", "width"])


class MeanShiftCluster(object):
    """ Class to help cluster together points in a 2d grid. the value
        of the point on the 2d grid describes its weighting.
    """

    def __init__(self, rectangle_shape: Shape):
        self._rectangle_shape = rectangle_shape

    @property
    def rectangle_shape(self):
        return self._rectangle_shape

    @staticmethod
    def _points_generator(shape: Shape):
        x_y_generator = itertools.product(
            range(shape.width),
            range(shape.height))
        for x, y in x_y_generator:
            yield Point(x, y)

    def group_grid(self, X: np.ndarray) -> dict:
        """ TODO: Docstring for fit.

        :param X: TODO
        :returns: TODO
        """
        grouped_points = dict()
        # keep track of points that are already set
        set_points = set()

        # find the modes for all the points
        matrix_shape = Shape(*X.shape)
        for point in self._points_generator(matrix_shape):
            if point in set_points:
                continue

            mode_point = find_point_mode(
                start_point=point,
                X=X, rectangle_shape=self.rectangle_shape)
            _logger.info("With start point {}, found mode point {}."
                         .format(point, mode_point))

            mode_point_neighbors = list(
                generate_region_points(
                    mode_point,
                    rectangle_shape=self.rectangle_shape,
                    map_shape=matrix_shape))

            for tmp_point in mode_point_neighbors:
                if tmp_point not in set_points:
                    set_points.add(tmp_point)
                    grouped_points.setdefault(mode_point, []).append(tmp_point)

        return grouped_points


def find_point_mode(
        start_point: Point,
        X: np.ndarray,
        rectangle_shape: Shape,
        n_iterations: Union[int, None]=None) -> Point:
    """
    Find the mode given a starting point via the mean shift algorithm.
    Will weight points in the grid with the value in 'X'.

    :param n_iterations: the number of iterations to run; if None,
        will run until the stopping criterion is satisfied
    """

    def _stopping_criterion(p1: Point, p2: Point, threshold: float=0.001):
        dist = distance.euclidean((p1.x, p1.y), (p2.x, p2.y))
        return dist < threshold

    if n_iterations is None:
        n_iterations = math.inf

    iteration_count = 0
    point, prev_point = start_point, start_point
    while iteration_count < n_iterations:
        prev_point = point
        point = calculate_mid(point, X, rectangle_shape)
        if _stopping_criterion(point, prev_point):
            break

        iteration_count += 1

    return point


def calculate_mid(
        center_point: Point,
        X: np.ndarray,
        rectangle_shape: Shape) -> Point:
    """
    Calculate the weighted 'mid' center_point among the points within
    the region of the rectangle

    :param X: TODO
    :param center_point: the point at which
    :param shape: the shape of the rectange kernel (width, height)
    """
    normalizing_constant = 0
    weighted_sum_x, weighted_sum_y = 0, 0
    for point in generate_region_points(center_point, rectangle_shape,
                                        map_shape=Shape(*X.shape)):
        point_weight = X[point.y][point.x]
        weighted_sum_x += point_weight * point.x
        weighted_sum_y += point_weight * point.y
        normalizing_constant += point_weight

    # none of the points in the matrix had values
    if normalizing_constant == 0:
        # then return the starting point
        return center_point

    return Point(
        weighted_sum_x / normalizing_constant,
        weighted_sum_y / normalizing_constant)


def generate_region_points(
        center_point: Point,
        rectangle_shape: Shape,
        map_shape: Shape):
    """
    Used to generate the points in the region of the rectangle
    centered at the passed 'center_point'

    :param center_point: TODO
    :param rectange_shape: TODO
    """
    left_boundary = math.floor(
        max(center_point.x - rectangle_shape.width, 0))
    right_boundary = math.floor(
        min(center_point.x + rectangle_shape.width + 1, map_shape.width))
    bottom_boundary = math.floor(
        max(center_point.y - rectangle_shape.height, 0))
    top_boundary = math.floor(
        min(center_point.y + rectangle_shape.height + 1, map_shape.height))

    x_y_generator = itertools.product(
        range(left_boundary, right_boundary),
        range(bottom_boundary, top_boundary))

    _logger.info("With center point: {}, found left/right boundaries: {} "
                 "and bottom/top boundaries: {}."
                 .format(center_point, (left_boundary, right_boundary),
                         (bottom_boundary, top_boundary)))
    for x, y in x_y_generator:
        _logger.info("Generating Point(x={}, y={})".format(x, y))
        yield Point(x, y)


if __name__ == "__main__":
    pass
