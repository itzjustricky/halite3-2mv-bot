"""

Test segmentation methodologies
"""

import logging
import argparse
import functools

from typing import List

import numpy as np
import matplotlib.pyplot as plt

import mean_shift
from mean_shift import Point
from mean_shift import Shape


_logger = logging.getLogger("deposit_clustering")


SAMPLE_HALITE_MAP = np.array([
    [0.8, 0.7, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.1, 0.7, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.0, 0.0],
    [0.2, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.7, 0.0, 0.3],
    [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.9, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.1, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
])


SEQUENTIAL_CMAPS = [
    'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
    'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
    'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']


def get_args(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Argument flag to determine whether or not to set "
             "logging configuration to debug.")
    return vars(parser.parse_args(args))


def main():

    parsed_args = get_args()
    log_level = logging.DEBUG if parsed_args["debug"] else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s %(levelname)s %(message)s')

    # this is a hardcoded hyper-parameter
    rectangle_shape = Shape(1, 1)

    ms_cluster = mean_shift.MeanShiftCluster(rectangle_shape)
    point_groupings = ms_cluster.group_grid(SAMPLE_HALITE_MAP)
    _logger.info("There are {} number of groups formed.".format(len(point_groupings)))

    keep_threshold = 0.05
    to_delete_points = []
    for mode_point, points in point_groupings.items():
        point_score = mean_shift.points_score(mode_point, SAMPLE_HALITE_MAP)
        if point_score < keep_threshold:
            _logger.debug("Will pop group with mode point {} because of score {} < {}"
                          .format(mode_point, point_score, keep_threshold))
            to_delete_points.append(mode_point)

    for mode_point in to_delete_points: point_groupings.pop(mode_point)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plot_map(SAMPLE_HALITE_MAP)
    for ind, (mode_point, points) in enumerate(point_groupings.items()):
        _logger.info("Group with mode point: {} (has {} members) is labeled '{}'."
                     .format(mode_point, len(points), ind))

        label_points(str(ind), points, ax)

    plt.show()


def test_find_point_mode():
    # starting_point = Point(8, 4)
    starting_point = Point(5, 4)
    rectangle_shape = Shape(2, 2)
    map_shape = Shape(*SAMPLE_HALITE_MAP.shape)

    mode_point_fn = functools.partial(
        mean_shift.find_point_mode,
        start_point=starting_point,
        X=SAMPLE_HALITE_MAP,
        rectangle_shape=rectangle_shape)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_map(SAMPLE_HALITE_MAP)

    mode_points = [
        mode_point_fn(n_iterations=iter_count)
        for iter_count in range(5)]
    for point in mode_points:
        plot_point(point, color="blue")

    plot_annotations(mode_points, ax)

    # point with the highest value
    highest_value_point = Point(x=7, y=6)
    plot_point(highest_value_point)

    xy = highest_value_point.x, highest_value_point.y
    ax.annotate("({:.1f}, {:.1f})".format(*xy),
                xy=xy, textcoords="data")

    # plot the transparent area of the rectangle surrounding first iteration
    neighborhood_points = list(
        mean_shift.generate_region_points(
            mode_points[0],
            rectangle_shape=rectangle_shape, map_shape=map_shape))
    window_matrix = matrix_from_points(neighborhood_points, map_shape)
    plot_map(window_matrix, color_map="Blues", alpha=0.5)

    # plot the transparent area of the rectangle surrounding fifth iteration
    neighborhood_points = list(
        mean_shift.generate_region_points(
            mode_points[-1],
            rectangle_shape=rectangle_shape, map_shape=map_shape))
    window_matrix = matrix_from_points(neighborhood_points, map_shape)
    plot_map(window_matrix, color_map="Purples", alpha=0.5)

    plt.show()


def plot_map(X: np.ndarray, *, color_map: str="Greys", alpha: float=1):
    """ Creates a grayscale plot of the representation of halite in the map

    :param X: is the matrix representation of the halite in the map
    """
    n_rows, n_cols = X.shape

    ax = plt.imshow(
        X, cmap=color_map,
        interpolation="nearest", alpha=alpha,
        origin="upper",
        extent=(0, n_cols, n_rows, 0),
    )
    return ax


def plot_point(point: Point, marker: str='o', color: str="red"):
    line_object = plt.plot(
        point.x, point.y,
        marker=marker, markersize=3, color=color)[0]
    return line_object


def plot_annotations(points: List[Point], ax):
    for ind, point in enumerate(points):
        xy = point.x, point.y
        ax.annotate(
            "[{}] ({:.1f}, {:.1f})".format(ind, *xy),
            xy=xy, textcoords="data")


def label_points(label: str, points: List[Point], ax):
    for point in points:
        # add some buffer to center it
        xy = point.x + 0.5, point.y + 0.5
        ax.annotate(label,
                    xy=xy, textcoords="data")


def matrix_from_points(points: List[Point], matrix_shape: Shape):
    """
    Create a matrix of 0/1 in which the matrix has value 1 at the points passed.

    :param points: the list of points to create the matrix from
    """
    X = np.empty((matrix_shape.height, matrix_shape.width))
    X[:] = np.nan
    for point in points:
        X[point.y][point.x] = 1
    return X


if __name__ == "__main__":
    # test_find_point_mode()
    main()
