from typing import Tuple
import numpy as np
import cv2

from numpy import ones, vstack
from numpy.linalg import lstsq


def line_orientation(
    frame: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int]
) -> str:
    """
    Determines the orientation of a line given two points (pt1 and pt2) and draws the line on a frame.

    Parameters:
    frame (np.ndarray): The frame on which to draw the line.
    pt1 (Tuple[int, int]): The first point (x1, y1).
    pt2 (Tuple[int, int]): The second point (x2, y2).

    Returns:
    str: The orientation of the line ('horizontal', 'vertical', or 'sloped').
    """

    cv2.line(frame, pt1, pt2, (255, 0, 0), 2)

    if abs(pt1[0] - pt2[0]) < 10:
        orientation = "Perfect_Horizantal"
    elif abs(pt1[1] - pt2[1]) < 10:
        orientation = "Perfect_Vertical"

    elif abs(pt1[0] - pt2[0]) > abs(pt1[1] - pt2[1]):
        orientation = "Horizantal"
    else:
        orientation = "Vertical"

    return orientation


def f(x: int, pt1: Tuple[int, int], pt2: Tuple[int, int]) -> float:
    """
    Calculates the y-coordinate of a point on a line given its x-coordinate,
    and the coordinates of two other points on the line (pt1 and pt2).

    Parameters:
    x (int): The x-coordinate of the point.
    pt1 (Tuple[int, int]): The first point (x1, y1) on the line.
    pt2 (Tuple[int, int]): The second point (x2, y2) on the line.

    Returns:
    float: The y-coordinate of the point.

    Raises:
    ZeroDivisionError: If the x-coordinates of pt1 and pt2 are the same.
    """

    if pt1[0] == pt2[0]:
        raise ZeroDivisionError("The x-coordinates of pt1 and pt2 cannot be the same.")

    points = [pt1, pt2]
    x_coords, y_coords = zip(*points)
    A = vstack([x_coords, ones(len(x_coords))]).T
    m, b = lstsq(A, y_coords, rcond=None)[0]

    y = m * x + b

    return y


def counting(frame, to, centroid, objectID, pt1, pt2, num_entries, num_exits):
    """
    Counts the number of "entries" and "exits" in a video frame.

    Parameters:
    frame (ndarray): The current video frame.
    to (TrackableObject): The object to be tracked.
    centroid (Tuple[int, int]): The centroid of the tracked object.
    objectID (int): The ID of the object.
    pt1 (Tuple[int, int]): The start point of the line.
    pt2 (Tuple[int, int]): The end point of the line.
    num_entries (int): The current count of entries.
    num_exits (int): The current count of exits.

    Returns:
    Tuple[int, int]: The updated counts of entries and exits.
    """

    orientation = line_orientation(frame, pt1, pt2)

    if orientation == "Perfect_Horizantal":
        # Difference between y-cor. of the 'current' centroid
        # and the mean of 'previous' centroids
        # -ve for 'up', +ve for 'down'
        y = [c[1] for c in to.centroids]
        direction = centroid[1] - np.mean(y)
        to.centroids.append(centroid)

        # Check if object has been counted or not
        # if direction<0 and centriod above line
        if not to.counted:
            if direction < 0 and centroid[1] < pt1[1]:
                num_entries += 1
                to.counted = True

            elif direction > 0 and centroid[1] > pt1[1]:
                num_exits += 1
                to.counted = True

    if orientation == "Perfect_Vertical":
        # -ve for 'left', +ve for 'right'
        x = [c[0] for c in to.centroids]
        direction = centroid[0] - np.mean(x)
        to.centroids.append(centroid)

        # Check if object has been counted or not
        # if direction<0 and centriod above line
        if not to.counted:
            if direction < 0 and centroid[0] > pt1[0]:
                num_entries += 1
                to.counted = True

            elif direction > 0 and centroid[0] < pt1[0]:
                num_exits += 1
                to.counted = True

    if orientation == "Horizantal":
        # -ve for 'up', +ve for 'down'
        y = [c[1] for c in to.centroids]
        direction = centroid[1] - np.mean(y)
        to.centroids.append(centroid)

        # Check if object has been counted or not
        # if direction<0 and centriod above line
        # at least one value of y should be below the line to validated entry
        if not to.counted:
            if (
                direction < 0
                and centroid[1] < f(centroid[0], pt1, pt2)
                and len([i for i in y if i > f(centroid[0], pt1, pt2)]) > 0
            ):
                num_entries += 1
                to.counted = True

            elif (
                direction > 0
                and centroid[1] > f(centroid[0], pt1, pt2)
                and len([i for i in y if i < f(centroid[0], pt1, pt2)]) > 0
            ):
                num_exits += 1
                to.counted = True

    if orientation == "Vertical":
        # -ve for 'up', +ve for 'down'
        x = [c[0] for c in to.centroids]
        direction = centroid[0] - np.mean(x)
        to.centroids.append(centroid)

        # x<0 -> direction<0 and object walking left
        if not to.counted:
            if (
                direction < 0
                and centroid[0] > f(centroid[1], pt1, pt2)
                and len([i for i in y if i > f(centroid[1], pt1, pt2)]) > 0
            ):
                num_entries += 1
                to.counted = True

            elif (
                direction > 0
                and centroid[0] < f(centroid[1], pt1, pt2)
                and len([i for i in y if i > f(centroid[1], pt1, pt2)]) > 0
            ):
                num_exits += 1
                to.counted = True

    return num_entries, num_exits
