from typing import Tuple, List
import cv2
import numpy as np


def birdView_plot(
    frame: np.ndarray,
    centroid_list: List[Tuple[int, int]],
    violations: List[Tuple[int, int]],
) -> np.ndarray:
    """
    Generate a bird's eye view plot on the frame.

    Parameters:
    frame (np.ndarray): The frame to plot on.
    centroid_list (list): A list of centroids of detected objects in the frame.
    violations (list): A list of tuples, each containing indices of two objects that are violating social distancing rules.

    Returns:
    background (np.ndarray): The frame with the bird's eye view plot.
    """

    # First, we create a black background with the same dimensions as the frame
    H, W = frame.shape[:2]
    background = np.zeros((H, W), np.uint8)
    background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)

    # Then we draw circles around centroids on the background
    for centroid in centroid_list:
        cv2.circle(background, (centroid[0], centroid[1]), 2, (0, 255, 255), -1)

    # Now, we draw lines between centroids that are too close to each other
    # and color the violating centroids in red
    for j in violations:
        cv2.circle(
            background,
            (centroid_list[j[0]][0], centroid_list[j[0]][1]),
            10,
            (0, 0, 255),
            1,
        )
        cv2.circle(
            background,
            (centroid_list[j[1]][0], centroid_list[j[1]][1]),
            10,
            (0, 0, 255),
            1,
        )
        cv2.line(background, centroid_list[j[0]], centroid_list[j[1]], (0, 0, 255), 4)

    return background


def motion_heatmap(
    frame, Frames, thres_heat, maxValue, accum_image, first_frame, background_subtractor
):
    """
    Function to generate a motion heatmap showing where movement has occurred throughout the video.
    The heatmap is updated at each frame based on the detected movement in the current frame.

    Parameters:
    frame (np.array): The current frame of the video.
    Frames (int): The current frame number.
    thres_heat (int): Threshold for the heatmap.
    maxValue (int): Maximum value for the threshold.
    accum_image (np.array): Accumulated image, which gets updated at each frame.
    first_frame (np.array): The first frame of the video.
    background_subtractor: The background subtractor object used to subtract the background from the current frame.


    Returns:
    heat_frame (np.array): The current frame with the heatmap applied.
    accum_image (np.array): The updated accumulated image.
    first_frame (np.array): The first frame of the video.
    """

    import copy

    # If first frame
    if Frames == 0:
        # Initialize the first_frame and accum_image
        first_frame = copy.deepcopy(frame)
        accum_image = np.zeros(frame.shape[:2], np.uint8)
        heat_frame = np.zeros(frame.shape[:2], np.uint8)

    else:
        # Apply a background subtractor to the current frame
        subtractor = background_subtractor.apply(frame)
        cv2.imwrite("heatmap/frame.jpg", frame)
        cv2.imwrite("heatmap/diff-bkgnd-frame.jpg", subtractor)

        # Apply a threshold to the subtracted image
        threshold = thres_heat
        maxValue = maxValue
        ret, th1 = cv2.threshold(subtractor, threshold, maxValue, cv2.THRESH_BINARY)

        # Add the threshold image to the accumulated image
        accum_image = cv2.add(accum_image, th1)

        # Apply a colormap to the accumulated image
        color_image_video = cv2.applyColorMap(accum_image, cv2.COLORMAP_JET)

        # Combine the colored accumulated image with the current frame to generate the heat_frame
        heat_frame = cv2.addWeighted(frame, 0.7, color_image_video, 0.7, 0)

    return heat_frame, accum_image, first_frame
