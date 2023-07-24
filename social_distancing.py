import numpy as np
import pandas as pd
import cv2


def violations(centroid_list, minSocialDistance, objectID_list):
    """
    Checks every pair of detected objects in a frame to see if the distance between them is less than `minSocialDistance`.

    Parameters:
    centroid_list (list): A list of centroids of detected objects in the frame.
    minSocialDistance (float): A threshold defining the minimum allowed distance between two objects.
    objectID_list (list): A list of IDs of the detected objects in the frame.

    Returns:
    violating (list): A list of tuples, where each tuple contains the indices of two objects that are violating social distancing rules.
    vio_ID (list): A list of tuples, where each tuple contains the IDs of two objects that are violating social distancing rules.
    vio_time (list): A list of datetime objects, where each datetime corresponds to the time of a violation.
    vio_indices (list): A list of integers, where each integer is the index of a violating object in the `centroid_list`.
    """
    from scipy.spatial import distance
    from datetime import datetime as dt

    distance_matrix = distance.cdist(centroid_list, centroid_list, metric="euclidean")

    # Find indices (centroids) violating social
    # Distancing criteria
    # (ind1,ind2) means object #ind1 and object #ind2 are violating
    violated = [
        (ind1, ind2)
        for ind1, row in enumerate(distance_matrix)
        for ind2, col in enumerate(row)
        if col < minSocialDistance and col > 0 and ind1 != ind2
    ]

    # Find pairs where the distance is less than the minimum allowed
    violating = np.argwhere(distance_matrix < minSocialDistance)

    # Keep only unique pairs and ensure the pair (i, j) is the same as (j, i)
    violating = [(min(i), max(i)) for i in violating if i[0] != i[1]]
    violating = list(set(violating))

    # Based on ObjectID
    vio_ID = [(objectID_list[i], objectID_list[j]) for (i, j) in violating]

    # Convert to a list of indices (then delete duplicate)
    vio_indices = [item for i in violating for item in i]
    vio_indices = list(dict.fromkeys(vio_indices))

    vio_time = dt.now()

    return violating, vio_ID, vio_time, vio_indices


def detect_violations(
    objects,
    centroid_list,
    minSocialDistance,
    objectID_list,
    rect_list,
    frame,
    num_frames,
    violations_df,
):
    """
    Detects social distancing violations in a given frame.

    Parameters:
    - objects: dictionary containing ID, centroid, and bounding box coordinates of detected objects.
    - centroid_list: list of tuples containing the centroid coordinates of detected objects.
    - minSocialDistance: integer specifying the minimum distance for social distancing.
    - objectID_list: list of IDs for the detected objects.
    - rect_list: list of tuples containing the bounding box coordinates of detected objects.
    - frame: the current frame being processed.
    - num_frames: total number of frames processed so far.
    - violations_df: DataFrame where violation data will be appended.

    Returns:
    - frame: the processed frame with marked violations.
    - violations_df: DataFrame containing information about social distancing violations.
    """

    # Ensure at least 2 objects are there in the frame
    if len(objects) >= 2:
        # List of tuples contains indices of violating centroids in a given frame
        violations_list, vio_ID, vio_time, vio_ind_list = violations(
            centroid_list, minSocialDistance, objectID_list
        )

        # Get the rects corresponding to the violated indx
        for k, rect in enumerate(rect_list):
            for j in violations_list:
                if k == j[1] or k == j[0]:
                    startX = rect[0]
                    startY = rect[1]
                    endX = rect[2]
                    endY = rect[3]

                    # Red rects for violations
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

        # List is updated each frame
        if vio_ID != []:
            violations_df = violations_df.append(
                {
                    "time": vio_time,
                    "ids": vio_ID,
                    "frame": num_frames,
                    "object_num": len(objects),
                    "violation_num": len(vio_ind_list),
                },
                ignore_index=True,
            )
    return frame, violations_list, violations_df
