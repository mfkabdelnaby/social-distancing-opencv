import cv2
import dlib
import numpy as np


def object_detection(frame, rects, net, ln, H, W, thres, nms_thres):
    """
    This function performs object detection in a video frame using YOLO.

    Args:
    frame (numpy.ndarray): The current video frame.
    rects (list): List of rectangles representing detected objects.
    net (dnn_Net): Pre-initialized YOLO model.
    ln (list): YOLO output layer names.
    H (int): Height of the video frame.
    W (int): Width of the video frame.
    thres (float): YOLO detection threshold.
    nms_thres (float): YOLO Non-maxima suppression threshold.

    Returns:
    trackers (list): List of dlib correlation trackers.
    rects (list): Updated list of rectangles representing detected objects.
    confidences (list): List of confidence values for each detected object.
    classIDs (list): List of class IDs for each detected object.
    """

    trackers = []

    # determine only the *output* layer names that we need from YOLO
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter person  > Class ID = 0
            if confidence > thres and classID == 0:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # Top and Left corner of box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, thres, nms_thres)
    # Ensure at least one detection
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # Instantiate dlib correlation tracker
            tracker = dlib.correlation_tracker()
            # Find object's bounding box
            rect = dlib.rectangle(x, y, x + w, y + h)
            # Start tracking
            tracker.start_track(frame, rect)
            # Append trackers
            trackers.append(tracker)

            rects.append((x, y, x + w, y + h))

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

    return trackers, rects, confidences, classIDs


def object_tracking(frame, trackers):
    """
    This function performs object tracking in a video frame using dlib.

    Args:
    frame (numpy.ndarray): The current video frame.
    trackers (list): List of dlib correlation trackers.

    Returns:
    rects (list): Updated list of rectangles representing tracked objects.
    """

    rects = []

    for tracker in trackers:
        # Update the tacker and grab update position
        tracker.update(frame)
        pos = tracker.get_position()

        # Unpack the object's position
        startX = int(pos.left())
        startY = int(pos.top())
        endX = int(pos.right())
        endY = int(pos.bottom())

        # Append the rects with the new bounding box coordinates
        rects.append((startX, startY, endX, endY))

        # Draw Rectangle
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 1)

    return rects
