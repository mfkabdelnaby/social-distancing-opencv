# imports

from centroid.centroidtracker_TimeRect import CentroidTracker
from centroid.trackableobject import TrackableObject

import numpy as np
import pandas as pd
from datetime import datetime as dt

import copy

import os
import sys

import cv2

# Calculate estimated frame/sec
import imutils
from imutils.video import FPS


from src.utlis import counting
from src.social_distancing import detect_violations
from src.visualizations import birdView_plot, motion_heatmap
from src.detect_track import object_detection, object_tracking
from src.constants import *


def main(input_dir, output_dir):
    # Load the line points
    pt1, pt2 = np.load("line_points.npy")

    video_path = os.path.join(input_dir, "TownCentreXVID.avi")

    print(video_path)
    video = cv2.VideoCapture(video_path)

    # Load the YOLO model
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    video = cv2.VideoCapture(video_path)
    cv2.namedWindow("count", cv2.WINDOW_AUTOSIZE)

    # Initiate the YOLO Model
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    # Heatmap Variables
    background_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # number of frames

    video = cv2.VideoCapture(video_path)
    cv2.namedWindow("count", cv2.WINDOW_AUTOSIZE)

    # ======================#
    # Initiatizations
    # ======================#

    frame_list = []
    background_list = []
    heat_frame_list = []

    # Initiate the dataframes

    # places
    place_df = pd.DataFrame(columns=["time", "entries", "exits", "ids", "centroids"])
    user_df = pd.DataFrame(columns=["id", "centroid"])

    # violations df
    violations_df = pd.DataFrame(
        columns=["time", "ids", "frame", "object_num", "violation_num"]
    )

    # Initiate the total # of frames, and entries and exists
    num_entries = 0
    num_exits = 0
    num_frames = 0

    # Initialize writer
    writer = None

    # Initiate the CentriodTracker
    ct = CentroidTracker(maxDisappeared, maxDistance=50)

    # Initiate list to store dlib correlation tracker
    trackers = []

    # Initiate a dict to map each unique ID to Trackableoject
    trackableObjects = {}

    # Start the f/s estimator
    fps = FPS().start()

    # while we read the video
    while True:
        res, frame = video.read()
        # res2, frame2 = video.read()  # for the heatmap

        if not res:
            break

        # Resize and convert the frame into RGB
        frame = imutils.resize(frame, width=500)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Set the dimentions of the frame H & W
        H = frame.shape[0]
        W = frame.shape[1]

        # Define Background for the BirdView
        background = np.zeros((H, W), np.uint8)
        background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)

        # Define Writers
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(
                os.path.join(output_dir, "out.avi"), fourcc, 30, (W, H), True
            )
            writer_bird = cv2.VideoWriter(
                os.path.join(output_dir, "out_bird.avi"), fourcc, 30, (W, H), True
            )
            writer_heat = cv2.VideoWriter(
                os.path.join(output_dir, "out_heat.avi"), fourcc, 30, (W, H), True
            )

        # Initialize status & list of bounding boxes
        status = "Waiting"
        rects = []

        #  Initializtion for the heatmap
        if num_frames == 0:
            first_frame = copy.deepcopy(frame)
            accum_image = np.zeros(frame.shape[:2], np.uint8)

        # ====================================================#
        # If: we utilize object "Detection" for skipped-based frames, else "Tracking"
        # ====================================================#

        # Check if we skip frames to avoid computational complexity
        if num_frames % skipframes == 0:
            status = "Detecting"

            # Call the object detection function
            trackers, rects, confidences, classIDs = object_detection(
                frame, rects, net, ln, H, W, thres, nms_thres
            )

        else:
            status = "Tracking"
            rects = object_tracking(frame, trackers)

        # ====================================================#
        # Update Centriod Tracker with newly computed centriod
        # ====================================================#
        objects = ct.update(rects)

        # Retrieve the dictionaries of the start and end time of each ID
        startID = ct.start_date()
        endID = ct.end_date()

        print("end", endID)

        # ====================================================#
        # Count Entries and Exits, write ID, and draw circle
        # ====================================================#

        # Initiate list of detected objects (rects, IDs, centroids)
        rect_list = []
        objectID_list = []
        centroid_list = []

        for objectID, (centroid, rect) in objects.items():
            # check to see if a trackable object exists for the current ID
            to = trackableObjects.get(objectID, None)
            # if there is no existing trackable object, create one
            if to is None:
                to = TrackableObject(objectID, centroid)

            # Otherwise, there is trackable object we utilize it to determine
            # Direction
            else:
                num_entries, num_exits = counting(
                    frame, to, centroid, objectID, pt1, pt2, num_entries, num_exits
                )

            # store the trackable object in our dictionary
            trackableObjects[objectID] = to

            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID:{}".format(objectID)
            cv2.putText(
                frame,
                text,
                (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (0, 255, 255),
                1,
            )
            cv2.circle(frame, (centroid[0], centroid[1]), 2, (0, 255, 255), -1)

            # Update the list of tracked objects
            objectID_list.append(objectID)
            centroid_list.append(tuple(centroid))
            rect_list.append(rect)

        # ====================================================#
        # Social Distancing Detection - outside loop
        # ====================================================#

        frame, violations_list, violations_df = detect_violations(
            objects,
            centroid_list,
            minSocialDistance,
            objectID_list,
            rect_list,
            frame,
            num_frames,
            violations_df,
        )

        # ====================================================#
        # BirdView
        # ====================================================#

        birdview = birdView_plot(background, centroid_list, violations_list)

        # ====================================================#
        # HeatMap function call
        # ====================================================#

        heat_frame, accum_image, first_frame = motion_heatmap(
            frame,
            num_frames,
            thres_heat,
            maxValue,
            accum_image,
            first_frame,
            background_subtractor,
        )

        # ====================================================#
        # Information to be displayed on the frame
        # ====================================================#
        info = [
            ("Time", dt.now()),
            ("Entry", num_entries),
            ("Exit", num_exits),
            ("Status", status),
        ]

        # loop over the info tuple and draw them on our frame
        for i, (k, v) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(
                frame,
                text,
                (7, H - ((i * 12) + 12)),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (0, 0, 255),
                1,
            )

        # ====================================================#
        # DATAFRAMES: place and user
        # ====================================================#

        # We need to do update the dataframe for every 10 seconds
        place_multiplier = place_seconds * video.get(cv2.CAP_PROP_FPS)

        # If you don't want to skip
        if place_seconds == 0:
            place_multiplier = 1

        if num_frames % place_multiplier == 0:
            place_df = place_df.append(
                {
                    "time": dt.now(),
                    "entries": num_entries,
                    "exits": num_exits,
                    "ids": objectID_list,
                    "centroids": centroid_list,
                },
                ignore_index=True,
            )

        user_multiplier = user_seconds * video.get(cv2.CAP_PROP_FPS)

        if user_seconds == 0:
            user_multiplier = 1

        # This will append all objects for all frames
        if num_frames % user_multiplier == 0:
            for objectID, (centroid, rect) in objects.items():
                if objectID in user_df["id"].values:
                    # The user is already in the DataFrame, update their centroids list
                    user_df.loc[user_df["id"] == objectID, "centroid"].apply(
                        lambda x: x.append(tuple(centroid))
                    )
                else:
                    # The user is not in the DataFrame, create a new row for them
                    user_df = user_df.append(
                        {"id": objectID, "centroid": [tuple(centroid)]},
                        ignore_index=True,
                    )

        # ====================================================#
        # Save Frames/Images
        # ====================================================#

        # SUGGESTION: for longer footage do it while skipping some frames
        frame_list.append(frame)
        background_list.append(background)
        heat_frame_list.append(heat_frame)

        # ====================================================#
        # Window name and exit condition
        # ====================================================#
        writer.write(frame)
        writer_bird.write(birdview)
        writer_heat.write(heat_frame)

        # show the output frame
        cv2.imshow("count", frame)
        cv2.imshow("birdview", birdview)
        cv2.imshow("heatmap", heat_frame)

        # if the ESC key was pressed, break from the loop
        if cv2.waitKey(1) & 0xFF == 27:
            break

        # increment the total number of frames processed thus far and
        # then update the FPS counter
        num_frames += 1
        fps.update()

    # ====================================================#
    # HeatMap final calculations
    # ====================================================#

    color_image = cv2.applyColorMap(accum_image, cv2.COLORMAP_JET)
    result_overlay = cv2.addWeighted(first_frame, 0.7, color_image, 0.7, 0)

    # save the final heatmap image
    cv2.imwrite(os.path.join(output_dir, "final_heatmap.jpg"), result_overlay)

    # stop the timer and display FPS information
    fps.stop()

    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    print("[INFO] Total Frames: {:.2f}".format(num_frames))

    # check to see if we need to release the video writer pointer
    writer.release()
    writer_bird.release()
    writer_heat.release()
    video.release()
    cv2.destroyAllWindows()

    ## DATA FRAMES
    user_df["StartTime"] = user_df.id.map(startID)
    user_df["EndTime"] = user_df.id.map(endID)

    dataframes_dir = os.path.join("", "dataframes")
    os.makedirs(dataframes_dir, exist_ok=True)

    # Save the DataFrames to CSV files
    place_df.to_csv(os.path.join(dataframes_dir, "place_df.csv"), index=False)
    user_df.to_csv(os.path.join(dataframes_dir, "user_df.csv"), index=False)
    violations_df.to_csv(os.path.join(dataframes_dir, "violations_df.csv"), index=False)


if __name__ == "__main__":
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    main(input_dir, output_dir)
