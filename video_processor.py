# imports

from classes.centroidtracker_TimeRect import CentroidTracker
from classes.trackableobject import TrackableObject

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


from utlis import counting
from social_distancing import detect_violations
from visualizations import birdView_plot, motion_heatmap
from detect_track import object_detection, object_tracking
from constants import *


class VideoProcessor:
    def __init__(self, net, ln, video):
        # Load the line points
        self.pt1, self.pt2 = np.load("line_points.npy")

        self.video = video

        self.output_dir = "/output"

        # Load the YOLO model
        self.net = net
        self.ln = ln

        # Heatmap Variables
        self.background_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()
        self.first_frame = None
        self.accum_image = None

        # Initiatizations
        self.frame_list = []
        self.background_list = []
        self.heat_frame_list = []

        # Initiate the dataframes
        self.place_df = pd.DataFrame(
            columns=["time", "entries", "exits", "ids", "centroids"]
        )
        self.user_df = pd.DataFrame(columns=["id", "centroid"])
        self.violations_df = pd.DataFrame(
            columns=["time", "ids", "frame", "object_num", "violation_num"]
        )

        # Initiate the total # of frames, and entries and exists
        self.num_entries = 0
        self.num_exits = 0
        self.num_frames = 0

        # Initiate the CentriodTracker
        self.ct = CentroidTracker(maxDisappeared, maxDistance=50)

        # Initiate list to store dlib correlation tracker
        self.trackers = []

        # Initiate a dict to map each unique ID to Trackableoject
        self.trackableObjects = {}

        # Start the f/s estimator
        self.fps = FPS().start()

    def process_frame(self, frame):
        # Resize and convert the frame into RGB
        frame = imutils.resize(frame, width=500)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Set the dimentions of the frame H & W
        H = frame.shape[0]
        W = frame.shape[1]

        # Define Background for the BirdView
        background = np.zeros((H, W), np.uint8)
        background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)

        # Initialize status & list of bounding boxes
        status = "Waiting"
        rects = []

        # Check if we skip frames to avoid computational complexity
        if self.num_frames % skipframes == 0:
            status = "Detecting"
            # Call the object detection function
            self.trackers, rects, confidences, classIDs = object_detection(
                frame, rects, self.net, self.ln, H, W, thres, nms_thres
            )
        else:
            status = "Tracking"
            rects = object_tracking(frame, self.trackers)

        # Update Centriod Tracker with newly computed centriod
        objects = self.ct.update(rects)

        # Retrieve the dictionaries of the start and end time of each ID
        self.startID = self.ct.start_date()
        self.endID = self.ct.end_date()

        # Count Entries and Exits, write ID, and draw circle
        rect_list = []
        objectID_list = []
        centroid_list = []

        for objectID, (centroid, rect) in objects.items():
            # check to see if a trackable object exists for the current ID
            to = self.trackableObjects.get(objectID, None)
            # if there is no existing trackable object, create one
            if to is None:
                to = TrackableObject(objectID, centroid)
            # Otherwise, there is trackable object we utilize it to determine
            # Direction
            else:
                self.num_entries, self.num_exits = counting(
                    frame,
                    to,
                    centroid,
                    objectID,
                    self.pt1,
                    self.pt2,
                    self.num_entries,
                    self.num_exits,
                )
            # store the trackable object in our dictionary
            self.trackableObjects[objectID] = to

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

        # Social Distancing Detection - outside loop
        frame, violations_list, self.violations_df = detect_violations(
            objects,
            centroid_list,
            minSocialDistance,
            objectID_list,
            rect_list,
            frame,
            self.num_frames,
            self.violations_df,
        )

        # BirdView
        birdview = birdView_plot(background, centroid_list, violations_list)

        # HeatMap
        # HeatMap function call
        heat_frame, self.accum_image, self.first_frame = motion_heatmap(
            frame,
            self.num_frames,
            thres_heat,
            maxValue,
            self.accum_image,
            self.first_frame,
            self.background_subtractor,
        )

        # Information to be displayed on the frame
        info = [
            ("Time", dt.now()),
            ("Entry", self.num_entries),
            ("Exit", self.num_exits),
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

        # DATAFRAMES: place and user
        # We need to do update the dataframe for every 10 seconds
        place_multiplier = place_seconds * self.video.get(cv2.CAP_PROP_FPS)

        # If you don't want to skip
        if place_seconds == 0:
            place_multiplier = 1

        if self.num_frames % place_multiplier == 0:
            self.place_df = self.place_df.append(
                {
                    "time": dt.now(),
                    "entries": self.num_entries,
                    "exits": self.num_exits,
                    "ids": objectID_list,
                    "centroids": centroid_list,
                },
                ignore_index=True,
            )

        user_multiplier = user_seconds * self.video.get(cv2.CAP_PROP_FPS)

        if user_seconds == 0:
            user_multiplier = 1

        # This will append all objects for all frames
        if self.num_frames % user_multiplier == 0:
            for objectID, (centroid, rect) in objects.items():
                if objectID in self.user_df["id"].values:
                    # The user is already in the DataFrame, update their centroids list
                    self.user_df.loc[self.user_df["id"] == objectID, "centroid"].apply(
                        lambda x: x.append(tuple(centroid))
                    )
                else:
                    # The user is not in the DataFrame, create a new row for them
                    self.user_df = self.user_df.append(
                        {"id": objectID, "centroid": [tuple(centroid)]},
                        ignore_index=True,
                    )

        # Save Frames/Images
        # SUGGESTION: for longer footage do it while skipping some frames
        self.frame_list.append(frame)
        self.background_list.append(background)
        self.heat_frame_list.append(heat_frame)

        # increment the total number of frames processed thus far and
        # then update the FPS counter
        self.num_frames += 1
        self.fps.update()

        return frame

    def finish_processing(self):
        # HeatMap final calculations
        color_image = cv2.applyColorMap(self.accum_image, cv2.COLORMAP_JET)
        result_overlay = cv2.addWeighted(self.first_frame, 0.7, color_image, 0.7, 0)

        # save the final heatmap image
        cv2.imwrite(os.path.join(self.output_dir, "final_heatmap.jpg"), result_overlay)

        # stop the timer and display FPS information
        self.fps.stop()

        print("[INFO] elapsed time: {:.2f}".format(self.fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(self.fps.fps()))
        print("[INFO] Total Frames: {:.2f}".format(self.num_frames))

        ## DATA FRAMES
        self.user_df["StartTime"] = self.user_df.id.map(self.startID)
        self.user_df["EndTime"] = self.user_df.id.map(self.endID)

        dataframes_dir = os.path.join("", "dataframes")
        os.makedirs(dataframes_dir, exist_ok=True)

        # Save the DataFrames to CSV files
        self.place_df.to_csv(os.path.join(dataframes_dir, "place_df.csv"), index=False)
        self.user_df.to_csv(os.path.join(dataframes_dir, "user_df.csv"), index=False)
        self.violations_df.to_csv(
            os.path.join(dataframes_dir, "violations_df.csv"), index=False
        )
