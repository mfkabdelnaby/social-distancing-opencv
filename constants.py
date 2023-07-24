video_path = "videos/TownCentreXVID.avi"

# load the YOLO model
weightsPath = "yolo-coco/yolov3.weights"
configPath = "yolo-coco/yolov3.cfg"

# Output paths
out_path = "output/social_distance.avi"
out_path_bird = "output/birdview.avi"
out_path_heat = "output/heatmap.avi"


# Initiate input varaibles
maxDisappeared = 7
maxDistance = 7
skipframes = 20
thres = 0.3
minSocialDistance = 20

# YOLO constants
nms_thres = 0.3
thres = 0.3


# Heatmap constants
thres_heat = 2
maxValue = 2

# how frequent you need to update the df (in seconds, 0 = every frame)
place_seconds = 0
user_seconds = 0
