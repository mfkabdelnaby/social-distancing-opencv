import cv2
import imutils
import numpy as np

# video_path = "videos/TownCentreXVID.avi"


def manual_draw(event, x, y, flags, param):
    """
    Lets the user draw a line on the first frame of the video specified by video_path.

    The line drawn by the user is intended to serve as the gate line for detecting entries and exits in the video.

    Parameters:
    video_path (str): The path to the video file.

    Outputs:
    A file named 'line_points.npy' is saved in the current directory. This file contains the start and end points
    of the line drawn by the user. The points are saved as a list of tuples in the format [(x1, y1), (x2, y2)],
    where (x1, y1) are the coordinates of the start point and (x2, y2) are the coordinates of the end point.

    Note:
    The window displaying the video frame will stay open until the user presses the ESC key.
    Make sure to draw the line and press the ESC key to close the window before running any subsequent script
    that uses the line points.
    """

    # Here is where global functions are updated
    global pt1, pt2, StartPoint_clicked, EndPoint_clicked

    # get left mouse is clicked (down)
    if event == cv2.EVENT_LBUTTONDOWN:
        # RESET THE RECTANGLE IF DRAWN
        if StartPoint_clicked == True and EndPoint_clicked == True:
            StartPoint_clicked = False
            EndPoint_clicked = False
            pt1 = (0, 0)
            pt2 = (0, 0)

        if StartPoint_clicked == False:
            pt1 = (x, y)
            StartPoint_clicked = True

        elif EndPoint_clicked == False:
            pt2 = (x, y)
            EndPoint_clicked = True


# GLOBAL VARIABLES
pt1 = (0, 0)  # top left
pt2 = (0, 0)  # bottum right
StartPoint_clicked = False
EndPoint_clicked = False


# CONNECT TO THE CALLBACK
cap = cv2.VideoCapture(video_path)

# Create a named window for connections
cv2.namedWindow("DrawLine")

# Bind draw_rectangle function to mouse cliks
cv2.setMouseCallback("DrawLine", manual_draw)


while True:
    # Capture frame-by-frame
    res, frame = cap.read()

    if not res:
        break

    # Resize and convert the frame into RGB
    frame = imutils.resize(frame, width=500)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create circle on the first point you click
    if StartPoint_clicked:
        cv2.circle(frame, center=pt1, radius=2, color=(0, 255, 255), thickness=-1)

    if StartPoint_clicked and EndPoint_clicked:
        cv2.line(frame, pt1, pt2, (0, 255, 255), 3)

    # Display the resulting frame
    cv2.imshow("DrawLine", frame)

    # save the points of the line
    np.save("line_points.npy", [pt1, pt2])

    # This command let's us quit with the ESC button on a keyboard.
    if cv2.waitKey(1) & 0xFF == 27:
        break


# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
