# import numpy as np
# import cv2
# from collections import deque

# # Default trackbar callback function
# def setValues(x):
#     pass

# # Creating trackbars for color adjustment
# cv2.namedWindow("Color detectors")
# cv2.createTrackbar("Upper Hue", "Color detectors", 153, 180, setValues)
# cv2.createTrackbar("Upper Saturation", "Color detectors", 255, 255, setValues)
# cv2.createTrackbar("Upper Value", "Color detectors", 255, 255, setValues)
# cv2.createTrackbar("Lower Hue", "Color detectors", 64, 180, setValues)
# cv2.createTrackbar("Lower Saturation", "Color detectors", 171, 255, setValues)
# cv2.createTrackbar("Lower Value", "Color detectors", 78, 255, setValues)

# # Initializing deques to handle color points
# bpoints = [deque(maxlen=1024)]
# gpoints = [deque(maxlen=1024)]
# rpoints = [deque(maxlen=1024)]
# ypoints = [deque(maxlen=1024)]

# # Indexes for different color points
# blue_index = 0
# green_index = 0
# red_index = 0
# yellow_index = 0

# # Define kernel for morphological transformations
# kernel = np.ones((5, 5), np.uint8)

# # Color list for drawing (BGR format)
# colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (8, 255, 255)]
# colorIndex = 0

# # Setting up the Paint window
# paintWindow = np.zeros((471, 636, 3)) + 255
# paintWindow = cv2.rectangle(paintWindow, (40, 1), (140, 65), (8, 8, 8), 2)
# paintWindow = cv2.rectangle(paintWindow, (160, 1), (255, 65), colors[0], -1)
# paintWindow = cv2.rectangle(paintWindow, (275, 1), (370, 65), colors[1], -1)
# paintWindow = cv2.rectangle(paintWindow, (390, 1), (485, 65), colors[2], -1)
# paintWindow = cv2.rectangle(paintWindow, (505, 1), (600, 65), colors[3], -1)

# cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
# cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
# cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
# cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
# cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

# cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# # Start capturing video from webcam
# cap = cv2.VideoCapture(0)

# while True:
#     # Read and flip the frame
#     ret, frame = cap.read()
#     frame = cv2.flip(frame, 1)
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     # Get current positions of trackbars
#     u_hue = cv2.getTrackbarPos("Upper Hue", "Color detectors")
#     u_saturation = cv2.getTrackbarPos("Upper Saturation", "Color detectors")
#     u_value = cv2.getTrackbarPos("Upper Value", "Color detectors")
#     l_hue = cv2.getTrackbarPos("Lower Hue", "Color detectors")
#     l_saturation = cv2.getTrackbarPos("Lower Saturation", "Color detectors")
#     l_value = cv2.getTrackbarPos("Lower Value", "Color detectors")

#     # Set HSV ranges
#     Upper_hsv = np.array([u_hue, u_saturation, u_value])
#     Lower_hsv = np.array([l_hue, l_saturation, l_value])

#     # Draw color selection buttons
#     frame = cv2.rectangle(frame, (40, 1), (140, 65), (122, 122, 122), -1)
#     frame = cv2.rectangle(frame, (160, 1), (255, 65), colors[0], -1)
#     frame = cv2.rectangle(frame, (275, 1), (370, 65), colors[1], -1)
#     frame = cv2.rectangle(frame, (390, 1), (485, 65), colors[2], -1)
#     frame = cv2.rectangle(frame, (505, 1), (600, 65), colors[3], -1)

#     cv2.putText(frame, "CLEAR ALL", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
#     cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
#     cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
#     cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
#     cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

#     # Masking and contour detection for pointer
#     Mask = cv2.inRange(hsv, Lower_hsv, Upper_hsv)
#     Mask = cv2.erode(Mask, kernel, iterations=1)
#     Mask = cv2.morphologyEx(Mask, cv2.MORPH_OPEN, kernel)
#     Mask = cv2.dilate(Mask, kernel, iterations=1)

#     # Find contours
#     cnts, _ = cv2.findContours(Mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     center = None

#     if len(cnts) > 0:
#         # Sort and find the largest contour
#         cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
#         ((x, y), radius) = cv2.minEnclosingCircle(cnt)
#         cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)

#         M = cv2.moments(cnt)
#         center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

#         if center[1] <= 65:
#             if 40 <= center[0] <= 140:
#                 bpoints, gpoints, rpoints, ypoints = [deque(maxlen=512)] * 4
#                 blue_index, green_index, red_index, yellow_index = 0, 0, 0, 0
#                 paintWindow[67:, :, :] = 255
#             elif 160 <= center[0] <= 255:
#                 colorIndex = 0  # Blue
#             elif 275 <= center[0] <= 370:
#                 colorIndex = 1  # Green
#             elif 390 <= center[0] <= 485:
#                 colorIndex = 2  # Red
#             elif 505 <= center[0] <= 600:
#                 colorIndex = 3  # Yellow
#         else:
#             if colorIndex == 0:
#                 bpoints[blue_index].appendleft(center)
#             elif colorIndex == 1:
#                 gpoints[green_index].appendleft(center)
#             elif colorIndex == 2:
#                 rpoints[red_index].appendleft(center)
#             elif colorIndex == 3:
#                 ypoints[yellow_index].appendleft(center)
#     # Append the next deques when nothing is detected to avoid index out of range
#     else:
#         bpoints.append(deque(maxlen=512))
#         blue_index += 1

#         gpoints.append(deque(maxlen=512))
#         green_index += 1

#         rpoints.append(deque(maxlen=512))
#         red_index += 1

#         ypoints.append(deque(maxlen=512))
#         yellow_index += 1


#     points = [bpoints, gpoints, rpoints, ypoints]
#     for i, color_points in enumerate(points):
#         for j, deque_points in enumerate(color_points):
#             for k in range(1, len(deque_points)):
#                 if deque_points[k] is None or deque_points[k - 1] is None:
#                     continue
#                 cv2.line(frame, deque_points[k - 1], deque_points[k], colors[i], 2)
#                 cv2.line(paintWindow, deque_points[k - 1], deque_points[k], colors[i], 2)

#     cv2.imshow("Tracking", frame)
#     cv2.imshow("Paint", paintWindow)
#     cv2.imshow("Mask", Mask)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()
import numpy as np
import cv2
from collections import deque

# Default trackbar callback function
def setValues(x):
    pass

# ✅ Color Detector Window
cv2.namedWindow("Color detectors", cv2.WINDOW_NORMAL)
cv2.createTrackbar("Upper Hue", "Color detectors", 153, 180, setValues)
cv2.createTrackbar("Upper Saturation", "Color detectors", 255, 255, setValues)
cv2.createTrackbar("Upper Value", "Color detectors", 255, 255, setValues)
cv2.createTrackbar("Lower Hue", "Color detectors", 64, 180, setValues)
cv2.createTrackbar("Lower Saturation", "Color detectors", 171, 255, setValues)
cv2.createTrackbar("Lower Value", "Color detectors", 78, 255, setValues)

# Drawing buffers
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

blue_index = green_index = red_index = yellow_index = 0
kernel = np.ones((5, 5), np.uint8)

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (8, 255, 255)]
colorIndex = 0

# ✅ Paint window setup
paintWindow = np.zeros((471, 636, 3)) + 255
paintWindow = cv2.rectangle(paintWindow, (40, 1), (140, 65), (8, 8, 8), 2)
paintWindow = cv2.rectangle(paintWindow, (160, 1), (255, 65), colors[0], -1)
paintWindow = cv2.rectangle(paintWindow, (275, 1), (370, 65), colors[1], -1)
paintWindow = cv2.rectangle(paintWindow, (390, 1), (485, 65), colors[2], -1)
paintWindow = cv2.rectangle(paintWindow, (505, 1), (600, 65), colors[3], -1)

cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# ✅ Start capturing from camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera open nahi ho raha bhai.")
    exit()

# ✅ Forcefully create "Tracking" window
cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get HSV values
    u_hue = cv2.getTrackbarPos("Upper Hue", "Color detectors")
    u_saturation = cv2.getTrackbarPos("Upper Saturation", "Color detectors")
    u_value = cv2.getTrackbarPos("Upper Value", "Color detectors")
    l_hue = cv2.getTrackbarPos("Lower Hue", "Color detectors")
    l_saturation = cv2.getTrackbarPos("Lower Saturation", "Color detectors")
    l_value = cv2.getTrackbarPos("Lower Value", "Color detectors")

    Upper_hsv = np.array([u_hue, u_saturation, u_value])
    Lower_hsv = np.array([l_hue, l_saturation, l_value])

    # Draw UI buttons
    frame = cv2.rectangle(frame, (40, 1), (140, 65), (122, 122, 122), -1)
    frame = cv2.rectangle(frame, (160, 1), (255, 65), colors[0], -1)
    frame = cv2.rectangle(frame, (275, 1), (370, 65), colors[1], -1)
    frame = cv2.rectangle(frame, (390, 1), (485, 65), colors[2], -1)
    frame = cv2.rectangle(frame, (505, 1), (600, 65), colors[3], -1)

    cv2.putText(frame, "CLEAR ALL", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Pointer detection
    Mask = cv2.inRange(hsv, Lower_hsv, Upper_hsv)
    Mask = cv2.erode(Mask, kernel, iterations=1)
    Mask = cv2.morphologyEx(Mask, cv2.MORPH_OPEN, kernel)
    Mask = cv2.dilate(Mask, kernel, iterations=1)

    cnts, _ = cv2.findContours(Mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None

    if cnts:
        cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)

        M = cv2.moments(cnt)
        if M["m00"] != 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            if center[1] <= 65:
                if 40 <= center[0] <= 140:
                    bpoints, gpoints, rpoints, ypoints = [deque(maxlen=512)] * 4
                    blue_index = green_index = red_index = yellow_index = 0
                    paintWindow[67:, :, :] = 255
                elif 160 <= center[0] <= 255:
                    colorIndex = 0
                elif 275 <= center[0] <= 370:
                    colorIndex = 1
                elif 390 <= center[0] <= 485:
                    colorIndex = 2
                elif 505 <= center[0] <= 600:
                    colorIndex = 3
            else:
                if colorIndex == 0:
                    bpoints[blue_index].appendleft(center)
                elif colorIndex == 1:
                    gpoints[green_index].appendleft(center)
                elif colorIndex == 2:
                    rpoints[red_index].appendleft(center)
                elif colorIndex == 3:
                    ypoints[yellow_index].appendleft(center)
    else:
        bpoints.append(deque(maxlen=512))
        blue_index += 1
        gpoints.append(deque(maxlen=512))
        green_index += 1
        rpoints.append(deque(maxlen=512))
        red_index += 1
        ypoints.append(deque(maxlen=512))
        yellow_index += 1

    points = [bpoints, gpoints, rpoints, ypoints]
    for i, color_points in enumerate(points):
        for deque_points in color_points:
            for k in range(1, len(deque_points)):
                if deque_points[k] is None or deque_points[k - 1] is None:
                    continue
                cv2.line(frame, deque_points[k - 1], deque_points[k], colors[i], 2)
                cv2.line(paintWindow, deque_points[k - 1], deque_points[k], colors[i], 2)

    # ✅ Show everything
    cv2.imshow("Tracking", frame)
    cv2.imshow("Paint", paintWindow)
    cv2.imshow("Mask", Mask)

    # ✅ Exit with 'q'
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
