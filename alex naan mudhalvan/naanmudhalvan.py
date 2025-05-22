import cv2
import numpy as np

# Load video file (update this path if needed)
cap = cv2.VideoCapture(r"C:\Users\jenis\OneDrive\Desktop\alex naan mudhalvan\input.mp4")

# Read the first frame
ret, first_frame = cap.read()
if not ret:
    print("Cannot read video file.")
    cap.release()
    exit()

# Convert first frame to grayscale
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Find initial feature points to track
prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

# Create a mask image for drawing the tracks
mask = np.zeros_like(first_frame)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert current frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)

    # Check if optical flow was found
    if next_pts is not None and status is not None:
        good_new = next_pts[status == 1]
        good_old = prev_pts[status == 1]

        # Draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            x_new, y_new = new.ravel()
            x_old, y_old = old.ravel()
            mask = cv2.line(mask, (int(x_new), int(y_new)), (int(x_old), int(y_old)), (0, 255, 0), 2)
            frame = cv2.circle(frame, (int(x_new), int(y_new)), 5, (0, 0, 255), -1)

        # Combine mask and frame
        output = cv2.add(frame, mask)
        cv2.imshow('Sports Performance Analysis - Player Movement', output)

        # Update previous frame and points
        prev_gray = gray.copy()
        prev_pts = good_new.reshape(-1, 1, 2)
    else:
        # If tracking fails, re-detect points
        print("Tracking lost. Reinitializing points.")
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

    # Press 'q' to exit
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()