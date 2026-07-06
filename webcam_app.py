# webcam_app.py
# ----------------------------------------
# Author : Prakhar Srivastava
# Date : 2026-03-10
# Description : Opens your webcam, captures the region where a user writes a handwritten equation.
# ----------------------------------------


# ===================================
# Importing the Neccessary Libraries
# ------------------------------
# OpenCV : Webcam Capture
# Deque : Double-ended queue used to store recent predictions.
# predict_equation from predict.py : Deep Learning infernence pipeline
# ===================================
from collections import deque

import cv2

from inference.predict import predict_equation, solve_equation

# Start Webcam
cap = cv2.VideoCapture(0)

# Check Webcam Status
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Initialize Variables
frame_counter = 0
prediction_history = deque(maxlen=5)
prediction = ""
result = ""

# Define Region of Interest
ROI_X1, ROI_Y1 = 120, 120
ROI_X2, ROI_Y2 = 520, 320

# Main loop
while True:
    # Capture Frame
    ret, frame = cap.read()
    if not ret:
        break
    # Flip Frame
    frame = cv2.flip(frame, 1)
    # Draw ROI Rectangle
    cv2.rectangle(frame, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (0,255,0), 2)
    # Extract ROI Image
    roi = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]

    # Run Prediction every 5 frames
    if frame_counter % 5 == 0:
        try:
            # Predict Equation
            eq = predict_equation(roi)
            if eq:
                # Add prediction to History
                prediction_history.append(eq)
                # Stabilize Prediction
                prediction = max(
                    set(prediction_history),
                    key=prediction_history.count
                )
                # Evaluate equation with sympy (never eval() on model output —
                # that would execute whatever string the model decodes).
                value, err = solve_equation(prediction)
                result = "" if err else str(value)
        
        # Handle prediction errors
        except Exception:
            prediction = ""
            result = ""

    # Increment Frame Counter
    frame_counter += 1

    # Draw UI Background
    cv2.rectangle(frame, (20,20), (620,100), (0,0,0), -1)

    # Display Predicted Equation
    cv2.putText(
        frame,
        f"Equation: {prediction}",
        (30,60),
        # Display result
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0,255,0),
        2
    )

    # Display instructions
    cv2.putText(
        frame,
        f"Result: {result}",
        (30,90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,255),
        2
    )

    cv2.putText(
        frame,
        "Write equation inside box | ESC to exit",
        (30,130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255,255,255),
        1
    )

    # Show webcam window
    cv2.imshow("AI Handwritten Math Solver", frame)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()