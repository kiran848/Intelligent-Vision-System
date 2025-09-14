import cv2 as cv
import mediapipe as mp
import math
import numpy as np
import time

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def calculate_scale(reference_length_pixels, actual_length_cm):
    return actual_length_cm / reference_length_pixels if reference_length_pixels else 1

def save_measurement_to_file(avg_chest_measurement_cm):
    try:
        with open('measurement.txt', 'a') as file:  # Open in append mode
            file.write(f"Average Chest Measurement: {avg_chest_measurement_cm:.2f} cm\n")
        print(f"Average chest measurement saved to 'measurement.txt': {avg_chest_measurement_cm:.2f} cm")
    except Exception as e:
        print(f"Error saving measurement to file: {e}")

def detect_chest_measurement(reference_length_cm=30):
    capture = cv.VideoCapture(0)

    # Check if the camera opened successfully
    if not capture.isOpened():
        print("Error: Camera could not be opened.")
        return

    scale_factor = None
    measurements = []  # Store the chest measurements over frames

    # Give camera some time to initialize
    time.sleep(2)

    while True:
        isTrue, img = capture.read()
        if not isTrue:
            print("Error: Failed to read from camera.")
            break

        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        result = pose.process(img_rgb)
        h, w, _ = img.shape
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            # Get shoulder points
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            # Calculate a point 15 cm below the shoulders for the chest measurement
            left_shoulder_x, left_shoulder_y = int(left_shoulder.x * w), int(left_shoulder.y * h)
            right_shoulder_x, right_shoulder_y = int(right_shoulder.x * w), int(right_shoulder.y * h)

            chest_left_y = left_shoulder_y + 50  # Rough estimate for height offset in pixels
            chest_right_y = right_shoulder_y + 50

            # Distance in pixels between the two points for the chest
            chest_width_pixels = math.sqrt((left_shoulder_x - right_shoulder_x) ** 2 + 
                                           (chest_left_y - chest_right_y) ** 2)
            # Set scale factor based on a known reference length
            if scale_factor is None:
                scale_factor = calculate_scale(chest_width_pixels, reference_length_cm)

            # Calculate chest circumference approximation
            chest_circumference_cm = chest_width_pixels * scale_factor * 2
            measurements.append(chest_circumference_cm)

            avg_chest_measurement_cm = round(np.mean(measurements), 2)

            # Draw points and display result
            cv.circle(img, (left_shoulder_x, left_shoulder_y), 5, (0, 255, 0), -1)
            cv.circle(img, (right_shoulder_x, right_shoulder_y), 5, (0, 255, 0), -1)
            cv.circle(img, (left_shoulder_x, chest_left_y), 5, (255, 0, 0), -1)
            cv.circle(img, (right_shoulder_x, chest_right_y), 5, (255, 0, 0), -1)
            cv.putText(img, f"Chest: {avg_chest_measurement_cm} cm", (50, 50), 
                       cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv.imshow("Chest Measurement Detection", img)

        # Exit on 'q' key press
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Calculate the final average of all frames
    if measurements:
        overall_avg_measurement = round(np.mean(measurements), 2)
        print(f"Chest Measurement: {overall_avg_measurement:.2f} cm")
        save_measurement_to_file(overall_avg_measurement)
    else:
        print("No measurements were collected.")

    capture.release()
    cv.destroyAllWindows()

def main():
    # Run the measurement in the main thread
    detect_chest_measurement()

if __name__ == "__main__":
    main()
