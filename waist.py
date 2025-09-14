import cv2
import mediapipe as mp
import math
import numpy as np

# Initialize Mediapipe Pose
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# Calibration: Known reference object size (in cm) and its pixel size
known_object_size_cm = 80  # Example: Known size of reference object in cm (e.g., ruler)
known_object_pixel_size = 100  # Measured pixel size of the reference object in the image

# Calculate the scaling factor for converting pixels to cm
scaling_factor = known_object_size_cm / known_object_pixel_size
print(f"Calibration Scaling Factor: {scaling_factor} cm per pixel")

# Function to smooth values for more stable output
def smooth_value(value, history, window_size=17):
    history.append(value)
    if len(history) > window_size:
        history.pop(0)
    return np.mean(history)

# Function to save final measurement to a text file
def save_measurement_to_file(average_measurement):
    with open("measurement.txt", "a") as file:
        file.write(f"Waist Circumference: {average_measurement:.2f} cm\n")
    print(f"Measurement saved to 'measurement.txt': {average_measurement:.2f} cm")

# Function to detect waist circumference in the video feed
def detect_waist_circumference_in_video():
    capture = cv2.VideoCapture(0)

    if not capture.isOpened():
        print("Error: Camera not accessible!")
        return

    waist_width_history = []  # Store the waist measurements for smoothing

    while True:
        isTrue, img = capture.read()
        if not isTrue:
            print("Failed to capture image.")
            break

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = pose.process(img_rgb)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            h, w, c = img.shape

            # Extract relevant landmarks for waist measurement (left and right hips)
            left_hip = landmarks[mpPose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[mpPose.PoseLandmark.RIGHT_HIP.value]

            # Calculate pixel distance between the left and right hips (waist measurement)
            left_hip_x, left_hip_y = int(left_hip.x * w), int(left_hip.y * h)
            right_hip_x, right_hip_y = int(right_hip.x * w), int(right_hip.y * h)
            distance_pixels_waist = math.sqrt((left_hip_x - right_hip_x) ** 2 + (left_hip_y - right_hip_y) ** 2)

            # Convert pixel distance to centimeters using the calibrated scaling factor
            waist_circumference_cm = distance_pixels_waist * scaling_factor

            # Smooth the waist circumference value by averaging over the last 17 frames
            waist_circumference_smoothed = smooth_value(waist_circumference_cm, waist_width_history, window_size=17)

            # Debugging output: Print raw distance and converted waist circumference
            print(f"Raw pixel distance (Waist): {distance_pixels_waist:.2f} px")
            print(f"Waist Circumference: {waist_circumference_cm:.2f} cm (Raw), {waist_circumference_smoothed:.2f} cm (Smoothed)")

            # Draw landmarks and waist circumference
            cv2.circle(img, (left_hip_x, left_hip_y), 10, (0, 255, 0), -1)  # Left Hip
            cv2.circle(img, (right_hip_x, right_hip_y), 10, (0, 0, 255), -1)  # Right Hip
            cv2.line(img, (left_hip_x, left_hip_y), (right_hip_x, right_hip_y), (255, 255, 0), 2)  # Waist Line

            # Display smoothed waist circumference on the video feed
            cv2.putText(img, f"Waist Circumference: {int(waist_circumference_smoothed)} cm", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Show the video feed with landmarks
        cv2.imshow("Waist Measurement", img)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Calculate and display the average of the last 17 frames when 'q' is pressed
    if waist_width_history:
        final_average = np.mean(waist_width_history)
        print(f"Final Average Waist Circumference: {final_average:.2f} cm")
        save_measurement_to_file(final_average)

    capture.release()
    cv2.destroyAllWindows()

# Run the main function to start video capture
if __name__ == "__main__":
    detect_waist_circumference_in_video()
