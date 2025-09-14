import cv2 as cv
import mediapipe as mp
import math
import numpy as np

# Initialize Mediapipe Pose
mpPose = mp.solutions.pose
pose = mpPose.Pose()

def calculate_stable_length(measurements):
    """Calculate the average length from the list of measurements for stability."""
    return round(np.mean(measurements), 2)

def save_measurement_to_file(length):
    """Save the final length measurement to a file."""
    with open('measurement.txt', 'a') as file:
        file.write(f"Final Lower Body Length: {length} cm\n")
    print(f"Measurement saved to 'measurement.txt': {length} cm")

def detect_lower_body_length_in_video():
    capture = cv.VideoCapture(0)  # Open the camera
    
    measurements = []  # List to store measurements for smoothing
    stable_measurement = None  # Variable to store stable length measurement

    while True:
        isTrue, img = capture.read()
        if not isTrue:
            break
        
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        result = pose.process(img_rgb)

        if result.pose_landmarks:
            # Retrieve the landmarks for length measurement
            landmarks = result.pose_landmarks.landmark
            h, w, c = img.shape

            # Use landmarks to measure lower body length
            left_hip = landmarks[mpPose.PoseLandmark.LEFT_HIP.value]
            left_ankle = landmarks[mpPose.PoseLandmark.LEFT_ANKLE.value]

            # Calculate pixel distances
            hip_x, hip_y = int(left_hip.x * w), int(left_hip.y * h)
            ankle_x, ankle_y = int(left_ankle.x * w), int(left_ankle.y * h)
            distance_pixels = math.sqrt((hip_x - ankle_x) ** 2 + (hip_y - ankle_y) ** 2)

            # Convert pixel distance to centimeters (adjust the scale as needed)
            length_cm = distance_pixels * 0.5  # Adjust the scale as needed
            measurements.append(length_cm)

            # Keep the last 17 measurements for stability
            if len(measurements) > 17:
                measurements.pop(0)

            stable_measurement = calculate_stable_length(measurements)

            # Draw black points at hip and ankle
            cv.circle(img, (hip_x, hip_y), 10, (0, 0, 0), -1)  # Hip point
            cv.circle(img, (ankle_x, ankle_y), 10, (0, 0, 0), -1)  # Ankle point

            # Display stable length on the video feed
            cv.putText(img, f"Lower Body Length: {stable_measurement} cm", (50, 150), 
                       cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Show the video feed with landmarks
        cv.imshow("Lower Body Length Detection", img)

        # Check for 'q' key press to exit
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Save the final stable measurement when exiting
    if measurements:
        final_measurement = calculate_stable_length(measurements)
        save_measurement_to_file(final_measurement)

    capture.release()
    cv.destroyAllWindows()

def main(mode='video'):
    if mode == 'video':
        # Run the lower body length detection from video
        detect_lower_body_length_in_video()

if __name__ == "__main__":
    main(mode='video')  # Run the program to detect lower body length in real-time and save to a file
