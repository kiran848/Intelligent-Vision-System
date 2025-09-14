import cv2
import mediapipe as mp
import math
import numpy as np

# Initialize Mediapipe Pose
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# Calibrate the scaling factor using a known reference object (in cm)
known_object_size_cm = 30  # Example: Known size of reference object in cm (e.g., ruler)
known_object_pixel_size = 100  # Measured pixel size of the reference object in the image

# Calculate the scaling factor for converting pixels to cm
scaling_factor = known_object_size_cm / known_object_pixel_size
print(f"Calibration Scaling Factor: {scaling_factor} cm per pixel")

# Function to save the overall average measurement to a file
def save_overall_average_to_file(overall_average):
    with open('measurement.txt', 'a') as file:
        file.write(f"Shoulder Width: {overall_average:.2f} cm\n")
    print(f"Overall average saved to 'measurement.txt': {overall_average:.2f} cm")

# Function to detect shoulder distance in the video feed
def detect_shoulder_distance_in_video():
    capture = cv2.VideoCapture(0)  # Capture video from the default camera
    if not capture.isOpened():
        print("Error: Camera not accessible!")
        return
    
    all_measurements = []  # Store all measurements for averaging
    
    print("Starting camera feed...")  # Debug message
    while True:
        isTrue, img = capture.read()
        if not isTrue:
            print("Failed to capture image.")
            break

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = pose.process(img_rgb)
        
        # Check if landmarks are detected
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            h, w, c = img.shape
            left_shoulder = landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER.value]

            # Calculate pixel distance between shoulders
            left_shoulder_x, left_shoulder_y = int(left_shoulder.x * w), int(left_shoulder.y * h)
            right_shoulder_x, right_shoulder_y = int(right_shoulder.x * w), int(right_shoulder.y * h)
            distance_pixels_shoulder = math.sqrt((left_shoulder_x - right_shoulder_x) ** 2 + 
                                                 (left_shoulder_y - right_shoulder_y) ** 2)
            shoulder_width_cm = distance_pixels_shoulder * scaling_factor

            # Add measurement to the list
            all_measurements.append(shoulder_width_cm)

            # Draw landmarks and shoulder line
            cv2.circle(img, (left_shoulder_x, left_shoulder_y), 10, (0, 255, 0), -1)
            cv2.circle(img, (right_shoulder_x, right_shoulder_y), 10, (0, 255, 255), -1)
            cv2.line(img, (left_shoulder_x, left_shoulder_y), 
                     (right_shoulder_x, right_shoulder_y), (255, 0, 255), 2)
            cv2.putText(img, f"Shoulder Width: {int(shoulder_width_cm)} cm", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Show the video feed
        cv2.imshow("Shoulder Measurement", img)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting program.")
            break

    capture.release()
    cv2.destroyAllWindows()

    # Calculate and save the overall average
    if all_measurements:
        overall_average = np.mean(all_measurements)
        save_overall_average_to_file(overall_average)
    else:
        print("No measurements were captured to calculate the average.")

# Run the main function
if __name__ == "__main__":
    detect_shoulder_distance_in_video()
