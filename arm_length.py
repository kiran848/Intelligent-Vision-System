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
print(f"Calibration Scaling Factor: {scaling_factor:.2f} cm per pixel")

# Function to save the result to a file
def save_measurement_to_file(average_arm_length_cm):
    try:
        with open('measurement.txt', 'a') as file:  # Open in append mode
            file.write(f" Arm Length: {average_arm_length_cm:.2f} cm\n")
        print(f"Average measurement saved to 'measurement.txt': {average_arm_length_cm:.2f} cm")
    except Exception as e:
        print(f"Error saving measurement to file: {e}")

# Function to detect arm length in the video feed
def detect_height_and_arm_length_in_video():
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Error: Camera not accessible!")
        return

    arm_length_history = []  # List to store all measurements for averaging

    print("Starting camera feed... Press 'q' to exit and calculate average.")  # Debug message

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
            # Extract relevant landmarks for arm length
            left_shoulder = landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value]
            left_wrist = landmarks[mpPose.PoseLandmark.LEFT_WRIST.value]
            
            # Calculate pixel distance for arm length
            shoulder_x, shoulder_y = int(left_shoulder.x * w), int(left_shoulder.y * h)
            wrist_x, wrist_y = int(left_wrist.x * w), int(left_wrist.y * h)
            distance_pixels_arm = math.sqrt((shoulder_x - wrist_x) ** 2 + (shoulder_y - wrist_y) ** 2)
            
            # Convert pixel distance to centimeters using the calibrated scaling factor
            arm_length_cm = distance_pixels_arm * scaling_factor
            arm_length_history.append(arm_length_cm)  # Store measurement

            # Debugging output: Print raw distance and converted arm length
            print(f"Raw pixel distance: {distance_pixels_arm:.2f} px")
            print(f"Arm Length: {arm_length_cm:.2f} cm")

            # Draw landmarks and arm length
            cv2.circle(img, (shoulder_x, shoulder_y), 10, (0, 0, 255), -1)  # Shoulder
            cv2.circle(img, (wrist_x, wrist_y), 10, (255, 0, 0), -1)  # Wrist
            cv2.line(img, (shoulder_x, shoulder_y), (wrist_x, wrist_y), (255, 255, 0), 2)
            
            # Display arm length on the video feed
            cv2.putText(img, f"Arm Length: {int(arm_length_cm)} cm", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Show the video feed with landmarks
        cv2.imshow(" Arm Length Detection", img)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Calculate the average arm length
    if arm_length_history:
        average_arm_length_cm = np.mean(arm_length_history)
        print(f"Average Arm Length: {average_arm_length_cm:.2f} cm")
        save_measurement_to_file(average_arm_length_cm)
    else:
        print("No arm length measurements captured.")

    capture.release()
    cv2.destroyAllWindows()

# Run the main function to start video capture
if __name__ == "__main__":
    detect_height_and_arm_length_in_video()
