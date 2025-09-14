import cv2 as cv
import mediapipe as mp
import math
import pyttsx3
import threading
import numpy as np

# Initialize Mediapipe Pose
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# Initialize Text-to-Speech
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

def calculate_stable_height(measurements):
    """Calculate the average height from the list of measurements for stability."""
    return round(np.mean(measurements), 2)

def save_measurement_to_file(height):
    """Save the final height measurement to a file."""
    with open('measurement.txt', 'a') as file:
        file.write(f"Final Height: {height} cm\n")
    print(f"Measurement saved to 'measurement.txt': {height} cm")

def detect_height_in_video(stop_event):
    capture = cv.VideoCapture(0)
    
    if not capture.isOpened():
        print("Error: Camera not accessible.")
        return

    measurements = []  # List to store measurements for smoothing
    stable_measurement = None  # Variable to store stable height measurement
    frame_count = 0  # To track how many frames we've processed

    while True:
        isTrue, img = capture.read()
        if not isTrue:
            print("Error: Unable to read from camera.")
            break  # Break if the video capture fails

        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        result = pose.process(img_rgb)

        # Check if landmarks are detected
        if result.pose_landmarks:
            # Retrieve the landmarks for height measurement
            landmarks = result.pose_landmarks.landmark
            h, w, c = img.shape

            # Use landmarks to measure height
            nose = landmarks[mpPose.PoseLandmark.NOSE.value]
            left_ankle = landmarks[mpPose.PoseLandmark.LEFT_ANKLE.value]

            # Calculate pixel distances
            nose_x, nose_y = int(nose.x * w), int(nose.y * h)
            ankle_x, ankle_y = int(left_ankle.x * w), int(left_ankle.y * h)
            distance_pixels = math.sqrt((nose_x - ankle_x) ** 2 + (nose_y - ankle_y) ** 2)

            # Convert pixel distance to centimeters (adjust the scale as needed)
            height_cm = distance_pixels * 0.5  # Adjust the scale as needed
            measurements.append(height_cm)

            # Average the measurements after capturing enough frames (20)
            if len(measurements) > 20:
                measurements.pop(0)
            
            stable_measurement = calculate_stable_height(measurements)

            # Draw landmarks: Head and Toe in Black, Other Landmarks in Green
            for i in range(len(landmarks)):
                lm = landmarks[i]
                x, y = int(lm.x * w), int(lm.y * h)

                if i == mpPose.PoseLandmark.NOSE.value or i == mpPose.PoseLandmark.LEFT_ANKLE.value:
                    # Draw head and toe in black
                    cv.circle(img, (x, y), 5, (0, 0, 0), -1)
                else:
                    # Draw other landmarks in green
                    cv.circle(img, (x, y), 5, (0, 255, 0), -1)

            # Display stable height on the video feed
            cv.putText(img, f"Height: {stable_measurement} cm", (50, 150), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        else:
            print("No pose landmarks detected in this frame.")

        # Show the video feed with landmarks
        cv.imshow("Height Detection", img)

        # Check if 'q' key is pressed to exit
        if cv.waitKey(1) & 0xFF == ord('q'):
            print(f"Stable Measurement: {stable_measurement} cm")
            save_measurement_to_file(stable_measurement)  # Save the measurement to the file
            print("Exiting...")
            break

    capture.release()
    cv.destroyAllWindows()

def main(mode='video'):
    if mode == 'video':
        # Create a stop event that can be set to signal the video thread to stop
        stop_event = threading.Event()

        # Start video capture in a separate thread to avoid blocking speech
        video_thread = threading.Thread(target=detect_height_in_video, args=(stop_event,))
        video_thread.start()

        # Start speaking while the video is running
        speak("Please come into the frame and stand at least 180 centimeters away from the camera.")
        speak("I am about to measure.")
        speak("Please stand still for a moment.")

        # Wait for the video thread to finish when 'q' is pressed
        video_thread.join()

    else:
        print("Invalid mode. Use 'video' for live capture.")

if __name__ == "__main__":
    main(mode='video')
