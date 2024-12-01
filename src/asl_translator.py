import cv2
from hand_detection import HandDetector
import pyttsx3
import time

# Global variable to track the time of the last spoken letter
last_spoken_time = 0

def text_to_speech(text):
    global last_spoken_time
    
    # Get the current time
    current_time = time.time()

    # Check if the cooldown period (1 second) has passed since the last spoken letter
    if current_time - last_spoken_time >= 1:
        # Initialize the TTS engine
        engine = pyttsx3.init()

        # Set properties (optional)
        engine.setProperty('rate', 250)  # Speed of speech
        engine.setProperty('volume', 1)  # Volume level (0.0 to 1.0)

        # Convert text to speech
        engine.say(text.lower())

        # Wait for the speech to finish
        engine.runAndWait()

        # Update the last spoken time
        last_spoken_time = current_time

def main():
    # Initialize the hand detector
    detector = HandDetector(model_path=r'C:\Users\Oscar\ASL Live - RowdyHacks\models\model0')

    # Open the camera
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        # Find hands in the frame
        frame_with_hands, bbox = detector.find_hands(frame)

        if bbox:
            # Pass the frame for classification
            predicted_letter, _ = detector.classify_asl_letter(frame_with_hands, bbox)
            if predicted_letter is not None:
                asl_letter_str, _ = detector.decode_predictions(predicted_letter)
                print(f"Predicted ASL Letter: {asl_letter_str}")

                # Convert ASL letter to speech with cooldown
                text_to_speech(asl_letter_str)

        # Display the frame
        cv2.imshow("ASL Translator", frame_with_hands)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
