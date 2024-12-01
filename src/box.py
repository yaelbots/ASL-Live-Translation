import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import time

class HandDetector:
    def __init__(self, mode=False, max_hands=2, model_complexity=1, detection_con=0.5, tracking_con=0.5, model_path='model1'):
        self.mode = mode
        self.max_hands = max_hands
        self.model_complexity = model_complexity
        self.detection_con = detection_con
        self.tracking_con = tracking_con

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.model_complexity, self.detection_con, self.tracking_con)
        self.mp_draw = mp.solutions.drawing_utils

        self.model = tf.keras.models.load_model(model_path)

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        bbox = []
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                x_min = min(lm.x for lm in hand_landmarks.landmark) * img.shape[1]
                x_max = max(lm.x for lm in hand_landmarks.landmark) * img.shape[1]
                y_min = min(lm.y for lm in hand_landmarks.landmark) * img.shape[0]
                y_max = max(lm.y for lm in hand_landmarks.landmark) * img.shape[0]
                
                # Expand the bounding box by 20%
                width_increase = (x_max - x_min) * 0.80
                height_increase = (y_max - y_min) * 0.40
                x_min = int(max(x_min - width_increase / 2, 0))
                x_max = int(min(x_max + width_increase / 2, img.shape[1]))
                y_min = int(max(y_min - height_increase / 2, 0))
                y_max = int(min(y_max + height_increase / 2, img.shape[0]))
                
                bbox.append((x_min, y_min, x_max, y_max))
        return img, bbox


    def find_position(self, img, hand_no=0, draw=True):
        landmark_list = []
        if self.results.multi_hand_landmarks:
            selected_hand = self.results.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(selected_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmark_list.append((id, cx, cy))
                if draw:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)
        return landmark_list

    def classify_asl_letter(self, img, bbox):
        if not bbox:  # Check if bbox list is empty
            return None, 0  # No hand detected, return None and 0 confidence
        
        # Assuming we only care about the first hand detected
        x_min, y_min, x_max, y_max = bbox[0]
        # Crop the image to the bounding box
        hand_img = img[y_min:y_max, x_min:x_max]

        if hand_img.size == 0:  # Check if the cropped image is empty
            return None, 0  # Return None and 0 confidence if the crop is empty

        preprocessed_img = self.extract_features(hand_img)
        predicted_letter = self.model.predict(np.array([preprocessed_img]))
        return predicted_letter, 1  # Returning a dummy confidence for simplicity

    def extract_features(self, img):
        # Ensure this handles the case where img might be empty or too small after cropping
        if img.size == 0:
            return np.zeros((150, 150, 1))  # Return a blank array if the crop is empty
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized_img = cv2.resize(gray_img, (150, 150))
        normalized_img = resized_img / 255.0
        return normalized_img.reshape((150, 150, 1))


    def decode_predictions(self, predictions):
        predictions = np.squeeze(predictions)
        predicted_index = np.argmax(predictions)
        confidence = np.max(predictions)

        # Mapping based on dataset indices
        mapping = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Delete', 'Nothing', 'Space']
        if predicted_index < len(mapping):
            predicted_letter = mapping[predicted_index]
        else:
            predicted_letter = "Unknown"
        
        return predicted_letter, confidence

def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(model_path='model1')  # Ensure the correct model path is provided

    while True:
        success, img = cap.read()
        if not success:
            break

        img, bbox = detector.find_hands(img)
        # Modify the call to classify_asl_letter to pass bbox instead of landmark_list
        if bbox:
            predicted_letter, _ = detector.classify_asl_letter(img, bbox)
            if predicted_letter is not None:
                asl_letter_str, confidence = detector.decode_predictions(predicted_letter)
                print(f"Predicted ASL Letter: {asl_letter_str} with confidence {confidence * 100:.2f}%")
                # Draw the bounding box and display the prediction
                for x_min, y_min, x_max, y_max in bbox:
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(img, f"{asl_letter_str}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Hand Tracking", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
