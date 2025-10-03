import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp


# load model
model = load_model("SIGN_MODEL.keras")

SIGN_LABELS_REVERSE={
   
    0:'A',
    1:'B',
    2:'C',
    3:'D',
    4:'E',
    5:'F',
    6:'G',
    7:'H',
    8:'I',
    9:'J',
    10:'K',
    11:'L',
    12:'M',
    13:'N',
    14:'O',
    15:'P',
    16:'Q',
    17:'R',
    18:'S',
    19:'T',
    20:'U',
    21:'V',
    22:'W',
    23:'X',
    24:'Y',
    25:'Z',
   
    
}


# cap =  cv2.VideoCapture(0)

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# while True:
    
#     ret, frame= cap.read()
#     if not ret:
#       print("Error in loading vedio / frame")
#       break
    
#     cv2.rectangle(frame, (15,15), (195,195), (0,165,250), 2)

#     resize = frame[15:195,15:195]
#     blur = cv2.GaussianBlur(resize, (5, 5), 0)
#     Image = cv2.resize(blur,(180,180))
#     black_bg = np.zeros((180, 180, 3), dtype=np.uint8)
#     black_bg[:,:] = Image

#     # Normalize and reshape
#     Image = black_bg / 255.0
   
#     Image = Image.reshape(1,180,180,3)

#     # predict 

#     logits = model.predict(Image)
#     probabilities = tf.nn.softmax(logits[0]).numpy()

#     confidence = np.max(probabilities)
#     if  confidence > 0.95:
       
#         predicted_class = np.argmax(probabilities)
#         letter = SIGN_LABELS_REVERSE[predicted_class]
#         text = f'Prediction: {letter}'
#         box_color = (0,255,0)
        
#     else:
        
#         text = "Waiting for input"
#         box_color = (0,165,255)
    
#     cv2.putText(frame, text, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#     cv2.rectangle(frame, (15,15), (195,195), box_color, 2)
        
    

#     cv2.imshow("HANDS SIGN DETECTION",frame)


#     if cv2.waitKey(1) == ord('q'):
#         break
    

# cap.release()
# cv2.destroyAllWindows()




# MediaPipe hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error in loading video / frame")
            break

        # Flip frame for natural selfie-view
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on original frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract bounding box coordinates from landmarks
                h, w, _ = frame.shape
                x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
                y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]

                x_min, x_max = max(min(x_coords) - 20, 0), min(max(x_coords) + 20, w)
                y_min, y_max = max(min(y_coords) - 20, 0), min(max(y_coords) + 20, h)

                # Crop hand region from frame
                hand_roi = frame[y_min:y_max, x_min:x_max]

                if hand_roi.size == 0:
                    # Skip if ROI is empty
                    continue

                # Preprocess ROI
                hand_roi = cv2.resize(hand_roi, (180, 180))
                hand_roi = hand_roi / 255.0
                input_img = np.expand_dims(hand_roi, axis=0)
 
                # Predict with your sign model
                logits = model.predict(input_img)
                probabilities = tf.nn.softmax(logits[0]).numpy()
                confidence = np.max(probabilities)

                if confidence > 0.95:
                    predicted_class = np.argmax(probabilities)
                    letter = SIGN_LABELS_REVERSE[predicted_class]
                    text = f'Prediction: {letter}'
                    box_color = (0, 255, 0)
                else:
                    text = "Waiting for input"
                    box_color = (0, 165, 255)

                # Draw bounding box and prediction on frame
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)
                cv2.putText(frame, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        else:
            # No hands detected
            cv2.putText(frame, "Show hand sign inside camera", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

        cv2.imshow("Sign Language Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
