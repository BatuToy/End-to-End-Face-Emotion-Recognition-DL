from tensorflow.keras.models import load_model
import cv2
import numpy as np
from mtcnn import MTCNN

# Load the pre-trained model
model_path = '/path/to/your/model.h5'
model = load_model(model_path)

# Define emotion dictionary
emotion_dict = {0: "Angry", 1: "Happy"}

# Initialize the MTCNN face detector
detector = MTCNN()

# Start the webcam feed
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to grayscale because the model expects grayscale inputs
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    results = detector.detect_faces(frame)
    for result in results:
        x, y, width, height = result['box']
        x2, y2 = x + width, y + height
        
        # Extract the face area
        face = gray_frame[y:y2, x:x2]

        # Resize the face area to 48x48, which is what the model expects
        face = cv2.resize(face, (48, 48))
        face = np.expand_dims(face, axis=-1)  # Add channel dimension
        face = np.expand_dims(face, axis=0)   # Add batch dimension
        face = face / 255.0                   # Normalize pixel values to [0, 1], if your training data was also normalized

        # Predict the emotion on the face
        prediction = model.predict(face)
        max_index = np.argmax(prediction)
        predicted_emotion = emotion_dict[max_index]

        # Draw rectangle around the face and put predicted emotion text
        cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
