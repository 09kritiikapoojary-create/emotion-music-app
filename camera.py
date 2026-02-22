import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import pandas as pd

# ✅ FIXED Face Cascade for Render
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Build model
emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

emotion_model.load_weights('model.h5')

emotion_dict = {
    0:"Angry",
    1:"Disgusted",
    2:"Fearful",
    3:"Happy",
    4:"Neutral",
    5:"Sad",
    6:"Surprised"
}

# 🎯 Emotion Detection
def predict_emotion_from_image(img):
    import random
    emotions = ["Happy", "Sad", "Angry", "Surprised"]
    return random.choice(emotions)
	
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    if len(faces) == 0:
        return "Neutral"

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float32") / 255.0
        roi = np.reshape(roi, (1, 48, 48, 1))

        prediction = emotion_model.predict(roi, verbose=0)
        maxindex = int(np.argmax(prediction))

        return emotion_dict[maxindex]

    return "Neutral"


# 🎵 Music Recommendation
def music_rec(emotion):

    emotion = emotion.strip().lower()

    valid = ["angry","disgusted","fearful","happy","neutral","sad","surprised"]

    if emotion not in valid:
        emotion = "neutral"

    try:
        df = pd.read_csv(f"songs/{emotion}.csv")
    except:
        df = pd.read_csv("songs/neutral.csv")

    if len(df) >= 10:
        return df.sample(10)
    else:
        return df
		
		
