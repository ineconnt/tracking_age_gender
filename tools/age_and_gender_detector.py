import cv2
import dlib
import numpy as np

from wide_resnet import WideResNet

weight = "network/models/age_and_gender.hdf5"
detector = dlib.get_frontal_face_detector()

model = WideResNet(64, depth=16, k=8)()
model.load_weights(weight)

def age_and_gender(image):
    height, width, depth = np.shape(image)

    detected_faces = detector(image, 1)
    resized_faces = np.empty((len(detected_faces), 64, 64, 3))

    for iterator, detected_face in enumerate(detected_faces):
        x1, y1 = detected_face.left(), detected_face.top()
        x2, y2 = detected_face.right() + 1, detected_face.bottom() + 1

        h1, w1 = detected_face.height(), detected_face.width()

        x1, y1 = max(int(x1 - 0.4 * w1), 0), max(int(y1 - 0.4 * h1), 0)
        x2, y2 = min(int(x2 + 0.4 * w1), width - 1), min(int(y2 + 0.4 * h1), height - 1)

        resized_faces[iterator, :, :, :] = cv2.resize(image[y1:y2 + 1, x1:x2 + 1, :], (64, 64))

    if len(detected_faces) > 0:
        predicted_gender = model.predict(resized_faces)[0]
        predicted_age = model.predict(resized_faces)[1].dot(np.arange(0, 101).reshape(101, 1)).flatten()

        for iterator, detected_face in enumerate(detected_faces):
            predicted_age = round(predicted_age[iterator], 3)
            predicted_gender = 'Female' if predicted_gender[iterator][0] > 0.5 else 'Male'
    else:
        predicted_gender = None
        predicted_age = None

    return predicted_age, predicted_gender

