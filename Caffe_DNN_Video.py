import cv2
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Initialize Detector & load pre-trained weight
prototxt_path = "./caffe_dnn_module/deploy.prototxt.txt"
weights_path = "./caffe_dnn_module/res10_300x300_ssd_iter_140000.caffemodel"

detector = cv2.dnn.readNet(prototxt_path, weights_path)

# Load pre-trained CNN model
CNN = load_model("./checkpoint/MobileNet_1")

# Cam Setting

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

# Threshold for detecting faces
threshold = 0.2

while True:

    # Read Image from Cam
    ret, frame = capture.read()

    # Convert image to blob in Caffe Framework
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    (h, w) = frame.shape[:2]

    # Detect faces
    detector.setInput(blob)
    detections = detector.forward()

    # Initialize array for storing information
    faces = []
    bboxes = []
    preds = []

    for i in range(0, detections.shape[2]):

        # Confidence of bounding box
        confidence = detections[0, 0, i, 2]

        if confidence > threshold:

            # Get minX, minY, maxX, maxY of bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            text =  "{:2f}%".format(confidence * 100)

            # Handle exception cases of negative value
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # Crop the face from frame
            face = frame[startY:endY, startX:endX]

            # Convert to gray scale
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            # Resize & Preprocessing the image to put in the CNN model
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            bboxes.append((startX, startY, endX, endY))

    # Classify the image into 2 classes; with_mask, without_mask
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = CNN.predict(faces, batch_size = 32)


    for (box, pred) in zip(bboxes, preds):
        (startX, startY, endX, endY) = box

        # Get probability of each class
        (withoutMask, withMask) = pred
        print(pred)

        # Labeling & differentiate color
        label = "Mask" if withMask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (255, 0, 0)

        label = "{}: {:.2f}%".format(label, max(withMask, withoutMask) * 100)

        # Print label & bounding box
        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) > 0: break

capture.release()
cv2.destroyAllWindows()