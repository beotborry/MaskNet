import cv2

# Camp Setting
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Load Haar Cascade Pretrained weight
detector = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")


while True:
    # Read image
    ret, frame = capture.read()

    # Flip image horizontally
    frame = cv2.flip(frame, 1)

    # Convert image to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector.detectMultiScale(gray, 1.05, 5)

    # Draw bounding box
    if len(faces):
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

    cv2.imshow("videoframe", frame)
    if cv2.waitKey(1) > 0: break

capture.release()
cv2.destroyAllWindows()