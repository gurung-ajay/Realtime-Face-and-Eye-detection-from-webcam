import cv2

face_model = cv2.CascadeClassifier('C:/Users/97798/Desktop/Python with AI/AI course/Computer Vision/Models/haarcascade_frontalface_default.xml')
eye_model = cv2.CascadeClassifier('C:/Users/97798/Desktop/Python with AI/AI course/Computer Vision/Models/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

if not cap.isOpened:
    exit()

while True:
    ret, frame = cap.read()
    
    if frame is None:
        break
    
    # detect face and draw rectangle
    faces = face_model.detectMultiScale(frame)
    for x,y,w,h in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 5)
        
    # detect eyes and draw rectangle
    eyes = eye_model.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=4)
    for x,y,w,h in eyes:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 10)
    
    cv2.imshow('Face Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('e'):
        exit()
        
cap.release()
cv2.destroyAllWindows()