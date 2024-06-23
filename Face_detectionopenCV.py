import cv2

video = cv2.VideoCapture(0)
eye_cascade = cv2.CascadeClassifier(r"C:\Users\Arnav\Desktop\haarcascade_lefteye_2splits.xml")
face_cascade = cv2.CascadeClassifier(r"C:\Users\Arnav\Desktop\frontal_face.xml")
face=cv2.imread(r"C:\Users\Arnav\Pictures\arnavpics.jpg")



while True:
    ret, frame = video.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.05, 5)
    eyes = eye_cascade.detectMultiScale(gray, 1.5, 10)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (200, 0, 0), 10)
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 300), 10)    

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

