## Defining function for predicting the face and giving required output
import cv2
import numpy as np
def predict_face(person_one, person_two):
    rec1=cv2.face.LBPHFaceRecognizer_create()
    rec2=cv2.face.LBPHFaceRecognizer_create()
    rec1.read( "{}".format(person_one)+".yml")
    rec2.read( "{}".format(person_two)+".yml")
    

    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    def face_detector(img, size=0.5):

        # Convert image to grayscale
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        if faces is ():
            return img, []


        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
            roi = img[y:y+h, x:x+w]
            roi = cv2.resize(roi, (200, 200))
        return img, roi


    # Open Webcam
    cap = cv2.VideoCapture(0)
    
    run_once = 0
    run_onc = 0
   
    while True:

        ret, frame = cap.read()

        image, face = face_detector(frame)

        try:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            # Pass face to prediction model
            # "results" comprises of a tuple containing the label and the confidence value

            results1 = rec1.predict(face)
            
            results2 = rec2.predict(face)

            if results1[1] < 500 :
                confidence1 = int( 100 * (1 - (results1[1])/400) )
                confidence2 = int( 100 * (1 - (results2[1])/400) )
                display_string1 = str(confidence1) + '% Confident it is {}'.format(person_one)
                display_string2 = str(confidence2) + '% Confident it is {}'.format(person_two)
            #if results2[1] < 500:
                #confidence2 = int( 100 * (1 - (results2[1])/400) )
                #display_string2 = str(confidence) + '% Confident it is User2'

            

            if confidence1 > 90 :
                cv2.putText(image, display_string1, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255,120,150), 2)
                cv2.putText(image, "Hey {}".format(person_one)+", How r you!", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                cv2.imshow('Face Recognition', image )
                #run_once = 0
                #while 1:
                if run_once == 0:
                    send_mail()
                    whatsapp()
                    run_once = 1
        
            elif confidence2 > 90:
                
                cv2.putText(image, display_string2, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255,120,150), 2)
                cv2.putText(image, "Hey {}".format(person_two)+", How r you!", (300, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                cv2.imshow('Face Recognition', image )
                #run_onc = 0
                #while 1:
                if run_onc == 0:
                    aws_setup()
                    run_onc = 1
                
            else:

                cv2.putText(image, "I dont know, who r u", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
                cv2.imshow('Face Recognition', image )

        except Exception as E:
            print(E)
            cv2.putText(image, "No Face Found", (220, 120) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv2.putText(image, "looking for face", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv2.imshow('Face Recognition', image )
            pass

        if cv2.waitKey(1) == 13: #13 is the Enter Key
            break

    cap.release()
    cv2.destroyAllWindows()     
    
## Asking model to analyze the face and do the required tasks
predict_face("Rahul","Vidyut")
