## Defining Function for taking images for training model
import cv2,os
import numpy as np
import subprocess as sb
from os import listdir
from os.path import isfile, join

# Load HAAR face classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load functions
def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    if faces is ():
        return None
    
    # Crop all faces found
    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face
def Take_sample_img(Person_Name):
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # Initialize Webcam
    cap = cv2.VideoCapture(0)
    count = 0
    try:
        # Collect 100 samples of your face from webcam input
        os.makedirs( "./faces/{}".format(Person_Name), 493 )
        while True:

            ret, frame = cap.read()
            if face_extractor(frame) is not None:
                count += 1
                face = cv2.resize(face_extractor(frame), (200, 200))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                # Save file in specified directory with unique name
        
                path = "./faces/{}".format(Person_Name)+"/"
                file_name_path = path + str(count) + '.jpg'.format(Person_Name)
                cv2.imwrite(file_name_path, face)

                # Put count on images and display live count
                cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                cv2.imshow('Face Cropper', face)

            else:
                #print("Face not found")
                pass

            if cv2.waitKey(1) == 13 or count == 100: #13 is the Enter Key
                break
        cap.release()
        cv2.destroyAllWindows()      
        print("Collecting Samples Complete")        
    except Exception as E :
        print ('Directory already  created. \nSample images already takens.{}'.format(E))
        
    
    
## Defining function for training the model
from os import listdir
from os.path import isfile, join
import joblib
def Train_model(Person_Name):
    data_path = './faces/{}'.format(Person_Name)+'/'
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

    # Create arrays for training data and labels
    Training_Data, Labels = [], []

    # Open training images in our datapath
    # Create a numpy array for training data
    for i, files in enumerate(onlyfiles):
        image_path = data_path + onlyfiles[i]
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        Training_Data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(i)

    # Create a numpy array for both training data and labels
    Labels = np.asarray(Labels, dtype=np.int32)

    # Initialize facial recognizer
    # model = cv2.face.createLBPHFaceRecognizer()
    # NOTE: For OpenCV 3.0 use cv2.face.createLBPHFaceRecognizer()
    # pip install opencv-contrib-python
    # model = cv2.createLBPHFaceRecognizer()
    
    Trained_model  = cv2.face_LBPHFaceRecognizer.create()
    # Let's train our model 
    Trained_model.train(np.asarray(Training_Data), np.asarray(Labels))
    print("Model trained sucessefully")

    #joblib.dump( Trained_model , '{}'.format(Person_Name)+'.pk1')
    Trained_model.save("{}".format(Person_Name)+".yml")
## Taking images for 1st Person and Training it
Take_sample_img("user1")
Train_model("user1")
## Taking images for 2nd Person and training it
Take_sample_img("user2")
Train_model("user2")

