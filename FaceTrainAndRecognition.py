# Face Recognition with OpenCV
#import OpenCV module
import cv2
#import os module for reading training data directories and paths
import os
#import numpy to convert python lists to numpy arrays as 
#it is needed by OpenCV face recognizers
import numpy as np
import copy
#function to detect face using OpenCV
def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    
    if (len(faces) == 0):
        return None, None
    
    (x, y, w, h) = faces[0]
    print(faces[0])
    return gray[y:y+w, x:x+h], faces[0]
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
# Function to Prepare Data
def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)
    faces = []
    labels = []
    
    #let's go through each directory and read images within it
    for dir_name in dirs:

        label = int(dir_name)
        Image_dir_path = data_folder_path + "/" + dir_name
        #get the images names that are inside the given subject directory
        All_images = os.listdir(Image_dir_path)
        
        #------STEP-3--------
        #go through each image name, read image, 
        #detect face and add face to list of faces
        for image_name in All_images:
            
            #ignore system files like .DS_Store
            if image_name.startswith("."):
                continue;
            image_path = Image_dir_path + "/" + image_name
            temp = cv2.imread(image_path)
            image=copy.copy(temp) 
            face, rect = detect_face(image)
            print(face)
            draw_rectangle(image, rect)
            cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            cv2.waitKey(500)
            #------STEP-4--------
            #for the purpose of this tutorial
            #we will ignore faces that are not detected
            if face is not None:
                #add face to list of faces
                faces.append(face)
                #add label for this face
                labels.append(label)
            
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    
    return faces, labels
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
if(not(os.path.isfile("TrainedAlgorith"))):
    print("Preparing data...")
    faces, labels = prepare_training_data("training-data")
    print("Data prepared")

    #print total faces and labels
    print("Total faces: ", len(faces))
    print("Total labels: ", len(labels))

    
    face_recognizer.train(faces, np.array(labels))
    #function to draw text on give image starting from
    #passed (x, y) coordinates. 
    face_recognizer.save("TrainedAlgorith")
else:
    face_recognizer.read("TrainedAlgorith")
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

#this function recognizes the person in image passed
def detect_face_out(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    
    if (len(faces) == 0):
        return None
    
    (x, y, w, h) = faces[0]
    print (faces)
    return faces

def predict_out(test_img):
    #make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    #detect face from the image
    faces = detect_face_out(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for face in faces:
    #predict the image using our face recognizer 
        print(face)
        (x, y, w, h) = face
        rect=face
        face=gray[y:y+w, x:x+h]
        label, confidence = face_recognizer.predict(face)
        print(confidence)
        #get name of respective label returned by face recognizer
        label_text = str(label)
        #draw a rectangle around face detected
        draw_rectangle(img, rect)
        #draw name of predicted person
        draw_text(img, label_text, rect[0], rect[1]-5)
    return img

print("Predicting images...")
#load test images
list_test=os.listdir("test-data")
for test in list_test:
    test_img=cv2.imread("test-data/"+test)
    predicted_img = predict_out(test_img)
    cv2.imshow("FirstImage", cv2.resize(predicted_img, (600, 800)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# test_img1 = cv2.imread("test-data/test2.jpg")
# test_img2 = cv2.imread("test-data/unknown.jpg")

# #perform a prediction
# predicted_img1 = predict(test_img1)
# predicted_img2 = predict(test_img2)
# print("Prediction complete")

# #display both images
# cv2.imshow("FirstImage", cv2.resize(predicted_img1, (400, 500)))
# cv2.imshow("SecondImage", cv2.resize(predicted_img2, (400, 500)))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
