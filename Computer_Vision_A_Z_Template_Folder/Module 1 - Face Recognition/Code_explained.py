#Face Recognition

#Importing the libraries
import cv2#Is basically opencv

#loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # We load the cascade for the face using class CascadeClassifer called by object face_cascade
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') # We load the cascade for the eyes using class CascadeClassifer called by object eye_cascade

#To reduce computation required, we do a little trick. We do the face cascade on global referential, but the eye cascade on referential of the face.
def detect(gray,frame):            #The input for this function will be images coming from the webcam. But this will work only for phtotos not videos. So the video will be split into infinite photos basically. We know the cascade uses the grayscale image , not original, but it outputs the rectangles on original image, so we take in both grayscale and original image
    faces=face_cascade.detectMultiScale(gray, 1.3, 5) #The arguement for this 'detectMutliScale' method will be the grayscale image and scale factor()which tells by how much size of image will be reduced or by ow much size of filters(like edge,line,rectangular) will be increased) and minimum number of neighbours(We know that for a certain zone to be accepted, it has to have some neighbours which are also accepted)
                                                        #1.3 and 5 are taken because of trial and error
                                                        #By the detect function we get the faces tuples and the coordinates of the Upper left corner of the rectangle that detects the face but also the length and width of these rectangles.
        # x and y are coordinates of upper left corner of rectangle
        #w and h is width and height of rectangle

#Basically the haarcascades detect the eyes and faces,we just get the rectangles from there 

    for (x, y, w, h) in faces:                          #faces contains these tuples
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)        #We draw the rectangles around the face. The arguements are frame(we draw the rectangle around the frame image) and the tuple of coordinates of the upper left corner of the rectangle(which is x and y) and third arguement is tuple of coordnates of lower right corner of rectangle and that we can get by taking x+w and y+h, 4th coordinate is the color(rgb code), 5th arguement is thickness of rectangles.
            
            #So now we use these rectangles to detect the eyes, as we save computational power by detecting for eyes only inside the face rectangle.
            #The first thing we do to detect the eyes is to get 2 region of interests() two because we need 1 for grayscale image and 1 for original color image
            roi_gray = gray[y:y+h, x:x+w]       #The zones of the rectangle
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)       #SImilar to detectMultiScale for faces.
            for (ex, ey, ew, eh) in eyes:                               #For 2 eyes, we get two rectangles.
                                                                        #These ex,ey,ew,eh are the coordinates of upper left corner of eye box, eh and ew are height and width of the rectangles.
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)#Similar to faces
            return frame#We want tor eturn the frame only as we have printed the rectangles of face and eyes on the frame
     
# Doing some Face Recognition with the webcam
video_capture = cv2.VideoCapture(0) # We turn the webcam on. We put 0 if its webcam of the laptop, we put 1 if its an external webcam
                                    #Video_capture is an object which has many methods
#We make a while loop as the webcam should be on and the faces should keep getting deteceted as long as the user doesnt manually switch it off.
while True:
    _, frame = video_capture.read()         #The read method returns 2 elements but we're only interested in the second one, so we put the first one as _ so that we wont get the first element returned by this method
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #cvtColor allows us to do some transformations on some frames. So the transformation which we do is change the the frame to grayscale. The second arguement cv2.COLOR_BGR2GRAY tells to do an average on Blue, green and red to get the right constrast when it is converted to grayscale.
    canvas = detect(gray, frame)                # we apply the detect function, and the frame returned is saved in canvas
    cv2.imshow('Video', canvas)             #imshow displays the video and the canvas(which is the image coming from the webcam with the rectangles on top of it)
    if cv2.waitKey(1) & 0xFF == ord('q'):       #This is to stop the webcam and face detection only if 'q' is typed.
        break#So if q is typed the infinite while loop is broken
video_capture.release()                         #To turn off the webcam
cv2.destroyAllWindows()                         #To destroy the window in which all cv2 images were displayed