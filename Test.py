import numpy as np
import cv2 as cv
import time

# OpenCV Facial Capture Test 

# _cap = cv.VideoCapture(0)
# _cap = cv.VideoCapture("C:/Users/faith/Documents/Captura/output.mp4")
# _cap = cv.VideoCapture("a.mp4")
_cap = cv.VideoCapture("0.mp4")
_cap.set(cv.CAP_PROP_FRAME_WIDTH, 512)
_cap.set(cv.CAP_PROP_FRAME_HEIGHT, 512)
_cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
time.sleep(0.5)

# pip install opencv-contrib-python
facemark = cv.face.createFacemarkLBF()
try:
    # Download the trained model lbfmodel.yaml:
    # https://github.com/kurnianggoro/GSOC2017/tree/master/data
    # and update this path to the file:
    facemark.loadModel("D:/vscode_workspace/GSOC2017-master/data/lbfmodel.yaml")
except cv.error:
    print("Model not found")

# cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_alt.xml")
cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_alt2.xml")
if cascade.empty() :
    print("cascade not found")
    exit()

print("Press ESC to stop")
while True:
    ret, frame = _cap.read()
    if not ret:
        break
    ##################################################### added by resized 
    scale_percent = 30 # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    frame = cv.resize(frame, dim) #, interpolation = cv2.INTER_AREA)
             
    faces = cascade.detectMultiScale(frame, 1.05,  6, cv.CASCADE_SCALE_IMAGE, (130, 130))
   
    try:
        print(faces.size)
    except:
        pass 
    #find biggest face, and only keep it
    if(type(faces) is np.ndarray and faces.size > 0):
        biggestFace = np.zeros(shape=(1,4))
        for face in faces:
            if face[2] > biggestFace[0][2]:
                biggestFace[0] = face

        # find landmarks
        ok, landmarks = facemark.fit(frame, faces=biggestFace)
        print("landmarks", len(landmarks)) 
        # draw landmarks
        for marks in landmarks:

            shape = marks[0]
            
            # #2D image points. If you change the image, you need to change vector
            # image_points = numpy.array([shape[30],     # Nose tip - 31
            #                             shape[8],      # Chin - 9
            #                             shape[36],     # Left eye left corner - 37
            #                             shape[45],     # Right eye right corne - 46
            #                             shape[48],     # Left Mouth corner - 49
            #                             shape[54]      # Right mouth corner - 55
            #                         ], dtype = numpy.float32)
                   
            print(len(shape)) 
            for idx, (x, y) in enumerate(shape):
                # if idx not in [30, 8, 36, 45, 48, 54]:
                #     continue
            # for (x, y) in marks[0]:
                try:
                    cv.circle(frame, (int(x), int(y)), 2, (0, 255, 255), -1)  
                except Exception as e:
                    print(e)
                    
                
        # draw detected face
        for (x,y,w,h) in faces:
            cv.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 1)

        for i,(x,y,w,h) in enumerate(faces):
            cv.putText(frame, "Face #{}".format(i), (x - 10, y - 10),
                            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv.imshow("Image Landmarks", frame)
    if(cv.waitKey(1) & 0xFF == ord('q')):
        exit()
# export DISPLAY=$(awk '/nameserver / {print $2; exit}' /etc/resolv.conf 2>/dev/null):0.0
# pip uninstall opencv-python
# pip install opencv-python-headless