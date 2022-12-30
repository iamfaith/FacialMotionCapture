import numpy as np
import cv2 as cv
import time
import matplotlib.pyplot as plt
# OpenCV Facial Capture Test 

# _cap = cv.VideoCapture(0)
# _cap = cv.VideoCapture("C:/Users/faith/Documents/Captura/output.mp4")
# _cap = cv.VideoCapture("a.mp4")
_cap = cv.VideoCapture("0.mp4")
_cap.set(cv.CAP_PROP_FRAME_WIDTH, 512)
_cap.set(cv.CAP_PROP_FRAME_HEIGHT, 512)
_cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
time.sleep(0.5)


from facemesh import FaceMesh

net = FaceMesh().to("cpu")
net.load_weights("D:/vscode_workspace/FacialMotionCapture_v2/facemesh.pth")

# cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_alt.xml")
cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_alt2.xml")
if cascade.empty() :
    print("cascade not found")
    exit()
    
#New Add
landmark_points_68 = [162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,71,63,105,66,107,336,
                296,334,293,301,168,197,5,4,75,97,2,326,305,33,160,158,133,153,144,362,385,387,263,373,
                380,61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87]

gap = 50
while True:
    ret, frame = _cap.read()
    if not ret:
        break
    ##################################################### added by resized 
    # scale_percent = 100 # percent of original size
    # width = int(frame.shape[1] * scale_percent / 100)
    # height = int(frame.shape[0] * scale_percent / 100)
    # dim = (width, height)
    # frame = cv.resize(frame, dim) #, interpolation = cv2.INTER_AREA)


    faces = cascade.detectMultiScale(frame, 1.05,  6, cv.CASCADE_SCALE_IMAGE, (130, 130))
   
    # try:
    #     print(faces.size)
    # except:
    #     pass 
    #find biggest face, and only keep it
    if(type(faces) is np.ndarray and faces.size > 0):
        biggestFace = np.zeros(shape=(1,4))
        for face in faces:
            if face[2] > biggestFace[0][2]:
                biggestFace[0] = face
                
        # print(biggestFace[0])
        for rect in biggestFace:
            x,y,w,h = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
            # cv.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 1)
            # x -= gap
            # y -= gap
            
            # w += gap
            h += 9 * gap
            h = int(h)
            # print(frame.shape, y, y+h, x, x+w)
            cropped_image = frame[y:y+h, x:x+w]
    
            img = cv.cvtColor(cropped_image, cv.COLOR_BGR2RGB)
            img = cv.resize(img, (192, 192))
            detections = net.predict_on_image(img).numpy()



            x, y = detections[landmark_points_68, 0], detections[landmark_points_68, 1]  
            # plt.imshow(img, zorder=1)
            # plt.scatter(x, y, zorder=2, s=1.0)
            # # plt.show()
            # plt.savefig("ret.png")

            shape = detections[landmark_points_68, :2]
            for x1, y1 in zip(x, y):  
                try:
                    cv.circle(img, (int(x1), int(y1)), 2, (0, 255, 255), -1)  
                except Exception as e:
                    print(e)
            cv.imshow("Image Landmarks", img)
    if(cv.waitKey(1) & 0xFF == ord('q')):
        exit()
# export DISPLAY=$(awk '/nameserver / {print $2; exit}' /etc/resolv.conf 2>/dev/null):0.0
# pip uninstall opencv-python
# pip install opencv-python-headless