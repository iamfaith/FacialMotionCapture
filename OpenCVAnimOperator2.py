import bpy
import cv2
import time
import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FaceMeshBlock(nn.Module):
    """This is the main building block for architecture
    which is just residual block with one dw-conv and max-pool/channel pad
    in the second branch if input channels doesn't match output channels"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super(FaceMeshBlock, self).__init__()

        self.stride = stride
        self.channel_pad = out_channels - in_channels

        # TFLite uses slightly different padding than PyTorch 
        # on the depthwise conv layer when the stride is 2.
        if stride == 2:
            self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
            padding = 0
        else:
            padding = (kernel_size - 1) // 2

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
                      kernel_size=kernel_size, stride=stride, padding=padding, 
                      groups=in_channels, bias=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                      kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.act = nn.PReLU(out_channels)

    def forward(self, x):
        if self.stride == 2:
            h = F.pad(x, (0, 2, 0, 2), "constant", 0)
            x = self.max_pool(x)
        else:
            h = x

        if self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)
        
        return self.act(self.convs(h) + x)


class FaceMesh(nn.Module):
    """The FaceMesh face landmark model from MediaPipe.

    Because we won't be training this model, it doesn't need to have
    batchnorm layers. These have already been "folded" into the conv 
    weights by TFLite.

    The conversion to PyTorch is fairly straightforward, but there are 
    some small differences between TFLite and PyTorch in how they handle
    padding on conv layers with stride 2.

    This version works on batches, while the MediaPipe version can only
    handle a single image at a time.
    """
    def __init__(self):
        super(FaceMesh, self).__init__()

        self.num_coords = 468
        self.x_scale = 192.0
        self.y_scale = 192.0
        self.min_score_thresh = 0.75

        self._define_layers()

    def _define_layers(self):
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=0, bias=True),
            nn.PReLU(16),

            FaceMeshBlock(16, 16),
            FaceMeshBlock(16, 16),
            FaceMeshBlock(16, 32, stride=2),
            FaceMeshBlock(32, 32),
            FaceMeshBlock(32, 32),
            FaceMeshBlock(32, 64, stride=2),
            FaceMeshBlock(64, 64),
            FaceMeshBlock(64, 64),
            FaceMeshBlock(64, 128, stride=2),
            FaceMeshBlock(128, 128),
            FaceMeshBlock(128, 128),
            FaceMeshBlock(128, 128, stride=2),
            FaceMeshBlock(128, 128),
            FaceMeshBlock(128, 128),
        )
        
        self.coord_head = nn.Sequential(
            FaceMeshBlock(128, 128, stride=2),
            FaceMeshBlock(128, 128),
            FaceMeshBlock(128, 128),
            nn.Conv2d(128, 32, 1),
            nn.PReLU(32),
            FaceMeshBlock(32, 32),
            nn.Conv2d(32, 1404, 3)
        )
        
        self.conf_head = nn.Sequential(
            FaceMeshBlock(128, 128, stride=2),
            nn.Conv2d(128, 32, 1),
            nn.PReLU(32),
            FaceMeshBlock(32, 32),
            nn.Conv2d(32, 1, 3)
        )
        
    def forward(self, x):
        # TFLite uses slightly different padding on the first conv layer
        # than PyTorch, so do it manually.
        x = nn.ReflectionPad2d((1, 0, 1, 0))(x)
        # x = nn.ConstantPad2d((0, 1, 0, 1), 0)(x)
        b = x.shape[0]      # batch size, needed for reshaping later

        x = self.backbone(x)            # (b, 128, 6, 6)
        
        c = self.conf_head(x)           # (b, 1, 1, 1)
        c = c.view(b, -1)               # (b, 1)
        
        r = self.coord_head(x)          # (b, 1404, 1, 1)
        r = r.reshape(b, -1)            # (b, 1404)
        
        return [r, c]

    def _device(self):
        """Which device (CPU or GPU) is being used by this model?"""
        return self.conf_head[1].weight.device
    
    def load_weights(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()        
    
    def _preprocess(self, x):
        """Converts the image pixels to the range [-1, 1]."""
        return x.float() / 127.5 - 1.0

    def predict_on_image(self, img):
        """Makes a prediction on a single image.

        Arguments:
            img: a NumPy array of shape (H, W, 3) or a PyTorch tensor of
                 shape (3, H, W). The image's height and width should be 
                 128 pixels.

        Returns:
            A tensor with face detections.
        """
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).permute((2, 0, 1))

        return self.predict_on_batch(img.unsqueeze(0))[0]

    def predict_on_batch(self, x):
        """Makes a prediction on a batch of images.

        Arguments:
            x: a NumPy array of shape (b, H, W, 3) or a PyTorch tensor of
               shape (b, 3, H, W). The height and width should be 128 pixels.

        Returns:
            A list containing a tensor of face detections for each image in 
            the batch. If no faces are found for an image, returns a tensor
            of shape (0, 17).

        Each face detection is a PyTorch tensor consisting of 17 numbers:
            - ymin, xmin, ymax, xmax
            - x,y-coordinates for the 6 keypoints
            - confidence score
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).permute((0, 3, 1, 2))

        assert x.shape[1] == 3
        assert x.shape[2] == 192
        assert x.shape[3] == 192

        # 1. Preprocess the images into tensors:
        x = x.to(self._device())
        x = self._preprocess(x)

        # 2. Run the neural network:
        with torch.no_grad():
            out = self.__call__(x)

        # 3. Postprocess the raw predictions:
        detections, confidences = out
        detections[0:-1:3] *= self.x_scale
        detections[1:-1:3] *= self.y_scale

        return detections.view(-1, 3), confidences

#New Add
landmark_points_68 = [162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,71,63,105,66,107,336,
                296,334,293,301,168,197,5,4,75,97,2,326,305,33,160,158,133,153,144,362,385,387,263,373,
                380,61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87]
gap = 30
# D:\blender3.4\blender-launcher.exe --python-use-system-env

class OpenCVAnimOperator(bpy.types.Operator):
    """Operator which runs its self from a timer"""
    bl_idname = "wm.opencv_operator"
    bl_label = "OpenCV Animation Operator"
    
    # Set paths to trained models downloaded above
    face_detect_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    #landmark_model_path = "/home/username/Documents/Vincent/lbfmodel.yaml"  #Linux
    #landmark_model_path = "/Users/username/Downloads/lbfmodel.yaml"         #Mac

    net = FaceMesh().to("cpu")
    net.load_weights("D:/vscode_workspace/FacialMotionCapture_v2/facemesh.pth")
    
    
    cas = cv2.CascadeClassifier(face_detect_path)
    
    _timer = None
    _cap  = None
    stop = False
    
    # Webcam resolution:
    width = 640
    height = 480
    
    # 3D model points. 
    model_points = numpy.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner
                            ], dtype = numpy.float32)
    # Camera internals
    camera_matrix = numpy.array(
                            [[height, 0.0, width/2],
                            [0.0, height, height/2],
                            [0.0, 0.0, 1.0]], dtype = numpy.float32
                            )
                            
    # Keeps a moving average of given length
    def smooth_value(self, name, length, value):
        if not hasattr(self, 'smooth'):
            self.smooth = {}
        if not name in self.smooth:
            self.smooth[name] = numpy.array([value])
        else:
            self.smooth[name] = numpy.insert(arr=self.smooth[name], obj=0, values=value)
            if self.smooth[name].size > length:
                self.smooth[name] = numpy.delete(self.smooth[name], self.smooth[name].size-1, 0)
        sum = 0
        for val in self.smooth[name]:
            sum += val
        return sum / self.smooth[name].size

    # Keeps min and max values, then returns the value in a range 0 - 1
    def get_range(self, name, value):
        if not hasattr(self, 'range'):
            self.range = {}
        if not name in self.range:
            self.range[name] = numpy.array([value, value])
        else:
            self.range[name] = numpy.array([min(value, self.range[name][0]), max(value, self.range[name][1])] )
        val_range = self.range[name][1] - self.range[name][0]
        if val_range != 0:
            return (value - self.range[name][0]) / val_range
        else:
            return 0.0
        
    # The main "loop"
    def modal(self, context, event):

        if (event.type in {'RIGHTMOUSE', 'ESC'}) or self.stop == True:
            self.cancel(context)
            return {'CANCELLED'}

        if event.type == 'TIMER':
            self.init_camera()
            ret, image = self._cap.read()
            if not ret:
                return {'PASS_THROUGH'}

            ##################################################### added by resized 
            # scale_percent = 30 # percent of original size
            # width = int(image.shape[1] * scale_percent / 100)
            # height = int(image.shape[0] * scale_percent / 100)
            # dim = (width, height)
            # image = cv2.resize(image, dim) #, interpolation = cv2.INTER_AREA)
            
            
            #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #gray = cv2.equalizeHist(gray)
            
            # find faces
            faces = self.cas.detectMultiScale(image, 
                scaleFactor=1.05,  
                minNeighbors=3, 
                flags=cv2.CASCADE_SCALE_IMAGE, 
                minSize=(int(self.width/5), int(self.width/5)))
            
            #find biggest face, and only keep it
            if type(faces) is numpy.ndarray and faces.size > 0: 
                biggestFace = numpy.zeros(shape=(1,4))
                for face in faces:
                    if face[2] > biggestFace[0][2]:
                        print(face)
                        biggestFace[0] = face

                for rect in biggestFace:
                    x,y,w,h = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
                    # cv.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 1)
                    # x -= gap
                    # y -= gap
                    
                    # w += gap
                    h += 9 * gap
                    h = int(h)
                    
                    cropped_image = image[y:y+h, x:x+w]
            
                    img = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (192, 192))
                    detections = self.net.predict_on_image(img).cpu().numpy()

                    shape = detections[landmark_points_68, :2]
                    # for x1, y1 in zip(x, y):  
                # find the landmarks.
                # _, landmarks = self.fm.fit(image, faces=biggestFace)
                # for mark in landmarks:
                #     shape = mark[0]
                    
                    #2D image points. If you change the image, you need to change vector
                    image_points = numpy.array([shape[30],     # Nose tip - 31
                                                shape[8],      # Chin - 9
                                                shape[36],     # Left eye left corner - 37
                                                shape[45],     # Right eye right corne - 46
                                                shape[48],     # Left Mouth corner - 49
                                                shape[54]      # Right mouth corner - 55
                                            ], dtype = numpy.float32)
                 
                    dist_coeffs = numpy.zeros((4,1)) # Assuming no lens distortion
                 
                    # determine head rotation
                    if hasattr(self, 'rotation_vector'):
                        (success, self.rotation_vector, self.translation_vector) = cv2.solvePnP(self.model_points, 
                            image_points, self.camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE, 
                            rvec=self.rotation_vector, tvec=self.translation_vector, 
                            useExtrinsicGuess=True)
                    else:
                        (success, self.rotation_vector, self.translation_vector) = cv2.solvePnP(self.model_points, 
                            image_points, self.camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE, 
                            useExtrinsicGuess=False)
                 
                    if not hasattr(self, 'first_angle'):
                        self.first_angle = numpy.copy(self.rotation_vector)
                    
                    # set bone rotation/positions
                    bones = bpy.data.objects["RIG-Vincent"].pose.bones
                    
                    # head rotation 
                    bones["head_fk"].rotation_euler[0] = self.smooth_value("h_x", 5, (self.rotation_vector[0] - self.first_angle[0])) / 1   # Up/Down
                    bones["head_fk"].rotation_euler[2] = self.smooth_value("h_y", 5, -(self.rotation_vector[1] - self.first_angle[1])) / 1.5  # Rotate
                    bones["head_fk"].rotation_euler[1] = self.smooth_value("h_z", 5, (self.rotation_vector[2] - self.first_angle[2])) / 1.3   # Left/Right
                    
                    bones["head_fk"].keyframe_insert(data_path="rotation_euler", index=-1)
                    
                    # mouth position
                    bones["mouth_ctrl"].location[2] = self.smooth_value("m_h", 2, -self.get_range("mouth_height", numpy.linalg.norm(shape[62] - shape[66])) * 0.06 )
                    bones["mouth_ctrl"].location[0] = self.smooth_value("m_w", 2, (self.get_range("mouth_width", numpy.linalg.norm(shape[54] - shape[48])) - 0.5) * -0.04)
                    
                    bones["mouth_ctrl"].keyframe_insert(data_path="location", index=-1)
                    
                    #eyebrows
                    bones["brow_ctrl_L"].location[2] = self.smooth_value("b_l", 3, (self.get_range("brow_left", numpy.linalg.norm(shape[19] - shape[27])) -0.5) * 0.04)
                    bones["brow_ctrl_R"].location[2] = self.smooth_value("b_r", 3, (self.get_range("brow_right", numpy.linalg.norm(shape[24] - shape[27])) -0.5) * 0.04)
                    
                    bones["brow_ctrl_L"].keyframe_insert(data_path="location", index=2)
                    bones["brow_ctrl_R"].keyframe_insert(data_path="location", index=2)
                    
                    # eyelids
                    l_open = self.smooth_value("e_l", 2, self.get_range("l_open", -numpy.linalg.norm(shape[48] - shape[44]))  )
                    r_open = self.smooth_value("e_r", 2, self.get_range("r_open", -numpy.linalg.norm(shape[41] - shape[39]))  )
                    eyes_open = (l_open + r_open) / 2.0 # looks weird if both eyes aren't the same...
                    bones["eyelid_up_ctrl_R"].location[2] =   -eyes_open * 0.025 + 0.005
                    bones["eyelid_low_ctrl_R"].location[2] =  eyes_open * 0.025 - 0.005
                    bones["eyelid_up_ctrl_L"].location[2] =   -eyes_open * 0.025 + 0.005
                    bones["eyelid_low_ctrl_L"].location[2] =  eyes_open * 0.025 - 0.005
                    
                    bones["eyelid_up_ctrl_R"].keyframe_insert(data_path="location", index=2)
                    bones["eyelid_low_ctrl_R"].keyframe_insert(data_path="location", index=2)
                    bones["eyelid_up_ctrl_L"].keyframe_insert(data_path="location", index=2)
                    bones["eyelid_low_ctrl_L"].keyframe_insert(data_path="location", index=2)
            
            ########################## not debug                    
                    # draw face markers
                    for (x, y) in shape:
                        cv2.circle(image, (int(x), int(y)), 2, (0, 255, 255), -1)
            
            # draw detected face
            # for (x,y,w,h) in faces:
            #     cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),1)
            
            # Show camera image in a window                     
            cv2.imshow("Output", image)
            if(cv2.waitKey(1) & 0xFF == ord('q')):
                self.cancel(context)
                return {'CANCELLED'}

        return {'PASS_THROUGH'}
    
    def init_camera(self):
        if self._cap == None:
            self._cap = cv2.VideoCapture(0)
            # self._cap = cv2.VideoCapture("D:/vscode_workspace/FacialMotionCapture_v2/0.mp4")
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            time.sleep(1.0)
    
    def stop_playback(self, scene):
        print(format(scene.frame_current) + " / " + format(scene.frame_end))
        if scene.frame_current == scene.frame_end:
            bpy.ops.screen.animation_cancel(restore_frame=False)
        
    def execute(self, context):
        bpy.app.handlers.frame_change_pre.append(self.stop_playback)

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.01, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        cv2.destroyAllWindows()
        self._cap.release()
        self._cap = None

def register():
    bpy.utils.register_class(OpenCVAnimOperator)

def unregister():
    bpy.utils.unregister_class(OpenCVAnimOperator)

if __name__ == "__main__":
    register()

    # test call
    bpy.ops.wm.opencv_operator()



