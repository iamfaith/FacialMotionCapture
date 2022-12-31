import trimesh
import pyrender
import numpy as np
from pyrender import Mesh, Scene, Viewer, PerspectiveCamera
import cv2
import gltflib
import renderer as renderer
from threading import Thread
import time
from enum import IntEnum
from typing import NamedTuple, List
from mediaPipeFace import Calculate_Face_Mocap
class BlendShapes(IntEnum):
    # Left eye blend shapes
    EyeBlinkLeft = 0
    EyeLookDownLeft = 1
    EyeLookInLeft = 2
    EyeLookOutLeft = 3
    EyeLookUpLeft = 4
    EyeSquintLeft = 5
    EyeWideLeft = 6
    # Right eye blend shapes
    EyeBlinkRight = 7
    EyeLookDownRight = 8
    EyeLookInRight = 9
    EyeLookOutRight = 10
    EyeLookUpRight = 11
    EyeSquintRight = 12
    EyeWideRight = 13
    # Jaw blend shapes
    JawForward = 14
    JawLeft = 15
    JawRight = 16
    JawOpen = 17
    # Mouth blend shapes
    MouthClose = 18
    MouthFunnel = 19
    MouthPucker = 20
    MouthLeft = 21
    MouthRight = 22
    MouthSmileLeft = 23
    MouthSmileRight = 24
    MouthFrownLeft = 25
    MouthFrownRight = 26
    MouthDimpleLeft = 27
    MouthDimpleRight = 28
    MouthStretchLeft = 29
    MouthStretchRight = 30
    MouthRollLower = 31
    MouthRollUpper = 32
    MouthShrugLower = 33
    MouthShrugUpper = 34
    MouthPressLeft = 35
    MouthPressRight = 36
    MouthLowerDownLeft = 37
    MouthLowerDownRight = 38
    MouthUpperUpLeft = 39
    MouthUpperUpRight = 40
    # Brow blend shapes
    BrowDownLeft = 41
    BrowDownRight = 42
    BrowInnerUp = 43
    BrowOuterUpLeft = 44
    BrowOuterUpRight = 45
    # Cheek blend shapes
    CheekPuff = 46
    CheekSquintLeft = 47
    CheekSquintRight = 48
    # Nose blend shapes
    NoseSneerLeft = 49
    NoseSneerRight = 50
    TongueOut = 51
    # Treat the head rotation as curves for LiveLink support
    HeadYaw = 52
    HeadPitch = 53
    HeadRoll = 54
    # Treat eye rotation as curves for LiveLink support
    LeftEyeYaw = 55
    LeftEyePitch = 56
    LeftEyeRoll = 57
    RightEyeYaw = 58
    RightEyePitch = 59
    RightEyeRoll = 60


gltf = gltflib.GLTF.load('D:/vscode_workspace/FacialMotionCapture_v2/DigitalHuman/model_me.glb', load_file_resources=True)
scene = pyrender.Scene.from_gltflib_scene(gltf)



for mesh in gltf.model.meshes:
        print(mesh.extras['targetNames'], mesh.weights, len(mesh.weights), len(mesh.extras['targetNames']))
#         mesh.weights = [1] * 52
class Livelink(NamedTuple):

    face_blend_shapes_values: List[float]

# for i in range(52):
#     Livelink.face_blend_shapes_values.append(0.5)

Livelink.face_blend_shapes_values = np.array([1.0, 0.49640795588493347, 0.11688006669282913, 0.0, 0.0, 0.05742459371685982, 0.0, 0.6556288003921509, 0.49774667620658875, 9.538116864860058e-05, 0.0, 0.0, 0.05742422491312027, 0.0, 0.023357238620519638, 0.0052541689947247505, 0.0, 0.0554061233997345, 0.07145385444164276, 0.09527645260095596, 0.36399880051612854, 0.0, 0.021346818655729294, 0.0, 0.0, 0.07755988091230392, 0.06911591440439224, 0.017455056309700012, 0.01804226078093052, 0.04070117697119713, 0.042244505137205124, 0.05893817916512489, 0.01590118370950222, 0.14774569869041443, 0.11772327125072479, 0.06696517020463943, 0.06838994473218918, 0.027602296322584152, 0.02812124229967594, 0.01904251240193844, 0.021279558539390564, 0.09746091067790985, 0.09731905162334442, 0.07517478615045547, 0.0, 0.0, 0.18365933001041412, 0.040523797273635864, 0.04445596784353256, 0.09587085992097855, 0.10142657160758972, 5.96365907767904e-08, 0.09523141384124756, 0.29515162110328674, 0.008784395642578602, -5.559242345043458e-05, 0.30405619740486145, -1.7444133845856413e-05, 0.06813524663448334, 0.3039661943912506, 0.21353380754590034])
# print(len(Livelink.face_blend_shapes_values), '---')


def update_face():
    time.sleep(1)
    print("---", "start!!!!!!!!!!!!")
    
    path = "C:/Users/faith/Downloads/mocap4face-0.5.1/js-example/public/m4f.mp4"
    # path = "C:/Users/faith/Documents/Captura/output.mp4"
    # path = None
    for i in Calculate_Face_Mocap(path, debug=True):
        # for idx, blend in enumerate(i["blendShapes"]):
            # pass
        # print(len(mesh.weights), len(i["blendShapes"]))
        blendValues = i["blendShapes"]
        blendList = []
        for mesh in gltf.model.meshes:
            if 'targetNames' in mesh.extras:
                targetNames = mesh.extras['targetNames']
                for targetName in targetNames:
                    key = targetName[0].upper() + targetName[1:]      
                    val = blendValues[key]              
                    blendList.append(val)
        
        
        # mesh_mappings = [
        #     [
        #         int(BlendShapes[target_name[0].upper() + target_name[1:]])
        #         for target_name in mesh.extras['targetNames']
        #     ] if 'targetNames' in mesh.extras else None
        #     for mesh in gltf.model.meshes
        # ]

        # print(mesh_mappings)
        # break
        
        for mesh_idx, mesh in enumerate(scene.meshes):
                if mesh.weights is not None:
                    # mesh.weights = np.array(i["blendShapes"][:len(mesh.weights)])
                    mesh.weights = np.array(blendList)
                    

# Viewer(scene)
v = renderer.OpenCvRenderer(scene)

t = Thread(target=update_face) #, args=(scene))


t.start()

v.start()

