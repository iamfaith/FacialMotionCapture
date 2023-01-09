import trimesh
import pyrender
import numpy as np
from pyrender import Mesh, Scene, Viewer, PerspectiveCamera
import cv2
import gltflib

tm = trimesh.load('model_me.glb')
print(tm)

bottle_trimesh = tm.geometry[list(tm.geometry.keys())[0]]
m = pyrender.Mesh.from_trimesh(bottle_trimesh)


gltf = gltflib.GLTF.load('./model_me.glb', load_file_resources=True)
for mesh in gltf.model.meshes:
        print(mesh.extras['targetNames'], mesh.weights, len(mesh.weights), len(mesh.extras['targetNames']))
        mesh.weights = [1] * 52

for mesh in gltf.model.meshes:
        print(mesh.extras['targetNames'], mesh.weights, len(mesh.weights), len(mesh.extras['targetNames']))

cam_node = pyrender.Node(
        camera=pyrender.PerspectiveCamera(yfov=1.0471975511965976, zfar=100, znear=0.0004926449574960696),
        matrix=np.array([[0.9999451448020004, 0.010473495475301227, -0.00011523643331192809, 2.632996104694026e-05], [-0.010456117703794985, 0.9988101239021526, 0.047634084363313206, 0.0021249022974223746], [0.0006139946832835854, -0.047630266460473435, 0.9988648460765015, 0.0058722449231459], [0.0, 0.0, 0.0, 1.0]]),
        scale=np.ones(3),
        translation=np.array([0.004293392833761634, 0.0018634976900897661, 0.004657140988373949]),
        rotation=np.array([0.27059805007309856, 0.27059805007309856, 0.6532814824381883, 0.6532814824381883]),
    )

# cam = PerspectiveCamera(yfov=(np.pi / 3.0))
# cam_pose = np.array([
#     [0.0,  -np.sqrt(2)/2, np.sqrt(2)/2, 0.5],
#     [1.0, 0.0,           0.0,           0.0],
#     [0.0,  np.sqrt(2)/2,  np.sqrt(2)/2, 0.4],
#     [0.0,  0.0,           0.0,          1.0]
# ])

# #==============================================================================
# # Scene creation
# #==============================================================================

# scene = Scene(ambient_light=np.array([0.02, 0.02, 0.02, 1.0]))
# v = Viewer(scene, shadows=True)
# #==============================================================================
# # Using the viewer with a pre-specified camera
# #==============================================================================
# cam_node = scene.add(cam, pose=cam_pose)
# Viewer(scene)


scene = Scene(ambient_light=np.array([1.0, 1.0, 1.0, 1.0]))
# scene.add_node(cam_node)
# scene.main_camera_node = cam_node
#         # CREATE LIGHTS
# for light in pyrender.Viewer._create_raymond_lights(None):
#         scene.add_node(light, parent_node=cam_node)
scene.add(m)

for mesh_idx, mesh in enumerate(scene.meshes):
        if mesh.weights is not None:
                mesh.weights = [1] * 52   
                print('seting')
# renderer = pyrender.OffscreenRenderer(640,480)
# while True:
    
#     color, _ = renderer.render(scene, pyrender.RenderFlags.ALL_SOLID)
#         # RGB -> BGR
#     color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
#     cv2.imshow('memoji', color_bgr)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cv2.destroyAllWindows()  
  
        
# scene.add(m)
Viewer(scene)