import pickle
import numpy as np
import trimesh
import pyrender

def load_variavle(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r

vertices = load_variavle('vertices_env_0.pkl')
print(vertices.shape)

trimesh.util.attach_to_log()
model_trimesh = trimesh.Trimesh(vertices=vertices)
mesh = pyrender.Mesh.from_trimesh(model_trimesh)
scene = pyrender.Scene()
scene.add(mesh)
pyrender.Viewer(scene, use_raymond_lighting=True)