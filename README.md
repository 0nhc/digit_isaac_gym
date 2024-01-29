# DIGIT & Isaac Gym
Simulation of DIGIT sensor in Isaac Gym.

## Prerequisites
* Ubuntu 20.04.
* [Isaac Gym](https://developer.nvidia.com/isaac-gym) Installed.

## Try this demo.
```sh
git clone https://github.com/0nhc/digit_isaac_gym.git
cd digit_isaac_gym
conda activate rlgpu # Your python/conda environment with isaac gym installed.
python digit_sim.py
```

## Reference: How to export a .tet file from a .stl file
Reference: [https://forums.developer.nvidia.com/t/softbody-simulation/160731](https://forums.developer.nvidia.com/t/softbody-simulation/160731)
1. Use [fTetWild](https://github.com/wildmeshing/fTetWild) to generate a .mesh file from a .stl file.
```sh
cd fTetWild/build
./FloatTetwild_bin -i ~/gel.stl -o ~/gel.mesh -e 1e-3 # resolution 1mm
```

2. Use [mesh_to_tet.py](https://gist.github.com/gavrielstate/a4b8910787c15fffbd4970c0ba862d60) to generate a .tet file from a .mesh file.</br>

**mesh_to_tet.py**:
```python
"""
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
Tet file generation example for Isaac Gym Soft Body meshes using fTetWild
"""

mesh_file = open("gel.mesh", "r")
tet_output = open("gel.tet", "w")

mesh_lines = list(mesh_file)
mesh_lines = [line.strip('\n') for line in mesh_lines]
vertices_start = mesh_lines.index('Vertices')
num_vertices = mesh_lines[vertices_start + 1]

vertices = mesh_lines[vertices_start + 2: vertices_start + 2 + int(num_vertices)]

tetrahedra_start = mesh_lines.index('Tetrahedra')
num_tetrahedra = mesh_lines[tetrahedra_start + 1]
tetrahedra = mesh_lines[tetrahedra_start + 2 : tetrahedra_start + 2 + int(num_tetrahedra)]

print(num_vertices, num_tetrahedra)

# Write to tet output
for v in vertices:
	tet_output.write("v " + v + "\n")
tet_output.write("\n")
tet_output.write("# " + num_tetrahedra + " tetrahedra\n")
for t in tetrahedra:
	l = t.split(' 0')[0]
	l = l.split(" ")
	l = [str(int(k) - 1) for k in l]
	l_text = ' '.join(l)
	tet_output.write("t " + l_text + "\n")
```

3. new origin of .tet file:
the link origin of the .stl file +- the joint origin that connects the .stl file with the base.