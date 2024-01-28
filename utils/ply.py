import numpy as np
from plyfile import PlyData 


def load_ply(file_name: str,
             with_faces: bool = False,
             with_color: bool = False) -> tuple:
    """
    Load a .ply file as a numpy array.
    
    :param file_name: str, path to the .ply file
    :param with_faces: bool, whether to load faces
    :param with_color: bool, whether to load color

    :return: points (np.ndarray), faces (Optional[np.ndarray]), colors (Optional[np.ndarray])
    """
    faces, colors = None, None
    ply_data = PlyData.read(file_name)
    points = ply_data['vertex']
    points = np.vstack([points['x'], points['y'], points['z']]).T
    if with_faces:
        faces = np.vstack(ply_data['face']['vertex_indices'])
    if with_color:
        colors = np.vstack([points['red'], points['green'], points['blue']]).T
    return points, faces, colors


def save_ply(points: np.ndarray, 
             file_name: str) -> None:
    """
    Save a numpy array as a .ply file.

    :param points: np.ndarray, points to save
    :param file_name: str, path to the .ply file
    """
    points_len = len(points)
    header = \
        "ply\n" \
        "format binary_little_endian 1.0\n" \
        "element vertex " + str(points_len) + "\n" \
        "property float x\n" \
        "property float y\n" \
        "property float z\n" \
        "end_header\n"
    vertex = np.empty(points_len, dtype=[('vertex', '<f4', (3))])
    vertex['vertex'] = points
    with open(file_name, 'wb') as f:
        f.write(bytes(header, 'utf-8'))
        f.write(vertex.tobytes())
