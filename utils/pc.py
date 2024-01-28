import numpy as np
from matplotlib import pyplot as plt


def plot_3d_point_cloud(point_cloud, edges=None, show=True, show_axis=True, in_u_sphere=False,
                        marker='.', s=8, alpha=.8, figsize=(5, 5), elev=10,
                        azim=240, axis=None, title=None, x1=None, y1=None, z1=None, 
                        backend='qtagg', *args, **kwargs):
    x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
    if backend:
        print(f'Switching backend from {plt.get_backend()} to {backend}...')
        plt.switch_backend(backend)
    if axis is None: 
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = axis
        fig = axis

    if title is not None:
        plt.title(title)

    if x1 is not None and y1 is not None and z1 is not None:
        ax.scatter(x1, y1, z1, color='r', marker=marker, s=s*3, alpha=1, zorder=2, *args, **kwargs)
        alpha = 0.3

    sc = ax.scatter(x, y, z, marker=marker, s=s, alpha=alpha, zorder=1, *args, **kwargs)
    ax.view_init(elev=elev, azim=azim)

    # TODO: add edges visualization
    if edges is not None:
        us, vs = edges
        for i in range(len(us)):
            u = us[i]
            v = vs[i]
            ax.plot([x[u], x[v]], [y[u], y[v]], [z[u], z[v]], color='r', alpha=0.5)

    if in_u_sphere:
        ax.set_xlim3d(-0.5, 0.5)
        ax.set_ylim3d(-0.5, 0.5)
        ax.set_zlim3d(-0.5, 0.5)
    else:
        # Multiply with 0.7 to squeeze free-space.
        miv = 0.7 * np.min([np.min(x), np.min(y), np.min(z)])
        mav = 0.7 * np.max([np.max(x), np.max(y), np.max(z)])
        ax.set_xlim(miv, mav)
        ax.set_ylim(miv, mav)
        ax.set_zlim(miv, mav)
        plt.tight_layout()

    if not show_axis:
        plt.axis('off')

    if 'c' in kwargs:
        plt.colorbar(sc)

    if show:
        plt.show()

    return fig
