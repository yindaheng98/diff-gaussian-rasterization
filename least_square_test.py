import numpy as np
from matplotlib import pyplot as plt


def least_square_np(x, y, x_dot, y_dot):
    """
    least square fitting for a set of points
    x,y: coordinates of the points before transformation
    x_dot,y_dot: coordinates of the points after transformation
    return: A, b: transformation matrix
    """
    # least square fitting
    A = np.vstack([x, y, np.ones(len(x))]).T
    kb_x = np.linalg.lstsq(A, np.array(x_dot), rcond=None)[0]
    kb_y = np.linalg.lstsq(A, np.array(y_dot), rcond=None)[0]
    kb = np.vstack([kb_x, kb_y])
    return kb[:, :2], kb[:, 2]

N = 500

A_gt = (np.random.rand(2, 2) - 0.5) * 10
b_gt = (np.random.rand(2) - 0.5) * 10
xy = np.random.rand(2, N)
xy_dot = A_gt @ xy + b_gt[:, None]
xy_dot += np.random.rand(2, N) * 0.1

x = xy[0, :]
y = xy[1, :]
x_dot = xy_dot[0, :]
y_dot = xy_dot[1, :]
plt.scatter(x, y, 2)
plt.scatter(x_dot, y_dot, 2)
xy_id = A_gt @ np.identity(2) + b_gt[:, None]
plt.plot([b_gt[0], xy_id[0, 0]], [b_gt[1], xy_id[1, 0]], color='black')
plt.plot([b_gt[0], xy_id[0, 1]], [b_gt[1], xy_id[1, 1]], color='black')

A_pred, b_pred = least_square_np(x, y, x_dot, y_dot)
xy_dot_pred = A_pred @ xy + b_pred[:, None]
x_dot_pred = xy_dot_pred[0, :]
y_dot_pred = xy_dot_pred[1, :]
plt.scatter(x_dot_pred, y_dot_pred, 2)
xy_id_pred = A_pred @ np.identity(2) + b_pred[:, None]
plt.plot([b_pred[0], xy_id_pred[0, 0]], [b_pred[1], xy_id_pred[1, 0]], color='red')
plt.plot([b_pred[0], xy_id_pred[0, 1]], [b_pred[1], xy_id_pred[1, 1]], color='red')

plt.show()
