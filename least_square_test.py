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


def least_square(x, y, x_dot):
    # least square fitting
    X = np.vstack([np.ones(len(x)), x, y]).T
    Y = x_dot[:, None]
    V11 = X.T @ X
    V12 = X.T @ Y
    B = np.linalg.inv(V11) @ V12
    b, A = B[:, 0][0], B[:, 0][1:]
    return A.T, b, V11, V12


Ax_pred, bx_pred, _, _ = least_square(x, y, x_dot)
Ay_pred, by_pred, _, _ = least_square(x, y, y_dot)
A_pred, b_pred = np.array([Ax_pred, Ay_pred]), np.array([bx_pred, by_pred])


def least_square_incremental_step(x, y, x_dot, V11, V12):
    # least square fitting
    X = np.vstack([np.ones(len(x)), x, y]).T
    Y = x_dot[:, None]
    V11_this = X.T @ X
    V12_this = X.T @ Y
    V11 = V11_this if V11 is None else V11 + V11_this
    V12 = V12_this if V12 is None else V12 + V12_this
    return V11, V12


def least_square_incremental(x, y, x_dot):
    _, _, V11, V12 = least_square(x[:3], y[:3], x_dot[:3])
    for sample in zip(x[3:], y[3:], x_dot[3:]):
        V11, V12 = least_square_incremental_step(*[np.array([s]) for s in sample], V11, V12)
    B = np.linalg.inv(V11) @ V12
    b, A = B[:, 0][0], B[:, 0][1:]
    return A.T, b


Ax_pred, bx_pred = least_square_incremental(x, y, x_dot)
Ay_pred, by_pred = least_square_incremental(x, y, y_dot)
A_pred, b_pred = np.array([Ax_pred, Ay_pred]), np.array([bx_pred, by_pred])


def manual_least_square_incremental_step(x, y, x_, y_, xyv11, xv12, yv12):
    # least square fitting
    x2 = x * x
    y2 = y * y
    xy = x * y
    xyv11[0] += 1
    xyv11[1] += x
    xyv11[2] += y
    xyv11[3] += x
    xyv11[4] += x2
    xyv11[5] += xy
    xyv11[6] += y
    xyv11[7] += xy
    xyv11[8] += y2
    
    xv12[0] += x_
    xv12[1] += x_ * x
    xv12[2] += x_ * y
    
    yv12[0] += y_
    yv12[1] += y_ * x
    yv12[2] += y_ * y
    
    return xyv11, xv12, yv12

def manual_least_square_incremental(x, y, x_, y_):
    xyv11 = [0] * 9
    xv12 = [0] * 3
    yv12 = [0] * 3
    for sample in zip(x, y, x_, y_):
        xyv11, xv12, yv12 = manual_least_square_incremental_step(*sample, xyv11, xv12, yv12)
    xyv11 = np.array(xyv11).reshape(3, 3)
    xv12 = np.array(xv12).reshape(3, 1)
    yv12 = np.array(yv12).reshape(3, 1)
    xB = np.linalg.inv(xyv11) @ xv12
    yB = np.linalg.inv(xyv11) @ yv12
    xb, xA = xB[:, 0][0], xB[:, 0][1:]
    yb, yA = yB[:, 0][0], yB[:, 0][1:]
    return xA.T, xb, yA.T, yb

Ax_pred, bx_pred, Ay_pred, by_pred = manual_least_square_incremental(x, y, x_dot, y_dot)

plt.show()
