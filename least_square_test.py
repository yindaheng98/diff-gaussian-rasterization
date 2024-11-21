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


N = 1000

A_gt = (np.random.rand(2, 2) - 0.5) * 10
b_gt = (np.random.rand(2) - 0.5) * 10
xy = np.random.rand(2, N)
xy_dot = A_gt @ xy + b_gt[:, None]
# xy_dot += np.random.rand(2, N) * 0.1

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


def manual_least_square_incremental_step(px, py, x_, y_, w, xyv11, x_v12, y_v12):
    # least square fitting
    x = w * px
    x2 = x * x
    y = w * py
    y2 = y * y
    xy = w * px * py
    xyv11[0] += w
    xyv11[1] += x
    xyv11[2] += y
    xyv11[3] += x2
    xyv11[4] += xy
    xyv11[5] += y2

    x_v12[0] += x_ * w
    x_v12[1] += x_ * x
    x_v12[2] += x_ * y

    y_v12[0] += y_ * w
    y_v12[1] += y_ * x
    y_v12[2] += y_ * y

    return xyv11, x_v12, y_v12


def manual_least_square_incremental(x, y, x_, y_, w):
    xyv11 = [0] * 6
    x_v12 = [0] * 3
    y_v12 = [0] * 3
    for sample in zip(x, y, x_, y_, w):
        xyv11, x_v12, y_v12 = manual_least_square_incremental_step(*sample, xyv11, x_v12, y_v12)

    v11 = xyv11
    m11, m12, m13 = v11[0], v11[1], v11[2]
    m22, m23 = v11[3], v11[4]
    m33 = v11[5]
    a11 = m33*m22-m23*m23
    a12 = m13*m23-m33*m12
    a13 = m12*m23-m13*m22
    a22 = m33*m11-m13*m13
    a23 = m12*m13-m11*m23
    a33 = m11*m22-m12*m12

    det = m11*a11+m12*a12+m13*a13

    xB, yB = [0] * 3, [0] * 3

    xB[0] = (a11*x_v12[0]+a12*x_v12[1]+a13*x_v12[2]) / det
    xB[1] = (a12*x_v12[0]+a22*x_v12[1]+a23*x_v12[2]) / det
    xB[2] = (a13*x_v12[0]+a23*x_v12[1]+a33*x_v12[2]) / det

    yB[0] = (a11*y_v12[0]+a12*y_v12[1]+a13*y_v12[2]) / det
    yB[1] = (a12*y_v12[0]+a22*y_v12[1]+a23*y_v12[2]) / det
    yB[2] = (a13*y_v12[0]+a23*y_v12[1]+a33*y_v12[2]) / det

    xb, xA = xB[0], xB[1:]
    yb, yA = yB[0], yB[1:]
    return xA, xb, yA, yb


weights = np.ones(N)  # np.random.rand(N)

Ax_pred, bx_pred, Ay_pred, by_pred = manual_least_square_incremental(x, y, x_dot, y_dot, weights)

plt.show()
