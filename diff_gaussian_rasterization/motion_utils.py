import math
import torch


def compute_Jacobian(mean, fovx, fovy, width, height, view_matrix):
    '''Compute the Jacobian matrix: diff-gaussian-rasterization/cuda_rasterizer/backward.cu#L191'''
    t = view_matrix.T[:3, :3] @ mean.T + view_matrix.T[:3, 3, None]
    tan_fovx = math.tan(fovx * 0.5)
    tan_fovy = math.tan(fovy * 0.5)
    focal_x = width / (2.0 * tan_fovx)
    focal_y = height / (2.0 * tan_fovy)
    limx = 1.3 * tan_fovx
    limy = 1.3 * tan_fovy
    txtz = t[0] / t[2]
    tytz = t[1] / t[2]
    t[0] = txtz.clamp(-limx, limx) * t[2]
    t[1] = tytz.clamp(-limy, limy) * t[2]
    J = torch.zeros((mean.shape[0], 2, 3), device=mean.device)
    J[:, 0, 0] = focal_x / t[2]
    J[:, 0, 2] = -focal_x * t[0] / (t[2] ** 2)
    J[:, 1, 1] = focal_y / t[2]
    J[:, 1, 2] = -focal_y * t[1] / (t[2] ** 2)
    return J


def compute_T(J, view_matrix):
    '''Compute the T matrix: diff-gaussian-rasterization/cuda_rasterizer/backward.cu#L205'''
    T = J @ view_matrix.T[:3, :3]
    return T


def compute_cov3D_equations(T, cov2D):
    '''Compute the equations for solving the 3D covariance matrix: diff-gaussian-rasterization/cuda_rasterizer/backward.cu#L375'''
    X = torch.zeros((T.shape[0], 3, 6), device=T.device)
    # 1st row for x
    X[..., 0, 0] = T[..., 0, 0] ** 2
    X[..., 0, 1] = 2 * T[..., 0, 1] * T[..., 0, 0]
    X[..., 0, 2] = 2 * T[..., 0, 2] * T[..., 0, 0]
    X[..., 0, 3] = T[..., 0, 1] ** 2
    X[..., 0, 4] = 2 * T[..., 0, 1] * T[..., 0, 2]
    X[..., 0, 5] = T[..., 0, 2] ** 2
    # 2nd row for y
    X[..., 1, 0] = T[..., 1, 0] * T[..., 0, 0]
    X[..., 1, 1] = T[..., 1, 1] * T[..., 0, 0] + T[..., 1, 0] * T[..., 0, 1]
    X[..., 1, 2] = T[..., 1, 2] * T[..., 0, 0] + T[..., 1, 0] * T[..., 0, 2]
    X[..., 1, 3] = T[..., 1, 1] * T[..., 0, 1]
    X[..., 1, 4] = T[..., 1, 1] * T[..., 0, 2] + T[..., 1, 2] * T[..., 0, 1]
    X[..., 1, 5] = T[..., 1, 2] * T[..., 0, 2]
    # 3rd row for z
    X[..., 2, 0] = T[..., 1, 0] ** 2
    X[..., 2, 1] = 2 * T[..., 1, 1] * T[..., 1, 0]
    X[..., 2, 2] = 2 * T[..., 1, 2] * T[..., 1, 0]
    X[..., 2, 3] = T[..., 1, 1] ** 2
    X[..., 2, 4] = 2 * T[..., 1, 1] * T[..., 1, 2]
    X[..., 2, 5] = T[..., 1, 2] ** 2
    # solve underdetermined system of equations
    Y = torch.zeros((T.shape[0], 3, 1), device=T.device)
    Y[..., 0, 0] = cov2D[..., 0, 0]  # for x
    Y[..., 1, 0] = cov2D[..., 0, 1]  # for y
    Y[..., 2, 0] = cov2D[..., 1, 1]  # for z
    return X, Y


def solve_cov3D(mean, fovx, fovy, width, height, view_matrix, cov2D):
    J = compute_Jacobian(mean, fovx, fovy, width, height, view_matrix)
    T = compute_T(J, view_matrix)
    X, Y = compute_cov3D_equations(T, cov2D)
    return X, Y


def unflatten_symmetry_2x2(A):
    '''Unflatten a 2x2 matrix'''
    m = torch.zeros((A.shape[0], 2, 2), dtype=A.dtype, layout=A.layout, device=A.device)
    m[..., 0, 0] = A[..., 0]
    m[..., 0, 1] = A[..., 1]
    m[..., 1, 0] = A[..., 1]
    m[..., 1, 1] = A[..., 2]
    return m


def unflatten_symmetry_3x3(A):
    '''Unflatten a 3x3 matrix'''
    m = torch.zeros((A.shape[0], 3, 3), dtype=A.dtype, layout=A.layout, device=A.device)
    m[..., 0, 0] = A[..., 0]
    m[..., 0, 1] = A[..., 1]
    m[..., 0, 2] = A[..., 2]
    m[..., 1, 0] = A[..., 1]
    m[..., 1, 1] = A[..., 3]
    m[..., 1, 2] = A[..., 4]
    m[..., 2, 0] = A[..., 2]
    m[..., 2, 1] = A[..., 4]
    m[..., 2, 2] = A[..., 5]
    return m


def compute_mean2D(projmatrix, W, H, p_orig):
    '''Compute the 2D mean'''
    p_hom = torch.cat([p_orig, torch.ones((p_orig.shape[0], 1), device=p_orig.device)], dim=1) @ projmatrix
    p_w = 1 / (p_hom[:, -1:] + 0.0000001)
    p_proj = p_hom[:, :-1] * p_w
    point_image = ((p_proj[:, :2] + 1) * torch.tensor([[W, H]], device=p_proj.device) - 1) * 0.5
    return point_image


def compute_mean2D_equations(projmatrix, W, H, point_image):
    p_proj = (2 * point_image + 1) / torch.tensor([[W, H]], device=point_image.device) - 1
    eq1 = projmatrix[:, 0] - projmatrix[:, 3] * p_proj[:, 0:1]
    eq2 = projmatrix[:, 1] - projmatrix[:, 3] * p_proj[:, 1:2]
    A = torch.stack([eq1, eq2], dim=1)
    return A


def compute_cov2D(T, cov3D):
    '''Compute the 2D covariance matrix'''
    return T.bmm(cov3D).bmm(T.transpose(1, 2))


def transform_cov2D(A, cov2D):
    return A.bmm(cov2D).bmm(A.transpose(1, 2))
