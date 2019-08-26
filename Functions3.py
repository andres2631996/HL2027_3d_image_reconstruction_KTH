import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt



eps = np.finfo(float).eps


def bwdist(a):
    """
    Intermediary function. 'a' has only True/False vals,
    so we convert them into 0/1 values - in reverse.
    True is 0, False is 1, distance_transform_edt wants it that way.
    """
    return nd.distance_transform_edt(a == 0)


# Displays the image with curve superimposed
def show_curve_and_phi(fig, I, phi, color):
    fig.axes[0].cla()
    fig.axes[0].imshow(I, cmap='gray')
    fig.axes[0].contour(phi, 0, colors=color)
    fig.axes[0].set_axis_off()
    plt.draw()

    fig.axes[1].cla()
    fig.axes[1].imshow(phi)
    fig.axes[1].set_axis_off()
    plt.draw()

    plt.pause(0.1)


def im2double(a):
    a = a.astype(np.float)
    a /= np.abs(a).max()
    return a


# Converts a mask to a SDF
def mask2phi(init_a):
    phi = bwdist(init_a) - bwdist(1 - init_a) + im2double(init_a) - 0.5
    return phi

# Compute curvature along SDF
def Evolution(phi,Feature,idx):
    dimy, dimx = phi.shape
    yx = np.array([np.unravel_index(i, phi.shape) for i in idx])  # subscripts
    y = yx[:, 0]
    x = yx[:, 1]

    # Get subscripts of neighbors
    ym1 = y - 1
    xm1 = x - 1
    yp1 = y + 1
    xp1 = x + 1

    # Bounds checking
    ym1[ym1 < 0] = 0
    xm1[xm1 < 0] = 0
    yp1[yp1 >= dimy] = dimy - 1
    xp1[xp1 >= dimx] = dimx - 1

    # Get indexes for 8 neighbors
    idup = np.ravel_multi_index((yp1, x), phi.shape)
    iddn = np.ravel_multi_index((ym1, x), phi.shape)
    idlt = np.ravel_multi_index((y, xm1), phi.shape)
    idrt = np.ravel_multi_index((y, xp1), phi.shape)
    idul = np.ravel_multi_index((yp1, xm1), phi.shape)
    idur = np.ravel_multi_index((yp1, xp1), phi.shape)
    iddl = np.ravel_multi_index((ym1, xm1), phi.shape)
    iddr = np.ravel_multi_index((ym1, xp1), phi.shape)

    # Get central derivatives of SDF at x,y
    phi_x = -phi.flat[idlt] + phi.flat[idrt]
    phi_y = -phi.flat[iddn] + phi.flat[idup]
    phi_xx = phi.flat[idlt] - 2 * phi.flat[idx] + phi.flat[idrt]
    phi_yy = phi.flat[iddn] - 2 * phi.flat[idx] + phi.flat[idup]
    phi_xy = 0.25 * (- phi.flat[iddl] - phi.flat[idur] +
                     phi.flat[iddr] + phi.flat[idul])
    phi_x2 = phi_x ** 2
    phi_y2 = phi_y ** 2

    # Compute curvature (Kappa)
    curvature = ((phi_x2 * phi_yy + phi_y2 * phi_xx - 2 * phi_x * phi_y * phi_xy) /
                 (phi_x2 + phi_y2 + eps) ** 1.5)

    # Compute norm of gradient
    phi_xm = phi.flat[idx]-phi.flat[idlt]
    phi_xp = phi.flat[idrt]-phi.flat[idx]
    phi_ym = phi.flat[idx]-phi.flat[iddn]
    phi_yp = phi.flat[idup]-phi.flat[idx]
    a1 = np.maximum(phi_xm,0)
    a2 = np.minimum(phi_xp,0)
    a3 = np.maximum(phi_ym,0)
    a4 = np.minimum(phi_yp,0)
    SumPower = np.power(a1,2)+np.power(a2,2)+np.power(a3,2)+np.power(a4,2)
    normGrad = SumPower**(0.5)

    # Compute scalar product between the feature image and the gradient of phi
    q1 = Feature.flat[idrt]
    q2 = Feature.flat[idlt]
    F_x = 0.5 * q1 - 0.5 * q2
    q3 = Feature.flat[idup]
    q4 = Feature.flat[iddn]
    F_y = 0.5 * q3 - 0.5 * q4
    a1 = np.maximum(F_x, 0)
    a2 = np.minimum(F_x, 0)
    a3 = np.maximum(F_y, 0)
    a4 = np.minimum(F_y, 0)
    FdotGrad = np.multiply(a1, phi_xp) + np.multiply(a2, phi_xm) + np.multiply(a3, phi_yp) + np.multiply(a4, phi_ym)

    return curvature,normGrad,FdotGrad

# Level set re-initialization by the sussman method
def sussman(D, dt):
    # forward/backward differences
    a = D - np.roll(D, 1, axis=1)
    b = np.roll(D, -1, axis=1) - D
    c = D - np.roll(D, -1, axis=0)
    d = np.roll(D, 1, axis=0) - D

    a_p = np.clip(a, 0, np.inf)
    a_n = np.clip(a, -np.inf, 0)
    b_p = np.clip(b, 0, np.inf)
    b_n = np.clip(b, -np.inf, 0)
    c_p = np.clip(c, 0, np.inf)
    c_n = np.clip(c, -np.inf, 0)
    d_p = np.clip(d, 0, np.inf)
    d_n = np.clip(d, -np.inf, 0)

    a_p[a < 0] = 0
    a_n[a > 0] = 0
    b_p[b < 0] = 0
    b_n[b > 0] = 0
    c_p[c < 0] = 0
    c_n[c > 0] = 0
    d_p[d < 0] = 0
    d_n[d > 0] = 0

    dD = np.zeros_like(D)
    D_neg_ind = np.flatnonzero(D < 0)
    D_pos_ind = np.flatnonzero(D > 0)

    dD.flat[D_pos_ind] = np.sqrt(
        np.max(np.concatenate(
            ([a_p.flat[D_pos_ind] ** 2], [b_n.flat[D_pos_ind] ** 2])), axis=0) +
        np.max(np.concatenate(
            ([c_p.flat[D_pos_ind] ** 2], [d_n.flat[D_pos_ind] ** 2])), axis=0)) - 1
    dD.flat[D_neg_ind] = np.sqrt(
        np.max(np.concatenate(
            ([a_n.flat[D_neg_ind] ** 2], [b_p.flat[D_neg_ind] ** 2])), axis=0) +
        np.max(np.concatenate(
            ([c_n.flat[D_neg_ind] ** 2], [d_p.flat[D_neg_ind] ** 2])), axis=0)) - 1

    D = D - dt * sussman_sign(D) * dD
    return D


def sussman_sign(D):
    return D / np.sqrt(D ** 2 + 1)


# Convergence Test
def convergence(p_mask, n_mask, thresh, c):
    diff = p_mask - n_mask
    n_diff = np.sum(np.abs(diff))
    if n_diff < thresh:
        c = c + 1
    else:
        c = 0
    return c


