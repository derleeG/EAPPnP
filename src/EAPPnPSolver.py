# modified EPPnP algorithm to handle unknown scales in the 3D model
import numpy as np
from ..lib import pyprocrutes as procrutes


def EAPPnP(P, p):
    '''
    P: nx3 matrix, points in world coordinates
    p: nx2 matrix, points in camera coordinates
    '''

    cP, mP = centralize(P, 0)
    M, Cw, Alph = prepare_data(cP, p)
    Km = kernel_noise(M, 4)
    R, T, S, err = generalized_kernel_PnP(Cw, Km)
    T = T - np.matmul(R*S, np.reshape(mP, (-1, 1)))

    return R, T, S, err


def EPPnP(P, p):
    '''
    P: nx3 matrix, points in world coordinates
    p: nx2 matrix, points in camera coordinates
    '''

    cP, mP = centralize(P, 0)
    M, Cw, Alph = prepare_data(cP, p)
    Km = kernel_noise(M, 4)
    R, T, err = kernel_PnP(Cw, Km)
    T = T - np.matmul(R, np.reshape(mP, (-1, 1)))

    return R, T, err


def EAPPnPMCS(P, p, t):
    '''
    P: nx3 matrix, points in world coordinates
    p: nx2 matrix, points in camera coordinates
    t: nx3 matrix, offset between cameras for each point
    '''

    cP, mP = centralize(P, 0)
    M, Cw, Alph = prepare_data(cP, p)
    b = prepare_offset(p, -t)
    Vt, Cc = kernel_noise_full(M, b, 4)
    R, T, S, err = generalized_kernel_PnP_MCS(Cw, Cc, Km)
    T = T - np.matmul(R*S, np.reshape(mP, (-1, 1)))

    return R, T, S, err


def EPPnPMCS(P, p, t):
    pass


def EAPPnL(L, l):
    pass


def EPPnL(L, l):
    pass


def EAPPnLMCS(L, l, t):
    pass


def EPPnLMCS(L, l, t):
    pass


def EAPPnX(P, p, L, l):
    pass


def EPPnX(P, p, L, l):
    pass


def EAPPnXMCS(P, p, tp, L, l, tl):
    pass


def EPPnXMCS(P, p, tp, L, l, tl):
    pass


def prepare_data(P, p):

    Cw = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.float32)
    Alph = np.concatenate((P, 1-np.sum(P, -1, keepdims=True)), -1)
    _Alph = np.reshape(Alph, (-1, 1))
    _Alph_u = np.reshape((Alph*np.reshape(p[:,0], (-1, 1))), (-1, 1))
    _Alph_v = np.reshape((Alph*np.reshape(p[:,1], (-1, 1))), (-1, 1))

    M = np.concatenate(
            (np.reshape(np.concatenate((_Alph, np.zeros(_Alph.shape), -_Alph_u), -1), (-1, 12)),
             np.reshape(np.concatenate((np.zeros(_Alph.shape), _Alph, -_Alph_v), -1), (-1, 12))),
            0).astype(np.float32)

    return M, Cw, Alph


def prepare_offset(p, t):
    b = np.concatenate(
            (np.reshape(p[:, 0]*t[:, 2] - t[:, 0], (-1, 1)),
             np.reshape(p[:, 1]*t[:, 2] - t[:, 1], (-1, 1))),
            0).astype(np.float32)
    return b


def kernel_noise(M, dims):
    _, _, Vt = np.linalg.svd(M, full_matrices=M.shape[0] < M.shape[1])
    return Vt.T[:, -dims:]


def kernel_noise_full(M, b, dims):
    U, s, Vt = np.linalg.svd(M, full_matrices=M.shape[0] < M.shape[1])
    Cc = np.matmul(np.matmul(b.T, U)/s, Vt[:s.size,:]).T
    return Vt.T[:, -dims:], Cc


def kernel_PnP(Cw, Km, iter_num=10):
    '''
    find R, t and S such that ||RSCw + t - Cc||^2 is minimized
    '''

    X = Cw.T
    cX, mX = centralize(X, -1)
    Y = Km[:,-1].reshape(-1, 3).T
    if Y[-1,:].mean() < 0:
        # control points should be in front of the camera
        Y = -Y

    cY, mY = centralize(Y, -1)
    Ynorm = np.linalg.norm(cY)
    nY = cY/Ynorm
    R, s = procrutes.isotropic_procrutes(cX, nY)
    s *= Ynorm

    for it in range(iter_num):
        Y = np.matmul(R, cX) + mY/s

        # project into the effective null space of M
        Y = np.matmul(Km, np.matmul(Km.T, (Y.T).reshape(-1, 1))).reshape(-1, 3).T
        newerr = np.linalg.norm(np.matmul(R.T, Y-Y.mean(-1, keepdims=True)) - cX)
        if it > 1 and newerr > err*0.95:
            break
        err = newerr
        cY, mY = centralize(Y, -1)
        R, s = procrutes.isotropic_procrutes(cX, cY)

    T = mY/s - np.matmul(R, mX)

    return R, T, err


def generalized_kernel_PnP(Cw, Km, iter_num=10):
    '''
    find R, t and S such that ||RSCw + t - Cc||^2 is minimized
    '''

    X = Cw.T
    cX, mX = centralize(X, -1)
    Y = Km[:,-1].reshape(-1, 3).T
    if Y[-1,:].mean() < 0:
        # control points should be in front of the camera
        Y = -Y

    cY, mY = centralize(Y, -1)
    Ynorm = np.linalg.norm(cY)
    nY = cY/Ynorm
    R, S = procrutes.anisotropic_procrutes(cX, nY)
    S *= Ynorm

    for it in range(iter_num):
        scale = np.cbrt(np.prod(S))
        Y, S = (np.matmul(R*S, cX) + mY)/scale, S/scale

        # project into the effective null space of M
        Y = np.matmul(Km, np.matmul(Km.T, (Y.T).reshape(-1, 1))).reshape(-1, 3).T
        newerr = np.linalg.norm(np.matmul(R.T, Y-Y.mean(-1, keepdims=True))/S.reshape(-1, 1) - cX)
        if it > 1 and newerr > err*0.95:
            break
        err = newerr
        cY, mY = centralize(Y, -1)
        R, S = procrutes.anisotropic_procrutes(cX, cY, S, 5)

    R, S = procrutes.anisotropic_procrutes(cX, cY, S, 15)
    scale = np.cbrt(np.prod(S))
    T, S = (mY - np.matmul(R*S, mX))/scale, S/scale

    return R, T, S, err


def generalized_kernel_PnP_MCS(Cw, Cc, Km, iter_num=10):
    '''
    find R, t and S such that ||RSCw + t - Cc||^2 is minimized
    '''

    X = Cw.T
    cX, mX = centralize(X, -1)
    Y = Cc.reshape(-1, 3).T
    if Y[-1,:].mean() < 0:
        # control points should be in front of the camera
        Y = -Y

    cY, mY = centralize(Y, -1)
    R, S = procrutes.anisotropic_procrutes(cX, cY)

    for it in range(iter_num):
        Y = np.matmul(R*S, cX) + mY

        # project into the effective null space of M
        Y = (np.matmul(Km, np.matmul(Km.T, (Y.T).reshape(-1, 1) - Cc)) + Cc).reshape(-1, 3).T
        newerr = np.linalg.norm(np.matmul(R.T, Y-Y.mean(-1, keepdims=True))/S.reshape(-1, 1) - cX)
        if it > 1 and newerr > err*0.95:
            break
        err = newerr
        cY, mY = centralize(Y, -1)
        R, S = procrutes.anisotropic_procrutes(cX, cY, S, 5)

    R, S = procrutes.anisotropic_procrutes(cX, cY, S, 15)
    T = mY - np.matmul(R*S, mX)

    return R, T, S, err


def centralize(X, dim=None):
    mX = X.mean(dim, keepdims=True)
    cX = X - mX

    return cX, mX



