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





def EAPPnP_planar(self, P, p):
    # degenerative case
    self.is_np = type(P) == np.ndarray
    P, p = self.sync_data(P, p)
    cP, mP = self.centralize(P, 0)

    R, T, S, err = self._EAPPnP_planar(cP, p)

    if self.is_np:
        T = T - np.matmul(R, S*np.reshape(mP, (-1, 1)))
    else:
        T = T - R.mm(S*mP.view(-1, 1))

    return R, T, S, err


def _EAPPnP_planar(self, P, p):

    M, Cw, Alph = self.prepare_data(P, p)

    if self.is_np:
        dim = np.argmin(np.power(P, 2).sum(0))
    else:
        dim = torch.argmin(P.pow(2).sum(0))

    idx = [x for x in range(4) if x != dim]
    idx3x = []
    for i in idx:
        idx3x.extend([i*3 + x for x in range(3)])

    M = M[:, idx3x]
    Cw = Cw[idx, :]
    Alph = Alph[:, idx]

    Km = self.kernel_noise(M, 3)
    R, T, S, err = self.kernel_PnP(Cw, Km)

    return R, T, S, err


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


def kernel_noise(M, dims):
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    return Vt.T[:, -dims:]


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
    scale = np.linalg.norm(cX)/np.linalg.norm(cY)
    cY, mY = scale*cY, scale*mY

    R = procrutes.procrutes(cX, cY)


    KmQ, _ = np.linalg.qr(Km)

    for it in range(iter_num):
        Y = np.matmul(R, cX) + mY

        # project into the effective null space of M
        Y = np.matmul(KmQ, np.matmul(KmQ.T, (Y.T).reshape(-1, 1))).reshape(-1, 3).T
        newerr = np.linalg.norm(np.matmul(R.T, Y-Y.mean(-1, keepdims=True)) - cX)
        if it > 1 and newerr > err*0.99:
            break
        err = newerr
        cY, mY = centralize(Y, -1)
        R = procrutes.procrutes(cX, cY)

    T = mY - np.matmul(R, mX)

    return R, T, err


def generalized_kernel_PnP(Cw, Km, iter_num=100):
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
    R, S = procrutes.anisotropic_procrutes(cX, cY, iter_num=30)

    
    e = np.linalg.norm(np.matmul(R*S, cX) -cY)
    vis = e > 1e-7
    if vis:
        print('init error: {}'.format(e))
    KmQ, _ = np.linalg.qr(Km)

    for it in range(iter_num):
        Y = np.matmul(R*S, cX) + mY

        # project into the effective null space of M
        Y2 = Y
        Y = np.matmul(KmQ, np.matmul(KmQ.T, (Y.T).reshape(-1, 1))).reshape(-1, 3).T
        if vis:
            print('project: {}'.format(np.linalg.norm(Y2-Y)))
            print(S)
        newerr = np.linalg.norm(np.matmul(R.T, Y-Y.mean(-1, keepdims=True))/S.reshape(-1, 1) - cX)
        if vis:
            print(newerr)
        if it > 1 and newerr > err*0.99:
            break
        err = newerr
        cY, mY = centralize(Y, -1)
        R, S = procrutes.anisotropic_procrutes(cX, cY, S, 30)
        if vis:
            print('error: {}'.format(np.linalg.norm(np.matmul(R*S, cX) -cY)))
    err = 0
    R, S = procrutes.anisotropic_procrutes(cX, cY, S, 30)
    T = mY - np.matmul(R*S, mX)
    T, S = T/S[0], S/S[0]

    return R, T, S, err


def centralize(X, dim=None):
    mX = X.mean(dim, keepdims=True)
    cX = X - mX

    return cX, mX



