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
    R, T, S, err = kernel_PnP(Cw, Km)
    T = T - np.matmul(R, S*np.reshape(mP, (-1, 1)))

    return R, T, S, err


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

    Cw = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
    Alph = np.concatenate((P, 1-np.sum(P, -1, keepdims=True)), -1)
    _Alph = np.reshape(Alph, (-1, 1))
    _Alph_u = np.reshape((Alph*np.reshape(p[:,0], (-1, 1))), (-1, 1))
    _Alph_v = np.reshape((Alph*np.reshape(p[:,1], (-1, 1))), (-1, 1))

    M = np.concatenate(
            (np.reshape(np.concatenate((_Alph, np.zeros(_Alph.shape), -_Alph_u), -1), (-1, 12)),
             np.reshape(np.concatenate((np.zeros(_Alph.shape), _Alph, -_Alph_v), -1), (-1, 12))),
            0)

    return M, Cw, Alph


def kernel_noise(M, dims):
    _, _, Vt = np.linalg.svd(M)
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
    R, S = procrutes.anisotropic_procrutes(cX, cY, 10)

    for it in range(iter_num):
        Y = np.matmul(R*S, cX) + mY

        # project into the effective null space of M
        coef = np.linalg.lstsq(Km, Y.T.reshape(-1, 1), rcond=None)[0]
        Y = np.matmul(Km, coef).reshape(-1, 3).T
        newerr = np.linalg.norm(np.matmul(R.T/S, Y-Y.mean(-1, keepdims=True)) - cX)

        if it > 1 and newerr > err:
            break
        err = newerr
        cY, mY = centralize(Y, -1)
        R, S = procrutes.anisotropic_procrutes(cX, cY, S, 10)

    T = mY - np.matmul(R*S, mX)
    T, S = T/S[0], S/S[0]

    return R, T, S, err


def centralize(X, dim=None):
    mX = X.mean(dim, keepdims=True)
    cX = X - mX

    return cX, mX



if __name__ == '__main__':

    solver = EAPPnPSolver()

    #test procrutes

    # random points X

    U, D, V = torch.svd(torch.rand(3, 3))
    D = torch.ones_like(D)
    D[-1] = U.mm(V.t()).det().sign()
    R = U.mm(torch.diag(D)).mm(V.t())

    Y = torch.rand(3, 10)
    Y -= Y.mean(1, True)
    # make sure all points are in front of the camera
    Y[-1, :] = Y[-1, :].abs() + 1

    S = torch.rand(3, 1)*2 + 0.5
    S = S/S[1]

    T = (torch.rand(3, 1)-0.5)*20

    X = R.t().mm(Y-T)/S
    #X[2,:] = X[2,:].mean()

    Y = R.mm(S*X) + T

    y = Y[:2,:]/Y[2,:]
    y += ((torch.rand(y.size()) - 0.5)*0)



    Re, Te, Se, err = solver.EAPPnP(X.t(), y.t())
    Rer, Ter, Ser, errr = solver.EAPPnP(X.t().numpy(), y.t().numpy())

    I = torch.ones(1).float()
    print(R)
    print(Re)
    print(Rer*I)
    print(T)
    print(Te)
    print(Ter*I)
    print(S)
    print(Se)
    print(Ser*I)

    print(err)
    print((Re.mm(Se*X) + Te - Y).abs().mean(-1))
    print(errr*I)
    print(np.abs(np.matmul(Rer, Ser*X.numpy()) + Ter - Y.numpy()).mean(-1)*I)


