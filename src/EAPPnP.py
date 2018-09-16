# modified EPPnP algorithm to handle unknown scales in the 3D model
import numpy as np
from ..lib import pyprocrutes as procrutes


class EAPPnPSolver:
    def __init__(self):
        pass


    def EAPPnP(self, P, p):
        '''
        P: nx3 matrix, points in world coordinates
        p: nx2 matrix, points in camera coordinates
        '''

        P, p = self.sync_data(P, p)
        cP, mP = self.centralize(P, 0)

        R, T, S, err = self._EAPPnP(cP, p)

        T = T - np.matmul(R, S*np.reshape(mP, (-1, 1)))

        return R, T, S, err


    def _EAPPnP(self, P, p):

        M, Cw, Alph = self.prepare_data(P, p)

        Km = self.kernel_noise(M, 4)
        R, T, S, err = self.kernel_PnP(Cw, Km)

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


    def prepare_data(self, P, p):

        if self.is_np:
            Cw = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1],
                           [0, 0, 0]])

            Alph = np.concatenate((P, 1-np.sum(P, -1, keepdims=True)), -1)

            _Alph = np.reshape(Alph, (-1, 1))
            _Alph_u = np.reshape((Alph*np.reshape(p[:,0], (-1, 1))), (-1, 1))
            _Alph_v = np.reshape((Alph*np.reshape(p[:,1], (-1, 1))), (-1, 1))

            M = np.concatenate(
                    (np.reshape(np.concatenate((_Alph, np.zeros(_Alph.shape), -_Alph_u), -1), (-1, 12)),
                     np.reshape(np.concatenate((np.zeros(_Alph.shape), _Alph, -_Alph_v), -1), (-1, 12))),
                    0)


        else:
            Cw = torch.tensor([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1],
                            [0, 0, 0]],
                            dtype=self.template.dtype,
                            device=self.template.device)

            Alph = torch.cat((P, 1-P.sum(-1, keepdim=True)), -1)

            _Alph = Alph.view(-1, 1)
            _Alph_u = (Alph*p[:,0].view(-1, 1)).view(-1, 1)
            _Alph_v = (Alph*p[:,1].view(-1, 1)).view(-1, 1)

            M = torch.cat(
                    (torch.cat((_Alph, torch.zeros_like(_Alph), -_Alph_u), -1).view(-1, 12),
                    torch.cat((torch.zeros_like(_Alph), _Alph, -_Alph_v), -1).view(-1, 12)),
                    0)

        return M, Cw, Alph


    def kernel_noise(self, M, dims):

        if self.is_np:
            _, _, Vt = np.linalg.svd(M)
            V = Vt.transpose()
        else:
            _, _, V = torch.svd(M)
        K = V[:, -dims:]
        return K


    def kernel_PnP(self, Cw, Km, iter_num=10):
        '''
        find R, t and S such that ||RSCw + t - Cc||^2 is minimized
        '''


        X = Cw.transpose() if self.is_np else Cw.t()
        cX, mX = self.centralize(X, -1)

        Y = Km[:,-1].reshape(-1, 3).transpose() if self.is_np else Km[:,-1].reshape(-1, 3).t()

        if Y[-1,:].mean() < 0:
            # control points should be in front of the camera
            Y = -Y

        cY, mY = self.centralize(Y, -1)

        R, S = self.anisotropic_procrutes(cX, cY)

        err = float('inf')

        for it in range(iter_num):
            newY = np.matmul(R, S*cX) + mY if self.is_np else R.mm(S*cX) + mY
            newY = newY/S[1]
            S = S/S[1]

            # project into the effective null space of M

            if self.is_np:
                coef = self.least_square(Km, newY.transpose().reshape(-1, 1))
                newY = np.matmul(Km, coef).reshape(-1, 3).transpose()
                newerr = self.norm(np.matmul(R.transpose(), newY-newY.mean(-1, keepdims=True))/S - cX)
            else:
                coef = self.least_square(Km, newY.t().reshape(-1, 1))
                newY = Km.mm(coef).reshape(-1, 3).t()
                newerr = self.norm(R.t().mm(newY-newY.mean(-1, True))/S - cX)

            if newerr > err and it > 1:
                break
            else:
                err = newerr

                Y = newY
                cY, mY = self.centralize(Y, -1)
                R, S = self.anisotropic_procrutes(cX, cY, S)

        if self.is_np:
            T = mY - np.matmul(R, S*mX)
        else:
            T = mY - R.mm(S*mX)
        T = T/S[1]
        S = S/S[1]
        return R, T, S, err


    def least_square(self, A, b):
        '''
        solve argmin{x} ||Ax - b||^2 for over constrained A
        '''

        return np.linalg.lstsq(A, b, rcond=None)[0]


    def norm(self, x):
        return np.sqrt(np.power(x, 2).sum())


    def centralize(self, X, dim=None):
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


