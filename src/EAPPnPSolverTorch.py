# modified EPPnP algorithm to handle unknown scales in the 3D model
import torch


def EAPPnP(P, p):
    '''
    P: nx3 matrix, points in world coordinates
    p: nx2 matrix, points in camera coordinates
    '''

    cP, mP = centralize(P, 0)
    M, Cw, Alph = prepare_data(cP, p)
    Km = kernel_noise(M, 4)
    R, T, S, err = generalized_kernel_PnP(Cw, Km)
    T = T - (R*S).mm(mP.view(-1, 1))

    return R, T, S, err


def EAPPnPMCS(P, p, t):
    '''
    P: nx3 matrix, points in world coordinates
    p: nx2 matrix, points in camera coordinates
    t: nx3 matrix, offset between cameras for each point
    '''
    cP, mP = centralize(P, 0)
    M, Cw, Alph = prepare_data(cP, p)
    b = prepare_offset(p, -t)
    Km, Cc = kernel_noise_full(M, b, 4)
    R, T, S, err = generalized_kernel_PnP_MCS(Cw, Cc, Km)
    T = T - (R*S).mm(mP.view(-1, 1))

    return R, T, S, err


def EAPPnPMCSCtrl(P, p, t):
    '''
    P: nx3 matrix, points in world coordinates
    p: nx2 matrix, points in camera coordinates
    t: nx3 matrix, offset between cameras for each point
    '''
    cP, mP = centralize(P, 0)
    M, Cw, Alph = prepare_data(cP, p)
    b = prepare_offset(p, -t)
    Km, Cc = kernel_noise_full(M, b, 4)

    return Cw, Cc, Km


def prepare_data(P, p):
    Cw = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]).to(P)
    Alph = torch.cat((P, 1-P.sum(-1, keepdim=True)), -1)
    _Alph = Alph.view(-1, 1)
    _Alph_u = (Alph*p[:,0].view(-1, 1)).view(-1, 1)
    _Alph_v = (Alph*p[:,1].view(-1, 1)).view(-1, 1)

    M = torch.cat(
            (torch.cat((_Alph, torch.zeros_like(_Alph), -_Alph_u), -1).view(-1, 12),
             torch.cat((torch.zeros_like(_Alph), _Alph, -_Alph_v), -1).view(-1, 12)),
            0)

    return M, Cw, Alph


def prepare_offset(p, t):
    b = torch.cat(
            ((p[:, 0]*t[:, 2] - t[:, 0]).view(-1, 1),
             (p[:, 1]*t[:, 2] - t[:, 1]).view(-1, 1)),
            0)
    return b


def kernel_noise(M, dims):
    _, _, V = torch.svd(M, some=M.shape[0] >= M.shape[1])
    return V[:, -dims:]


def kernel_noise_full(M, b, dims):
    U, s, V = torch.svd(M, some=M.shape[0] >= M.shape[1])

    Cc = (b.t().mm(U)/s).mm(V.t()[:s.numel(),:]).t()
    return V[:, -dims:], Cc


def generalized_kernel_PnP(Cw, Km, iter_num=10):
    '''
    find R, t and S such that ||RSCw + t - Cc||^2 is minimized
    '''

    X = Cw.t()
    cX, mX = centralize(X, -1)
    Y = Km[:,-1].reshape(-1, 3).t()
    if Y[-1,:].mean() < 0:
        # control points should be in front of the camera
        Y = -Y

    cY, mY = centralize(Y, -1)
    Ynorm = torch.norm(cY)
    nY = cY/Ynorm
    R, S = procrutes.anisotropic_procrutes(cX, nY)
    S *= Ynorm

    for it in range(iter_num):
        scale = torch.prod(S).pow(1/3)
        Y, S = ((R*S).mm(cX) + mY)/scale, S/scale

        # project into the effective null space of M
        Y = Km.mm(Km.t().mm(Y.t().view(-1, 1))).reshape(-1, 3).t()
        newerr = torch.norm(R.t().mm(Y-Y.mean(-1, keepdim=True))/S.view(-1, 1) - cX)
        if it > 1 and newerr > err*0.95:
            break
        err = newerr
        cY, mY = centralize(Y, -1)
        R, S = procrutes.anisotropic_procrutes(cX, cY, S, 5)

    R, S = procrutes.anisotropic_procrutes(cX, cY, S, 15)
    scale = torch.prod(S).pow(1/3)
    T, S = (mY - (R*S).mm(mX))/scale, S/scale

    return R, T, S, err


def generalized_kernel_PnP_MCS(Cw, Cc, Km, iter_num=10):
    '''
    find R, t and S such that ||RSCw + t - Cc||^2 is minimized
    '''

    X = Cw.t()
    cX, mX = centralize(X, -1)
    Y = Cc.reshape(-1, 3).t()
    if Y[-1,:].mean() < 0:
        # control points should be in front of the camera
        Y = -Y

    cY, mY = centralize(Y, -1)
    R, S = procrutes.anisotropic_procrutes(cX, cY)

    for it in range(iter_num):
        Y = (R*S).mm(cX) + mY

        # project into the effective null space of M
        Y = (Km.mm(Km.t().mm(Y.t().reshape(-1, 1) - Cc)) + Cc).reshape(-1, 3).t()
        newerr = torch.norm(R.t().mm(Y-Y.mean(-1, keepdim=True))/S.view(-1, 1) - cX)
        if it > 1 and newerr > err*0.95:
            break
        err = newerr
        cY, mY = centralize(Y, -1)
        R, S = procrutes.anisotropic_procrutes(cX, cY, S, 5)

    R, S = procrutes.anisotropic_procrutes(cX, cY, S, 15)
    T = (mY - (R*S).mm(mX))

    return R, T, S, err


def centralize(X, dim=None):
    mX = X.mean(dim, True)
    cX = X - mX

    return cX, mX


class procrutes:
    def anisotropic_procrutes(X, Y, S=None, iter_num=30):
        A = Y.mm(X.t())
        if S is None:
            S = torch.ones(3).to(X)

        Xs = (X*X).sum(-1)
        for _ in range(iter_num):
            R = procrutes.orthogonal_polar_factor(A*S)
            S = (A*R).sum(0) / Xs

        return R, S


    def orthogonal_polar_factor(A):
        U, S, V = torch.svd(A)
        S = torch.ones_like(S)
        S[-1] = U.mm(V.t()).det().sign()
        R = (U*S).mm(V.t())

        return R







