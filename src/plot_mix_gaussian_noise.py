import matplotlib.pyplot as plt
import unit_test as test
import numpy as np



def cal_rot_err(Rgt, R):
    err = []
    for rgt, r in zip(Rgt, R):
        rgt = rgt/np.linalg.norm(rgt)
        r = r/np.linalg.norm(r)
        err.append(np.arccos(np.clip(np.dot(rgt, r), -1.0, 1.0)))

    return max(err)




if __name__ == '__main__':


    aniso_data = test.gen_stretched_transform_with_gt
    iso_data = test.gen_rigid_transform_with_gt
    aniso_func = test.EAPPnP.EAPPnP
    iso_func = test.EAPPnP.EPPnP
    Mpoints = range(10, 210, 10)
    focal_length = 800
    sigma = np.array(range(1, 11), dtype=np.float32)/focal_length
    ratio = np.ones_like(sigma)/sigma.size


    aniso_rot, aniso_trans = [], []
    iso_rot, iso_trans = [], []
    aniso_err, iso_err = [], []

    for M in Mpoints:
        test.M = M

        # for comparison we are testing on isotropic data
        P, p, R, T = iso_data(test.N)
        noise = test.mix_gaussian_noise(p.shape, sigma, ratio)
        datas = (P, p + noise, R, T)

        rot_err, trans_err = 0, 0
        p_err = 0
        for P, p, rgt, tgt in zip(*datas):
            r, t, _, err = aniso_func(P, p)

            rot_err += cal_rot_err(rgt, r)
            trans_err += np.linalg.norm(t-tgt)/np.linalg.norm(t)
            p_err += err

        aniso_rot.append(rot_err/test.N*180/np.pi)
        aniso_trans.append(trans_err/test.N*100)
        aniso_err.append(p_err/test.N)

        rot_err, trans_err = 0, 0
        p_err = 0
        for P, p, rgt, tgt in zip(*datas):

            r, t, err = iso_func(P, p)

            rot_err += cal_rot_err(rgt, r)
            trans_err += np.linalg.norm(t-tgt)/np.linalg.norm(t)
            p_err += err

        iso_rot.append(rot_err/test.N*180/np.pi)
        iso_trans.append(trans_err/test.N*100)
        iso_err.append(p_err/test.N)


        print('N points: {}\n\
                , aniso rot: {:.3E}, aniso trans: {:.3E}\n\
                , iso rot: {:.3E}, iso trans: {:.3E}\n\
                , aniso err: {:.3E}, iso err: {:.3E}'\
                .format(M, aniso_rot[-1], aniso_trans[-1], iso_rot[-1], iso_trans[-1], aniso_err[-1], iso_err[-1]))


    plt.scatter(Mpoints, aniso_rot, marker='^', label='EAPPnP')
    plt.scatter(Mpoints, iso_rot, marker='o', label='EPPnP')
    plt.ylim(ymax=1)
    plt.xlabel("n points")
    plt.ylabel("mean rotation error (deg)")
    plt.legend()
    plt.show()



