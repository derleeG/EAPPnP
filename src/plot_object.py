import cv2
import numpy as np
import EAPPnP
import stochastic.continuous

def gen_point_on_box(n):
    x = np.linspace(-1, 1, n)
    P = []
    for i in x:
        for j in x:
            for k in x:
                if np.sum([np.abs(m) == 1 for m in [i,j,k]]) >= 0:
                    P.append([i,j,k])

    P = np.array(P, dtype=np.float32)
    return P

def coorrelated_uniform_generator(limit=(-1, 1)):
    # not uniform not all
    # using Fractional Brownian Motion with wrap around

    FBM = stochastic.continuous.FractionalBrownianMotion(0.90, 10)
    x0, sign, offset = 0, 1, 0

    while True:
        noises = FBM.sample(2000)

        for noise in noises:
            x = (x0 + noise)*sign + offset

            # check boundary
            while x > limit[1] or x < limit[0]:
                if x > limit[1]:
                    sign, offset = -sign, -offset + 2*limit[1]
                    x = (x0 + noise)*sign + offset
                if x < limit[0]:
                    sign, offset = -sign, -offset + 2*limit[0]
                    x = (x0 + noise)*sign + offset

            yield x
        x0 = x


def RTS_state_generator():
    # parameters for the random motion
    limits = [(-2, 2), (-2, 2), (-3, 3), (-3, 3), (-2, 2), (1.2, 5), (-1, 1), (-1, 1)]

    rngs = [coorrelated_uniform_generator(limit) for limit in limits]

    for state in zip(*rngs):
        r = cv2.Rodrigues(np.array(state[:3]))[0].astype(np.float32)
        t = np.array(state[3:6], dtype=np.float32)
        t[-1] *= 5
        s = np.power(2, np.array([0, *state[6:]])).astype(np.float32)

        yield r, t, s


def perspective_project(x):
    x = x[:, :2]/x[:, 2].reshape(-1, 1)
    return x


def draw_perspective_point(p, f, img, color):

    y, x, _ = img.shape
    imp = p*f + np.array([x/2, y/2])
    for x, y in imp.astype(np.int32):
        cv2.circle(img, (x, y), 1, color, -1)

    return


def draw_perspective_box(box, f, img, color):

    y, x, _ = img.shape
    p = box.reshape(2, 2)
    imp = p*f + np.array([x/2, y/2])
    x1, y1, x2, y2 = imp.reshape(-1).astype(np.int32)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    return


def find_2D_bounding_box(p):
    pmin = np.amin(p, 0)
    pmax = np.amax(p, 0)
    pmin, pmax = pmin*1.1 - pmax*0.1, pmax*1.1 - pmin*0.1
    box = np.concatenate((pmin, pmax))

    return box


def orthogonal_project(x, dim):
    dims = [x for x in range(3) if x != dim]
    x = x[:, dims]
    return x


def gen_orthogonal_view_setting(p, dim, show_size):

    p = orthogonal_project(p, dim)
    box = find_2D_bounding_box(p).reshape(2, 2)
    center = np.mean(box, 0)
    scale = show_size / np.max(box[1, :] - box[0, :])

    return dim, center, scale


def draw_orthogonal_point(p, view_setting, img, color):
    dim, center, scale = view_setting
    y, x, _ = img.shape
    p = (orthogonal_project(p, dim)-center)*scale + np.array([x/2, y/2])

    for x, y in p.astype(np.int32):
        cv2.circle(img, (x, y), 1, color, -1)

    return





if __name__ == '__main__':

    # build reference P
    X = gen_point_on_box(4)
    image_size = (360, 640) # 1/3 of full HD
    f = 1400/3 # approximately the same field of view as Iphone 6

    rng = RTS_state_generator()
    while True:
        R_gt, T_gt, S_gt = next(rng)
        Y_gt = np.matmul(X*S_gt, R_gt.T) + T_gt
        p_gt = perspective_project(Y_gt)
        box = find_2D_bounding_box(p_gt)

        p_noise = p_gt
        R, T, S, _ = EAPPnP.EAPPnP(X, p_noise)

        Y = np.matmul(X*S, R.T) + T.reshape(-1)
        p = perspective_project(Y)
        


        # visualize
        p_view = np.zeros((*image_size, 3), dtype=np.uint8)
        draw_perspective_point(p_gt, f, p_view, (255, 255, 255))
        draw_perspective_point(p_noise, f, p_view, (40, 40, 255))
        draw_perspective_point(p, f, p_view, (255, 100, 100))
        draw_perspective_box(box, f, p_view, (80, 80, 80))

        t_view_setting = gen_orthogonal_view_setting(\
                np.concatenate((Y_gt, Y), 0), 1, image_size[0]/2)
        t_view = np.zeros((*image_size, 3), dtype=np.uint8)
        draw_orthogonal_point(Y_gt, t_view_setting, t_view, (255, 255, 255))
        draw_orthogonal_point(Y, t_view_setting, t_view, (100, 100, 255))


        vis = np.concatenate((p_view, t_view), 1)

        cv2.imshow('vis', vis)

        q = cv2.waitKey(10)
        if q == 27:
            break




