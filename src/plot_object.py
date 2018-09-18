import cv2
import numpy as np
import EAPPnP
import stochastic.continuous
import matplotlib.pyplot as plt

def gen_point_on_box(n):
    x = np.linspace(-1, 1, n)
    P = []
    for i in x:
        for j in x:
            for k in x:
                if np.sum([np.abs(m) == 1 for m in [i,j,k]]) >= 2:
                    P.append([i,j,k])

    P = np.array(P, dtype=np.float32)
    return P

def coorrelated_uniform_generator(limit=(-1, 1)):
    # not uniform not all
    # using Fractional Brownian Motion with wrap around

    FBM = stochastic.continuous.FractionalBrownianMotion(0.90, 1)
    x0, sign, offset = 0, 1, 0

    while True:
        noises = FBM.sample(10000)

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
    limits = [(-2, 2), (-2, 2), (-2, 2), (-2, 2), (-2, 2), (4, 8), (-1, 1), (-1, 1)]

    rngs = [coorrelated_uniform_generator(limit) for limit in limits]

    for state in zip(*rngs):
        r = cv2.Rodrigues(np.array(state[:3]))[0].astype(np.float32)
        t = np.array(state[3:6], dtype=np.float32)
        s = np.power(2, np.array([1, *state[6:]])).astype(np.float32)

        yield r, t, s


def perspective_project(x):
    x = x[:, :2]/x[:, 2].reshape(-1, 1)
    return x



if __name__ == '__main__':

    # build reference P
    P = gen_point_on_box(5)
    image_size = (360, 640) # 1/3 of full HD
    f = 1400/3 # approximately the same field of view as Iphone 6
    K = np.array([[f, 0, image_size[1]/2], [0, f, image_size[0]/2], [0, 0, 1]])

    rng = RTS_state_generator()
    while True:
        R, T, S = next(rng)
        p = perspective_project(np.matmul(P*S, R.T) + T)

        imp = p*f + np.array([image_size[1]/2, image_size[0]/2])
        
        vis = np.zeros((*image_size, 3), dtype=np.uint8)
        for x, y in imp.astype(np.int32):
            cv2.circle(vis, (x, y), 2, (255, 255, 255), -1)

        cv2.imshow('vis', vis)
        q = cv2.waitKey(10)
        if q == 27:
            break




