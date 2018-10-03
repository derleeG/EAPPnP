import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool


def float_to_float(p, r, n):
    rr = r*r
    x, y = np.meshgrid(range(n), range(n))
    x = x.reshape(1, n, n)
    y = y.reshape(1, n, n)
    px = p[:, 0].reshape(-1, 1, 1)
    py = p[:, 1].reshape(-1, 1, 1)

    mask = ((px - x)**2 + (py - y)**2 < rr)

    px = np.sum((mask*x).reshape(-1, n*n), -1)/np.sum(mask.reshape(-1, n*n), -1)
    py = np.sum((mask*y).reshape(-1, n*n), -1)/np.sum(mask.reshape(-1, n*n), -1)

    p = np.stack((px, py), 1)
    return p

def float_to_float2(p, r, n):
    return np.round(p)



def discretize_error(r):

    r_c = np.ceil(r).astype(np.int32)
    x = np.random.rand(10000, 2) + r_c
    x_noise = float_to_float2(x, r, 2*r_c+2)

    error = (x - x_noise)
    error = np.sqrt(np.sum(error*error, -1))

    return np.mean(error), np.max(error)



if __name__ == '__main__':


    x = np.linspace(0.71, 20, 2)
    with Pool() as p:
        results = p.map(discretize_error, x)
    results = np.array(results)

    plt.figure(1)
    plt.scatter(x, results[:, -1], 1)
    plt.xlabel('radius')
    plt.ylabel('max error')
    plt.figure(2)
    plt.scatter(x, results[:, -2], 1)
    plt.xlabel('radius')
    plt.ylabel('mean error')
    plt.show()
