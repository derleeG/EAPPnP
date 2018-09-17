import numpy as np
from timeit import default_timer as timer
import EAPPnP

N = 1000


def flatten(X):
    mX = np.mean(X, -1, keepdims=True)
    cX = X - mX
    _, v = np.linalg.eigh(np.matmul(cX, cX.T))
    v = v[:, 1:]
    X = mX + np.matmul(v, np.matmul(v.T, cX))

    return X


def gen_rigid_transform(n):
    np.random.seed(2018)
    R = np.random.randn(n, 3, 3).astype(np.float32)
    T = np.random.randn(n, 3, 1).astype(np.float32)
    X = np.random.randn(n, 3, 80).astype(np.float32)
    Y = np.zeros_like(X)
    for r, t, x, y in zip(R, T, X, Y):
        r[...] = EAPPnP.procrutes.np_orthogonal_polar_factor(r)
        y[...] = np.matmul(r, x) + t
        y[-1, :] += 10

    Y = Y[:,:-1,:]/np.expand_dims(Y[:,-1,:], 1)
    X = np.swapaxes(X, -1, -2)
    Y = np.swapaxes(Y, -1, -2)

    return X, Y


def gen_rigid_transform_planar(n):
    np.random.seed(2018)
    R = np.random.randn(n, 3, 3).astype(np.float32)
    T = np.random.randn(n, 3, 1).astype(np.float32)
    X = np.random.randn(n, 3, 80).astype(np.float32)
    Y = np.zeros_like(X)
    for r, t, x, y in zip(R, T, X, Y):
        x[...] = flatten(x)
        r[...] = EAPPnP.procrutes.np_orthogonal_polar_factor(r)
        y[...] = np.matmul(r, x) + t
        y[-1, :] += 10

    Y = Y[:,:-1,:]/np.expand_dims(Y[:,-1,:], 1)
    X = np.swapaxes(X, -1, -2)
    Y = np.swapaxes(Y, -1, -2)

    return X, Y


def gen_stretched_transform(n):
    R = np.random.randn(n, 3, 3).astype(np.float32)
    S = np.exp(np.random.randn(n, 3).astype(np.float32)/2)
    T = np.random.randn(n, 3, 1).astype(np.float32)
    X = np.random.randn(n, 3, 80).astype(np.float32)
    Y = np.zeros_like(X)
    for r, s, t, x, y in zip(R, S, T, X, Y):
        s[0] = 1
        r[...] = EAPPnP.procrutes.np_orthogonal_polar_factor(r)
        y[...] = np.matmul(r*s, x) + t
        y[-1, :] += 40000

    Y = Y[:,:-1,:]/np.expand_dims(Y[:,-1,:], 1)
    X = np.swapaxes(X, -1, -2)
    Y = np.swapaxes(Y, -1, -2)

    return X, Y


def gen_stretched_transformi_planar(n):
    R = np.random.randn(n, 3, 3).astype(np.float32)
    S = np.exp(np.random.randn(n, 3).astype(np.float32)/2)
    T = np.random.randn(n, 3, 1).astype(np.float32)
    X = np.random.randn(n, 3, 80).astype(np.float32)
    Y = np.zeros_like(X)
    for r, s, t, x, y in zip(R, S, T, X, Y):
        x[...] = flatten(x)
        s[0] = 1
        r[...] = EAPPnP.procrutes.np_orthogonal_polar_factor(r)
        y[...] = np.matmul(r*s, x) + t
        y[-1, :] += 40000

    Y = Y[:,:-1,:]/np.expand_dims(Y[:,-1,:], 1)
    X = np.swapaxes(X, -1, -2)
    Y = np.swapaxes(Y, -1, -2)

    return X, Y


def get_func(method):
    if method == 'EAPPnP':
        func = EAPPnP.EAPPnP
        data_func = gen_rigid_transform
        transform = lambda x, o: np.matmul(o[0]*o[2], x.T) + o[1]
        project = lambda x: x[:-1, :]/x[-1,:]
        stat_func = lambda x, y, o: (x, y,\
                project(transform(x, o)).T, *o[:-1], \
                np.linalg.norm(project(transform(x, o)) - y.T)/\
                np.linalg.norm(y-y.mean(0)))
        print_fmt = 'matrix X:\n{}\nmatrix Y:\n{}\nmatrix Yp:\n{}\nmatrix R:\n{}\n' \
                   +'matrix T:\n{}\nmatrix S:\n{}\nerror: {}'

    elif method == 'EPPnP':
        func = EAPPnP.EPPnP
        data_func = gen_rigid_transform
        transform = lambda x, o: np.matmul(o[0], x.T) + o[1]
        project = lambda x: x[:-1, :]/x[-1,:]
        stat_func = lambda x, y, o: (x, y,\
                project(transform(x, o)).T, *o[:-1], \
                np.linalg.norm(project(transform(x, o)) - y.T)/\
                np.linalg.norm(y-y.mean(0)))
        print_fmt = 'matrix X:\n{}\nmatrix Y:\n{}\nmatrix Yp:\n{}\nmatrix R:\n{}\n' \
                   +'matrix T:\n{}\nerror: {}'

    return data_func, func, stat_func, print_fmt


def test_correctness(data_func, func, stat_func, print_fmt):

    datas = data_func(1)
    for data in zip(*datas):
        out = stat_func(*data, func(*data))
        print(print_fmt.format(*out))
    return


def benchmark_accuracy(data_func, func, stat_func):

    datas = data_func(N)
    err_sum = 0
    for data in zip(*datas):
        err_sum += stat_func(*data, func(*data))[-1]

    print('Average error over {} random samples: {}'.format(N, err_sum/N))

    return


def benchmark_speed(data_func, func):

    datas = data_func(N)

    start = timer()
    for data in zip(*datas):
        _ = func(*data)
    end = timer()

    print('Average execution time over {} random samples: {} us'.format(N, (end-start)/N*1e6))

    return


def test_method_correctness(method):
    print('Testing method: {}'.format(method))
    funcs = get_func(method)
    test_correctness(*funcs)
    return


def benchmark_method_accuracy(method):
    print('Benchmarking accuracy: {}'.format(method))
    funcs = get_func(method)
    benchmark_accuracy(*funcs[:-1])
    return


def benchmark_method_speed(method):
    print('Benchmarking speed: {}'.format(method))
    funcs = get_func(method)
    benchmark_speed(*funcs[:-2])
    return



if __name__ == '__main__':
    #test_method_correctness('EPPnP')
    benchmark_method_accuracy('EPPnP')
    benchmark_method_speed('EPPnP')
    #test_method_correctness('EAPPnP')
    benchmark_method_accuracy('EAPPnP')
    benchmark_method_speed('EAPPnP')



