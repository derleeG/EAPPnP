import matplotlib.pyplot as plt
import unit_test as test

if __name__ == '__main__':


    aniso_data_func, aniso_func, stat_func, _ = test.get_func('EAPPnP')
    iso_data_func, iso_func, _, _ = test.get_func('EPPnP')

    iso_time, aniso_time = [], []

    Mpoints = [10, *range(100, 2100, 100)]
    for M in Mpoints:
        test.M = M

        # for comparison we are testing on isotropic data
        datas = iso_data_func(test.N)

        start = test.timer()
        for data in zip(*datas):
            _ = aniso_func(*data)
        end = test.timer()

        aniso_time.append((end-start)/test.N*1e6)

        start = test.timer()
        for data in zip(*datas):
            _ = iso_func(*data)
        end = test.timer()

        iso_time.append((end-start)/test.N*1e6)

        print('N points: {}, aniso time: {:.3f}us, iso time: {:.3f}us'\
                .format(M, aniso_time[-1], iso_time[-1]))


    plt.scatter(Mpoints, aniso_time, marker='^', label='EAPPnP')
    plt.scatter(Mpoints, iso_time, marker='o', label='EPPnP')
    plt.xlabel("n points")
    plt.ylabel("execution time (us)")
    plt.legend()
    plt.show()



