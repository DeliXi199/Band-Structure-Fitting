import numpy as np
import os
import multiprocessing
import time


def fun(kpointsk, band_index, coes, star_R_len, star_R):
    result = 0
    for m in range(0, coes.shape[1]):
        star_R_m = star_R[m, : star_R_len[m], :]
        exp_factors = np.cos(np.dot(kpointsk, star_R_m.T))  # 向量化的计算
        result += (
            coes[band_index, m] * np.sum(exp_factors) / star_R_m.shape[0]
        )  # 使用NumPy的sum加速计算
        # if np.all(kpointsk == [0.3692308e-0, 0.0000000e00, 0.0000000e00]*2*np.pi):
        #     print(result)
    return result


def get_fermi_data(
    i, j, k, size, bands_fermi_level, Kijk, reciprocal_lattice, coes, star_R_len, star_R
):
    count = i * size**2 + j * size + k
    result = np.zeros(bands_fermi_level.shape[0])
    for band_index in range(bands_fermi_level.shape[0]):
        result[band_index] = fun(
            np.dot(Kijk, reciprocal_lattice), band_index, coes, star_R_len, star_R
        )
    return result, count, Kijk


if __name__ == "__main__":

    cores = 8

    bands_fermi_level = np.array([5, 6])

    reciprocal_lattice = np.array(
        [[1.38609868, 0.80043799, 0.0], [0.0, 1.60087599, 0.0], [0.0, -0.0, 0.53265389]]
    )

    filename = os.path.join(os.path.dirname(__file__), "..", "Data", "coes_Bi.txt")
    coes = np.loadtxt(filename)
    filename = os.path.join(
        os.path.dirname(__file__), "..", "Data", "star_R_len_Bi.txt"
    )
    star_R_len = np.loadtxt(filename, dtype=int)
    filename = os.path.join(os.path.dirname(__file__), "..", "Data", "star_R_Bi.npy")
    star_R = np.load(filename)

    size = 16
    # 生成从 -1 到 1 之间的 16 个均匀分布的数据
    x = np.linspace(0, 1, size + 1)
    y = np.linspace(0, 1, size + 1)
    z = np.linspace(0, 1, size + 1)

    # 使用 meshgrid 生成三维网格
    X, Y, Z = np.meshgrid(x, y, z)

    # 将 X, Y, Z 堆叠成一个三维数组
    ks = np.stack((X, Y, Z), axis=-1)
    fermi_data = np.zeros((size**3, 3 + bands_fermi_level.shape[0]))

    time_start = time.time()
    print("Start multiprocessing...")

    with multiprocessing.Pool(processes=cores) as pool:
        results = pool.starmap(
            get_fermi_data,
            [
                (
                    i,
                    j,
                    k,
                    size,
                    bands_fermi_level,
                    ks[i, j, k],
                    reciprocal_lattice,
                    coes,
                    star_R_len,
                    star_R,
                )
                for i in range(size)
                for j in range(size)
                for k in range(size)
            ],
        )
        for result, count, Kijk in results:
            fermi_data[count, 0:3] = Kijk
            fermi_data[count, 3:] = result
    time_end = time.time()
    print("Multiprocessing finished, time used:", time_end - time_start)

    np.savetxt(
        os.path.join(os.path.dirname(__file__), "..", "Data", "fermi_data_Bi.txt"),
        fermi_data,
    )
