import numpy as np
import time
import spglib
import multiprocessing
import os


def fun(kpointsk, band_index):
    result = 0
    for m in range(0, len(coe)):
        star_R_m = star_R[m, : star_R_len[m], :]
        exp_factors = np.cos(np.dot(kpointsk, star_R_m.T))  # 向量化的计算
        result += (
            coes[band_index, m] * np.sum(exp_factors) / star_R_m.shape[0]
        )  # 使用NumPy的sum加速计算
        # if np.all(kpointsk == [0.3692308e-0, 0.0000000e00, 0.0000000e00]*2*np.pi):
        #     print(result)
    return result


def velocity(kkk, band_index):
    result = np.zeros(3)
    for m in range(1, len(coe)):
        star_R_m = star_R[m, : star_R_len[m], :]
        # if m == 1:
        #     print(star_R_m)
        exp_factors = np.zeros(3)
        for l in range(star_R_m.shape[0]):
            exp_factors += (
                -np.sin(np.dot(kkk, star_R_m[l, :])) * star_R_m[l, :]
            )  # 向量化的计算
        result += (
            coes[band_index, m] * exp_factors / star_R_m.shape[0]
        )  # 使用NumPy的sum加速计算
        # print(result)
    return result


def remove_opposite_coordinates(coordinates):
    """
    去除互为相反数的坐标，只保留一半。

    Parameters:
    coordinates (np.ndarray): 输入的二维数组，包含坐标。

    Returns:
    np.ndarray: 去除相反数后的坐标数组。
    """
    # 创建一个哈希集合来记录已出现的坐标
    seen = set()
    result = []

    # 遍历坐标数组
    for coord in coordinates:
        # 将坐标的相反数表示为元组，以便存储到集合中
        neg_coord = tuple(-coord)

        # 如果相反数不存在于集合中，则将当前坐标添加到结果中，并标记其相反数已处理
        if neg_coord not in seen:
            result.append(coord)
            seen.add(tuple(coord))  # 添加当前坐标到集合中

    return np.array(result)


## 定义读取 EIGENVAL 文件的函数，读取文件中的 k 点和能量数据
def read_eigenval(filename):
    with open(filename, "r") as file:
        lines = file.readlines()

    # 获取电子数、k 点数和能带数
    num_electrons, num_kpoints, num_bands = map(int, lines[5].split())

    # 初始化存储 k 点和能量数据的列表
    kpoints = []
    energies = []

    index = 7  # 从第7行开始读取 k 点和能量数据
    for _ in range(num_kpoints):
        if not lines[index].strip():  # 跳过空行
            index += 1

        # 读取 k 点信息
        kpoint_data = list(map(float, lines[index].split()[:3]))
        kpoints.append(kpoint_data)
        index += 1  # 移到下一个位置，读取能带信息

        # 读取能带能量
        band_energies = []
        for _ in range(num_bands):
            if not lines[index].strip():  # 跳过空行
                index += 1

            band_energy = float(lines[index].split()[1])  # 提取能量值
            band_energies.append(band_energy)
            index += 1

        energies.append(band_energies)

    return kpoints, energies


## 读取 POSCAR 文件，获取晶胞的缩放因子和晶胞格矢量
def read_poscar(file_name):
    with open(file_name, "r") as f:
        lines = f.readlines()

    # 读取缩放因子
    scale_factor = float(lines[1].strip())

    # 读取晶胞格矢量
    lattice_vectors = []
    for i in range(2, 5):
        vector = list(map(float, lines[i].split()))
        lattice_vectors.append(vector)

    # 输出结果
    return scale_factor, lattice_vectors


## 计算倒易晶格
def compute_reciprocal_lattice(lattice):
    # 计算倒易晶格
    a1, a2, a3 = lattice
    volume = np.dot(a1, np.cross(a2, a3))  # 计算晶胞体积

    # 计算倒易晶格向量
    b1 = 2 * np.pi * np.cross(a2, a3) / volume
    b2 = 2 * np.pi * np.cross(a3, a1) / volume
    b3 = 2 * np.pi * np.cross(a1, a2) / volume

    return np.array([b1, b2, b3])


def get_symmetry_operations(
    cell, cell_reciprocal, lattice, reciprocal_lattice, kpoints
):
    ## 提取对称性操作
    dataset = spglib.get_symmetry_dataset(cell)
    rotations = dataset.rotations  # 获取旋转操作

    ## 获取实空间不可约网格
    number_mesh = 50  # 控制网格的分辨率
    mesh = [number_mesh, number_mesh, number_mesh]  # 网格大小 (你可以根据需要调整)

    ir_Rpoints = spglib.get_ir_reciprocal_mesh(mesh, cell_reciprocal)
    Rpoints_index = np.unique(ir_Rpoints[0])
    Rpoints_ir_end = ir_Rpoints[1][Rpoints_index]
    # 计算每个点到原点的欧几里得距离
    distances = np.linalg.norm(Rpoints_ir_end, axis=1)
    # 按照距离对数据排序
    sorted_indices = np.argsort(distances)  # 获取排序后的索引
    sorted_Rpoints = Rpoints_ir_end[sorted_indices]  # 根据排序后的索引重新排列数据
    Rpoints_end = sorted_Rpoints[0:M]

    ## 使用分数坐标
    kpoints = np.dot(kpoints, reciprocal_lattice)  # 转换坐标
    Rvec = np.dot(Rpoints_end, lattice)  # 转换坐标

    return rotations, kpoints, Rvec


def get_H(i, j, M, exppim, exppjm, distance):
    Hijm = 0.0
    for m in range(1, M):
        Hijm += exppim[m] * exppjm[m] / distance[m]
    return i, j, Hijm


## 计算get_distance(R)
def get_distance(coord):
    C1 = 2
    C2 = 0.75
    r = np.linalg.norm(coord)
    r = r / np.linalg.norm(lattice[0])
    result = (1 - C1 * r**2) ** 2 + C2 * r**6
    return result


def get_fermi_data():
    size = 16
    # 生成从 -1 到 1 之间的 16 个均匀分布的数据
    x = np.linspace(0, 1, size + 1)
    y = np.linspace(0, 1, size + 1)
    z = np.linspace(0, 1, size + 1)

    # 使用 meshgrid 生成三维网格
    X, Y, Z = np.meshgrid(x, y, z)

    # 将 X, Y, Z 堆叠成一个三维数组
    ks = np.stack((X, Y, Z), axis=-1)
    fermi_data = np.zeros(size**3)
    count = 0
    for i in range(size):
        for j in range(size):
            for k in range(size):
                fermi_data[count] = fun(
                    np.dot(ks[i, j, k], reciprocal_lattice), band_index
                )
                count += 1

    return fermi_data


def write_files():
    filename = os.path.join(os.path.dirname(__file__), "..", "Data", "coes.txt")
    np.savetxt(filename, coes)
    filename = os.path.join(os.path.dirname(__file__), "..", "Data", "star_R_len.txt")
    np.savetxt(filename, star_R_len, fmt="%d")
    filename = os.path.join(os.path.dirname(__file__), "..", "Data", "star_R.txt")
    np.savetxt(filename, star_R.reshape(-1, star_R.shape[-1]))
    filename = os.path.join(os.path.dirname(__file__), "..", "Data", "star_R.npy")
    np.save(filename, star_R)
    filename = os.path.join(os.path.dirname(__file__), "..", "Data", "Rvec.txt")
    np.savetxt(filename, Rvec)
    filename = os.path.join(os.path.dirname(__file__), "..", "Data", "NMS.txt")
    np.savetxt(
        filename, np.array([N, M, rotations.shape[0], Nband_Fermi_Level]), fmt="%d"
    )
    filename = os.path.join(os.path.dirname(__file__), "..", "Data", "H.txt")
    np.savetxt(filename, H)
    filename = os.path.join(os.path.dirname(__file__), "..", "Data", "expp.txt")
    np.savetxt(filename, expp)
    filename = os.path.join(os.path.dirname(__file__), "..", "Data", "k_v.txt")
    np.savetxt(filename, kpoints / 2 / np.pi)
    filename = os.path.join(os.path.dirname(__file__), "..", "Data", "fermi_data.txt")
    np.savetxt(filename, fermi_data)


if __name__ == "__main__":
    ## 用于计算某个kpoints点的能量

    time_start = time.time()

    cores = 8

    ## 从文件中读数据，并将数据转换为np数组
    filename = os.path.join(os.path.dirname(__file__), "..", "Data", "EIGENVAL_fit")
    kpoints, energies = read_eigenval(filename)
    kpoints = np.array(kpoints)
    energies = np.array(energies)

    # fermi_level = 7.7083
    fermi_level = -1.3888
    Nband_Fermi_Level = 0
    bands_fermi_level = []
    for i in range(energies.shape[1]):
        if min(energies[:, i]) < fermi_level and max(energies[:, i]) > fermi_level:
            Nband_Fermi_Level += 1
            bands_fermi_level.append(i + 1)
    bands_fermi_level = np.array(bands_fermi_level)

    filename = os.path.join(os.path.dirname(__file__), "..", "Data", "POSCAR")
    scale, lattice = read_poscar(filename)
    lattice = np.array(lattice)

    reciprocal_scale = 1 / scale
    reciprocal_lattice = compute_reciprocal_lattice(lattice)
    reciprocal_lattice = np.array(reciprocal_lattice)

    N = len(kpoints)  # 用于拟合的数据点数
    M = round(N * 4)  # 用于拟合的基函数数
    print("N:", N)
    ## 获取晶体的空间群信息，
    positions = [[0.0, 0.0, 0.0]]  # 原子位置，仍然是列表
    numbers = [14]  # Si 的原子编号，仍然是列表
    # 将这些值合成一个元组传递给 spglib
    cell = (lattice, positions, numbers)
    cell_reciprocal = (reciprocal_lattice, positions, numbers)

    ## 使用分数坐标
    reciprocal_lattice = np.eye(3) * 2 * np.pi
    lattice = np.eye(3)

    rotations, kpoints, Rvec = get_symmetry_operations(
        cell, cell_reciprocal, lattice, reciprocal_lattice, kpoints
    )

    print("initialize done")

    ## 计算转移矩阵
    H = np.zeros((N - 1, N - 1))

    star_Rpoints = np.zeros((M, rotations.shape[0], 3))
    for m in range(M):
        for r in range(rotations.shape[0]):
            star_Rpoints[m, r, :] = np.dot(rotations[r], Rvec[m, :])
    print("star_Rpoints done")

    expp = np.zeros((N, M))
    distance = np.zeros((M))
    star_R = np.zeros((M, rotations.shape[0], 3))
    for m in range(M):
        for r in range(rotations.shape[0]):
            star_R[m, r, :] = np.dot(star_Rpoints[m, r, :], lattice)
    star_R_temp = np.zeros((M, rotations.shape[0], 3))
    star_R_len = np.zeros(M, dtype=int)
    for m in range(M):
        star_R_m = star_R[m, :, :]
        star_R_m = np.unique(star_R_m, axis=0)
        star_R_m = remove_opposite_coordinates(star_R_m)
        star_R_temp[m, : len(star_R_m), :] = star_R_m
        star_R_len[m] = len(star_R_m)

    star_R = star_R_temp
    print("star_R done")

    for m in range(1, M):
        distance[m] = get_distance(Rvec[m, :])
        star_R_m = star_R[m, : star_R_len[m], :]
        # star_R_m = star_R[m, :, :]
        # star_R_m = np.unique(star_R_m, axis=0)
        for n in range(N):
            for r in range(star_R_m.shape[0]):
                expp[n, m] += (
                    np.cos(np.dot(kpoints[n, :], star_R_m[r, :])) / star_R_m.shape[0]
                )
    for m in range(1):
        distance[m] = get_distance(Rvec[m, :])
        star_R_m = star_R[m, : star_R_len[m], :]
        # star_R_m = star_R[m, :, :]
        # star_R_m = np.unique(star_R_m, axis=0)
        for n in range(N):
            for r in range(star_R_m.shape[0]):
                expp[n, m] += (
                    np.cos(np.dot(kpoints[n, :], star_R_m[r, :])) / star_R_m.shape[0]
                )
    print("expp done")

    exppi = np.zeros((N, M))
    exppj = np.zeros((N, M))
    for m in range(1, M):
        exppi[:, m] = expp[:, m] - expp[-1, m]
    exppj = np.conj(exppi)

    with multiprocessing.Pool(processes=cores) as pool:
        results = pool.starmap(
            get_H,
            [
                (
                    i,
                    j,
                    M,
                    exppi[i, :],
                    exppj[j, :],
                    distance,
                )
                for i in range(N - 1)
                for j in range(i + 1)
            ],
        )
        for i, j, Hijm in results:
            H[i, j] = Hijm
            H[j, i] = np.conj(H[i, j])
    print("H done")
    # # 检查矩阵是否可逆（行列式是否为非零）
    # det = np.linalg.det(H)
    # if det != 0:
    #     # 计算矩阵的逆
    #     inverse_H = np.linalg.inv(H)
    # else:
    #     print("\n此矩阵不可逆(行列式为零)。")
    #     print(H)

    # epsilons = np.arange(0, N, 1)

    coes = np.zeros((Nband_Fermi_Level, M))

    band_index = 0
    for iband in bands_fermi_level:
        print("iband:", iband)
        eV2Hartree = 27.211385

        epsilons = (energies[:, iband - 1] - fermi_level) / eV2Hartree

        delta_epsilons = epsilons[0 : N - 1] - epsilons[-1]

        # lam = np.dot(inverse_H, delta_epsilons)
        lam = np.linalg.solve(H, delta_epsilons)
        print("lam done")

        # 拟合的基矢前面的系数
        coe = np.zeros(M)
        for i in range(1, M):
            for j in range(N - 1):
                coe[i] += (
                    (np.conj(expp[j, i]) - np.conj(expp[-1, i]))
                    * lam[j]
                    / get_distance(Rvec[i, :])
                )

        sum_es = 0
        for i in range(1, M):
            sum_es += coe[i] * expp[-1, i]
        coe[0] = epsilons[-1] - sum_es

        coes[band_index, :] = coe

        energy = np.zeros(N)
        for i in range(N):
            energy[i] = fun(kpoints[i, :], band_index)

        print("minus", epsilons - energy)

    print("mean star_R_len:", np.mean(star_R_len))

    time_end = time.time()
    print("Time used:", time_end - time_start)
    # print("energy:", energy)
    # print("epsilons:", epsilons)

    ## 获取画费米面的数据
    # fermi_data = get_fermi_data()

    # write_files()

    time_end = time.time()
    print("Time used:", time_end - time_start)
