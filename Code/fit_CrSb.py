import numpy as np
import time
import spglib
import multiprocessing
import os
import re
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.spatial import cKDTree
import subprocess

def fun(kpointsk, band_index):
    result = 0
    for m in range(0, len(coe)):
        star_R_m = star_R[m, : star_R_len[m], :]
        exp_factors = np.exp(-1j * np.dot(kpointsk, star_R_m.T))  # 向量化的计算
        result += (
            coes[band_index, m] * np.sum(exp_factors) / star_R_m.shape[0]
        )  # 使用NumPy的sum加速计算
        # if np.all(kpointsk == [0.3692308e-0, 0.0000000e00, 0.0000000e00]*2*np.pi):
        #     print(result)
    return result.real


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

    # 读取元素类型和数量
    element_names = lines[5].split()
    element_counts = list(map(int, lines[6].split()))

    total_atoms = sum(element_counts)

    # 读取坐标
    coord_start = 8  # Direct 坐标从第 9 行开始
    positions = []
    for i in range(total_atoms):
        pos = list(map(float, lines[coord_start + i].split()[:3]))
        positions.append(pos)

    # 不映射到元素周期表编号，只编号为 1, 2, 3,...
    numbers = []
    for idx, count in enumerate(element_counts):
        numbers.extend([idx + 1] * count)  # Cr = 1, Sb = 2

    return scale_factor, lattice_vectors, positions, numbers



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

    # 读取元素类型和数量
    element_names = lines[5].split()
    element_counts = list(map(int, lines[6].split()))

    total_atoms = sum(element_counts)

    # 读取坐标
    coord_start = 8  # Direct 坐标从第 9 行开始
    positions = []
    for i in range(total_atoms):
        pos = list(map(float, lines[coord_start + i].split()[:3]))
        positions.append(pos)

    # 不映射到元素周期表编号，只编号为 1, 2, 3,...
    numbers = []
    for idx, count in enumerate(element_counts):
        numbers.extend([idx + 1] * count)  # Cr = 1, Sb = 2

    return scale_factor, lattice_vectors, positions, numbers



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


def canonical_form(p, rotations, A, A_inv, tol=1e-6):
    """
    对于晶格点 p（笛卡尔坐标），先转换到分数坐标，
    对所有旋转操作（分数坐标下）计算 p 的轨道，
    再将旋转后的分数坐标转换回笛卡尔坐标，
    将结果四舍五入后，取字典序最小的表示作为标准形式。
    """
    # 转换到分数坐标
    f = A_inv @ p
    orbit = []
    for R in rotations:
        f_rot = R @ f
        p_rot = A @ f_rot
        orbit.append(tuple(np.round(p_rot, 6)))
    return min(orbit)

def orbit_size(p, rotations, A, A_inv, tol=1e-6):
    """
    计算晶格点 p 在所有旋转操作下的轨道大小
    """
    f = A_inv @ p
    orbit_set = {tuple(np.round(A @ (R @ f), 6)) for R in rotations}
    return len(orbit_set)


def get_irmesh(sym_filename):
    # 这里假设 parse_symmetry_operations、canonical_form 和 orbit_size 是已定义的函数
    rotations = parse_symmetry_operations(sym_filename)
    print("加载旋转操作个数：", len(rotations))

    # 构造晶胞矩阵 A（列为晶格向量）

    # 3. Generate all integer grid points
    index_range = 12  # 网格范围
    tol = 1e-6  # 精度
    idx = np.arange(-index_range, index_range + 1)
    grid = np.vstack(np.meshgrid(idx, idx, idx, indexing='ij')).reshape(3, -1).T
    M = grid.shape[0]
    
    # 4. Build mapping and index lookup
    mapping = np.arange(M, dtype=int)
    index_map = {tuple(grid[i]): i for i in range(M)}
    
    # 5. Build full rotation list
    full_rots = []
    for R in rotations:
        full_rots.append(R)
    
    # 6. Find irreducible representative for each point
    for i in range(M):
        # Cartesian coordinate of point i
        if mapping[i] == i:
            for R in full_rots:
                frac = R @ grid[i]
                idx_r = np.rint(frac).astype(int)
                # Check if exactly matches within tolerance
                if not np.allclose(frac, idx_r, atol=tol):
                    continue
                j = index_map.get(tuple(idx_r))
                if j is not None and j > i:
                    mapping[j] = i

    
    # 7. Extract inequivalent (representative) points
    reps = np.unique(mapping)
    unique_grid = grid[reps]
    return rotations,unique_grid






def get_symmetry_operations(lattice, kpoints):
    ## 提取对称性操作
    sym_filename = os.path.join(os.path.dirname(__file__), "..", "Data", "rotation_Cu.txt")
    time_end1 = time.time()
    rotations,  ir_Rpoints= get_irmesh(sym_filename)
    time_end2 = time.time()
    print("加载对称性操作耗时：", time_end2 - time_end1)

    ## 获取实空间不可约网格
    # ir_Rpoints = spglib.get_ir_reciprocal_mesh(mesh, cell_reciprocal)
    inv_lattice = np.linalg.inv(lattice)
    # 计算每个点到原点的欧几里得距离
    ir_Rpoints_cart = np.dot(ir_Rpoints, lattice)
    distances = np.linalg.norm(ir_Rpoints_cart, axis=1)
    # 按照距离排序，获取排序后的索引
    sorted_indices = np.argsort(distances)
    # 根据排序后的索引重新排列数据
    sorted_Rpoints = ir_Rpoints[sorted_indices]
    # 选择前 M 个点作为最终结果（确保 M 已定义）
    Rvec = sorted_Rpoints[0:M]
    ## 使用分数坐标进行转换
    # 假设 kpoints, reciprocal_lattice, lattice 已经定义
    kpoints = kpoints*2*np.pi # 转换坐标
    # Rvec = np.dot(Rpoints_end, lattice)            # 转换坐标

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


def fun_fermi(kpointsk, band_index, star_R, star_R_len, coes):
    result = 0
    # coes.shape 应为 (n_bands, n_shells)
    for m in range(star_R.shape[0]):
        # star_R[m,:,:] 中只有前 star_R_len[m] 行有效
        Rm = star_R[m, : star_R_len[m], :]
        exp_factors = np.exp(-1j * np.dot(kpointsk, Rm.T))
        result += coes[band_index, m] * np.sum(exp_factors) / Rm.shape[0]
    return result.real

def _compute_fermi_point(args):
    k_point, band_index, reciprocal_lattice, star_R, star_R_len, coes = args
    kdotR = np.dot(k_point, reciprocal_lattice)
    return fun_fermi(kdotR, band_index, star_R, star_R_len, coes)

def get_fermi_data(bands_fermi_level,
                   reciprocal_lattice,
                   star_R,
                   star_R_len,
                   coes,
                   band_index=3,
                   size=16,
                   nprocs=8):
    # 1) 生成网格并展平
    x = np.linspace(0, 1, size + 1)
    y = np.linspace(0, 1, size + 1)
    z = np.linspace(0, 1, size + 1)
    X, Y, Z = np.meshgrid(x[0:size], y[0:size], z[0:size], indexing='ij')
    ks = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)  # shape = (4913, 3)

    # 2) 准备参数，顺序要和 _compute_fermi_point 的解包一致
    tasks = [
        (k, band_index, reciprocal_lattice, star_R, star_R_len, coes)
        for k in ks
    ]

    # 3) 并行计算
    with multiprocessing.Pool(processes=nprocs) as pool:
        fermi_data = pool.map(_compute_fermi_point, tasks)

    # 4) 返回正确长度的数组
    return np.array(fermi_data)


def coes_python2fortran():
    # 构造文件路径
    filename = os.path.join(os.path.dirname(__file__), "..", "Data", "coes_Cu.txt")
    filename2 = os.path.join(os.path.dirname(__file__), "..", "Data", "coes_Cu2.txt")

    # 改进后的复数匹配：匹配科学计数法格式的复数
    complex_pattern = re.compile(
        r"\(\s*([+-]?\d+(?:\.\d+)?[Ee][+-]?\d+)\s*([+-]\d+(?:\.\d+)?[Ee][+-]?\d+)j\s*\)"
    )

    with open(filename) as fin, open(filename2, "w") as fout:
        for line in fin:
            # 替换为 Fortran 格式 (real, imag)
            line_out = complex_pattern.sub(r"(\1,\2)", line)
            fout.write(line_out)



def write_files():
    filename = os.path.join(os.path.dirname(__file__), "..", "Data", "coes_Cu.txt")
    np.savetxt(filename, coes)
    filename = os.path.join(
        os.path.dirname(__file__), "..", "Data", "star_R_len_Cu.txt"
    )
    np.savetxt(filename, star_R_len, fmt="%d")
    filename = os.path.join(os.path.dirname(__file__), "..", "Data", "star_R_Cu.txt")
    np.savetxt(filename, star_R.reshape(-1, star_R.shape[-1]))
    filename = os.path.join(os.path.dirname(__file__), "..", "Data", "star_R_Cu.npy")
    np.save(filename, star_R)
    filename = os.path.join(os.path.dirname(__file__), "..", "Data", "Rvec_Cu.txt")
    np.savetxt(filename, Rvec)
    filename = os.path.join(os.path.dirname(__file__), "..", "Data", "NMS_Cu.txt")
    np.savetxt(
        filename, np.array([N, M, rotations.shape[0], Nband_Fermi_Level]), fmt="%d"
    )
    filename = os.path.join(os.path.dirname(__file__), "..", "Data", "H_Cu.txt")
    np.savetxt(filename, H)
    filename = os.path.join(os.path.dirname(__file__), "..", "Data", "expp_Cu.txt")
    np.savetxt(filename, expp)
    # filename = os.path.join(os.path.dirname(__file__), "..", "Data", "k_v_Cu.txt")
    # np.savetxt(filename, kpoints / 2 / np.pi)
    if 'fermi_data' in locals() or 'fermi_data' in globals():
        filename = os.path.join(os.path.dirname(__file__), "..", "Data", "fermi_data_Cu.txt")
        np.savetxt(filename, fermi_data)



def parse_symmetry_operations(filename):
    """
    解析文件中所有对称操作，只提取 unitary（纯空间）操作的 rotation 和 translation。
    忽略注释行，只处理每个 '#  idx   flag' 后面的 3 行矩阵数据。

    参数:
        filename (str): 文件路径

    返回:
        rotations: np.ndarray, shape=(N,3,3)
            仅包含 flag=0 的旋转矩阵
        translations: np.ndarray, shape=(N,3)
            仅包含 flag=0 的平移向量
    """
    rotations = []
    translations = []

    header_pattern = re.compile(r"#\s*(\d+)\s+([01])")  # 匹配 '#  idx   flag'

    with open(filename, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # 跳过无关注释
        if line.startswith("#symm"):
            i += 1
            continue

        # 匹配 '#  idx   flag'
        m = header_pattern.match(line)
        if m:
            flag = int(m.group(2))
            # 读取接下来的三行数据
            if i + 3 < len(lines):
                rot = []
                trans = []
                for j in range(1, 4):
                    parts = lines[i + j].strip().split()
                    # 前三列是旋转矩阵整数，第四列是平移量浮点数
                    rot.append([int(parts[0]), int(parts[1]), int(parts[2])])
                    trans.append(float(parts[3]))
                if flag == 0:
                    rotations.append(np.array(rot, dtype=int))
                    translations.append(np.array(trans, dtype=float))
                # 跳过这三行
                i += 4
                continue

        # 其他行跳过
        i += 1

    rotations = np.array(rotations)       # shape (N, 3, 3)
    translations = np.array(translations) # shape (N, 3)
    return rotations




if __name__ == "__main__":
    ## 用于计算某个kpoints点的能量

    time_start = time.time()

    cores = 8

    ## 从文件中读数据，并将数据转换为np数组
    filename = os.path.join(os.path.dirname(__file__), "..", "Data", "EIGENVAL_fit_Cu")
    kpoints, energies = read_eigenval(filename)
    kpoints = np.array(kpoints)
    energies = np.array(energies)
    # 8.0758 15
    # 8.0700 13
    # 8.0657 17
    # 8.0636 19
    fermi_level =  8.0636
    Nband_Fermi_Level = 0
    bands_fermi_level = []
    for i in range(energies.shape[1]):
        if min(energies[:, i]) < fermi_level and max(energies[:, i]) > fermi_level:
            Nband_Fermi_Level += 1
            bands_fermi_level.append(i + 1)
    bands_fermi_level = np.array(bands_fermi_level)

    print("Nband_Fermi_Level:", Nband_Fermi_Level)

    filename = os.path.join(os.path.dirname(__file__), "..", "Data", "POSCAR_Cu")
    scale, lattice, positions, numbers = read_poscar(filename)
    lattice = np.array(lattice)

    reciprocal_scale = 1 / scale
    reciprocal_lattice = compute_reciprocal_lattice(lattice)
    reciprocal_lattice = np.array(reciprocal_lattice)

    print("reciprocal_lattice:", reciprocal_lattice)

    N = len(kpoints)  # 用于拟合的数据点数
    M = round(N * 2)  # 用于拟合的基函数数
    print("N:", N)
    ## 获取晶体的空间群信息，
    # 2. 原子坐标（6 个原子，Direct 格式）
    # positions = np.array([
    #     [0.0, 0.0, 0.0],               # Cr
    #     [0.0, 0.0, 0.5],               # Cr
    #     [0.33333333, 0.66666667, 0.25],# Sb
    #     [0.66666667, 0.33333333, 0.75] # Sb
    # ])

    # # 3. 原子种类序号（W = 74, P = 15）
    # numbers = [24, 24, 51, 51]
    # 将这些值合成一个元组传递给 spglib
    cell = (lattice, positions, numbers)
    cell_reciprocal = (reciprocal_lattice, positions, numbers)

    ## 使用分数坐标

    rotations, kpoints, Rvec = get_symmetry_operations(lattice, kpoints)
    reciprocal_lattice = np.eye(3) * 2 * np.pi
    lattice = np.eye(3)
    

    # print("rotations:", rotations)


    print("initialize done")

    ## 计算转移矩阵
    H = np.zeros((N - 1, N - 1), dtype=complex)

    star_Rpoints = np.zeros((M, rotations.shape[0], 3))
    for m in range(M):
        for r in range(rotations.shape[0]):
            star_Rpoints[m, r, :] = np.dot(rotations[r], Rvec[m, :])
    print("star_Rpoints done")

    expp = np.zeros((N, M), dtype=complex)
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
        # star_R_m = remove_opposite_coordinates(star_R_m)
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
                    np.exp(-1j * np.dot(kpoints[n, :], star_R_m[r, :])) / star_R_m.shape[0]
                )
    for m in range(1):
        distance[m] = get_distance(Rvec[m, :])
        star_R_m = star_R[m, : star_R_len[m], :]
        # star_R_m = star_R[m, :, :]
        # star_R_m = np.unique(star_R_m, axis=0)
        for n in range(N):
            for r in range(star_R_m.shape[0]):
                expp[n, m] += (
                    np.exp(-1j * np.dot(kpoints[n, :], star_R_m[r, :])) / star_R_m.shape[0]
                )
    print("expp done")

    exppi = np.zeros((N, M),dtype=complex)
    exppj = np.zeros((N, M),dtype=complex)
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

    coes = np.zeros((Nband_Fermi_Level, M), dtype=complex)
    print("bands_fermi_level:", bands_fermi_level)

    band_index = 0
    for iband in bands_fermi_level:
        print("iband:", iband)
        eV2Hartree = 1 / 27.211385

        epsilons = (energies[:, iband - 1] - fermi_level) * eV2Hartree

        delta_epsilons = epsilons[0 : N - 1] - epsilons[-1]

        # lam = np.dot(inverse_H, delta_epsilons)
        lam = np.linalg.solve(H, delta_epsilons)
        print("lam done")

        # 拟合的基矢前面的系数
        coe = np.zeros(M, dtype=complex)
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

        band_index += 1

        print("band_index:", band_index)
    print("mean star_R_len:", np.mean(star_R_len))

    time_end = time.time()
    print("Time used:", time_end - time_start)
    # print("energy:", energy)
    # print("epsilons:", epsilons)

    # 获取画费米面的数据
    # fermi_data = get_fermi_data(bands_fermi_level,reciprocal_lattice,star_R,star_R_len,coes,
    #     band_index=0,size=40,nprocs=8)


    write_files()
    coes_python2fortran()

    time_end = time.time()
    print("Time used:", time_end - time_start)
    # filename = os.path.join(os.path.dirname(__file__), "..", "Code", "plot_fitten_bands_CrSb.py")
    # subprocess.run(["python", filename])

