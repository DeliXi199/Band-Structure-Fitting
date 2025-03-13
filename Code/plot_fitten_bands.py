import matplotlib.pyplot as plt
import numpy as np
import os

N = 40
number_of_bands = 6
lowest_band = 3
fermi_level = -1.3875


# 定义读取 EIGENVAL 文件的函数
def read_eigenval(filename="EIGENVAL"):
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


# 读取 EIGENVAL 文件的数据
filename = os.path.join(os.path.dirname(__file__), "..", "Data", "EIGENVAL_band")
kpoints, energies = read_eigenval(filename)


trans_k = np.array([[1, 1, -1], [-1, 1, 1], [1, -1, 1]]) * np.pi
trans_k = np.eye(3) * 2 * np.pi  # 转换到三维坐标系
kpoints = np.dot(kpoints, trans_k)  # 转换到三维坐标系

num_delta_e = round(len(energies) / N - 1)
indices_to_delete = np.zeros(num_delta_e, dtype=int)
for i in range(num_delta_e):
    indices_to_delete[i] = i * N - 1

## 去除连接点处的重复数据
kpoints = np.delete(kpoints, indices_to_delete, axis=0)
energies = np.delete(energies, indices_to_delete, axis=0)


# 将 k 点序列转换为一维数组用于绘图
k_distances = np.arange(kpoints.shape[0])

# 绘制能带结构图，仅选择前 4 条能带
plt.figure(figsize=(10, 6))
for band in range(number_of_bands):  # 仅绘制前 4 条能带
    band_energies = [energies[k][band] for k in range(len(kpoints))]
    plt.plot(k_distances, band_energies, color="b")

plt.xlabel("k-point Index")
plt.ylabel("Energy (eV)")
plt.title("Band Structure (Lowest 4 Bands)")
plt.grid(True)

# 将 k 点和能量数据转换为可用于绘图的格式
k_x = [kp[0] for kp in kpoints]
k_y = [kp[1] for kp in kpoints]
k_z = [kp[2] for kp in kpoints]

# 选择第一个能带进行 3D 能量分布图绘制
energy_band = [energies[k][0] for k in range(len(kpoints))]

# 绘制三维能量分布图
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection="3d")

# 使用散点图展示 k 空间中的能量分布
sc = ax.scatter(k_x, k_y, k_z, c=energy_band, cmap="viridis")

# 添加颜色条和标签
cbar = plt.colorbar(sc)
cbar.set_label("Energy (eV)")
ax.set_xlabel("k_x")
ax.set_ylabel("k_y")
ax.set_zlabel("k_z")
ax.set_title("Energy Distribution in k-space")

filename = os.path.join(os.path.dirname(__file__), "..", "Data", "coes.txt")
coe = np.loadtxt(filename, dtype=complex)
filename = os.path.join(os.path.dirname(__file__), "..", "Data", "Rvec.txt")
Rvec = np.loadtxt(filename)
filename = os.path.join(os.path.dirname(__file__), "..", "Data", "star_R.npy")
star_R = np.load(filename)
filename = os.path.join(os.path.dirname(__file__), "..", "Data", "star_R_len.txt")
star_R_len = np.loadtxt(filename, dtype=int)
filename = os.path.join(os.path.dirname(__file__), "..", "Data", "H.txt")
H = np.loadtxt(filename)
filename = os.path.join(os.path.dirname(__file__), "..", "Data", "expp.txt")
expp = np.loadtxt(filename)


def C(coord):
    C1 = 2
    C2 = 0.75
    r = np.linalg.norm(coord)
    # r = r / np.linalg.norm(R1 + R2 + R3)
    result = (1 - C1 * r**2) ** 2 + C2 * r**6
    return result


## 用于计算某个kk点的能量
def fun(kpointsk):
    result = 0
    for m in range(1, len(coe)):
        star_R_m = star_R[m, : star_R_len[m], :]
        exp_factors = np.cos(np.dot(kpointsk, star_R_m.T))  # 向量化的计算
        result += (
            coe[m] * np.sum(exp_factors) / star_R_m.shape[0]
        )  # 使用NumPy的sum加速计算
    for m in range(1):
        star_R_m = star_R[m, : star_R_len[m], :]
        exp_factors = np.cos(np.dot(kpointsk, star_R_m.T))  # 向量化的计算
        result += (
            coe[m] * np.sum(exp_factors) / star_R_m.shape[0]
        )  # 使用NumPy的sum加速计算
    return result


filename = os.path.join(os.path.dirname(__file__), "..", "Data", "EIGENVAL_fit")
_, energies_fit = read_eigenval(filename)
energies_fit = np.array(energies_fit)


def get_coe(band_index):
    N = expp.shape[0]
    M = expp.shape[1]
    epsilons = energies_fit[:, band_index]

    delta_epsilons = epsilons[0 : N - 1] - epsilons[-1]

    # lam = np.dot(inverse_H, delta_epsilons)
    lam = np.linalg.solve(H, delta_epsilons)
    print(band_index, "lam done")

    # 拟合的基矢前面的系数

    coe = np.zeros(M, dtype=complex)
    for i in range(1, M):
        for j in range(N - 1):
            coe[i] += (
                (np.conj(expp[j, i]) - np.conj(expp[-1, i])) * lam[j] / C(Rvec[i, :])
            )

    sum_es = 0
    for i in range(1, M):
        sum_es += coe[i] * expp[-1, i]
    coe[0] = epsilons[-1] - sum_es

    return coe


energies_new = np.zeros((number_of_bands, (len(kpoints))))
for j in range(lowest_band - 1, number_of_bands):
    coe = get_coe(j)
    for i in range(len(kpoints)):
        energies_new[j, i] = fun(kpoints[i, :])


plt.figure(figsize=(8, 6))

# 绘制原始能带（VASP计算的能带）
for i in range(lowest_band - 1, number_of_bands):
    plt.plot(
        k_distances,
        energies[:, i] - fermi_level,
        color="#1f77b4",
        lw=2,
        label="VASP Calculated Bands" if i == 0 else "",
    )  # 蓝色实线，仅第一个线条加上标签

# 绘制修改后的能带（SKW方法拟合得到的能带）
for i in range(lowest_band - 1, number_of_bands):
    plt.plot(
        k_distances,
        energies_new[i, :] - fermi_level,
        color="#ff7f0e",
        lw=2,
        linestyle="--",
        label="SKW Fitted Bands" if i == 0 else "",  # 橙色虚线，仅第一个线条加上标签
    )

# 添加图例
plt.legend(loc="upper right")

# 绘制费米能量标线
plt.axhline(0, color="r", linestyle="--", lw=2)  # 黑色虚线标记费米能量
# 在费米能量位置添加标签，明确说明是费米能量
plt.text(
    k_distances[len(k_distances) - 1],  # 让标签在中间位置显示
    0,  # 能量为0的位置
    r"$E_F$",  # 费米能标签
    color="r",  # 黑色标签
    ha="center",  # 标签居中
    va="bottom",  # 标签位置稍微在费米能量上方
    fontsize=14,
)

# 假设这些是高对称点在 k_distances 中的索引
high_symmetry_indices = np.arange(0, 39 * 5, 39)
# 对应的高对称点标签
high_symmetry_labels = ["X", "W", "L", "Γ", "X", "K", "U", "Γ"]

# 循环添加高对称点标记
for index, label in zip(high_symmetry_indices, high_symmetry_labels):
    high_symmetry_k = k_distances[index]
    plt.axvline(
        x=high_symmetry_k, color="#2ca02c", linestyle="--", lw=1
    )  # 绿色虚线标记高对称点
    plt.text(
        high_symmetry_k,
        plt.ylim()[0],
        label,
        color="#2ca02c",
        ha="center",
        va="top",
        fontsize=14,
    )  # 添加标签，绿色

# 设置坐标轴标签和标题
# plt.xlabel("k-point Index", fontsize=14)
plt.ylabel("Energy (eV)", fontsize=14)
plt.title(f"Band Structure (Lowest {number_of_bands} Bands)", fontsize=16)


# 设置网格
plt.grid(True)

# 隐藏x轴的数字刻度
plt.xticks([])

# 显示图形
plt.show()


# # 定义 k 空间的范围和分辨率
# kx, ky, kz = np.mgrid[-1:1:50j, -1:1:50j, -1:1:50j]

# energies_fermi = np.zeros((len(kx), len(ky), len(kz)))

# # 计算能带在 k 空间的值
# for i in range(len(kx)):
#     for j in range(len(ky)):
#         for k in range(len(kz)):
#             kkk = [kx[i, j, k], ky[i, j, k], kz[i, j, k]]
#             energies_fermi[i, j, k] = fun(kkk)

# E_F = 2
# # 绘制等值面，值为费米能量 E_F
# contour = mlab.contour3d(kx, ky, kz, energies_fermi, contours=[E_F], opacity=0.5)

# mlab.colorbar(contour, title="Energy", orientation="vertical")
# mlab.xlabel("kx")
# mlab.ylabel("ky")
# mlab.zlabel("kz")
# mlab.title("Fermi Surface")
# mlab.show()
