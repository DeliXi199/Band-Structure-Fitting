import spglib
import numpy as np

# 定义 Bi 晶格结构（单位元胞）
lattice = np.array([[4.54, 0, 0], [-2.27, 3.92, 0], [0, 0, 11.86]])  # 近似 Bi 晶格参数
positions = np.array([[0, 0, 0], [0.234, 0.234, 0.234]])  # 原子坐标
numbers = [83, 83]  # Bi 原子序数

cell = (lattice, positions, numbers)
sym_data = spglib.get_symmetry_dataset(cell)

# 输出所有对称操作
for rot in sym_data.rotations:
    print(rot)
