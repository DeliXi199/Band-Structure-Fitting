import numpy as np
import matplotlib.pyplot as plt
import os

filename = os.path.join(os.path.dirname(__file__), "..", "Data", "mesh10.txt")
mesh10 = np.loadtxt(filename)

filename = os.path.join(os.path.dirname(__file__), "..", "Data", "mesh11.txt")
mesh11 = np.loadtxt(filename)

diff = mesh11 - mesh10

filename = os.path.join(os.path.dirname(__file__), "..", "Data", "wannierk.txt")
wannierk = np.loadtxt(filename)

# Create line plot
plt.figure(figsize=(10, 6))
plt.plot(mesh10[:, 0], "-b", label="mesh10", marker="o")
plt.plot(mesh10[:, 1], "-b", label="mesh10", marker="o")
plt.plot(mesh10[:, 2], "-b", label="mesh10", marker="o")
plt.plot(mesh11[:, 0], "-g", label="mesh11", marker="o")
plt.plot(mesh11[:, 1], "-g", label="mesh11", marker="o")
plt.plot(mesh11[:, 2], "-g", label="mesh11", marker="o")
plt.title("1D Array Line Plot")
plt.xlabel("Index")
plt.ylabel("Value")
plt.grid(True)
plt.legend()


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# 绘制所有点
ax.scatter(wannierk[:, 0], wannierk[:, 1], wannierk[:, 2], c="b", marker="o", s=10)


plt.show()
