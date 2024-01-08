import numpy as np
import open3d as o3d
# 读取npz文件
data2 = np.load('runs/graphcnn_norm/20231220-125201/predictions.npz')

# 打印npz文件的所有数据
for key in data2.files:
    print("Data under key '%s':" % key)
    print(data2[key])
# 使用numpy的genfromtxt函数读取txt文件
data = np.genfromtxt('Data\Surface_Defects_pcd_extend_2000_estnorm_noise0001/raised/raised10.txt', delimiter=',')
# 读取npz文件
data1 = np.load('Data\Surface_Defects_pcd_extend_2000_estnorm_noise0001/raised/raised10.npz')
labels = data1['data']  # 假设你的标签存储在'labels'字段下

# 检查数据的形状
print(data.shape)

# 如果你只关心前三列（假设它们是x, y, z坐标），你可以这样做：
points = data[:, :3]
# 创建一个颜色数组
colors = np.zeros(points.shape)
colors[labels == 0] = [1, 0, 0]  # 红色
colors[labels == 1] = [0, 0, 1]  # 蓝色
print(points.shape)
# 假设你的点云数据存储在一个形状为 (N, 3) 的 NumPy 数组中
# points = np.random.rand(N, 3)
# 创建一个点云对象
pcd = o3d.geometry.PointCloud()

# 设置点云的点和颜色
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# 可视化点云
o3d.visualization.draw_geometries([pcd])