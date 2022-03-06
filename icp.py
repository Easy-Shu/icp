import matplotlib.pyplot as plt
import torch
from pytorch3d.io import load_obj, save_obj
import numpy as np


DEVICE = torch.device("cuda:6") if torch.cuda.is_available() else torch.device("cpu")


def read_obj(objFilePath):
    """
    obj格式有4种数据，分别以一下字母开头：
　　１.　v顶点
　　２.　vt纹理坐标
　　３.　vn顶点法向量
　　４.　f 面
    :param objFilePath:
    :return: points
    """
    with open(objFilePath) as file:
        points = []
        while 1:
            line = file.readline()
            if not line:
                break
            string = line.split(" ")
            if string[0] == "v":
                points.append((float(string[1]), float(string[2]), float(string[3])))
            if string[0] == "vt":
                break
    # points原本为列表，需要转变为矩阵，方便处理
    points = torch.tensor(points)
    return points


# 在pt_set点集中找到距(p_x，p_y)最近点的id
def get_closest_index(point_, set_):
    dist = set_ - point_.unsqueeze(1)
    dist = dist * dist
    dist = torch.sum(dist, dim=0)
    idx = torch.argmin(dist)
    min_dist = torch.sqrt(dist[idx])
    return idx, min_dist


# 求两个点集之间的平均点距
def get_set_dist(set1, set2):
    dists = 0
    for i in range(set1.shape[1]):
        _, dist = get_closest_index(set1[:, i], set2)
        dists = dists + dist
    return dists / set1.shape[1]


def plot_pointcloud(points, title=""):
    from mpl_toolkits.mplot3d import Axes3D

    # Sample points uniformly from the surface of the mesh.
    x, y, z = points.cpu()
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.scatter3D(x, z, -y)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title(title)
    ax.view_init(190, 220)
    plt.show()


def ICP(source, target):

    # 1. init settings
    source = source.to(DEVICE).T
    target = target.to(DEVICE).T

    # mean
    source_mean = torch.tensor([source[0].mean(), source[1].mean(), source[2].mean()]).to(DEVICE).unsqueeze(-1)
    target_mean = torch.tensor([target[0].mean(), target[1].mean(), target[2].mean()]).to(DEVICE).unsqueeze(-1)

    # barycentric coordinates
    source_bary = source - source_mean
    target_bary = target - target_mean

    # variable
    n_points = source.shape[1]
    iterations = 0  # 迭代次数为0
    dist_now = 1e5
    dist_best = 1e5
    R_best = R = torch.eye(3).to(DEVICE)
    t_best = t = torch.zeros((3, 1)).to(DEVICE)
    source_to_target = torch.zeros(n_points).to(DEVICE)

    while iterations < 5 and dist_now > 1e-3:  # 迭代次数小于10 并且 距离提升大于0.001时，继续迭代

        # 2. find the closest point
        source_new = torch.mm(R, source) + t
        for i in range(n_points):
            idx, _ = get_closest_index(source_new[:, i], target)
            source_to_target[i] = idx

        # 3. find best transformation
        # 计算矩阵H
        H = torch.zeros((3, 3)).to(DEVICE)
        for i in range(n_points):
            H += torch.mm(
                source_bary[:, i].unsqueeze(1),
                target_bary[:, int(source_to_target[i])].unsqueeze(0))
        # 对H求SVD分解，根据公式求得R
        u, _, v = np.linalg.svd(H.cpu().numpy())
        u = torch.from_numpy(u).to(DEVICE)
        v = torch.from_numpy(v).to(DEVICE)
        weight = torch.eye(3).to(DEVICE)
        weight[2, 2] = torch.det(torch.mm(v, u.T))
        R = torch.mm(torch.mm(v, weight), u.T)
        # 根据公式计算t
        t = target_mean - torch.mm(R, source_mean)

        iterations = iterations + 1  # 迭代次数+1
        source_new = torch.mm(R, source) + t
        dist_now = get_set_dist(source_new, target)  # 变换后两个点云之间的距离
        print("迭代第%d次, 损失是%.6f" % (iterations, dist_now))  # 打印迭代次数、损失距离、损失提升
        if dist_now < dist_best:
            dist_best = dist_now
            R_best = R
            t_best = t

        plot_pointcloud(source_new)
    return torch.mm(R_best, source) + t_best


def main():
    source_verts, faces, _ = load_obj("data/mean_shape.obj")
    target_verts, _, _ = load_obj("data/target_shape.obj")
    plot_pointcloud(source_verts.T)
    plot_pointcloud(target_verts.T)
    result = ICP(source_verts, target_verts).T
    print("图片保存到 results/icp/result.obj")
    save_obj("results/icp/result.obj", result, faces[0])


if __name__ == '__main__':
    with torch.no_grad():
        main()
