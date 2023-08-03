# -*- coding: utf-8 -*-
"""
@Time ： 12/26/2022 4:34 PM
@Auth ： YY
@File ：使用SVD进行3D rigid registration配准.py
@IDE ：PyCharm
@state:
@Function：
"""
import numpy as np

def rigid_registration_3D(X, Y):
  # 计算X和Y中点的平均值
  mean_X = np.mean(X, axis=0)
  mean_Y = np.mean(Y, axis=0)

  # 计算X和Y的中心化点云
  X_centered = X - mean_X
  Y_centered = Y - mean_Y

  # 计算协方差矩阵
  C = np.dot(X_centered.T, Y_centered)

  # 使用SVD求解最小二乘问题
  U, S, Vt = np.linalg.svd(C)

  # 计算旋转矩阵R
  R = np.dot(Vt.T, U.T)

  # 如果矩阵的行列式为负，则将R乘以单位阵
  if np.linalg.det(R) < 0:
      Vt[2,:] *= -1
      R = np.dot(Vt.T, U.T)

  # 计算平移向量t
  t = mean_Y - np.dot(R, mean_X)

  return R, t

# 示例：使用随机生成的点云X和Y进行3D rigid registration
X = np.random.random((100, 3))
Y = np.random.random((100, 3))

R, t = rigid_registration_3D(X, Y)

# 输出旋转矩阵R和平移向量t
print("Rotation matrix:")
print(R)
print("Translation vector:")
print(t)
