# -*- coding: utf-8 -*-
"""
@Time ： 12/26/2022 8:26 PM
@Auth ： YY
@File ：python获取dicom上某点的真实空间位置.py
@IDE ：PyCharm
@state:
@Function：
"""
import SimpleITK as sitk
import numpy as np

# 读取 DICOM 文件
reader = sitk.ImageFileReader()
reader.SetFileName(r'C:\Users\Yang\Desktop\MRI_IMAGE\16\PROSTATE_221219.MR.K-MEDI_HUB_NECK.0016.0001.2022.12.19.19.19.15.797509.410989463.IMA')
image = reader.Execute()

# 将 DICOM 图像转换为 NumPy 数组
image_array = sitk.GetArrayFromImage(image)
spacing = image.GetSpacing()
origin = image.GetOrigin()

def getposition(p):
    # 访问图像中的特定像素
    x, y, z = p#191, 191,0  # 特定像素的索引

    # 获取图像的空间尺寸和像素间隔


    # 计算特定像素的真实位置
    x_real = origin[0] + x * spacing[0]
    y_real = origin[1] + y * spacing[1]
    z_real = origin[2] + z * spacing[2]

    print(f'特定像素的真实位置: ({x_real}, {y_real}, {z_real})')

getposition((0,0,0))
getposition((191,0,0))
getposition((191,191,0))
getposition((0,191,0))
