import itk

def get_transformation_matrix(dicom_directory):
    # 创建 ImageSeriesReader 对象
    reader = itk.ImageSeriesReader[itk.Image[itk.ctype('signed short'), 3]].New()
    reader.SetFileNames(dicom_directory)

    # 读取 DICOM 序列数据
    image = reader.Execute()

    # 获取图像的空间方向
    spatial_orientation = itk.SpatialOrientation.New(image)

    # 获取 DICOM 图像坐标系到世界坐标系的转换矩阵
    transformation_matrix = itk.Orientation.GetOrientationMatrixFromImageOrientation(spatial_orientation)

    return transformation_matrix



dicom_directory = r'C:\Users\Yang\Desktop\MRI_IMAGE\16'

# 获取 DICOM 图像坐标系到世界坐标系的转换矩阵
transformation_matrix = get_transformation_matrix(dicom_directory)

print(transformation_matrix)
