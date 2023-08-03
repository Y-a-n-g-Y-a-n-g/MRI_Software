# -*- coding: utf-8 -*-
"""
@Time ： 12/26/2022 11:17 PM
@Auth ： YY
@File ：python获取dicom文件某帧上某像素点在空间中的真实位置（转换矩阵）.py
@IDE ：PyCharm
@state:
@Function：
"""
import SimpleITK as sitk
import numpy as np


def get_real_position(dicom_file, x, y, z=None):
    """
    Calculates the real-world position of a point on a DICOM image.

    Parameters:
      - dicom_file (str): Path to the DICOM file
      - x (int): x-coordinate of the point in pixels
      - y (int): y-coordinate of the point in pixels
      - z (int, optional): z-coordinate of the point in pixels (if the image is 3D). Default is None.

    Returns:
      tuple: The real-world position of the point (x, y, z)
    """

    # Read the DICOM image
    image = sitk.ReadImage(dicom_file)


    # Get the origin of the image (the position of the first pixel in physical space)
    origin = image.GetOrigin()

    # Get the spacing between pixels in the image (the size of each pixel in physical space)
    spacing = image.GetSpacing()

    # Get the direction of the image (the orientation of the axes in physical space)
    direction = image.GetDirection()

    # Convert the direction to a 3x3 matrix
    direction_matrix = np.array(direction).reshape((3, 3))

    # Create a 3D point with the pixel coordinates
    point = np.array([x, y, z]) if z is not None else np.array([x, y, 0])

    # Transform the point using the direction matrix
    point_real = direction_matrix.dot(point)

    # Add the origin to the transformed point to get the real-world position
    x_real = origin[0] + point_real[0] * spacing[0]
    y_real = origin[1] + point_real[1] * spacing[1]
    z_real = origin[2] + point_real[2] * spacing[2]

    if z is not None:
        # If the image is 3D, return the real-world position of the point
        return (x_real, y_real, z_real)
    else:
        # If the image is 2D, return the x and y positions only
        return (x_real, y_real)
# Calculate the real-world position of a point on a 2D DICOM image
position = get_real_position(r'C:\Users\Yang\Desktop\MRI_IMAGE\16\PROSTATE_221219.MR.K-MEDI_HUB_NECK.0016.0001.2022.12.19.19.19.15.797509.410989463.IMA', 10, 20)
print("The real-world position of the point is:", position)

# Calculate the real-world position of a point on a 3D DICOM image
position = get_real_position(r'C:\Users\Yang\Desktop\MRI_IMAGE\16\PROSTATE_221219.MR.K-MEDI_HUB_NECK.0016.0001.2022.12.19.19.19.15.797509.410989463.IMA', 0, 0, 0)
print("The real-world position of the point is:", position)
