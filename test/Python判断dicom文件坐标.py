import SimpleITK as sitk

import numpy as np

def transform_coordinate(image, coordinate):
    # Get the image origin and direction
    origin = image.GetOrigin()
    direction = image.GetDirection()
    # Convert the direction matrix to a 3x3 numpy array
    direction_matrix = np.array(direction).reshape(3, 3)
    # Create a numpy array from the coordinate
    coord_array = np.array(coordinate)
    # Subtract the origin from the coordinate
    coord_array -= origin
    # Multiply the coordinate by the direction matrix to transform it
    transformed_coord = np.matmul(direction_matrix, coord_array)
    # Return the transformed coordinate as a tuple
    return tuple(transformed_coord)


print(transform_coordinate(r'C:\Users\Yang\Desktop\MRI_IMAGE\16\PROSTATE_221219.MR.K-MEDI_HUB_NECK.0016.0001.2022.12.19.19.19.15.797509.410989463.IMA', (48, 0, 0)))