# -*- coding: utf-8 -*-
"""
@Time ： 12/23/2022 11:31 PM
@Auth ： YY
@File ：utlis.py
@IDE ：PyCharm
@state:
@Function：
"""
import cv2
import SimpleITK as sitk
from os import listdir, scandir
from os.path import splitext
from pathlib import Path
import numpy as np
import vtk
from PIL import Image
from torch.utils.data import Dataset
import torch

from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray, vtkPolygon
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper, vtkProperty


def getdatawithfail(reader, tag):
    try:
        return reader.GetMetaData(tag)
    except:
        return None


def dicom_to_numpy(fname):
    DCM_Img = sitk.ImageFileReader()
    DCM_Img.SetFileName(fname)
    DCM_Img.ReadImageInformation()
    rows = int(DCM_Img.GetMetaData("0028|0010"))  # Get number of rows from tag (0028, 0010)
    cols = int(DCM_Img.GetMetaData("0028|0011"))  # Get number of cols from tag (0028, 0011)

    Instance_Number = int(DCM_Img.GetMetaData("0020|0013"))  # Get actual slice instance number from tag (0020, 0013)

    data00281050 = getdatawithfail(DCM_Img, "0028|1050")
    if data00281050:
        Window_Center = int(data00281050)  # Get window center from tag (0028, 1050)
    else:
        Window_Center = int(rows / 2)

    data00281051 = getdatawithfail(DCM_Img, "0028|1051")
    if data00281051:
        Window_Width = int(data00281051)
    else:
        Window_Width = int(cols / 2)

    Window_Max = int(Window_Center + Window_Width / 2)
    Window_Min = int(Window_Center - Window_Width / 2)

    data00281052 = getdatawithfail(DCM_Img, "0028|1052")
    if data00281052 is None:
        Rescale_Intercept = 0
    else:
        Rescale_Intercept = int(data00281052)

    data00281053 = getdatawithfail(DCM_Img, "0028|1053")
    if data00281053 is None:
        Rescale_Slope = 1
    else:
        Rescale_Slope = int(data00281053)

    New_Img = np.zeros((rows, cols), np.uint8)
    # Pixels = DCM_Img.pixel_array

    image = DCM_Img.Execute()
    Pixels = sitk.GetArrayFromImage(image)[0]
    for i in range(0, rows):
        for j in range(0, cols):
            Pix_Val = Pixels[i][j]
            Rescale_Pix_Val = Pix_Val * Rescale_Slope + Rescale_Intercept

            if (Rescale_Pix_Val > Window_Max):  # if intensity is greater than max window
                New_Img[i][j] = 255
            elif (Rescale_Pix_Val < Window_Min):  # if intensity is less than min window
                New_Img[i][j] = 0
            else:
                New_Img[i][j] = int(((Rescale_Pix_Val - Window_Min) / (Window_Max - Window_Min)) * 255)  # Normalize the intensities

    return New_Img


class BasicDataset(Dataset):
    def __init__(self, unet_type, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.unet_type = unet_type
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        # logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

            img_ndarray = img_ndarray / 255

        return img_ndarray

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext == '.npy':
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)


def put_mask(image, mask):
    # image(a,b,c) mask(a,b)
    zeros1 = np.zeros((image.shape), dtype=np.uint8)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] > 0:
                zeros1[i, j] = (0, 255, 0)
    # alpha 为第一张图片的透明度
    alpha = 1
    # beta 为第二张图片的透明度
    beta = 0.2
    gamma = 0
    # cv2.addWeighted 将原始图片与 mask 融合
    mask_img = cv2.addWeighted(image, alpha, zeros1, beta, gamma)
    return mask_img
    # cv2.imwrite('mask_img.jpg', mask_img)


def traversal_files(path):
    dirs = []
    files = []
    for item in scandir(path):
        if item.is_dir() and item.name[0] != '.':
            dirs.append(item)
        elif item.is_file() and item.name[0] != '.':
            files.append(item)
    return dirs, files


def draw_axes(Translate=(0, 0, 0), RotateX=0, RotateY=0, RotateZ=0, XYZLength=(0, 0, 0)):
    transform = vtkTransform()

    transform.Translate(Translate)
    transform.RotateX(RotateX)
    transform.RotateY(RotateY)
    transform.RotateZ(RotateZ)

    axes = vtkAxesActor()
    #  The axes are positioned with a user transform
    axes.SetUserTransform(transform)
    axes.SetTotalLength(XYZLength)

    return axes


def DrawaPlane(x, Translate=(0, 0, 0)):
    colors = vtkNamedColors()
    points = vtkPoints()
    points.InsertNextPoint(-35, 35, x)
    points.InsertNextPoint(35.0, 35.0, x)
    points.InsertNextPoint(35.0, -35.0, x)
    points.InsertNextPoint(-35.0, -35.0, x)

    # Create the polygon
    polygon = vtkPolygon()
    polygon.GetPointIds().SetNumberOfIds(4)  # make a quad
    polygon.GetPointIds().SetId(0, 0)
    polygon.GetPointIds().SetId(1, 1)
    polygon.GetPointIds().SetId(2, 2)
    polygon.GetPointIds().SetId(3, 3)

    # Add the polygon to a list of polygons
    polygons = vtkCellArray()
    polygons.InsertNextCell(polygon)

    # Create a PolyData
    polygonPolyData = vtkPolyData()
    polygonPolyData.SetPoints(points)
    polygonPolyData.SetPolys(polygons)

    # Create a mapper and actor
    mapper = vtkPolyDataMapper()
    mapper.SetInputData(polygonPolyData)

    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(colors.GetColor3d('Silver'))
    Property = actor.GetProperty()

    transform = vtkTransform()
    transform.Translate(Translate)
    actor.SetUserTransform(transform)

    # 设置透明度（0.0 为完全透明，1.0 为完全不透明）
    Property.SetColor(colors.GetColor3d('red'))
    Property.SetOpacity(0.2)
    actor.SetProperty(Property)
    return actor


def DrawAMarker(point=(12, 12, -12), color='yellow',Radius=3):
    # 创建一个球体，球心位于0，0，0的位置，半径0.5
    sphereSource = vtkSphereSource()
    sphereSource.SetCenter(point)

    sphereSource.SetRadius(Radius)
    colors = vtkNamedColors()
    # 创建数据映射器，连接到球体
    sphereMapper = vtk.vtkPolyDataMapper()
    sphereMapper.SetInputConnection(sphereSource.GetOutputPort())
    property = vtkProperty()
    property.SetColor(colors.GetColor3d(color))
    property.SetDiffuse(0.8)
    property.SetOpacity(0.5)

    # 创建演员，可以理解为表演者
    sphereActor = vtk.vtkActor()
    sphereActor.SetMapper(sphereMapper)
    sphereActor.SetProperty(property)
    return sphereActor


def DrawProstateCenter(point=(-10, -10, -10)):
    # 创建一个球体，球心位于0，0，0的位置，半径0.5
    sphereSource = vtkSphereSource()
    sphereSource.SetCenter(point)

    sphereSource.SetRadius(5)
    colors = vtkNamedColors()
    # 创建数据映射器，连接到球体
    sphereMapper = vtk.vtkPolyDataMapper()
    sphereMapper.SetInputConnection(sphereSource.GetOutputPort())
    property = vtkProperty()
    property.SetColor(colors.GetColor3d('white'))
    property.SetDiffuse(0.8)
    property.SetOpacity(1)

    # 创建演员，可以理解为表演者
    sphereActor = vtk.vtkActor()
    sphereActor.SetMapper(sphereMapper)
    sphereActor.SetProperty(property)
    return sphereActor


def getrealposition(ImagePosition=(0, 0, 0), ImageOrientation=(0, 0, 0, 0, 0, 0), pixelsposition=(0, 0), PixelSpacing=(0, 0)):
    Sx, Sy, Sz = ImagePosition
    Xx, Xy, Xz, Yx, Yy, Yz = ImageOrientation
    pixelx, pixely = pixelsposition
    PixelSpacingx, PixelSpacingy = PixelSpacing
    a = np.asarray([[Xx * PixelSpacingx, Yx * PixelSpacingy, 0, Sx],
                    [Xy * PixelSpacingx, Yy * PixelSpacingy, 0, Sy],
                    [Xz * PixelSpacingx, Yy * PixelSpacingy, 0, Sz],
                    [0, 0, 0, 1]])
    # print(a.shape)
    b = np.asarray([[pixelx],
                    [pixely],
                    [0],
                    [1]])
    # print(b.shape)
    result=np.dot(a, b)
    result=result.flatten()
    return result[0],result[1],result[2]

def draw3dBox(pointA, pointB):

    minX, minY, minZ = pointA
    maxX, maxY, maxZ = pointB

    boxGridPoints = vtk.vtkPoints()
    boxGridPoints.SetNumberOfPoints(8)
    boxGridPoints.SetPoint(0, minX, maxY, minZ)
    boxGridPoints.SetPoint(1, maxX, maxY, minZ)
    boxGridPoints.SetPoint(2, maxX, minY, minZ)
    boxGridPoints.SetPoint(3, minX, minY, minZ)

    boxGridPoints.SetPoint(4, minX, maxY, maxZ)
    boxGridPoints.SetPoint(5, maxX, maxY, maxZ)
    boxGridPoints.SetPoint(6, maxX, minY, maxZ)
    boxGridPoints.SetPoint(7, minX, minY, maxZ)

    boxGridCellArray = vtk.vtkCellArray()
    for i in range(12):
        boxGridCell = vtk.vtkLine()
        if i < 4:
            temp_data = (i + 1) if (i + 1) % 4 != 0 else 0
            boxGridCell.GetPointIds().SetId(0, i)
            boxGridCell.GetPointIds().SetId(1, temp_data)
        elif i < 8:
            temp_data = (i + 1) if (i + 1) % 8 != 0 else 4
            boxGridCell.GetPointIds().SetId(0, i)
            boxGridCell.GetPointIds().SetId(1, temp_data)
        else:
            boxGridCell.GetPointIds().SetId(0, i % 4)
            boxGridCell.GetPointIds().SetId(1, i % 4 + 4)
        boxGridCellArray.InsertNextCell(boxGridCell)

    boxGridData = vtk.vtkPolyData()
    boxGridData.SetPoints(boxGridPoints)
    boxGridData.SetLines(boxGridCellArray)
    boxGridMapper = vtk.vtkPolyDataMapper()
    boxGridMapper.SetInputData(boxGridData)
    actor2 = vtk.vtkActor()
    actor2.SetMapper(boxGridMapper)
    return actor2
def draw3Dtext(text="",Scale=(5,5,5),Position=(0,0,25)):

    colors = vtkNamedColors()
    view = vtk.vtkVectorText()
    view.SetText(text)
    textMapper1 = vtk.vtkPolyDataMapper()
    textMapper1.SetInputConnection(view.GetOutputPort())
    textActorFview = vtk.vtkFollower()
    textActorFview.SetMapper(textMapper1)
    textActorFview.SetScale(Scale)
    textActorFview.AddPosition(Position)
    textActorFview.GetProperty().SetColor(colors.GetColor3d('blue'))
    return textActorFview
def DrawaPlanep1p2p3p4(p1,p2,p3,p4,corlors='green'):

    colors = vtkNamedColors()
    points = vtkPoints()
    points.InsertNextPoint(p1)
    points.InsertNextPoint(p2)
    points.InsertNextPoint(p3)
    points.InsertNextPoint(p4)

    # Create the polygon
    polygon = vtkPolygon()
    polygon.GetPointIds().SetNumberOfIds(4)  # make a quad
    polygon.GetPointIds().SetId(0, 0)
    polygon.GetPointIds().SetId(1, 1)
    polygon.GetPointIds().SetId(2, 2)
    polygon.GetPointIds().SetId(3, 3)

    # Add the polygon to a list of polygons
    polygons = vtkCellArray()
    polygons.InsertNextCell(polygon)

    # Create a PolyData
    polygonPolyData = vtkPolyData()
    polygonPolyData.SetPoints(points)
    polygonPolyData.SetPolys(polygons)

    # Create a mapper and actor
    mapper = vtkPolyDataMapper()
    mapper.SetInputData(polygonPolyData)

    actor = vtkActor()
    actor.SetMapper(mapper)
    Property = actor.GetProperty()



    # 设置透明度（0.0 为完全透明，1.0 为完全不透明）
    Property.SetColor(colors.GetColor3d(corlors))
    Property.SetOpacity(0.4)
    actor.SetProperty(Property)
    return actor
def MarkercheckAixs(p1,p2,p3,p4):
    # 将四个点的坐标放入一个矩阵中
    points = np.vstack((p1, p2, p3, p4))
    # 对矩阵进行 SVD 分解
    U, S, V = np.linalg.svd(points)

    # 使用 SVD 分解得到的三个向量作为新的坐标轴
    axes = V.T

    # 将四个点投影到新的坐标轴上
    projected_points = np.dot(points, axes)

    # 计算投影误差
    error = np.sum((points - projected_points) ** 2)
    return error


