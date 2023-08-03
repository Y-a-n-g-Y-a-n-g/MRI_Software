import vtk


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
    return boxGridMapper


if __name__ == '__main__':
    pointA = [-1, -1, -1]
    pointB = [1, 1, 1]
    boxGridMapper = draw3dBox(pointA, pointB)
    actor = vtk.vtkActor()
    actor.SetMapper(boxGridMapper)

    # 数据源 圆柱
    cylinder = vtk.vtkCylinderSource()
    cylinder.SetHeight(1.0)
    cylinder.SetRadius(1.0)
    cylinder.SetResolution(360)
    print("高、半径、面:", cylinder.GetHeight(), cylinder.GetRadius(), cylinder.GetResolution())
    # 映射
    cylinderMapper = vtk.vtkPolyDataMapper()
    cylinderMapper.SetInputConnection(cylinder.GetOutputPort())

    # 绘制对象/演员
    cylinderActor = vtk.vtkActor()
    # 绘制对象添加映射器
    cylinderActor.SetMapper(cylinderMapper)
    # 添加颜色
    colors = vtk.vtkNamedColors()
    actor.GetProperty().SetColor(0.67, 1, 1)
    # 绘制器
    renderer = vtk.vtkRenderer()
    # 绘制器添加对象
    renderer.AddActor(cylinderActor)
    renderer.AddActor(actor)
    # 绘制器设置背景
    renderer.SetBackground(0.1, 0.2, 0.4)
    print("Renderer bg:", renderer.GetBackground())
    # 绘制窗口
    renWin = vtk.vtkRenderWindow()
    # 绘制窗口添加绘制器
    renWin.AddRenderer(renderer)
    renWin.SetSize(1200, 1200)
    print("Window size:", renWin.GetSize())
    renWin.SetWindowName("ooooooooooooooo")
    # 绘制窗口内所有绘制器同步渲染绘制
    renWin.Render()
    pointPicker = vtk.vtkPointPicker()
    # 交互器
    i_ren = vtk.vtkRenderWindowInteractor()
    i_ren.SetPicker(pointPicker)
    # 交互器绑定绘制窗口
    i_ren.SetRenderWindow(renWin)

    style = vtk.vtkInteractorStyleTrackballCamera()
    i_ren.SetInteractorStyle(style)
    renderer.SetBackground(colors.GetColor3d("Silver"))

    renderer.ResetCamera()
    # 交互器初始化
    i_ren.Initialize()
    # 交互器启动
    i_ren.Start()
