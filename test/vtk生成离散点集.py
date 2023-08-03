# -*- coding: utf-8 -*-
"""
@Time ： 12/27/2022 3:42 PM
@Auth ： YY
@File ：vtk生成离散点集.py
@IDE ：PyCharm
@state:
@Function：
"""
import vtk

# 创建点的列表
points = vtk.vtkPoints()

# 添加数据点
points.InsertNextPoint(1.0, 2.0, 3.0)
points.InsertNextPoint(2.0, 3.0, 4.0)

# 创建点的数据集
polydata = vtk.vtkPolyData()
polydata.SetPoints(points)

# 创建点的图元数据集
vertexFilter = vtk.vtkVertexGlyphFilter()
vertexFilter.SetInputData(polydata)
vertexFilter.Update()

glyph = vertexFilter.GetOutput()

# 创建点的图元数据集的映射器
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(glyph.GetOutputPort())

# 创建点的演员
actor = vtk.vtkActor()
actor.SetMapper(mapper)

# 创建渲染器
renderer = vtk.vtkRenderer()

# 创建渲染窗口
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)

# 设置窗口大小
render_window.SetSize(800, 600)

# 创建交互式渲染窗口
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)

# 将演员添加到渲染器中
renderer.AddActor(actor)

# 开始渲染并进入交互模式
interactor.Initialize()
interactor.Start()
