# -*- coding: utf-8 -*-
"""
@Time ： 12/25/2022 6:29 PM
@Auth ： YY
@File ：在vtk中绘制一个有大小的平面.py
@IDE ：PyCharm
@state:
@Function：
"""
import vtk

# Create a plane source
plane = vtk.vtkPlaneSource()

# Set the plane center and normal
plane.SetCenter(0, 0, 0)
plane.SetNormal(0, 0, 1)

# Set the plane dimensions
plane.SetPoint1(-10, -30, 0)
plane.SetPoint2(10, 30, 0)

# Update the plane
plane.Update()

# Get the plane polydata
polydata = plane.GetOutput()

# Create a mapper
mapper = vtk.vtkPolyDataMapper()

# Set the input data for the mapper
mapper.SetInputData(polydata)

# Create an actor
actor = vtkshengcheng.vtkActor()

# Set the mapper for the actor
actor.SetMapper(mapper)

# Create a renderer
renderer = vtk.vtkRenderer()

# Add the actor to the renderer
renderer.AddActor(actor)

# Create a render window
render_window = vtk.vtkRenderWindow()

# Add the renderer to the render window
render_window.AddRenderer(renderer)

# Create an interactor
interactor = vtkshengcheng.vtkRenderWindowInteractor()

# Set the interactor for the render window
render_window.SetInteractor(interactor)

# Initialize the interactor
interactor.Initialize()

# Render the scene and start the interactor
render_window.Render()
interactor.Start()
