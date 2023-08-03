#!/usr/bin/env python

# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import (
    vtkCellArray,
    vtkPolyData,
    vtkPolygon
)
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)

def DrawAPlane(z):
    colors = vtkNamedColors()
    points = vtkPoints()
    points.InsertNextPoint(0.0, 0.0, z)
    points.InsertNextPoint(10.0, 0.0, z)
    points.InsertNextPoint(10.0, 10.0, z)
    points.InsertNextPoint(0.0, 10.0, z)

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
    return actor

def main():
    colors = vtkNamedColors()

    # Setup four points


    # Visualize
    renderer = vtkRenderer()
    renderWindow = vtkRenderWindow()
    renderWindow.SetWindowName('Polygon')
    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
    actor=DrawAPlane(10)
    actor2 = DrawAPlane(20)
    renderer.AddActor(actor)
    renderer.AddActor(actor2)
    renderer.SetBackground(colors.GetColor3d('Salmon'))
    renderWindow.Render()
    renderWindowInteractor.Start()


if __name__ == '__main__':
    main()
