# -*- coding: utf-8 -*-
"""
@Time ： 12/25/2022 4:36 PM
@Auth ： YY
@File ：在vtk中显示坐标系.py
@IDE ：PyCharm
@state:
@Function：
"""
#!/usr/bin/env python

# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)

def draw_axes(Translate=(0,0,0),RotateX=0,RotateY=0,RotateZ=0,XYZLength=(0,0,0)):
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

def main():
    colors = vtkNamedColors()

    # # create a Sphere
    # sphereSource = vtkSphereSource()
    # sphereSource.SetCenter(0.0, 0.0, 0.0)
    # sphereSource.SetRadius(0.5)
    #
    # # create a mapper
    # sphereMapper = vtkPolyDataMapper()
    # sphereMapper.SetInputConnection(sphereSource.GetOutputPort())
    #
    # # create an actor
    # sphereActor = vtkActor()
    # sphereActor.SetMapper(sphereMapper)

    # a renderer and render window
    renderer = vtkRenderer()
    renderWindow = vtkRenderWindow()
    renderWindow.SetWindowName('Axes')
    renderWindow.AddRenderer(renderer)

    # an interactor
    renderWindowInteractor = vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    # add the actors to the scene
    # renderer.AddActor(sphereActor)
    renderer.SetBackground(colors.GetColor3d('SlateGray'))

    transform = vtkTransform()

    transform.Translate(0, 0.0, 0.0)

    axes = vtkAxesActor()
    #  The axes are positioned with a user transform
    axes.SetUserTransform(transform)

    # properties of the axes labels can be set as follows
    # this sets the x axis label to red
    # axes.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetColor(colors.GetColor3d('Red'));

    # the actual text of the axis label can be changed:
    # axes->SetXAxisLabelText('test');
    AxesActor=draw_axes(Translate=(0,0,0),RotateX=30,RotateY=0,RotateZ=0,XYZLength=(10,10,10))
    renderer.AddActor(axes)
    renderer.AddActor(AxesActor)

    # renderer.GetActiveCamera().Azimuth(50)
    # renderer.GetActiveCamera().Elevation(-30)

    renderer.ResetCamera()
    renderWindow.SetWindowName('Axes')
    renderWindow.Render()

    # begin mouse interaction
    renderWindowInteractor.Start()


if __name__ == '__main__':
    main()
