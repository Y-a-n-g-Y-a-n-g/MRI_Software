import numpy as np
import vtk
from vtk.util import numpy_support


numpy_data=np.random.randint(0,2,(100,100,100))
# numpy_data is a 3D numpy array
shape = numpy_data.shape[::-1]
vtk_data = numpy_support.numpy_to_vtk(numpy_data.ravel(), 1, vtkshengcheng.VTK_SHORT)

vtk_image_data = vtk.vtkImageData()
vtk_image_data.SetDimensions(shape)
vtk_image_data.SetSpacing((1,1,1))
vtk_image_data.SetOrigin((0,0,0))
vtk_image_data.GetPointData().SetScalars(vtk_data)
def main():
    # fileName = get_program_parameters()

    colors = vtk.vtkNamedColors()

    # This is a simple volume rendering example that
    # uses a vtkFixedPointVolumeRayCastMapper

    # Create the standard renderer, render window
    # and interactor.
    ren1 = vtk.vtkRenderer()

    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren1)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # Create the reader for the data.
    # reader = vtk.vtkStructuredPointsReader()
    # reader.SetFileName(fileName)

    # Create transfer mapping scalar value to opacity.
    opacityTransferFunction = vtkshengcheng.vtkPiecewiseFunction()
    opacityTransferFunction.AddPoint(20, 0.0)
    opacityTransferFunction.AddPoint(255, 0.2)

    # Create transfer mapping scalar value to color.
    colorTransferFunction = vtkshengcheng.vtkColorTransferFunction()
    colorTransferFunction.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
    colorTransferFunction.AddRGBPoint(64.0, 1.0, 0.0, 0.0)
    colorTransferFunction.AddRGBPoint(128.0, 0.0, 0.0, 1.0)
    colorTransferFunction.AddRGBPoint(192.0, 0.0, 1.0, 0.0)
    colorTransferFunction.AddRGBPoint(255.0, 0.0, 0.2, 0.0)

    # The property describes how the data will look.
    volumeProperty = vtkshengcheng.vtkVolumeProperty()
    volumeProperty.SetColor(colorTransferFunction)
    volumeProperty.SetScalarOpacity(opacityTransferFunction)
    volumeProperty.ShadeOn()
    volumeProperty.SetInterpolationTypeToLinear()

    # The mapper / ray cast function know how to render the data.
    volumeMapper = vtkshengcheng.vtkFixedPointVolumeRayCastMapper()
    volumeMapper.SetInput(vtk_image_data)
    #
    # The volume holds the mapper and the property and
    # can be used to position/orient the volume.
    volume = vtkshengcheng.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)

    ren1.AddVolume(volume)
    ren1.SetBackground(colors.GetColor3d('Wheat'))
    ren1.GetActiveCamera().Azimuth(45)
    ren1.GetActiveCamera().Elevation(30)
    ren1.ResetCameraClippingRange()
    ren1.ResetCamera()

    renWin.SetSize(600, 600)
    renWin.SetWindowName('SimpleRayCast')
    renWin.Render()

    iren.Start()


def get_program_parameters():
    import argparse
    description = 'Volume rendering of a high potential iron protein.'
    epilogue = '''
    This is a simple volume rendering example that uses a vtkFixedPointVolumeRayCastMapper.
    '''
    parser = argparse.ArgumentParser(description=description, epilog=epilogue,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # parser.add_argument('filename', help='ironProt.vtk.')
    args = parser.parse_args()
    return args.filename


if __name__ == '__main__':
    main()