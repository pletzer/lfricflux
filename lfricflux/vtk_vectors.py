import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy


class VtkVectors(object):

    RADIUS = 1.0

    def __init__(self, x, y, vector_field_name='vectors', cartesian=False):

        self.cartesian = cartesian
        self.x = x
        self.y = y

        # last dimension is number of edges
        self.nedges = x.shape[-1]
        self._buildMesh()
        self.vectorField = vtk.vtkDoubleArray()
        self.vectorField.SetNumberOfComponents(3)
        self.vectorField.SetNumberOfTuples(self.nedges)
        self.vectorField.SetName(vector_field_name)
        self.vxyz = numpy.zeros((self.nedges, 3), numpy.float64)
        self.vectorField.SetVoidArray(self.vxyz, 3*self.nedges, 1)
        self.pointMesh.GetPointData().AddArray(self.vectorField)


    def _buildMesh(self):

        self.pointCoordArray = vtk.vtkDoubleArray()
        self.pointCoordArray.SetNumberOfComponents(3)
        self.pointCoordArray.SetNumberOfTuples(self.nedges)


        self.pointCoords = vtk.vtkPoints()

        self.pointMesh = vtk.vtkUnstructuredGrid()

        # connect
        self.pointCoords.SetData(self.pointCoordArray)
        self.pointMesh.SetPoints(self.pointCoords)

        # unstructured grid is a cloud of points
        self.pointMesh.AllocateExact(self.nedges, 1)
        ptIds = vtk.vtkIdList()
        ptIds.SetNumberOfIds(1)
        point = numpy.empty((3,), numpy.float64)
        for i in range(self.nedges):
            if self.cartesian:
                # in cartesian
                rho = self.RADIUS * numpy.cos(self.y[i] * numpy.pi/180.)
                point[0] = rho * numpy.cos(self.x[i] * numpy.pi/180.)
                point[1] = rho * numpy.sin(self.x[i] * numpy.pi/180.)
                point[2] = self.RADIUS * numpy.sin(self.y[i] * numpy.pi/180.)
                self.pointCoordArray.SetTuple(i, point)
            else:
                # x, y are lon-lat
                self.pointCoordArray.SetTuple(i, (self.x[i], self.y[i], 0.))
            ptIds.SetId(0, i)
            self.pointMesh.InsertNextCell(vtk.VTK_VERTEX, ptIds)


    def getMesh(self):
        return self.pointMesh


    def setField(self, u_east, v_north):
        self.uEast = u_east
        self.vNorth = v_north


    def setSlice(self, inds):

        slab = inds + (slice(None, None),)
        u = self.uEast[slab]
        v = self.vNorth[slab]

        if self.cartesian:
            # in cartesian
            cos_the = numpy.cos(self.y * numpy.pi/180.)
            sin_the = numpy.sin(self.y * numpy.pi/180.)
            cos_lam = numpy.cos(self.x * numpy.pi/180.)
            sin_lam = numpy.sin(self.x * numpy.pi/180.)
            self.vxyz[:, 0] = -u*sin_lam - v*sin_the*cos_lam
            self.vxyz[:, 1] = +u*cos_lam - v*sin_the*sin_lam
            self.vxyz[:, 2] = +v*cos_the
        else:
            # in lon-lat coords
            self.vxyz[:, 0] = u
            self.vxyz[:, 1] = v


    def getPointArray(self):
        return vtk_to_numpy(self.pointCoordArray)


    def getFieldArray(self):
        return self.vxyz


    def save(self, fileName, inds):

        writer = vtk.vtkUnstructuredGridWriter()
        fn = fileName.split('.')[0] + '_' + '_'.join([f'{idx}' for idx in inds]) + '.vtk'
        writer.SetFileName(fn)
        writer.SetInputData(self.pointMesh)
        self.setSlice(inds)
        writer.Update()

