import mint
import numpy

class LFRicFlux(object):


    def __init__(self, fileName, meshName, uName='u_in_w2h', vName='v_in_w2h', rhoName='rho', dz=1000.0):
        
        # build the mesh
        self.grid = mint.Grid()
        self.grid.setFlags(1, 1, 1)
        self.grid.loadFromUgrid2DFile(f'{fileName}${meshName}')

        # read the velocity fields on edges
        self.u = mint.NcFieldRead(fileName, uName).data()
        self.v = mint.NcFieldRead(fileName, vName).data()

        # read the cell centred density 
        rho_w3 = mint.NcFieldRead(fileName, rhoName).data()

        # compute the density one edges
        edge2face = numpy.array(mint.NcFieldRead(fileName, f'{meshName}_edge_face_links').data(), dtype=int)
        # number of edges is assumed to be the last dimension
        self.nedges = self.u.shape[-1]
        self.rho_edge = numpy.empty(rho_w3.shape[:-1] + (self.nedges,), numpy.float64)
        for i in range(self.nedges):
            faceid1, faceid2 = edge2face[i,:]
            self.rho_edge[..., i] = 0.5*(rho_w3[..., faceid1] + rho_w3[..., faceid2])

        # flow across edges
        self.rho_u_dz = self.rho_edge * self.u * dz
        self.rho_v_dz = self.rho_edge * self.v * dz

        # store the locations of the mid-edge positions
        self.xEdge = mint.NcFieldRead(fileName, f'{meshName}_edge_x').data()
        self.yEdge = mint.NcFieldRead(fileName, f'{meshName}_edge_y').data()



    def saveFlowVTK(self, fileName):
        
        import vtk
        pointCoordArray = vtk.vtkDoubleArray()
        pointCoordArray.SetNumberOfComponents(3)
        pointCoordArray.SetNumberOfTuples(self.nedges)
        pointCoords = vtk.vtkPoints()
        pointMesh = vtk.vtkUnstructuredGrid()
        pointMeshWriter = vtk.vtkUnstructuredGridWriter()

        pointCoords.SetData(pointCoordArray)
        pointMesh.SetPoints(pointCoords)
        # unstructured grid is a cloud of points
        pointMesh.AllocateExact(self.nedges, 1)
        pointMeshWriter.SetInputData(pointMesh)

        ptIds = vtk.vtkIdList()
        ptIds.SetNumberOfIds(1)
        for i in range(self.nedges):
            pointCoordArray.SetTuple(i, (self.xEdge[i], self.yEdge[i], 0.))
            ptIds.SetId(0, i)
            pointMesh.InsertNextCell(vtk.VTK_VERTEX, ptIds)

        # need to attach the u, v fields
        rhoVelocity = vtk.vtkDoubleArray()
        rhoVelocity.SetNumberOfComponents(3)
        rhoVelocity.SetNumberOfTuples(self.nedges)
        rhoVelocity.SetName('rho_velocity')
        vxyz = numpy.zeros((self.nedges, 3), numpy.float64)
        rhoVelocity.SetVoidArray(vxyz, 3*self.nedges, 1)
        pointMesh.GetPointData().AddArray(rhoVelocity)

        extra_dims = self.rho_u_dz.shape[:-1]
        mai = mint.MultiArrayIter(extra_dims)
        mai.begin()
        for _ in range(mai.getNumIters()):
            inds = tuple(mai.getIndices())
            slab = inds + (slice(None, None),)
            vxyz[:, 0] = self.rho_u_dz[slab]
            vxyz[:, 1] = self.rho_v_dz[slab]
            fn = fileName.split('.')[0] + '_'.join([f'{idx}' for idx in inds]) + '.vtk'
            pointMeshWriter.SetFileName(fn)
            pointMeshWriter.Update()


    def computeFlow(self, xy):

        pli = mint.PolylineIntegral()
        pli.setGrid(self.grid)
        pli.buildLocator(numCellsPerBucket=128, periodX=360., enableFolding=True)
        xyz = numpy.array([(p[0], p[1], 0.) for p in xy])
        pli.computeWeights(xyz)

        extraDims = self.rho_u_dz.shape[:-1]
        mai = mint.MultiArrayIter(extraDims)
        res = numpy.empty(extraDims, numpy.float64)
        for _ in range(mai.getNumIters()):
            inds = tuple(mai.getIndices())
            slab = inds + (slice(0, self.nedges),)
            res[inds] = pli.vectorGetIntegral(self.rho_u_dz[slab], self.rho_v_dz[slab],fs=mint.FUNC_SPACE_W2)
            mai.next()

        return res
