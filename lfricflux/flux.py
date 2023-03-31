import mint
import numpy
from configparser import ConfigParser
import vtk

class LFRicFlux(object):


    def __init__(self, configFile: str, inputFile: str):


        cfg = ConfigParser()
        cfg.read(configFile)

        meshName = cfg['mesh']['name']
        uName = cfg['velocity']['u']
        vName = cfg['velocity']['v']
        rhoName = cfg['density']['name']
        zName = cfg['vertical.axis']['name']
        
        # build the mesh
        self.grid = mint.Grid()
        self.grid.setFlags(1, 1, 1)
        self.grid.loadFromUgrid2DFile(f'{inputFile}${meshName}')

        # read the velocity fields on edges
        self.u = mint.NcFieldRead(inputFile, uName).data()
        self.v = mint.NcFieldRead(inputFile, vName).data()

        # compute the density one edges
        edge2face = numpy.array(mint.NcFieldRead(inputFile, f'{meshName}_edge_face_links').data(), dtype=int)
        # number of edges is assumed to be the last dimension
        self.nedges = self.u.shape[-1]

        # read the cell centred fields
        z_w3 = mint.NcFieldRead(inputFile, zName).data()
        rho_w3 = mint.NcFieldRead(inputFile, rhoName).data()

        self.rho_edge = numpy.empty(rho_w3.shape[:-1] + (self.nedges,), numpy.float64)
        self.z_edge = numpy.empty(z_w3.shape[:-1] + (self.nedges,), numpy.float64)


        for i in range(self.nedges):
            faceid1, faceid2 = edge2face[i,:]
            self.rho_edge[..., i] = 0.5*(rho_w3[..., faceid1] + rho_w3[..., faceid2])
            self.z_edge[..., i] = 0.5*(z_w3[..., faceid1] + z_w3[..., faceid2])

        # flow across edges
        dz = self.z_edge[:, 1:, :] - self.z_edge[:, :-1, :]
        self.rho_u_dz = self.rho_edge * self.u * dz
        self.rho_v_dz = self.rho_edge * self.v * dz

        # store the locations of the mid-edge positions
        self.xEdge = mint.NcFieldRead(inputFile, f'{meshName}_edge_x').data()
        self.yEdge = mint.NcFieldRead(inputFile, f'{meshName}_edge_y').data()

        # points defining the flux target line
        self.targetPoints = vtk.vtkDoubleArray()
        self.targetPoints.SetNumberOfComponents(3)


    def getFlows(self):
        return self.rho_u_dz, self.rho_v_dz


    def getEdgeLonLat(self):
        return self.xEdge, self.yEdge


    def computeFlow(self, xy):

        xya = numpy.array(xy)
        self.targetLons = xya[:, 0]
        self.targetLats = xya[:, 1]

        pli = mint.PolylineIntegral()
        pli.setGrid(self.grid)
        pli.buildLocator(numCellsPerBucket=128, periodX=360., enableFolding=True)
        xyz = numpy.zeros((xya.shape[0], 3))
        xyz[:, :2] = xya
        pli.computeWeights(xyz)

        extraDims = self.rho_u_dz.shape[:-1]
        mai = mint.MultiArrayIter(extraDims)
        self.flux = numpy.empty(extraDims, numpy.float64)
        for _ in range(mai.getNumIters()):
            inds = tuple(mai.getIndices())
            slab = inds + (slice(0, self.nedges),)
            self.flux[inds] = pli.vectorGetIntegral(self.rho_u_dz[slab], self.rho_v_dz[slab],fs=mint.FUNC_SPACE_W2)
            mai.next()

        return self.flux


    def buildFluxSurface(self, extrusion=0.1, cartesian=False, radius=1.0):
        
        npts = len(self.targetLats)
        n = npts * 2 # factor 2 because of extrusion
        ncells = npts - 1

        self.targetFlux = vtk.vtkDoubleArray()
        self.targetFlux.SetNumberOfComponents(1)
        self.targetFlux.SetNumberOfTuples(ncells)
        self.targetFlux.SetName("flux")

        self.targetSegmentLength = vtk.vtkDoubleArray()
        self.targetSegmentLength.SetNumberOfComponents(1)
        self.targetSegmentLength.SetNumberOfTuples(ncells)
        self.targetSegmentLength.SetName("length")

        self.targetGrid = vtk.vtkUnstructuredGrid()
        self.targetPointData = vtk.vtkDoubleArray()
        self.targetPoints = vtk.vtkPoints()

        # set the target points
        self.targetPointArray = numpy.empty((n, 3), numpy.float64)

        self.targetPointData.SetNumberOfComponents(3)
        self.targetPointData.SetNumberOfTuples(n)

        if cartesian:
            lam = self.targetLons * numpy.pi / 180.
            the = self.targetLats * numpy.pi / 180.
            rho = numpy.cos(the)
            r0 = radius
            r1 = r0*(1. + extrusion)
            self.targetPointArray[:npts, 0] = r0*rho*numpy.cos(lam)
            self.targetPointArray[:npts, 1] = r0*rho*numpy.sin(lam)
            self.targetPointArray[:npts, 2] = r0*numpy.sin(the)
            self.targetPointArray[npts:, 0] = r1*rho*numpy.cos(lam)
            self.targetPointArray[npts:, 1] = r1*rho*numpy.sin(lam)
            self.targetPointArray[npts:, 2] = r1*numpy.sin(the)
        else:
            self.targetPointArray[:npts, 0] = self.targetLons
            self.targetPointArray[:npts, 1] = self.targetLats
            self.targetPointArray[:npts, 2] = 0.0
            self.targetPointArray[npts:, 0] = self.targetLons
            self.targetPointArray[npts:, 1] = self.targetLats
            self.targetPointArray[npts:, 2] = extrusion

        self.targetPointData.SetVoidArray(self.targetPointArray, n*3, 1)
        self.targetPoints.SetData(self.targetPointData)
        self.targetGrid.SetPoints(self.targetPoints)

        # connectivity
        self.targetGrid.AllocateExact(ncells, 4)
        ptIds = vtk.vtkIdList()
        ptIds.SetNumberOfIds(4)
        # unit normals to the sphere
        n0 = numpy.zeros((3,), numpy.float64)
        n1 = numpy.zeros((3,), numpy.float64)
        for i in range(npts - 1):
            ptIds.SetId(0, i)
            ptIds.SetId(1, i + 1)
            ptIds.SetId(2, i + 1 + npts)
            ptIds.SetId(3, i + npts)
            self.targetGrid.InsertNextCell(vtk.VTK_QUAD, ptIds)

            # compute the arc length assuming a unit radius
            lam = self.targetLons[i] * numpy.pi / 180.
            the = self.targetLats[i] * numpy.pi / 180.
            rho = numpy.cos(the)
            n0[:] = rho*numpy.cos(lam), rho*numpy.sin(lam), numpy.sin(the)

            lam = self.targetLons[i + 1] * numpy.pi / 180.
            the = self.targetLats[i + 1] * numpy.pi / 180.
            rho = numpy.cos(the)
            n1[:] = rho*numpy.cos(lam), rho*numpy.sin(lam), numpy.sin(the)
            
            dAngle = n0.dot(n1)
            segLength = radius * dAngle
            self.targetSegmentLength.SetTuple(i, (segLength,))

        # add the field to the grid
        self.targetGrid.GetCellData().AddArray(self.targetFlux)
        self.targetGrid.GetCellData().AddArray(self.targetSegmentLength)


    def setSlice(self, inds):

        slab = inds + (slice(None, None),)
        n = self.targetFlux.GetNumberOfTuples()
        self.targetFlux.SetVoidArray(self.flux[inds], n, 1)


    def save(self, fileName, inds):

        writer = vtk.vtkUnstructuredGridWriter()
        fn = fileName.split('.')[0] + '_' + '_'.join([f'{idx}' for idx in inds]) + '.vtk'
        writer.SetFileName(fn)
        writer.SetInputData(self.targetGrid)
        self.setSlice(inds)
        writer.Update()

