import mint
import numpy
from configparser import ConfigParser

class LFRicFlux(object):


    def __init__(self, configFile: str, inputFile: str):


        cfg = ConfigParser()
        cfg.read(configFile)

        meshName = cfg['mesh']['name']
        uName = cfg['velocity']['u']
        vName = cfg['velocity']['v']
        rhoName = cfg['density']['name']
        dz = float(cfg['vertical.axis']['dz'])
        
        # build the mesh
        self.grid = mint.Grid()
        self.grid.setFlags(1, 1, 1)
        self.grid.loadFromUgrid2DFile(f'{inputFile}${meshName}')

        # read the velocity fields on edges
        self.u = mint.NcFieldRead(inputFile, uName).data()
        self.v = mint.NcFieldRead(inputFile, vName).data()

        # read the cell centred density 
        rho_w3 = mint.NcFieldRead(inputFile, rhoName).data()

        # compute the density one edges
        edge2face = numpy.array(mint.NcFieldRead(inputFile, f'{meshName}_edge_face_links').data(), dtype=int)
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
        self.xEdge = mint.NcFieldRead(inputFile, f'{meshName}_edge_x').data()
        self.yEdge = mint.NcFieldRead(inputFile, f'{meshName}_edge_y').data()


    def getFlows(self):
        return self.rho_u_dz, self.rho_v_dz


    def getEdgeLonLat(self):
        return self.xEdge, self.yEdge


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


