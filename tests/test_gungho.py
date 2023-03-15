from pathlib import Path
import lfricflux
import mint

TESTS_DIR = Path(__file__).absolute().parent.parent / Path('tests')
DATA_DIR = Path(__file__).absolute().parent.parent / Path('data')

def test1():

    fn = str(DATA_DIR / 'gungho' / 'original' / 'lfric_diag.nc')
    cf = str(TESTS_DIR/ 'tests.cfg')

    lf = lfricflux.LFRicFlux(configFile=cf, inputFile=fn)

    flow = lf.computeFlow(xy=[(0., -90.), (0., 90.)])

    x, y = lf.getEdgeLonLat()
    vc = lfricflux.VtkVectors(x=x, y=y, vector_field_name='horiz_flow', cartesian=True)

    rho_u_dz, rho_v_dz = lf.getFlows()
    extra_dims = rho_u_dz.shape[:-1]
    mai = mint.MultiArrayIter(extra_dims)
    mai.begin()
    for _ in range(mai.getNumIters()):
        inds = tuple(mai.getIndices())
        vc.save('flow.vtk', rho_u_dz, rho_v_dz, inds)
        mai.next()

