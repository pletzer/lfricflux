from pathlib import Path
import lfricflux

DATA_DIR = Path(__file__).absolute().parent.parent.parent / Path('data')

def test1():
	fn = str(DATA_DIR/ 'gungho' / 'uniform_extrusion')
	mesh = 'Mesh2d'
	lf = lfricflux.LFRicFlux(fileName=fn, meshName=mesh,)
	flow = lf.computeFlow(xy=[(0., -90.), (0., 90.)])