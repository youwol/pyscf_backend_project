import numpy
from pyscf import lib
from pyscf.dft import numint

from pyscf_backend.cube import GaussianCubeFile


class CacheStore:

    entries = {}

    @staticmethod
    def set(key, value):
        CacheStore.entries[key] = value

    @staticmethod
    def get(key):
        if key in CacheStore.entries:
            return CacheStore.entries[key]


def density(mol, dm, nx=80, ny=80, nz=80, margin=3.0):

    from pyscf.pbc.gto import Cell

    cc = GaussianCubeFile(mol=mol, nx=nx, ny=ny, nz=nz, resolution=None, margin=margin)

    gto_val = "GTOval" if not isinstance(mol, Cell) else "PBCGTOval"

    coords = cc.get_coords()
    ngrids = cc.get_ngrids()

    blksize = min(8000, ngrids)
    rho = numpy.empty(ngrids)
    for ip0, ip1 in lib.prange(0, ngrids, blksize):
        ao = mol.eval_gto(gto_val, coords[ip0:ip1])
        rho[ip0:ip1] = numint.eval_rho(mol, ao, dm)

    rho = rho.reshape(cc.nx, cc.ny, cc.nz)

    return rho, cc
