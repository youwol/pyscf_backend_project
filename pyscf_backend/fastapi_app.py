# third parties
import asyncio
import json

import numpy as np
from fastapi import FastAPI
from pyscf import gto
from starlette.requests import Request
from starlette.responses import Response
from youwol.utils.context import ContextFactory

from pyscf_backend.auto_generated import version
from pyscf_backend.dependencies import dependencies
from pyscf_backend.implementation import CacheStore, density
from pyscf_backend.models import CubeBody, CubeResponse, MoleBody, MoleResponse

app: FastAPI = FastAPI(
    title="PySCF backend",
    root_path=f"http://localhost:{dependencies().yw_port}/backends/pyscf_backend/{version}",
)


@app.get("/")
async def home():
    # When proxied through py-youwol, this end point is always triggered, when testing weather a backend
    # is listening. The line is `if not self.is_listening():` in `RedirectSwitch`
    return Response(status_code=200)


@app.post("/mole")
async def mole(request: Request, body: MoleBody) -> MoleResponse:

    async with ContextFactory.proxied_backend_context(request).start(action="/mole"):
        mol = gto.Mole(atom=body.atom, basis=body.basis)
        mol.build()
        return MoleResponse(
            elements=mol.elements,
            coordinates=mol.atom_coords().tolist(),
            groupName=mol.groupname,
            basis=mol.basis,
            charge=mol.charge,
            electronCount=mol.nelectron,
        )


@app.post("/cube", response_model=CubeResponse)
async def cube(request: Request, body: CubeBody) -> Response:

    async with ContextFactory.proxied_backend_context(request).start(
        action="/cube"
    ) as ctx:

        def iteration_callback(envs):
            print(f"Iteration {envs['cycle']}, E={envs['e_tot']}")
            asyncio.ensure_future(
                ctx.info(f"Iteration {envs['cycle']}, E={envs['e_tot']}")
            )

        await ctx.info("Start computation of cube data", data=body)
        mol = gto.Mole(atom=body.mole.atom, basis=body.mole.basis)
        mol.build()
        cache_key = json.dumps({**body.dict(), "bbox": {}})
        mf = CacheStore.get(cache_key)
        if not mf:
            if body.method.type == "scf.rhf":
                from pyscf import scf

                mf = scf.RHF(mol)
            if body.method.type == "dft.rks":
                from pyscf import dft

                mf = dft.RKS(mol)
            if mf is None:
                raise RuntimeError(f"Method '{body.method.type}' not known")
            for key, value in body.method.params.items():
                setattr(mf, key, value)
            mf.callback = iteration_callback
            mf.run()
            CacheStore.set(cache_key, mf)

        density_matrix = mf.make_rdm1()
        rho, cc = density(
            mol=mol,
            dm=density_matrix,
            nx=body.bbox.nx,
            ny=body.bbox.ny,
            nz=body.bbox.nz,
            margin=body.bbox.margin,
        )
        cube_meta = {
            "shape": [cc.nx, cc.ny, cc.nz],
            "min": np.min(cc.get_coords(), axis=0).tolist(),
            "max": np.max(cc.get_coords(), axis=0).tolist(),
        }
        meta = {
            "cube": cube_meta,
            "mole": MoleResponse(
                elements=mol.elements,
                coordinates=mol.atom_coords().tolist(),
                groupName=mol.groupname,
                basis=mol.basis,
                charge=mol.charge,
                electronCount=mol.nelectron,
            ).dict(),
        }
        return Response(
            content=rho.tobytes(), headers={"X-Content-Metadata": json.dumps(meta)}
        )
