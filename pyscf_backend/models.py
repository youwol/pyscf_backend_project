from typing import Literal

from pydantic import BaseModel


class MoleBody(BaseModel):
    atom: str
    basis: str


class BBoxBody(BaseModel):
    nx: int = 80
    ny: int = 80
    nz: int = 80
    margin: float = 3.0


class MethodBody(BaseModel):
    type: Literal["scf.rhf", "dft.rks"] = "scf.rhf"
    params: dict[str, str | float | int] = {}


class CubeBody(BaseModel):
    mole: MoleBody
    bbox: BBoxBody
    method: MethodBody = MethodBody()


class CubeResponse(BaseModel):
    values: list[float]


class MoleResponse(BaseModel):
    elements: list[str]
    coordinates: list[list[float]]
    groupName: str
    basis: str
    charge: int
    electronCount: int
