from typing import Optional

from pyscf_backend.auto_generated import default_port


class Dependencies:
    port: int = default_port
    yw_port: int = 2000


def dependencies():
    return Dependencies()


def set_dependencies(port: Optional[int], yw_port: Optional[int]):
    if port:
        Dependencies.port = port
    if yw_port:
        Dependencies.yw_port = yw_port
