# standard library
import argparse
import traceback

# third parties
import uvicorn

from pyscf_backend.dependencies import dependencies, set_dependencies
from pyscf_backend.fastapi_app import app

parser = argparse.ArgumentParser()

parser.add_argument("--port", help="Specify the port on which the service is running")
parser.add_argument(
    "--yw_port", help="Specify the port on which the youwol server is running"
)


args = parser.parse_args()

set_dependencies(
    port=int(args.port) if args.port else None,
    yw_port=int(args.yw_port) if args.yw_port else None,
)


def start():
    uvicorn_log_level = "info"

    try:
        uvicorn.run(
            app,
            host="localhost",
            port=dependencies().port,
            log_level=uvicorn_log_level,
        )
    except BaseException as e:
        print("".join(traceback.format_exception(type(e), value=e, tb=e.__traceback__)))
        raise e


start()
