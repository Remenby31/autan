"""Version and build identity of the SDK and of the schemas it implements."""

from importlib.metadata import version

from autan import _build_info
from autan.meshinfo import SCHEMA_ID as MESH_INFO_SCHEMA
from autan.results import SCHEMA_ID as RESULTS_SCHEMA
from autan.schema import SCHEMA_ID as CASE_SCHEMA

__version__ = version("autan")
COMMIT: str = _build_info.COMMIT
DIRTY: bool = _build_info.DIRTY

SCHEMAS = {"case": CASE_SCHEMA, "mesh-info": MESH_INFO_SCHEMA, "results": RESULTS_SCHEMA}
