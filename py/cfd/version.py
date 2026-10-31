"""Version and build identity of the SDK and of the schemas it implements."""

from importlib.metadata import version

from cfd import _build_info
from cfd.meshinfo import SCHEMA_ID as MESH_INFO_SCHEMA
from cfd.results import SCHEMA_ID as RESULTS_SCHEMA
from cfd.schema import SCHEMA_ID as CASE_SCHEMA

__version__ = version("cfd")
COMMIT: str = _build_info.COMMIT
DIRTY: bool = _build_info.DIRTY

SCHEMAS = {"case": CASE_SCHEMA, "mesh-info": MESH_INFO_SCHEMA, "results": RESULTS_SCHEMA}
