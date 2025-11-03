import os
import sys
import re
import pandas as pd
from climate_viewer_config import FOLDER

def resolve_folder() -> str:
    if not os.path.isdir(FOLDER):
        sys.exit(f"Invalid folder (metrics root): {FOLDER}")
    return FOLDER

def dedupe_preserve_order(items):
    seen, out = set(), []
    for x in items or []:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")

def parse_data_type(series: pd.Series) -> pd.DataFrame:
    return series.str.extract(
        r"^(?P<Type>[^ ]+) \((?P<Name>[^,]+), (?P<Location>[^)]+)\)$"
    )
