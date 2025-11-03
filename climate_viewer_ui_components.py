from typing import Literal
from climate_viewer_helpers import dedupe_preserve_order, slugify

def multi_selector(
        container,
        label: str,
        options,
        default=None,
        widget: Literal["checkbox", "toggle"] = "checkbox",
        columns: int = 1,
        key_prefix: str = "sel",
        namespace: str = None
):
    opts = dedupe_preserve_order(options or [])
    default = set(default or [])
    selected = []

    container.markdown(f"**{label}**")
    cols = container.columns(columns)
    widget_fn = container.checkbox if widget == "checkbox" else container.toggle

    import hashlib
    opts_hash = hashlib.md5(str(sorted(opts)).encode()).hexdigest()[:8]

    for i, opt in enumerate(opts):
        key = f"{namespace + '_' if namespace else ''}{key_prefix}_{i}_{opts_hash}_{slugify(opt)}"
        with cols[i % columns]:
            if widget_fn(str(opt), value=(opt in default), key=key):
                selected.append(opt)
    return selected
