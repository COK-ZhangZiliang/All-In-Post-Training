from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

from .catalog import DEFAULT_DATA_PATH, load_catalog


PACKAGE_ROOT = Path(__file__).resolve().parent
TEMPLATE_ROOT = PACKAGE_ROOT / "templates"
PROJECT_ROOT = PACKAGE_ROOT.parents[1]


def build_site(
    data_path: Path | str = DEFAULT_DATA_PATH,
    out_dir: Path | str = Path("site"),
) -> Path:
    data = load_catalog(data_path)
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)

    html_template = (TEMPLATE_ROOT / "index.html").read_text(encoding="utf-8")
    rendered = html_template.replace("__PANORAMA_JSON__", _json_script_payload(data))
    rendered = rendered.replace("__PANORAMA_TITLE__", html.escape(data["meta"]["name"]))

    (destination / "index.html").write_text(rendered, encoding="utf-8")
    shutil.copyfile(TEMPLATE_ROOT / "styles.css", destination / "styles.css")
    shutil.copyfile(TEMPLATE_ROOT / "app.js", destination / "app.js")
    shutil.copyfile(PROJECT_ROOT / "assets" / "icon.svg", destination / "icon.svg")
    return destination / "index.html"


def _json_script_payload(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2).replace("</", "<\\/")

