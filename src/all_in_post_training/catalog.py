from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_DATA_PATH = Path("data/panorama.json")


class CatalogError(ValueError):
    """Raised when the panorama catalog is structurally invalid."""


@dataclass(frozen=True)
class CatalogStats:
    tracks: int
    references: int
    nodes: int
    edges: int
    tags: int


def load_catalog(path: Path | str = DEFAULT_DATA_PATH) -> dict[str, Any]:
    catalog_path = Path(path)
    with catalog_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    validate_catalog(data)
    return data


def validate_catalog(data: dict[str, Any]) -> None:
    required_sections = ("meta", "tracks", "references", "nodes", "edges")
    for section in required_sections:
        if section not in data:
            raise CatalogError(f"missing required section: {section}")

    track_ids = _validate_unique_items(data["tracks"], "tracks")
    reference_ids = _validate_unique_items(data["references"], "references")
    node_ids = _validate_unique_items(data["nodes"], "nodes")

    for track in data["tracks"]:
        _require_fields(track, ("id", "title", "summary"), "track")

    for reference in data["references"]:
        _require_fields(reference, ("id", "title", "year", "url", "kind"), "reference")
        if not str(reference["url"]).startswith(("https://", "http://")):
            raise CatalogError(f"reference {reference['id']} has a non-web URL")

    for node in data["nodes"]:
        _require_fields(
            node,
            ("id", "track", "title", "phase", "status", "summary", "why", "tags", "references"),
            "node",
        )
        if node["track"] not in track_ids:
            raise CatalogError(f"node {node['id']} uses unknown track {node['track']}")
        for reference_id in node["references"]:
            if reference_id not in reference_ids:
                raise CatalogError(f"node {node['id']} uses unknown reference {reference_id}")

    for edge in data["edges"]:
        _require_fields(edge, ("source", "target", "relation", "summary"), "edge")
        if edge["source"] not in node_ids:
            raise CatalogError(f"edge source {edge['source']} is not a known node")
        if edge["target"] not in node_ids:
            raise CatalogError(f"edge target {edge['target']} is not a known node")
        if edge["source"] == edge["target"]:
            raise CatalogError(f"edge {edge['source']} -> {edge['target']} points to itself")


def catalog_stats(data: dict[str, Any]) -> CatalogStats:
    validate_catalog(data)
    tags = {tag for node in data["nodes"] for tag in node.get("tags", [])}
    return CatalogStats(
        tracks=len(data["tracks"]),
        references=len(data["references"]),
        nodes=len(data["nodes"]),
        edges=len(data["edges"]),
        tags=len(tags),
    )


def _validate_unique_items(items: list[dict[str, Any]], label: str) -> set[str]:
    seen: set[str] = set()
    for item in items:
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id:
            raise CatalogError(f"{label} contains an item without a string id")
        if item_id in seen:
            raise CatalogError(f"{label} contains duplicate id: {item_id}")
        seen.add(item_id)
    return seen


def _require_fields(item: dict[str, Any], fields: tuple[str, ...], label: str) -> None:
    missing = [field for field in fields if field not in item]
    if missing:
        item_id = item.get("id", "<unknown>")
        raise CatalogError(f"{label} {item_id} missing fields: {', '.join(missing)}")

