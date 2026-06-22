from __future__ import annotations

import unittest

from all_in_post_training.catalog import CatalogError, catalog_stats, load_catalog, validate_catalog


class CatalogTest(unittest.TestCase):
    def test_seed_catalog_is_valid(self) -> None:
        data = load_catalog("data/panorama.json")
        stats = catalog_stats(data)
        self.assertGreaterEqual(stats.tracks, 6)
        self.assertGreaterEqual(stats.nodes, 10)
        self.assertGreaterEqual(stats.references, 10)

    def test_rejects_unknown_edge_endpoint(self) -> None:
        data = load_catalog("data/panorama.json")
        broken = dict(data)
        broken["edges"] = [*data["edges"], {"source": "missing", "target": "grpo", "relation": "x", "summary": "x"}]
        with self.assertRaises(CatalogError):
            validate_catalog(broken)


if __name__ == "__main__":
    unittest.main()

