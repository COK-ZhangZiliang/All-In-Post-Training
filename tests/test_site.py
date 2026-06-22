from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from all_in_post_training.site import build_site


class SiteBuildTest(unittest.TestCase):
    def test_build_site_writes_static_assets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = build_site("data/panorama.json", Path(directory))
            self.assertTrue(output.exists())
            self.assertTrue((Path(directory) / "styles.css").exists())
            self.assertTrue((Path(directory) / "app.js").exists())
            self.assertIn("All-In Post-Training", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

