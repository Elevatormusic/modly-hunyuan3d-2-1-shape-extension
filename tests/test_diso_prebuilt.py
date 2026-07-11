# tests/test_diso_prebuilt.py
import hashlib
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BUNDLE = REPO / "prebuilt" / "win_amd64-cp311-cu128"
WHEEL = BUNDLE / "diso-0.1.4-cp311-cp311-win_amd64.whl"
SHA = "e060f36edf5b79fd4be8d6db4c782e7729ebbf838f2ae78f07321424baf093fa"


class TestDisoPrebuilt(unittest.TestCase):
    def test_wheel_present_and_hash_matches(self):
        self.assertTrue(WHEEL.is_file(), "diso wheel missing from prebuilt bundle")
        h = hashlib.sha256(WHEEL.read_bytes()).hexdigest()
        self.assertEqual(h, SHA)

    def test_provenance_records_diso_hash_and_nc_license(self):
        prov = (BUNDLE / "PROVENANCE.md").read_text(encoding="utf-8")
        self.assertIn(SHA, prov)
        self.assertIn("diso", prov.lower())
        self.assertIn("BY-NC", prov)              # NonCommercial license recorded

    def test_license_text_shipped(self):
        lic = (BUNDLE / "diso.LICENSE").read_text(encoding="utf-8")
        self.assertIn("Attribution-NonCommercial", lic)

    def test_notice_attributes_diso(self):
        notice = (REPO / "NOTICE").read_text(encoding="utf-8")
        self.assertIn("diso", notice.lower())


if __name__ == "__main__":
    unittest.main()
