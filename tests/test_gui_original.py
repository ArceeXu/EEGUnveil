"""Keep Xu GUI unchanged and isolated from SDAD training code."""
import hashlib
import importlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class OriginalGuiTests(unittest.TestCase):
    def test_online_sources_match_xu_original_commit(self):
        record = json.loads((ROOT / 'docs/integration/gui_restore.json').read_text(encoding='utf-8'))
        for rel, expected in record['git_blob_sha256'].items():
            with self.subTest(path=rel):
                self.assertEqual(hashlib.sha256((ROOT / rel).read_bytes()).hexdigest(), expected)

    def test_original_gui_uses_original_receiver(self):
        sys.path.insert(0, str(ROOT / 'src/online'))
        gui = importlib.import_module('main_app')
        receiver = importlib.import_module('receiver')
        self.assertIs(gui.BCIProcessor, receiver.BCIProcessor)
        self.assertIs(gui.load_trials, receiver.load_trials)
        self.assertFalse((ROOT / 'src/online/model_registry.py').exists())
        self.assertFalse((ROOT / 'integration/sdad_backend.py').exists())


if __name__ == '__main__':
    unittest.main()
