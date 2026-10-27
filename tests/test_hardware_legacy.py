"""Restored hardware code checks without opening a socket or serial port."""
import importlib.util
import json
from pathlib import Path
import queue
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_hardware_launcher_tests', ROOT / 'hardware_legacy/launch.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class HardwareLegacyTests(unittest.TestCase):
    def test_real_v50_weights_load_and_synthetic_prediction(self):
        torch.set_num_threads(2)
        receiver = launcher.load_original('receiver.py')
        cfg = json.loads((ROOT / 'hardware_legacy/models/online/v50pro/model_config.json').read_text())
        receiver.ARTIFACTS_DIR = str(ROOT / 'hardware_legacy/models/online/v50pro')
        receiver.TIMELINE_FILE = str(ROOT / 'hardware_legacy/outputs/timelines/cue_timeline0tt.txt')
        processor = receiver.BCIProcessor(cfg['n_channels'])
        processor.buffer[:] = np.random.default_rng(42).normal(size=processor.buffer.shape).astype(np.float32) * 1e-6
        result = processor.run_inference()
        self.assertIsNotNone(result)
        label, confidence = result
        self.assertIn(label, ('Left', 'Right', 'Foot', 'Tongue'))
        self.assertTrue(np.isfinite(confidence))
        self.assertGreaterEqual(confidence, .25)

    def test_car_command_mapping_uses_memory_serial_only(self):
        control = launcher.load_original('receive_control_utf8.py')
        worker = control.SerialControlWorker('unused-test-port', 115200, queue.Queue(),
                                             cruise_speed=300, turn_speed=400)
        expected = {'Left':'$spd:-400,-400,400,400#',
                    'Right':'$spd:400,400,-400,-400#',
                    'Foot':'$pwm:0,0,0,0#',
                    'Tongue':'$spd:300,300,300,300#'}
        sent = []
        class MemorySerial:
            def write(self, data): sent.append(data)
        worker._ser = MemorySerial()
        for label, command in expected.items():
            self.assertEqual(worker._label_to_cmd(label), command)
            worker._send(command)
        self.assertEqual(sent, [(command+'\r\n').encode('utf-8') for command in expected.values()])
        self.assertIsNone(worker._label_to_cmd('unknown'))
        self.assertFalse(worker.is_alive())


if __name__ == '__main__':
    unittest.main()
