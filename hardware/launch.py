"""Original v50 hardware/replay chain. Default check never opens devices."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def load_original(filename):
    online = ROOT / 'src/online'
    sys.path.insert(0, str(online))
    name = '_legacy_car_' + Path(filename).stem
    spec = importlib.util.spec_from_file_location(name, online / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', default='check', choices=['check', 'sender', 'receiver', 'car'])
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'models/online/v50pro')
    parser.add_argument('--timeline', type=Path, help='Cue timeline matching the actual recording/task')
    parser.add_argument('--data-dir', type=Path, help='Original BCICIV2a GDF directory, sender only')
    parser.add_argument('--subject', type=int, default=3)
    parser.add_argument('--host', help='Sender bind address or receiver server address')
    parser.add_argument('--port', type=int, default=65432)
    parser.add_argument('--serial-port', help='Actual car port, for example COM3 or /dev/ttyUSB2')
    parser.add_argument('--baud', type=int, default=115200)
    args = parser.parse_args()
    if args.action == 'check':
        manifest = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))
        errors = []
        for entry in manifest['files']:
            path = ROOT / entry['hardware_path']
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
                errors.append(entry['hardware_path'])
        cfg = json.loads((args.model_dir / 'model_config.json').read_text(encoding='utf-8-sig'))
        print(json.dumps({'model': 'legacy v50pro four-class', 'config': cfg,
                          'files_checked': len(manifest['files']), 'errors': errors,
                          'devices_opened': False}, ensure_ascii=False, indent=2))
        return int(bool(errors))
    if args.action == 'sender':
        if args.data_dir is None or not args.data_dir.is_dir():
            parser.error('sender requires an existing --data-dir containing original GDF files')
        module = load_original('sender.py')
        module.DATA_DIR_GDF = args.data_dir.resolve()
        module.SUBJECT_ID = args.subject
        module.HOST, module.PORT = args.host or '0.0.0.0', args.port
        module.run_server()
        return 0
    if args.timeline is None or not args.timeline.is_file():
        parser.error('receiver/car requires an existing --timeline matching the recording or task')
    if args.action == 'car' and not args.serial_port:
        parser.error('car requires an explicit --serial-port for the actual device')
    filename = 'receive_control_utf8.py' if args.action == 'car' else 'receiver.py'
    module = load_original(filename)
    module.ARTIFACTS_DIR = str(args.model_dir.resolve())
    module.TIMELINE_FILE = str(args.timeline.resolve())
    module.HOST, module.PORT = args.host or '127.0.0.1', args.port
    if args.action == 'car':
        module.SERIAL_PORT, module.SERIAL_BAUD = args.serial_port, args.baud
    module.run_visual_receiver()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
