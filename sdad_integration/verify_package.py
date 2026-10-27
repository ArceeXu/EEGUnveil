"""Standard-library integrity and syntax check. Does not import or train models."""
from pathlib import Path
import ast
import hashlib
import json

def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    errors = []
    checked = 0
    syntax = 0
    for item in manifest['files']:
        if not item['package_path']:
            continue
        path = root / item['package_path']
        if not path.is_file():
            errors.append('Missing: ' + item['package_path'])
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            errors.append('Changed source: ' + item['package_path'])
        checked += 1
        if item['package_path'].startswith('code/') and path.suffix == '.py':
            try:
                ast.parse(path.read_bytes(), filename=str(path))
                syntax += 1
            except SyntaxError as exc:
                errors.append(str(exc))
    print(json.dumps({'source_commit': manifest['source_commit'], 'source_files_checked': checked,
                      'runtime_python_files_parsed': syntax, 'errors': errors}, indent=2))
    raise SystemExit(1 if errors else 0)

if __name__ == '__main__':
    main()
