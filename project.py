"""One project, separate original training protocols. No training on invocation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SDAD = ROOT / 'sdad_integration/code'
TARGETS = {
    'gen1': ('正常组、无蒸馏', ROOT, ['src/offline/run_openbmi_holdout.py', '--train-population', 'normal', '--sampling', 'natural', '--no-distillation']),
    'gen2': ('混合训练、同步蒸馏', ROOT, ['src/offline/run_openbmi_holdout.py', '--train-population', 'all', '--sampling', 'mixed', '--mix-lambda', '0.5625']),
    'sdad42': ('后期配对42/42', SDAD, ['repeat_group_domain42.py']),
    'sdad43': ('后期配对43/43', SDAD, ['late_subject_domain.py']),
    'sdad44': ('后期配对43/44', SDAD, ['repeat_group_domain44.py']),
    'test-sdad42': ('固定检查点测试42/42', SDAD, ['evaluate_group_domain42.py']),
    'test-sdad43': ('固定检查点测试43/43', SDAD, ['evaluate_subject_domain.py']),
    'test-sdad44': ('固定检查点测试43/44', SDAD, ['evaluate_group_domain44.py']),
    'csp-reference': ('SDAD协议传统参考', SDAD, ['compare_pooled_csp_lda.py']),
    'gui': ('Xu原版OpenBMI界面，不增加SDAD模型适配', ROOT, ['src/online/main_app.py']),
    'hardware': ('原v50在线推理、小车TCP/串口链路；默认只检查文件', ROOT, ['hardware_legacy/launch.py']),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('list', help='List methods; does not load EEG or models')
    run = sub.add_parser('run', help='Forward arguments to the original entry point')
    run.add_argument('target', choices=TARGETS)
    run.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.action == 'list':
        print(json.dumps({k: {'description': v[0], 'working_directory': str(v[1]), 'command': v[2]} for k, v in TARGETS.items()}, ensure_ascii=False, indent=2))
        return 0
    forwarded = args.arguments
    if forwarded[:1] == ['--']:
        forwarded = forwarded[1:]
    if args.target in ('gen1', 'gen2'):
        reserved = {'--train-population', '--sampling', '--mix-lambda', '--no-distillation'}
        if any(token.split('=')[0] in reserved for token in forwarded):
            parser.error('Generation presets are fixed; use original entry for other ablations.')
    _, cwd, command = TARGETS[args.target]
    return subprocess.call([sys.executable, *command, *forwarded], cwd=cwd)


if __name__ == '__main__':
    raise SystemExit(main())
