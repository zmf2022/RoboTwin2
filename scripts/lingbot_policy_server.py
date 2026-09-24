"""LingBot-VLA 2.0 websocket policy server that loads its training config from $LINGBOT_CLI_YAML.

The upstream loader reads <model_path>/../../../lingbotvla_cli.yaml, which a flat model dir
(e.g. the released base model) does not have. Same CLI as deploy/lingbot_vla_v2_policy.py.
"""
import builtins
import os
import sys
from pathlib import Path

LINGBOT = Path(__file__).resolve().parents[1] / "lingbot-vla-v2"
sys.path.insert(0, str(LINGBOT))
os.chdir(LINGBOT)

import deploy.lingbot_vla_v2_policy as policy  # noqa: E402

CLI_YAML = os.environ["LINGBOT_CLI_YAML"]


def _open(file, *args, **kwargs):
    if str(file).endswith("lingbotvla_cli.yaml"):
        file = CLI_YAML
    return builtins.open(file, *args, **kwargs)


policy.open = _open

if __name__ == "__main__":
    policy.main()
