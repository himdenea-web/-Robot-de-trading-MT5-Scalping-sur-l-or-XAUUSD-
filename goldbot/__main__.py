"""Run the bot live on MetaTrader 5.

    python -m goldbot                    # settings from .env (alert mode by default)
    python -m goldbot --mode trade       # place orders (demo account unless GB_REQUIRE_DEMO=false)
    python -m goldbot --llm on           # enable the committee (provider from GB_LLM_PROVIDER)
    python -m goldbot --check            # connect, print settings and account state, then exit

For unattended operation (VPS or always-on laptop) see deploy/README.md.
"""
import argparse
import sys

from goldbot.config import Config


def main():
    ap = argparse.ArgumentParser(prog="goldbot")
    ap.add_argument("--env", default=".env")
    ap.add_argument("--mode", choices=["alert", "trade"])
    ap.add_argument("--llm", choices=["on", "off"])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    cfg = Config.from_env(a.env)
    if a.mode:
        cfg.mode = a.mode
    if a.llm:
        cfg.judge.enabled = a.llm == "on"
    cfg.validate()

    from goldbot.live import FatalError, run
    try:
        return run(cfg, check=a.check)
    except FatalError as e:
        print(f"goldbot: {e}", file=sys.stderr)
        return e.code      # the watchdog does not restart on these codes


if __name__ == "__main__":
    sys.exit(main())
