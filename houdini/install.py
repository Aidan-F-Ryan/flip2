#!/usr/bin/env python3
"""Installs flip2's Houdini package for this user: writes flip2.json, pointing at this folder, into the packages directory of every Houdini version
whose preferences are here (or into the one given). Houdini picks it up when it next starts. Run it again after moving this folder; delete the
flip2.json it wrote to uninstall.

  python3 houdini/install.py [packages directory]
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def preference_directories():
    """every Houdini version's user preferences directory on this machine"""
    home = Path.home()
    if sys.platform == "darwin":
        roots = sorted((home / "Library" / "Preferences" / "houdini").glob("[0-9]*.[0-9]*"))
    elif sys.platform == "win32":
        roots = sorted((home / "Documents").glob("houdini[0-9]*.[0-9]*"))
    else:
        roots = sorted(home.glob("houdini[0-9]*.[0-9]*"))
    return [root for root in roots if root.is_dir()]


def main():
    targets = [Path(sys.argv[1])] if len(sys.argv) > 1 else [root / "packages" for root in preference_directories()]
    if not targets:
        sys.exit("found no Houdini preferences: start Houdini once, or give the packages directory")
    package = {"env": [{"FLIP2_HOUDINI": str(HERE)}, {"PYTHONPATH": {"value": "$FLIP2_HOUDINI/python", "method": "prepend"}}],
               "path": "$FLIP2_HOUDINI"}
    for target in targets:
        target.mkdir(parents=True, exist_ok=True)
        (target / "flip2.json").write_text(json.dumps(package, indent=4) + "\n")
        print("wrote", target / "flip2.json")


if __name__ == "__main__":
    main()
