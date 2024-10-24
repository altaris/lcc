"""
Kills every `/nlnas/.venv/bin/python3` process listed by `nvidia-smi`.
"""

import re
import subprocess

if __name__ == "__main__":
    raw = subprocess.check_output(["nvidia-smi"])
    pids = []
    r = re.compile(
        r"\|\s+\d\s+N/A\s+N/A\s+(\d+)\s+C.*/nlnas/\.venv/bin/python3"
    )
    for line in raw.decode("utf-8").split("\n"):
        if m := re.search(r, line):
            pids.append(m.group(1))
    print("PIDS:", pids)
    subprocess.run(["kill", "-9"] + pids, check=False)
