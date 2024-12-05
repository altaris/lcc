"""
Kills every `/lcc/.venv/bin/python3` process listed by `nvidia-smi`, and then
every `/lcc/.venv/bin/python3 -m lcc` process listed by `ps`.
"""

import re
import subprocess

if __name__ == "__main__":
    print()
    print("        ⣀⣠⣀⣀  ⣀⣤⣤⣄⡀           ")
    print("   ⣀⣠⣤⣤⣾⣿⣿⣿⣿⣷⣾⣿⣿⣿⣿⣿⣶⣿⣿⣿⣶⣤⡀    ")
    print(" ⢠⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷    ")
    print(" ⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⡀ ")
    print(" ⢀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇ ")
    print(" ⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠟⠁ ")
    print("  ⠻⢿⡿⢿⣿⣿⣿⣿⠟⠛⠛⠋⣀⣀⠙⠻⠿⠿⠋⠻⢿⣿⣿⠟    ")
    print("      ⠈⠉⣉⣠⣴⣷⣶⣿⣿⣿⣿⣶⣶⣶⣾⣶        ")
    print("        ⠉⠛⠋⠈⠛⠿⠟⠉⠻⠿⠋⠉⠛⠁        ")
    print("              ⣶⣷⡆             ")
    print("      ⢀⣀⣠⣤⣤⣤⣤⣶⣿⣿⣷⣦⣤⣤⣤⣤⣀⣀      ")
    print("    ⢰⣿⠛⠉⠉⠁   ⢸⣿⣿⣧    ⠉⠉⠙⢻⣷    ")
    print("     ⠙⠻⠷⠶⣶⣤⣤⣤⣿⣿⣿⣿⣦⣤⣤⣴⡶⠶⠟⠛⠁    ")
    print("          ⢀⣴⣿⣿⣿⣿⣿⣿⣷⣄          ")
    print("         ⠒⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠓         ")
    print()

    raw = subprocess.check_output(["nvidia-smi"])
    pids = []
    r = re.compile(r"\|\s+\d\s+N/A\s+N/A\s+(\d+)\s+C.*/lcc/\.venv/bin/python3")
    for line in raw.decode("utf-8").split("\n"):
        if m := re.search(r, line):
            pids.append(m.group(1))

    if not pids:
        print("[nvidia-smi] No matching processes found.")
    else:
        print("[nvidia-smi] Nuking processes:", pids)
        subprocess.run(["kill", "-9"] + pids, check=True)

    raw = subprocess.check_output(["ps", "-u", "cedric", "-eo", "pid,comm"])
    pids = []
    r = re.compile(r"(\d+) .*/lcc/\.venv/bin/python3.*m lcc.*")
    for line in raw.decode("utf-8").split("\n"):
        if m := re.search(r, line):
            pids.append(m.group(1))

    if not pids:
        print("[ps] No matching processes found.")
    else:
        print("[ps] Nuking processes:", pids)
        subprocess.run(["kill", "-9"] + pids, check=True)
