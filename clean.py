"""
Looks at a sweep output folder and deletes all the `version_*` folders that are
not mention on `results.*.json` output files.
"""

import json
import shutil
import sys
from pathlib import Path
from sys import exit

from loguru import logger as logging

if __name__ == "__main__":
    answer = input(
        "This script MUST NOT be run while a sweep is running. Continue? [y/n] "
    )
    if answer.lower() != "y":
        logging.info("Aborted")
        exit(0)

    sweep_dir = Path(sys.argv[1])
    logging.info("Sweep path: {}", sweep_dir)

    results_files = list(sweep_dir.glob("results.*.json"))
    if not results_files:
        logging.warning("No results files found")
        exit(0)

    results_versions = set()
    for fn in results_files:
        try:
            with fn.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
            v = data["training"]["best_checkpoint"]["version"]
            results_versions.add(v)
        except Exception as e:
            logging.error("Error reading {}: {}", fn, e)

    logging.info(
        "Found {} versions from results json files: {}",
        len(results_versions),
        results_versions,
    )

    pl_dirs = list(sweep_dir.glob("*_logs/*/version_*"))
    logging.info("Found {} version folders", len(pl_dirs))
    log_folders = []
    for dn in pl_dirs:
        v = int(dn.name.split("_")[-1])
        found = v in results_versions
        if not found:
            log_folders.append(dn)
    if not log_folders:
        logging.info("No folders to delete")
        exit(0)

    logging.info(
        "Found {} folders to delete: {}",
        len(log_folders),
        [str(dn) for dn in log_folders],
    )
    answer = input("Delete? [y/n] ")
    if answer.lower() != "y":
        logging.info("Aborted")
        exit(0)
    for dn in log_folders:
        logging.info("Deleting {}", dn)
        shutil.rmtree(dn, ignore_errors=True)
