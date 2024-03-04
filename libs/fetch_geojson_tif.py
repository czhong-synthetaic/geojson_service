import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import geojson as gj
import numpy as np
from tqdm import tqdm


GLOBAL_PROD_SAS_KEY = os.getenv("GLOBAL_PROD_SAS_KEY", "?sv=2022-11-02&ss=bfqt&srt=sco&sp=rltf&se=2025-02-22T00:56:36Z&st=2024-02-21T16:56:36Z&spr=https&sig=8eiBI1IhuRXam9TsFNI2ucSCGndaHM8w7CzOdEh4fzI%3D")
GUARD_TIF_STORAGE_ACCOUNT = os.getenv("GUARD_TIF_STORAGE_ACCOUNT", "https://guardstscus.blob.core.windows.net/planet-dls")
GUARD_TIF_SAS_KEY = os.getenv("GUARD_TIF_SAS_KEY", "?sp=rl&st=2024-02-14T00:45:32Z&se=2025-02-14T08:45:32Z&spr=https&sv=2022-11-02&sr=c&sig=HJdHWPNMzRxjwiaEF1vsbnFQDeU1cHImFgOW%2Bu46G0M%3D")
GUARD_DATAFRAME_STORAGE_ACCOUNT = os.getenv("GUARD_DATAFRAME_STORAGE_ACCOUNT", "https://guardstscus.blob.core.windows.net/reference-dataframes")
GUARD_DATAFRAME_SAS_KEY = os.getenv("GUARD_DATAFRAME_SAS_KEY", "?sp=rl&st=2024-02-14T02:43:18Z&se=2025-02-14T10:43:18Z&spr=https&sv=2022-11-02&sr=c&sig=xHidffOdzqBjSlEyo6Qbva7RECRe6wOayMnKT%2BivX7A%3D")


def main(
    report_geojson_path: Path,
    output_folder: Path,
):
    with open(report_geojson_path, "r") as f:
        dsources = json.load(f)
    
    guardguids = [k["job_id"] for k in dsources]
    dsetguids = [k["data_source_container_name"] for k in dsources]
    all_cmds = []
    
    # command to download all tiffs:
    tiffs_cmd = [
        f'azcopy cp --overwrite=false --recursive "{GUARD_TIF_STORAGE_ACCOUNT}/{k}{GUARD_TIF_SAS_KEY}" "{output_folder}"'
        for k in guardguids
    ]
    all_cmds.extend(tiffs_cmd)

    # command to download all dataframes
    dataframes_cmd = [
        f'azcopy cp "{GUARD_DATAFRAME_STORAGE_ACCOUNT}/{k}-meta.parquet{GUARD_DATAFRAME_SAS_KEY}" "{output_folder}"'
        for k in guardguids
    ]
    all_cmds.extend(dataframes_cmd)    

    # Run commands
    all_cmds = " && ".join(all_cmds)
    print(f"Running Azcopy Commands:\n\n{all_cmds}\n\n")

    run_process = subprocess.run(all_cmds, shell=True)
    print(f"Azcopy Run Process:\n\n{run_process}\n\n")
    

if __name__ == "__main__":
    t0: float = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "report_geojson_path",
        type=str,
        help="Path to the GeoJSON report file.",
    )
    parser.add_argument(
        "output_folder",
        default="/datadrive/D8_2024-02-26/",
        type=str,
        help="Path to where resulting baby images should be saved.",
    )
    args = parser.parse_args()

    report_geojson_path = Path(args.report_geojson_path)
    output_folder = Path(args.output_folder)
    main(
        report_geojson_path,
        output_folder,
    )

    print(f"Finished running script in {time.time() - t0} seconds...")
