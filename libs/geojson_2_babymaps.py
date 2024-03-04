import argparse
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
from uuid import uuid4

import geojson as gj
import numpy as np
import pandas as pd
import PIL
import rasterio as rio
from osgeo import gdal
from PIL import Image
from rasterio import warp as rio_warp
from rasterio import windows as rio_windows
from tqdm import tqdm

PIL.Image.MAX_IMAGE_PIXELS = None

GEO_MAP_EXTENSIONS: List[str] = [".tif", ".tiff", ".vrt"]
GEO_IMAGE_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png"]


def rio_get_dim(
    src_ds: Union[rio.DatasetReader, Path, str],
) -> Tuple[int, int]:
    if type(src_ds) in [Path, str]:
        src_ds: rio.DatasetReader = rio.open(src_ds)
    dim: Tuple[int, int] = (src_ds.width, src_ds.height)
    return dim


def rio_convert_coordinates_to_pixels_bbox(
    src_ds: Union[rio.DatasetReader, Path],
    coordinates: Union[np.ndarray, List],
) -> np.ndarray:
    if isinstance(src_ds, Path):
        src_ds: rio.DatasetReader = rio.open(src_ds)
    if isinstance(coordinates, list):
        coordinates: np.ndarray = np.array(coordinates)
    min_lng, min_lat = coordinates.min(axis=0)
    max_lng, max_lat = coordinates.max(axis=0)
    min_x, max_y = rio_convert_lnglats_to_pixels(src_ds, min_lng, min_lat)
    max_x, min_y = rio_convert_lnglats_to_pixels(src_ds, max_lng, max_lat)
    bbox: np.ndarray = np.array([min_x, min_y, max_x, max_y])
    return bbox


def rio_convert_lnglats_to_pixels(
    src_ds: Union[rio.DatasetReader, Path, str],
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    options: Dict[str, Any] = {"src_crs": "EPSG:4326"},
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert arrays of longitudes-latitudes coordinates to pixel coordinates using rasterio.

    Args:
        src_ds (Union[rio.DatasetReader, Path, str]): Path to the GeoTIFF dataset.
        longitudes (np.ndarray): Array of longitudes values.
        latitudes (np.ndarray): Array of latitudes values.
        options (Dict[str, Any]): Settings.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays of x-pixel and y-pixel coordinates.
    """
    if type(src_ds) in [Path, str]:
        src_ds: rio.DatasetReader = rio.open(src_ds)
    src_crs: str = options.get("src_crs", "EPSG:4326")
    dst_crs: str = options.get("dst_crs", src_ds.crs)
    dst_lnglats: np.ndarray = np.vstack((longitudes, latitudes))
    dst_xs: List[int] = dst_lnglats[0, :]
    dst_ys: List[int] = dst_lnglats[1, :]
    # Transform coordinates from WGS84 to the dataset's coordinate reference system
    eastings_northings: List[int] = rio_warp.transform(
        src_crs=src_crs,
        dst_crs=dst_crs,
        xs=dst_xs,
        ys=dst_ys,
    )
    # Calculate pixel coordinates
    pixels_yx: Tuple[np.ndarray, np.ndarray] = rio.transform.rowcol(
        src_ds.transform,
        eastings_northings[0],
        eastings_northings[1],
    )
    pixels_y: np.ndarray = np.array(pixels_yx[0])
    pixels_x: np.ndarray = np.array(pixels_yx[1])
    return pixels_x, pixels_y


def crop_with_padding(
    src_ds: Union[rio.DatasetReader, Path],
    dst_ds: Path,
    bbox: np.ndarray,
    format: str = ".tif",
):
    if isinstance(src_ds, Path):
        src_ds = rio.open(src_ds)
    dst_width: int = math.floor(bbox[2] - bbox[0])
    dst_height: int = math.floor(bbox[3] - bbox[1])
    dst_window = rio_windows.Window(bbox[0], bbox[1], dst_width, dst_height)
    dst_transform = src_ds.window_transform(dst_window)
    dst_profile = src_ds.profile.copy()
    dst_profile.update(
        {
            "width": dst_width,
            "height": dst_height,
            "transform": dst_transform,
        }
    )
    crop: np.ndarray = src_ds.read(window=dst_window)
    if format in [".tif", ".tiff", ".vrt"]:
        with rio.open(dst_ds, "w", **dst_profile) as dst_ds:
            dst_ds.write(crop)
    elif format in [".png", ".jpg", ".jpeg"]:
        crop: np.ndarray = crop.transpose(1, 2, 0).astype(np.uint8)
        image: Image.Image = Image.fromarray(crop)
        if format == ".png":
            image: Image.Image = image.convert("RGBA")
        else:
            image: Image.Image = image.convert("RGB")
        image.save(dst_ds)
    else:
        raise ValueError()


def adjust_bbox(
    src_bbox: np.ndarray,
    src_width: int,
    src_height: int,
    padding: int = 50,
) -> np.ndarray:
    """
    x: (latitude, width)
    y: (longitude, height)
    src_bbox: [
        (x1, left),
        (y1, top, upper),
        (x2, right),
        (y2, bottom, lower),
    ]
    """
    height = src_bbox[3] - src_bbox[1]
    width = src_bbox[2] - src_bbox[0]
    
    y1 = np.maximum(src_bbox[1] - padding, 0)
    x1 = np.maximum(src_bbox[0] - padding, 0)
        
    height = height + padding * 2
    width = width + padding * 2
    
    width = np.minimum(width, max(src_width - x1, 1))
    height = np.minimum(height, max(src_height - y1, 1))
    # width = np.minimum(width, src_width - x1)
    # height = np.minimum(height, src_height - y1)
    bbox: np.ndarray = np.array([x1, y1, (x1 + width), (y1 + height)])
    return bbox, width, height


def create_baby_map(
    src_path: Path,
    dst_path: Path,
    coordinates: np.ndarray,
    padding: int = 50,
    format: str = ".jpg",
):
    src_ds: rio.DatasetReader = rio.open(src_path)
    src_bbox: np.ndarray = rio_convert_coordinates_to_pixels_bbox(src_ds, coordinates)
    src_width, src_height = rio_get_dim(src_ds)
    adj_bbox, adj_width, adj_height = adjust_bbox(
        src_bbox,
        src_width,
        src_height,
        padding=padding,
    )
    crop_with_padding(
        src_ds,
        dst_path,
        adj_bbox,
        format=format,
    )


def main(
    detection_geojson_path: Path,
    detection_output_folder: Path,
    stem_geotiff_folder: Path,
    padding: int = 50,
    format: str = ".jpg",
):
    with open(detection_geojson_path, "r") as f:
        gdf: Dict[str, Any] = gj.load(f)
    reference_paths: List[Path] = [
        path
        for path in stem_geotiff_folder.glob(f"**/*.*")
        if path.suffix in [".tif", ".tiff", ".vrt"]
    ]
    reference_stems: List[str] = [k.stem for k in reference_paths]
    reference_df: pd.DataFrame = pd.DataFrame(
        {
            "path": reference_paths,
            "stem": reference_stems,
        }
    )
    reference_df.set_index("stem", drop=True, inplace=True)
    detection_output_folder.mkdir(parents=True, exist_ok=True)

    failed_maps: List = []
    ntotal_features: int = len(gdf["features"])
    print(f"Processing {ntotal_features} features...")
    for k in tqdm(range(ntotal_features)):
        try:
            geojson_feature: Dict[str, Any] = gdf["features"][k]
            reference_key: str = geojson_feature["properties"]["SourceImagery"]

            src_path: Path = reference_df.loc[reference_key, "path"]
            if not isinstance(src_path, Path):
                src_path: Path = src_path[0]
                
            if "crop_url" not in geojson_feature["properties"].keys():
                dst_path: Path = Path(detection_output_folder, f"{str(uuid4())}{format}")
            else:
                dst_path: Path = Path(
                    geojson_feature["properties"]["crop_url"].replace(
                        "https://guardstscus.blob.core.windows.net/planet-detect-crops",
                        str(detection_output_folder),
                    )
                ).with_suffix(format)
            coordinates: np.ndarray = np.array(
                geojson_feature["geometry"]["coordinates"][0]
            )

            create_baby_map(
                src_path,
                dst_path,
                coordinates,
                padding=padding,
                format=format,
            )
        except BaseException as e:
            # failed_maps.append(src_path)
            failed_maps.append(reference_key)
            print(f"Encounted error on element {k} {src_path}:\n\n {str(e)}\n\n, skipping")
            # break

    failed_maps = np.unique(failed_maps)
    print(f"Failed Maps: {len(failed_maps)}\n{failed_maps}")


if __name__ == "__main__":
    """    
    python fetch_geojson_tif.py /datadrive/geojson/D8-aoi-water-data-sources.json /datadrive/D8_2024-02-26/
    
    python geojson_2_babymaps.py /datadrive/geojson/D8-aoi-water-Final.geojson /datadrive/geojson/D8-aoi-water-Final-outputs/ /datadrive/D8_2024-02-26 --format .jpg
    """
    t0: float = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "detection_geojson_path",
        type=str,
        help="Path to the GeoJSON file.",
    ) 
    parser.add_argument(
        "detection_output_folder",
        type=str,
        help="Path to where resulting baby images should be saved.",
    )
    parser.add_argument(
        "stem_geotiff_folder",
        type=str,
        help="Path to the folder of GeoTIFF files (recursively searched).",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--format",
        type=str,
        default=".jpg",
    )
    args = parser.parse_args()

    detection_geojson_path = Path(args.detection_geojson_path)
    detection_output_folder = Path(args.detection_output_folder)
    stem_geotiff_folder = Path(args.stem_geotiff_folder)
    main(
        detection_geojson_path,
        detection_output_folder,
        stem_geotiff_folder,
        padding=args.padding,
        format=args.format,
    )

    print(f"Finished running script in {time.time() - t0} seconds...")
