import os
import sys
import glob
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap


def resolve_result_dir(start_path: Path) -> Path:
    """Resolve the result directory relative to where this script lives.

    The script is expected to live in code/. Results typically live in ../result/.
    Fallbacks try result/ in CWD and absolute path under user's workspace.
    """
    candidates = [
        start_path.parent / "result",            # code/ -> ../result (if script moved)
        start_path.parent.parent / "result",     # code/ -> ../result (normal layout)
        Path.cwd() / "result",
        Path("/Users/shirley/Documents/Research/demand/parkchargebike/result"),
    ]
    for cand in candidates:
        if cand.exists() and cand.is_dir():
            return cand
    # Default to ../result even if not present yet; will error later if needed
    return start_path.parent.parent / "result"


def load_stations(result_dir: Path) -> pd.DataFrame:
    stations_path = None
    for cand in [
        result_dir / "master_stations.csv",
        Path("/Users/shirley/Documents/Research/demand/parkchargebike/result/master_stations.csv"),
    ]:
        if cand.exists():
            stations_path = cand
            break
    if stations_path is None:
        raise FileNotFoundError(
            f"master_stations.csv not found in {result_dir}. Please ensure it exists."
        )
    stations_df = pd.read_csv(stations_path, dtype={"station_id": str})
    if not {"station_id", "lat", "lng"}.issubset(set(stations_df.columns)):
        raise ValueError("master_stations.csv must include columns: station_id, lat, lng")
    stations_df["station_id"] = stations_df["station_id"].astype(str)
    return stations_df[["station_id", "lat", "lng"]].copy()


def build_heatmap(df_plot: pd.DataFrame, title: str, out_html: Path) -> None:
    # Center map at median of points or fallback to Chicago
    center_lat = float(df_plot["lat"].median()) if df_plot["lat"].notna().any() else 41.8781
    center_lng = float(df_plot["lng"].median()) if df_plot["lng"].notna().any() else -87.6298

    m = folium.Map(location=[center_lat, center_lng], zoom_start=10, tiles="cartodbpositron")

    # Prepare heat data with weights from abs_error
    # Robust cap to reduce dominance of extreme outliers
    weights = df_plot["abs_error"].astype(float).values
    if len(weights) == 0:
        # Still write an empty map for consistency
        m.save(str(out_html))
        return
    cap = np.percentile(weights, 99.0) if len(weights) > 10 else weights.max()
    cap = max(cap, 1e-6)
    weights_capped = np.clip(weights, 0, cap)
    # Normalize to [0,1]
    weights_norm = (weights_capped / cap).astype(float)

    heat_data = list(zip(df_plot["lat"].astype(float), df_plot["lng"].astype(float), weights_norm))

    HeatMap(
        heat_data,
        radius=12,
        blur=15,
        max_zoom=12,
        min_opacity=0.2,
    ).add_to(m)

    # Title marker
    folium.map.Marker(
        [center_lat, center_lng],
        icon=folium.DivIcon(html=f"<div style='font-size:14px;font-weight:bold'>{title}</div>")
    ).add_to(m)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))


def main():
    parser = argparse.ArgumentParser(description="Create heatmaps from instance error CSVs.")
    parser.add_argument(
        "--result_dir",
        type=str,
        default=None,
        help="Path to result directory containing instance_errors_*.csv and master_stations.csv",
    )
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    result_dir = Path(args.result_dir) if args.result_dir else resolve_result_dir(script_path)

    stations = load_stations(result_dir)

    csv_paths = sorted(glob.glob(str(result_dir / "instance_errors_*.csv")))
    if not csv_paths:
        print(f"No instance_errors_*.csv found in {result_dir}")
        sys.exit(0)

    for csv_path in csv_paths:
        csv_path = Path(csv_path)
        target = csv_path.stem.replace("instance_errors_", "")
        print(f"Processing {csv_path.name} -> target={target}")

        df_err = pd.read_csv(csv_path)
        # Ensure required columns
        missing_cols = {"station_id", "abs_error"} - set(df_err.columns)
        if missing_cols:
            print(f"  Skipping: missing columns {missing_cols}")
            continue

        df_err["station_id"] = df_err["station_id"].astype(str)
        df_plot = df_err.merge(stations, on="station_id", how="left")
        df_plot = df_plot.dropna(subset=["lat", "lng"])
        if df_plot.empty:
            print("  Skipping: no rows with coordinates after merge")
            continue

        out_html = result_dir / f"error_heatmap_{target}.html"
        build_heatmap(df_plot, f"Error Heatmap - {target}", out_html)
        print(f"  Wrote {out_html}")


if __name__ == "__main__":
    main()


