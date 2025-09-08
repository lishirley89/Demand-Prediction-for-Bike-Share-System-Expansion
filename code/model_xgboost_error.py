import os
import sys
import glob
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap
import json


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


def build_heatmap(df_plot: pd.DataFrame, title: str, out_html: Path, error_type: str = "abs") -> None:
    # Center map at median of points or fallback to Chicago
    center_lat = float(df_plot["lat"].median()) if df_plot["lat"].notna().any() else 41.8781
    center_lng = float(df_plot["lng"].median()) if df_plot["lng"].notna().any() else -87.6298

    m = folium.Map(location=[center_lat, center_lng], zoom_start=10, tiles="cartodbpositron")

    # Calculate sMAPE if needed
    if error_type == "smape":
        # Calculate sMAPE: |y_true - y_pred| / ((|y_true| + |y_pred|) / 2) * 100
        y_true_abs = np.abs(df_plot["y_true"].astype(float))
        y_pred_abs = np.abs(df_plot["y_pred"].astype(float))
        denominator = (y_true_abs + y_pred_abs) / 2
        # Avoid division by zero by setting minimum denominator
        denominator = np.maximum(denominator, 1e-6)
        df_plot = df_plot.copy()
        df_plot["smape"] = (df_plot["abs_error"] / denominator) * 100
        error_col = "smape"
        error_label = "sMAPE (%)"
    else:
        error_col = "abs_error"
        error_label = "abs_error"

    # Prepare heat data with weights from error column
    # Robust cap to reduce dominance of extreme outliers
    weights = df_plot[error_col].astype(float).values
    if len(weights) == 0:
        # Still write an empty map for consistency
        m.save(str(out_html))
        return
    cap = np.percentile(weights, 99.0) if len(weights) > 10 else weights.max()
    cap = max(cap, 1e-6)
    weights_capped = np.clip(weights, 0, cap)
    # Normalize to [0,1] and apply a small floor so very small weights remain visible
    weights_norm = (weights_capped / cap).astype(float)
    weights_vis = np.maximum(weights_norm, 0.05)

    # Create data array for JS: [lat, lng, weight_norm, error_value]
    heat_data_full = [
        [float(lat), float(lng), float(w_norm), float(err_val)]
        for lat, lng, w_norm, err_val in zip(
            df_plot["lat"].values,
            df_plot["lng"].values,
            weights_norm,
            df_plot[error_col].values,
        )
    ]

    # Add an empty HeatMap to ensure Leaflet.heat assets are loaded; actual data added via JS
    HeatMap([], radius=12, blur=15, max_zoom=12, min_opacity=0.2).add_to(m)

    # Add slider UI and JS to (re)build heat layer by error threshold
    slider_html = f"""
    <div id='err-slider-container' style='position: fixed; top: 10px; left: 50px; z-index:9999; background: white; padding: 8px; border: 1px solid #ccc; border-radius: 4px;'>
      <div style='font-weight:600;margin-bottom:4px;'>{title}</div>
      {error_label} ≥ <span id='thrVal'>0.00</span> (max {cap:.2f})<br>
      <input type='range' id='errSlider' min='0' max='{cap:.6f}' step='{max(cap/200.0, 0.000001):.6f}' value='0' style='width:260px;'>
    </div>
    """

    script = f"""
    <script>
    (function() {{
      var data = {json.dumps(heat_data_full)}; // [lat,lng,weight_norm,error_value]
      var map = null;

      function getMap() {{
        if (!map) {{
          // Find the map object from the global scope
          for (var key in window) {{
            if (window[key] && window[key]._container && window[key]._container.className.includes('leaflet-container')) {{
              map = window[key];
              break;
            }}
          }}
        }}
        return map;
      }}

      function buildHeat(thr) {{
        var mapObj = getMap();
        if (!mapObj) {{ return; }}
        var pts = [];
        for (var i = 0; i < data.length; i++) {{
          if (data[i][3] >= thr) {{ pts.push([data[i][0], data[i][1], data[i][2]]); }}
        }}
        if (window._heatLayer) {{ mapObj.removeLayer(window._heatLayer); }}
        window._heatLayer = L.heatLayer(pts, {{radius:12, blur:15, maxZoom:12, minOpacity:0.2}}).addTo(mapObj);
      }}

      function initBindings() {{
        var slider = document.getElementById('errSlider');
        var thrSpan = document.getElementById('thrVal');
        if (!slider || !thrSpan) {{ return false; }}
        slider.addEventListener('input', function() {{
          thrSpan.textContent = parseFloat(this.value).toFixed(2);
          buildHeat(parseFloat(this.value));
        }});
        thrSpan.textContent = parseFloat(slider.value).toFixed(2);
        buildHeat(parseFloat(slider.value));
        return true;
      }}

      // Wait for map to be ready
      window.addEventListener('load', function() {{
        var tries = 0;
        var timer = setInterval(function() {{
          tries += 1;
          if (getMap() && initBindings()) {{
            clearInterval(timer);
          }} else if (tries > 100) {{
            clearInterval(timer);
          }}
        }}, 100);
      }});
    }})();
    </script>
    """

    # Title marker for quick context
    folium.map.Marker(
        [center_lat, center_lng],
        icon=folium.DivIcon(html=f"<div style='font-size:14px;font-weight:bold'>{title}</div>")
    ).add_to(m)

    m.get_root().html.add_child(folium.Element(slider_html))
    m.get_root().html.add_child(folium.Element(script))

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

        # For sMAPE, we also need y_true and y_pred
        if "y_true" not in df_err.columns or "y_pred" not in df_err.columns:
            print(f"  Skipping: missing y_true or y_pred columns for sMAPE calculation")
            continue

        df_err["station_id"] = df_err["station_id"].astype(str)
        df_plot = df_err.merge(stations, on="station_id", how="left")
        df_plot = df_plot.dropna(subset=["lat", "lng"])
        if df_plot.empty:
            print("  Skipping: no rows with coordinates after merge")
            continue

        # Create absolute error heatmap
        out_html_abs = result_dir / f"error_heatmap_abs_{target}.html"
        build_heatmap(df_plot, f"Absolute Error Heatmap - {target}", out_html_abs, "abs")
        print(f"  Wrote {out_html_abs}")

        # Create sMAPE heatmap
        out_html_smape = result_dir / f"error_heatmap_smape_{target}.html"
        build_heatmap(df_plot, f"sMAPE Heatmap - {target}", out_html_smape, "smape")
        print(f"  Wrote {out_html_smape}")


if __name__ == "__main__":
    main()


