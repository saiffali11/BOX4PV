"""
Plot4PV: Solar Data Visualization Tool
--------------------------------------

This Streamlit dashboard allows users to visualise photovoltaic
performance data—including extracted parameters, IPCE curves and JV
characteristics—using interactive plots.  It combines clean UI design
with powerful customisation options such as mismatch correction,
variation colouring, smoothing, curve normalisation and exportable
graphics.

Select a plot type in the sidebar (Box Plot, IPCE Curve or JV Curve) to
reveal the relevant inputs.  The box plot mode supports grouped and
individual variation views of Jsc, Voc, FF and PCE distributions with
custom colours and markers.  Mismatch correction is applied only to
Jsc, and corrected PCE is recalculated as

    PCE = (Voc × Jsc_corr × FF) / 100

with optional toggles to enable or disable this correction.  IPCE
curves are plotted on dual axes (IPCE vs wavelength and integrated
current density on a secondary axis) with the ability to overlay
multiple devices or view them separately.  JV curves display forward
and reverse scans in different colours with optional smoothing and
normalisation.

The interface has been carefully designed to minimise errors and
clipping: slider values are clamped to valid ranges, invalid Plotly
properties have been removed, and default themes are configurable.  All
plots are exportable in high resolution via Kaleido.
"""

import io
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns
# Import matplotlib.colors for RGBA conversion when setting box face colours with opacity
import matplotlib.colors as mcolors

def extract_variation(file_name: str) -> str:
    """Extracts a variation identifier from a file name.

    The variation is assumed to be the substring after the final hyphen
    in the base name.  If no hyphen is present, the entire stem
    (without extension) is returned.
    """
    if not file_name:
        return ""
    base = file_name.split("/")[-1].strip()
    stem = base.rsplit(".", 1)[0] if "." in base else base
    return stem.rsplit("-", 1)[-1] if "-" in stem else stem

def parse_txt(file_buffer) -> pd.DataFrame:
    """Parses a tab‑delimited summary file into a tidy DataFrame.

    Attempts multiple encodings and standardises column names.  A
    ``Variation`` column is derived from the file name.
    """
    encodings = ["utf-8", "utf-8-sig", "utf-16", "utf-16le", "utf-16be", "latin-1"]
    df: Optional[pd.DataFrame] = None
    for enc in encodings:
        try:
            file_buffer.seek(0)
            df = pd.read_csv(file_buffer, sep="\t", encoding=enc)
            break
        except Exception:
            continue
    if df is None:
        file_buffer.seek(0)
        raw = file_buffer.read()
        text = raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else str(raw)
        from io import StringIO
        df = pd.read_csv(StringIO(text), sep="\t")
    rename_map = {
        "File": "File Name",
        "Voc (V)": "Voc",
        "Jsc (mA/cm²)": "Jsc",
        "V_MPP (V)": "V_MPP",
        "J_MPP (mA/cm²)": "J_MPP",
        "P_MPP (mW/cm²)": "P_MPP",
        "Rs (Ohm)": "Rs",
        "R// (Ohm)": "R//",
        "FF (%)": "FF",
        "Eff (%)": "Eff",
    }
    df = df.rename(columns=rename_map)
    for col in ["Voc", "Jsc", "V_MPP", "J_MPP", "P_MPP", "Rs", "R//", "FF", "Eff"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Variation"] = df["File Name"].apply(extract_variation)
    return df

def parse_excel_raw(file_buffer) -> Dict[str, Tuple[float, pd.DataFrame]]:
    """Parses sun‑simulator Excel workbooks for reverse and forward data."""
    result: Dict[str, Tuple[float, pd.DataFrame]] = {}
    xls = pd.ExcelFile(file_buffer)
    mapping = {"RV Data": "RV", "For Data": "FW"}
    for sheet_name, key in mapping.items():
        if sheet_name not in xls.sheet_names:
            continue
        raw = pd.read_excel(xls, sheet_name=sheet_name, header=None)
        factor = 1.0
        try:
            cell = raw.iloc[1, 11]
            if isinstance(cell, (int, float)):
                factor = float(cell)
        except Exception:
            pass
        df = pd.read_excel(xls, sheet_name=sheet_name, header=2, skiprows=[3])
        df = df.dropna(axis=1, how="all")
        rename_map = {
            "Voc (V)": "Voc",
            "Jsc (mA/cm²)": "Jsc",
            "V_MPP (V)": "V_MPP",
            "J_MPP (mA/cm²)": "J_MPP",
            "P_MPP (mW/cm²)": "P_MPP",
            "Rs (Ohm)": "Rs",
            "R// (Ohm)": "R//",
            "FF (%)": "FF",
            "Eff (%)": "Eff",
            "Eff.1": "Eff (Corr)",
        }
        df = df.rename(columns=rename_map)
        df = df[df["File Name"].astype(str).str.lower() != "file"]
        for col in ["Voc", "Jsc", "V_MPP", "J_MPP", "P_MPP", "Rs", "R//", "FF", "Eff", "Jsc Corrected", "Eff (Corr)"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["Variation"] = df["File Name"].apply(extract_variation)
        result[key] = (factor, df)
    return result

def compute_corrected_values(df: pd.DataFrame, mismatch: float) -> pd.DataFrame:
    """Applies mismatch correction to Jsc and recalculates PCE.

    Correction is applied only to the magnitude of Jsc.  PCE is
    recomputed as `(Voc × Jsc_corr × FF) / 100` using FF in percent.
    """
    df = df.copy()
    if mismatch != 0:
        df["Jsc_corrected"] = df["Jsc"].abs() / mismatch
    else:
        df["Jsc_corrected"] = df["Jsc"].abs()
    df["PCE_corrected"] = (df["Voc"] * df["Jsc_corrected"] * df["FF"]) / 100.0
    return df

def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics by variation and scan."""
    metrics = {
        "Jsc_corrected": "Jsc_corrected",
        "Voc": "Voc",
        "FF": "FF",
        "PCE_corrected": "PCE_corrected",
    }
    agg = {m: ["mean", "median", "std", "min", "max"] for m in metrics.values()}
    summary = df.groupby(["Variation", "Scan"]).agg(agg)
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()

def parse_ipce(file_buffer) -> pd.DataFrame:
    """Parse IPCE .txt files to extract wavelength, IPCE and integrated current."""
    file_buffer.seek(0)
    raw = file_buffer.read()
    text = raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else str(raw)
    lines = text.splitlines()
    header_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("Wavelength"):
            header_idx = i
            break
    data_lines = [l for l in lines[header_idx:] if l.strip()]
    if not data_lines:
        return pd.DataFrame()
    csv_str = "\n".join(data_lines)
    df = pd.read_csv(io.StringIO(csv_str), sep="\t")
    rename_map = {
        "Wavelength (nm)": "Wavelength",
        "IPCE (%)": "IPCE",
        "J integrated (mA/cm2)": "J_integrated",
        "J integrated (mA/cm²)": "J_integrated",
    }
    df = df.rename(columns=rename_map)
    for col in ["Wavelength", "IPCE", "J_integrated"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Wavelength", "IPCE"])
    return df[[c for c in ["Wavelength", "IPCE", "J_integrated"] if c in df.columns]]

def parse_jv(file_buffer) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Parse JV summary logs to extract forward and reverse curves."""
    file_buffer.seek(0)
    raw = file_buffer.read()
    text = raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else str(raw)
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("V_FW") or line.startswith("V_FW (V)"):
            header_idx = i
            break
    if header_idx is None:
        return pd.DataFrame(), {}
    data_lines = [l for l in lines[header_idx:] if l.strip()]
    csv_str = "\n".join(data_lines)
    df = pd.read_csv(io.StringIO(csv_str), sep="\t")
    rename_map = {
        "V_FW (V)": "V_FW",
        "J_FW (mA/cm)": "J_FW",
        "J_FW (mA/cm²)": "J_FW",
        "V_RV (V)": "V_RV",
        "J_RV (mA/cm)": "J_RV",
        "J_RV (mA/cm²)": "J_RV",
    }
    df = df.rename(columns=rename_map)
    for col in ["V_FW", "J_FW", "V_RV", "J_RV"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[[c for c in ["V_FW", "J_FW", "V_RV", "J_RV"] if c in df.columns]], {}

def main() -> None:
    """Main entry for Streamlit app."""
    st.set_page_config(page_title="Plot4PV: Solar Data Visualization Tool", layout="wide")
    st.title("Plot4PV: Solar Data Visualization Tool")
    st.markdown(
        """
        Visualize photovoltaic performance with Jsc, Voc, FF, PCE, IPCE, and JV curves.  Upload Excel or
        TXT files to generate clean, export‑ready plots.  Use the sidebar to select a plot type and
        customise your analysis.
        """
    )
    # Global theme selection
    theme_choice = st.sidebar.selectbox("Theme", options=["Light", "Dark"], index=0)
    template_name = "plotly_white" if theme_choice == "Light" else "plotly_dark"

    # Initialise global font settings with sensible defaults.  These will be
    # overridden when the user adjusts font settings in the Box Plot mode.
    font_size_global: int = 12
    bold_title_global: bool = False
    italic_axes_global: bool = False
    font_family_global: str = "Arial"

    # Select plot mode
    mode = st.sidebar.radio("What would you like to plot?", options=["Box Plot", "IPCE Curve", "JV Curve"], index=0)
    # Colour palette for variations
    from plotly.colors import qualitative as qc
    base_palette = qc.Dark24

    if mode == "Box Plot":
        # Upload data
        st.sidebar.header("Upload Summary Files")
        excel_files = st.sidebar.file_uploader("Excel (.xlsx) file(s)", type=["xlsx"], accept_multiple_files=True)
        rv_files = st.sidebar.file_uploader("Reverse scan summary (.txt) file(s)", type=["txt"], accept_multiple_files=True)
        fw_files = st.sidebar.file_uploader("Forward scan summary (.txt) file(s)", type=["txt"], accept_multiple_files=True)
        # Correction toggle and factors
        st.sidebar.header("Mismatch Correction")
        apply_corr = st.sidebar.checkbox("Apply mismatch correction", value=True)
        # Default factors from first Excel file if present
        default_rv = 1.0
        default_fw = 1.0
        excel_data_list: List[Dict[str, Tuple[float, pd.DataFrame]]] = []
        if excel_files:
            for ef in excel_files:
                try:
                    ed = parse_excel_raw(ef)
                    excel_data_list.append(ed)
                    if default_rv == 1.0 and "RV" in ed:
                        default_rv = ed["RV"][0]
                    if default_fw == 1.0 and "FW" in ed:
                        default_fw = ed["FW"][0]
                except Exception as e:
                    st.sidebar.warning(f"Could not parse Excel file '{ef.name}': {e}")
        rv_factor = default_rv
        fw_factor = default_fw
        if apply_corr:
            rv_factor = st.sidebar.number_input("Reverse mismatch factor", value=float(default_rv), min_value=0.0, step=0.01)
            fw_factor = st.sidebar.number_input("Forward mismatch factor", value=float(default_fw), min_value=0.0, step=0.01)
        # Load and correct data
        frames: List[pd.DataFrame] = []
        raw_previews: List[Tuple[str, pd.DataFrame]] = []
        for ed in excel_data_list:
            for key, (fac, df) in ed.items():
                scan = "Reverse" if key == "RV" else "Forward"
                corr = rv_factor if scan == "Reverse" else fw_factor
                raw_previews.append((f"{scan} (Excel)", df.copy()))
                frames.append(compute_corrected_values(df, corr if apply_corr else 1.0).assign(Scan=scan))
        if rv_files:
            for f in rv_files:
                try:
                    dfr = parse_txt(f)
                    raw_previews.append((f"Reverse (TXT) - {f.name}", dfr.copy()))
                    frames.append(compute_corrected_values(dfr, rv_factor if apply_corr else 1.0).assign(Scan="Reverse"))
                except Exception as e:
                    st.sidebar.error(f"Failed to parse {f.name}: {e}")
        if fw_files:
            for f in fw_files:
                try:
                    dff = parse_txt(f)
                    raw_previews.append((f"Forward (TXT) - {f.name}", dff.copy()))
                    frames.append(compute_corrected_values(dff, fw_factor if apply_corr else 1.0).assign(Scan="Forward"))
                except Exception as e:
                    st.sidebar.error(f"Failed to parse {f.name}: {e}")
        if frames:
            combined = pd.concat(frames, ignore_index=True)
        else:
            combined = pd.DataFrame()
        # Show raw preview
        if not combined.empty:
            st.subheader("Raw Data Preview Before Correction")
            for name, df in raw_previews:
                with st.expander(name, expanded=False):
                    st.dataframe(df)
        else:
            st.info("Upload at least one summary file to begin.")
        if combined.empty:
            return
        # Variation definition
        with st.sidebar.expander("Define Variations and Visuals", expanded=True):
            st.write(
                "Enter the number of variations and assign labels, colours and markers.  Rows are matched by substring search on the file name."
            )
            nvars = st.number_input("Number of variations", min_value=1, max_value=100, value=5, step=1)
            marker_opts = [
                "circle", "square", "diamond", "triangle-up", "triangle-down", 
                "triangle-left", "triangle-right", "pentagon", "hexagon", "hexagon2", 
                "star", "star-square", "star-diamond", "cross", "x", "asterisk", 
                "bowtie", "hourglass"
            ]
            user_vars: List[Tuple[str, str, str]] = []
            for i in range(1, int(nvars) + 1):
                default_col = base_palette[(i - 1) % len(base_palette)]
                lbl = st.text_input(f"Variation {i} label", key=f"var_lbl_{i}").strip()
                col = st.color_picker(f"Colour for variation {i}", default_col, key=f"var_col_{i}")
                mrk = st.selectbox(f"Marker for variation {i}", marker_opts, index=(i - 1) % len(marker_opts), key=f"var_mrk_{i}")
                if lbl:
                    user_vars.append((lbl, col, mrk))
        var_color_map: Dict[str, str] = {}
        var_marker_map: Dict[str, str] = {}
        if user_vars:
            def assign(fname: str) -> str:
                fname_lower = str(fname).lower() if pd.notna(fname) else ""
                for lbl, _, _ in user_vars:
                    if lbl.lower() in fname_lower:
                        return lbl
                return "Unassigned"
            combined["Variation"] = combined["File Name"].apply(assign)
            for lbl, col, mrk in user_vars:
                var_color_map[lbl] = col
                var_marker_map[lbl] = mrk
            combined = combined[combined["Variation"] != "Unassigned"].copy()
        # Plot settings
        with st.sidebar.expander("Plot Settings", expanded=True):
            scans = sorted(combined["Scan"].unique())
            selected_scans = st.multiselect("Select scan directions", options=scans, default=scans)
            variations = sorted(combined["Variation"].unique())
            selected_variations = st.multiselect("Select variations", options=variations, default=variations)
            metric_map = {"Jsc": "Jsc_corrected", "Voc": "Voc", "FF": "FF", "PCE": "PCE_corrected"}
            selected_metrics = st.multiselect("Select metrics", list(metric_map.keys()), default=list(metric_map.keys()))
            mode_choice = st.radio("Combine scans or separate", ["Combine", "Separate"], index=0)
            show_boxes = st.checkbox("Show box outlines", value=True)
            # Thickness of the box border
            outline_px = st.slider("Box border thickness (px)", min_value=1, max_value=6, value=2)
            # Whisker width clamp to [0.05,1.0].  Currently unused but retained for UI completeness
            whisker_frac = st.slider("Whisker width (0–1)", min_value=0.05, max_value=1.0, value=0.4)
            # Overlay marker size and opacity
            marker_sz = st.slider("Overlay marker size", min_value=2, max_value=12, value=4)
            marker_opacity = st.slider("Overlay marker opacity", min_value=0.2, max_value=1.0, value=0.7, step=0.05)
            # Overlay style: scatter points, marker shapes using variation markers, or none
            overlay_style = st.selectbox(
                "Overlay style for data points",
                options=["Scatter", "Markers", "None"],
                index=0,
                help="Choose how to display individual data points on the box plot"
            )
            # Jitter spread to avoid overlapping points when overlaying data.  0 = no jitter, 1 = maximal jitter
            jitter_spread = st.slider(
                "Jitter spread (0–1)", min_value=0.0, max_value=1.0, value=0.3, step=0.05,
                help="Controls horizontal spreading of data points when overlaying"
            )
            # Opacity for the filled portion of each box (0=transparent, 1=opaque)
            box_opacity = st.slider(
                "Box opacity", min_value=0.1, max_value=1.0, value=0.7, step=0.05,
                help="Controls the transparency of the coloured boxes"
            )
            # Whether to display a legend for the variations and where to place it
            show_legend = st.checkbox(
                "Show legend for variations", value=True,
                help="Toggle display of the variation legend with coloured markers"
            )
            legend_position = st.selectbox(
                "Legend position",
                options=["bottom", "top", "left", "right"],
                index=1,
                help="Select where to place the legend on the figure"
            )
            # Spacing between box groups
            spacing_val = st.slider("Adjust spacing between box groups (1–100)", 1, 100, 10, help="Smaller values produce tighter plots")
            box_gap = spacing_val / 100.0
            plot_width = st.slider("Plot width (inches)", 4, 20, 10)
            plot_height = st.slider("Plot height (inches)", 3, 15, 6)
            download_fmt = st.selectbox("Download format", ["PNG", "SVG"], index=0).lower()
            export_base = st.text_input("Export file base name", value="plot4pv")
            show_combined = st.checkbox("Show Combined Box Plot (All Selected Variations)", value=False, help="Display all selected variations together for each metric")
            # Font customization
            st.markdown("**Font Settings**")
            font_size = st.slider("Font size", min_value=10, max_value=24, value=12)
            bold_title = st.checkbox("Bold plot titles", value=False)
            italic_axes = st.checkbox("Italic axis labels", value=False)
            font_family = st.selectbox("Font family", options=["Arial", "Times New Roman", "Roboto", "Courier New"], index=0)
            x_axis_rotation = st.slider("X-axis label rotation (degrees)", 0, 90, 30)
            # Axis label visibility options
            # Hide x-axis labels by default and show y-axis labels by default
            show_x_label = st.checkbox("Show X-axis labels", value=False)
            show_y_label = st.checkbox("Show Y-axis labels", value=True)
            # Propagate font settings to global variables for use in other modes
            font_size_global = font_size
            bold_title_global = bold_title
            italic_axes_global = italic_axes
            font_family_global = font_family
            # Propagate axis label visibility to global variables
            show_x_label_global = show_x_label
            show_y_label_global = show_y_label
        # Filters and axes
        with st.sidebar.expander("Filters and Axis Limits", expanded=False):
            filter_ranges: Dict[str, Tuple[float, float]] = {}
            axis_limits: Dict[str, Optional[Tuple[float, float]]] = {}
            for m_key, col in metric_map.items():
                if col not in combined.columns:
                    continue
                vals = combined[col].dropna()
                if vals.empty:
                    continue
                vmin, vmax = float(vals.min()), float(vals.max())
                sel_range = st.slider(f"{m_key} filter range", min_value=vmin, max_value=vmax, value=(vmin, vmax), step=(vmax - vmin) / 100 if vmax != vmin else 1.0)
                filter_ranges[col] = sel_range
                manual_lim = st.checkbox(f"Set {m_key} axis limits", value=False, key=f"lim_{m_key}")
                if manual_lim:
                    a_min = st.number_input(f"{m_key} axis min", value=vmin, key=f"amin_{m_key}")
                    a_max = st.number_input(f"{m_key} axis max", value=vmax, key=f"amax_{m_key}")
                    if a_min > a_max:
                        a_min, a_max = a_max, a_min
                    axis_limits[col] = (a_min, a_max)
                else:
                    axis_limits[col] = None
        # Custom labels
        with st.sidebar.expander("Custom Labels", expanded=False):
            custom_labels: Dict[str, str] = {}
            for m in ["Jsc", "Voc", "FF", "PCE"]:
                custom_labels[m] = st.text_input(f"Label for {m} axis", value=m, key=f"lab_{m}")
            x_label = st.text_input("Label for x‑axis", value="Variation & Scan")
        # Apply filters
        data = combined[combined["Scan"].isin(selected_scans) & combined["Variation"].isin(selected_variations)].copy()
        for col, (mi, ma) in filter_ranges.items():
            data = data[(data[col] >= mi) & (data[col] <= ma)]
        # Data previews and summary
        st.subheader("Corrected Data Preview")
        st.dataframe(data)
        st.subheader("Per‑Variation Preview")
        for v in selected_variations:
            with st.expander(f"{v} data", expanded=False):
                st.dataframe(data[data["Variation"] == v])
        st.subheader("Summary Statistics")
        st.dataframe(build_summary_table(data))
        # Colour and marker fallback
        c_map: Dict[str, str] = {}
        m_map: Dict[str, str] = {}
        for i, v in enumerate(selected_variations):
            c_map[v] = var_color_map.get(v, base_palette[i % len(base_palette)])
            m_map[v] = var_marker_map.get(v, marker_opts[i % len(marker_opts)])
        # Box plots
        st.subheader("Box Plots")
        # Set Matplotlib style based on theme
        if theme_choice == "Dark":
            plt.style.use('dark_background')
        else:
            plt.style.use('default')
        # Determine layout spacing (for grouping in plots)
        gap = box_gap
        group_gap = box_gap / 2
        # Define marker shape mapping from Plotly names to Matplotlib
        marker_map_plotly_to_mpl = {
            "circle": "o", "square": "s", "diamond": "D", "triangle-up": "^", 
            "triangle-down": "v", "triangle-left": "<", "triangle-right": ">", 
            "pentagon": "p", "hexagon": "h", "hexagon2": "H", "star": "*", 
            "star-square": "*", "star-diamond": "*", "cross": "+", "x": "x", 
            "asterisk": "*", "bowtie": "X", "hourglass": "X"
        }
        # Configure outlier (flier) and mean point style for box plots
        # Configure outlier (flier) size: hide flier markers when overlaying data points to avoid clutter
        flierprops = {
            "marker": "o",
            "markerfacecolor": "black",
            "markeredgecolor": "black",
            # If overlay_style is not 'None', suppress flier markers by setting their size to zero
            "markersize": (0 if overlay_style != "None" else marker_sz)
        }
        meanprops = {"marker": "D", "markerfacecolor": "white", "markeredgecolor": ("white" if theme_choice == "Dark" else "black"), "markersize": marker_sz + 2}
        # Compute box_width from spacing for Seaborn (width of boxes)
        box_width = 1.0 - box_gap
        if box_width < 0.05:
            box_width = 0.05
        # Combined metrics figure
        # When requested, display a 2×2 grid of box plots for the four key metrics (Voc, Jsc, FF, PCE).
        # Each subplot shows the distribution of the selected metric for all variations and scans, with
        # scan labels (RV/FW) on the x‑axis and variation colours represented in the legend.  Variation
        # markers and colours are used for overlay points when chosen.  The legend location and
        # visibility are configurable.
        if show_combined:
            # Determine the ordered list of metrics to display in the grid.  Even if only a subset
            # of metrics is selected, the grid will allocate up to four panels and hide unused ones.
            all_metrics_order = ["Voc", "Jsc", "FF", "PCE"]
            metrics_to_plot = [m for m in all_metrics_order if m in selected_metrics]
            # Create a 2×2 subplot grid.  The size of the figure is scaled up relative to a single
            # plot to maintain readability when multiple subplots are present.
            fig_grid, axes_grid = plt.subplots(2, 2, figsize=(plot_width * 2, plot_height * 2))
            axes_flat = axes_grid.flatten()
            # Build a mapping from combined category (Variation + Scan) to colours based on variation.
            # This mapping is reused for all metrics.
            var_scan_order: List[str] = []
            for v in selected_variations:
                for s in selected_scans:
                    # Only include categories present in the data
                    if not data[(data["Variation"] == v) & (data["Scan"] == s)].empty:
                        var_scan_order.append(f"{v} {s}")
            colour_mapping = {cat: c_map[cat.split()[0]] for cat in var_scan_order}
            # Loop through each of the four possible metric positions
            for idx, metric_name in enumerate(all_metrics_order):
                ax = axes_flat[idx]
                if metric_name not in metrics_to_plot:
                    # Hide unused subplot
                    ax.axis('off')
                    continue
                col_name = metric_map[metric_name]
                # Prepare DataFrame for plotting: combine variation and scan into a single category
                df_plot = data[["Variation", "Scan", col_name]].copy()
                df_plot["VarScan"] = df_plot["Variation"] + " " + df_plot["Scan"]
                # Drop rows without data for this metric
                df_plot = df_plot.dropna(subset=[col_name])
                # Draw the box plot with custom palette based on variation
                sns.boxplot(
                    x="VarScan",
                    y=col_name,
                    data=df_plot,
                    order=var_scan_order,
                    ax=ax,
                    palette=colour_mapping,
                    width=box_width,
                    linewidth=(outline_px if show_boxes else 0),
                    flierprops=flierprops,
                    showmeans=True,
                    meanprops=meanprops
                )
                # Set titles and axis labels
                title_text = f"{metric_name} Distribution"
                ax.set_title(
                    title_text,
                    fontsize=font_size + 2,
                    fontname=font_family,
                    fontweight=('bold' if bold_title else 'normal')
                )
                # Label axes.  X-axis shows only scan names (e.g. RV/FW) rather than variation names.
                xlabel_text = "" if not show_x_label_global else "Scan"
                ylabel_text = "" if not show_y_label_global else custom_labels.get(metric_name, metric_name)
                ax.set_xlabel(xlabel_text, fontsize=font_size, fontname=font_family, fontstyle=('italic' if italic_axes else 'normal'))
                ax.set_ylabel(ylabel_text, fontsize=font_size, fontname=font_family, fontstyle=('italic' if italic_axes else 'normal'))
                # Set custom x‑tick labels to only the scan portion (e.g. RV/FW)
                ax.set_xticks(range(len(var_scan_order)))
                ax.set_xticklabels([cat.split()[1] for cat in var_scan_order], rotation=x_axis_rotation)
                # Adjust tick label alignment if rotated
                if x_axis_rotation:
                    for tick in ax.get_xticklabels():
                        tick.set_ha('right')
                # Apply custom font to all tick labels
                ax.tick_params(axis='both', labelsize=font_size)
                for tick in ax.get_xticklabels() + ax.get_yticklabels():
                    tick.set_fontname(font_family)
                # Apply axis limits if set
                if axis_limits.get(col_name):
                    ax.set_ylim(axis_limits[col_name])
                # Apply opacity to each box patch and set face colours
                # The patch order corresponds to the category order defined above
                # Apply opacity and colours to each box.  Seaborn stores the box patches in ax.patches.
                for patch, cat in zip(ax.patches, var_scan_order):
                    var_name = cat.split()[0]
                    colour = c_map.get(var_name, base_palette[0])
                    rgba = mcolors.to_rgba(colour, alpha=box_opacity)
                    patch.set_facecolor(rgba)
                # Overlay points according to the selected style
                if overlay_style != "None":
                    for v in selected_variations:
                        sub_v = df_plot[df_plot["Variation"] == v]
                        if sub_v.empty:
                            continue
                        # Determine x positions for all scans of this variation
                        positions = [var_scan_order.index(f"{v} {s}") for s in sub_v["Scan"]]
                        y_vals = sub_v[col_name].tolist()
                        # Generate jitter offsets.  Spread is relative to half the box width
                        jitter_range = jitter_spread * box_width / 2.0
                        jitter_offsets = np.random.uniform(-jitter_range, jitter_range, size=len(positions))
                        x_jittered = [p + j for p, j in zip(positions, jitter_offsets)]
                        # Determine marker shape: scatter uses simple circles; markers uses variation marker
                        if overlay_style == "Scatter":
                            mshape = 'o'
                        else:
                            mshape = marker_map_plotly_to_mpl.get(m_map.get(v, 'circle'), 'o')
                        ax.scatter(
                            x_jittered,
                            y_vals,
                            s=marker_sz ** 2,
                            marker=mshape,
                            color=c_map[v],
                            alpha=marker_opacity,
                            edgecolors='none'
                        )
            # Remove any individual axes legends to avoid duplication
            for ax in axes_flat:
                if hasattr(ax, 'get_legend') and ax.get_legend():
                    ax.get_legend().remove()
            # Create a global legend for variations if requested
            if show_legend:
                legend_handles = []
                legend_labels = []
                for v in selected_variations:
                    # Determine appropriate marker for legend (variation marker)
                    mshape = marker_map_plotly_to_mpl.get(m_map.get(v, 'circle'), 'o')
                    handle = plt.Line2D([], [], linestyle='None', marker=mshape,
                                        markerfacecolor=c_map[v], markeredgecolor=c_map[v], markersize=marker_sz,
                                        label=v)
                    legend_handles.append(handle)
                    legend_labels.append(v)
                # Map legend_position to Matplotlib loc strings
                loc_map = {
                    'bottom': ('lower center', 0.5, -0.05),
                    'top': ('upper center', 0.5, 1.05),
                    'left': ('center left', -0.05, 0.5),
                    'right': ('center right', 1.05, 0.5),
                }
                loc, xoff, yoff = loc_map.get(legend_position, ('lower center', 0.5, -0.05))
                fig_grid.legend(
                    handles=legend_handles,
                    labels=legend_labels,
                    loc=loc,
                    bbox_to_anchor=(xoff, yoff),
                    ncol=max(1, int(len(selected_variations) / 4) + 1),
                    frameon=False,
                    fontsize=font_size
                )
            fig_grid.tight_layout()
            st.pyplot(fig_grid)
            # Offer download of the combined figure
            buf = io.BytesIO()
            fig_grid.savefig(buf, format=download_fmt, dpi=300)
            st.download_button(
                label="Download combined box plot",
                data=buf.getvalue(),
                file_name=f"{export_base}_combined_box_plot.{download_fmt}",
                mime=f"image/{download_fmt}"
            )
            plt.close(fig_grid)
        # Individual metric plots
        for metric in selected_metrics:
            col = metric_map[metric]
            if mode_choice == "Combine":
                # Combined scans in one plot per metric.  Each variation may have multiple scan directions (e.g. RV/FW).
                df_plot = data.copy()
                df_plot["VarScan"] = df_plot["Variation"] + " " + df_plot["Scan"]
                # Establish the order of categories for plotting (variation and scan)
                order = []
                for v in selected_variations:
                    for s in selected_scans:
                        category = f"{v} {s}"
                        if not df_plot[df_plot["VarScan"] == category].empty:
                            order.append(category)
                # If no categories, skip this metric
                if not order:
                    continue
                fig_metric, ax_metric = plt.subplots(figsize=(plot_width, plot_height))
                # Create palette mapping from category to variation colour
                palette_mapping = {cat: c_map[cat.split()[0]] for cat in order}
                sns.boxplot(
                    x="VarScan",
                    y=col,
                    data=df_plot,
                    order=order,
                    ax=ax_metric,
                    palette=palette_mapping,
                    width=box_width,
                    linewidth=(outline_px if show_boxes else 0),
                    flierprops=flierprops,
                    showmeans=True,
                    meanprops=meanprops
                )
                # Title and labels
                metric_title = f"{metric} distribution"
                ax_metric.set_title(
                    metric_title,
                    fontsize=font_size + 2,
                    fontname=font_family,
                    fontweight=('bold' if bold_title else 'normal')
                )
                # X-axis label shows only scan names; allow toggling visibility
                xlabel_text = "" if not show_x_label_global else "Scan"
                ylabel_text = "" if not show_y_label_global else custom_labels.get(metric, metric)
                ax_metric.set_xlabel(
                    xlabel_text,
                    fontsize=font_size,
                    fontname=font_family,
                    fontstyle=('italic' if italic_axes else 'normal')
                )
                ax_metric.set_ylabel(
                    ylabel_text,
                    fontsize=font_size,
                    fontname=font_family,
                    fontstyle=('italic' if italic_axes else 'normal')
                )
                # Apply axis limits if provided
                if axis_limits.get(col):
                    ax_metric.set_ylim(axis_limits[col])
                # Adjust tick labels: show only scan part (second element) and apply rotation
                ax_metric.set_xticks(range(len(order)))
                ax_metric.set_xticklabels([cat.split()[1] for cat in order], rotation=x_axis_rotation)
                if x_axis_rotation:
                    for tick in ax_metric.get_xticklabels():
                        tick.set_ha('right')
                # Apply custom font to ticks
                ax_metric.tick_params(axis='both', labelsize=font_size)
                for tick in ax_metric.get_xticklabels() + ax_metric.get_yticklabels():
                    tick.set_fontname(font_family)
                # Set box opacity and colours
                # Seaborn stores box patches in ax_metric.patches, not in ax_metric.artists
                for patch, cat in zip(ax_metric.patches, order):
                    var_name = cat.split()[0]
                    colour = c_map.get(var_name, base_palette[0])
                    rgba = mcolors.to_rgba(colour, alpha=box_opacity)
                    patch.set_facecolor(rgba)
                # Overlay points based on selected style
                if overlay_style != "None":
                    for v in selected_variations:
                        sub_v = df_plot[df_plot["Variation"] == v]
                        if sub_v.empty:
                            continue
                        positions = [order.index(f"{v} {s}") for s in sub_v["Scan"]]
                        y_vals = sub_v[col].tolist()
                        jitter_range = jitter_spread * box_width / 2.0
                        jitter_offsets = np.random.uniform(-jitter_range, jitter_range, size=len(positions))
                        x_jittered = [p + j for p, j in zip(positions, jitter_offsets)]
                        # Determine marker shape based on overlay style
                        if overlay_style == "Scatter":
                            mshape = 'o'
                        else:
                            mshape = marker_map_plotly_to_mpl.get(m_map.get(v, 'circle'), 'o')
                        ax_metric.scatter(
                            x_jittered,
                            y_vals,
                            s=marker_sz ** 2,
                            marker=mshape,
                            color=c_map[v],
                            alpha=marker_opacity,
                            edgecolors='none'
                        )
                # Remove default legend created by seaborn
                if ax_metric.get_legend():
                    ax_metric.get_legend().remove()
                # Create a legend for variations if requested
                if show_legend:
                    legend_handles = []
                    legend_labels = []
                    for v in selected_variations:
                        # Skip variations with no data
                        if df_plot[df_plot["Variation"] == v].empty:
                            continue
                        mshape = marker_map_plotly_to_mpl.get(m_map.get(v, 'circle'), 'o')
                        handle = plt.Line2D([], [], linestyle='None', marker=mshape,
                                            markerfacecolor=c_map[v], markeredgecolor=c_map[v], markersize=marker_sz,
                                            label=v)
                        legend_handles.append(handle)
                        legend_labels.append(v)
                    loc_map = {
                        'bottom': ('lower center', 0.5, -0.15),
                        'top': ('upper center', 0.5, 1.15),
                        'left': ('center left', -0.1, 0.5),
                        'right': ('center right', 1.1, 0.5),
                    }
                    loc, xoff, yoff = loc_map.get(legend_position, ('lower center', 0.5, -0.15))
                    ax_metric.legend(
                        handles=legend_handles,
                        labels=legend_labels,
                        loc=loc,
                        bbox_to_anchor=(xoff, yoff),
                        ncol=max(1, int(len(legend_labels) / 4) + 1),
                        frameon=False,
                        fontsize=font_size
                    )
                fig_metric.tight_layout()
                st.pyplot(fig_metric)
                buf = io.BytesIO()
                fig_metric.savefig(buf, format=download_fmt, dpi=300)
                st.download_button(label=f"Download {metric} box plot", data=buf.getvalue(),
                                   file_name=f"{export_base}_{metric}_box_plot.{download_fmt}", mime=f"image/{download_fmt}")
                plt.close(fig_metric)
            else:
                # Separate plots per scan
                for s in selected_scans:
                    sub_df = data[data["Scan"] == s]
                    if sub_df.empty:
                        continue
                    fig_metric, ax_metric = plt.subplots(figsize=(plot_width, plot_height))
                    # Define the order of variations for this scan based on data availability
                    var_order = [v for v in selected_variations if not sub_df[sub_df["Variation"] == v].empty]
                    sns.boxplot(
                        x="Variation",
                        y=col,
                        data=sub_df,
                        ax=ax_metric,
                        order=var_order,
                        palette={v: c_map[v] for v in var_order},
                        width=box_width,
                        linewidth=(outline_px if show_boxes else 0),
                        flierprops=flierprops,
                        showmeans=True,
                        meanprops=meanprops
                    )
                    title_text = f"{metric} ({s}) distribution"
                    ax_metric.set_title(
                        title_text,
                        fontsize=font_size + 2,
                        fontname=font_family,
                        fontweight=('bold' if bold_title else 'normal')
                    )
                    # X-axis label shows variation; allow toggling visibility
                    xlabel_text = "" if not show_x_label_global else "Variation"
                    ylabel_text = "" if not show_y_label_global else custom_labels.get(metric, metric)
                    ax_metric.set_xlabel(
                        xlabel_text,
                        fontsize=font_size,
                        fontname=font_family,
                        fontstyle=('italic' if italic_axes else 'normal')
                    )
                    ax_metric.set_ylabel(
                        ylabel_text,
                        fontsize=font_size,
                        fontname=font_family,
                        fontstyle=('italic' if italic_axes else 'normal')
                    )
                    # Apply axis limits if provided
                    if axis_limits.get(col):
                        ax_metric.set_ylim(axis_limits[col])
                    # Adjust tick labels and rotation
                    ax_metric.set_xticks(range(len(var_order)))
                    ax_metric.set_xticklabels(var_order, rotation=x_axis_rotation)
                    if x_axis_rotation:
                        for tick in ax_metric.get_xticklabels():
                            tick.set_ha('right')
                    ax_metric.tick_params(axis='both', labelsize=font_size)
                    for tick in ax_metric.get_xticklabels() + ax_metric.get_yticklabels():
                        tick.set_fontname(font_family)
                    # Set box opacity.  The box patches are stored in ax_metric.patches.
                    for patch, v in zip(ax_metric.patches, var_order):
                        colour = c_map.get(v, base_palette[0])
                        rgba = mcolors.to_rgba(colour, alpha=box_opacity)
                        patch.set_facecolor(rgba)
                    # Overlay points according to selected style
                    if overlay_style != "None":
                        for v in var_order:
                            sub_v = sub_df[sub_df["Variation"] == v]
                            if sub_v.empty:
                                continue
                            x_pos = var_order.index(v)
                            y_vals = sub_v[col].tolist()
                            jitter_range = jitter_spread * box_width / 2.0
                            jitter_offsets = np.random.uniform(-jitter_range, jitter_range, size=len(y_vals))
                            x_jittered = [x_pos + j for j in jitter_offsets]
                            if overlay_style == "Scatter":
                                mshape = 'o'
                            else:
                                mshape = marker_map_plotly_to_mpl.get(m_map.get(v, 'circle'), 'o')
                            ax_metric.scatter(
                                x_jittered,
                                y_vals,
                                s=marker_sz ** 2,
                                marker=mshape,
                                color=c_map[v],
                                alpha=marker_opacity,
                                edgecolors='none'
                            )
                    # Remove default legend
                    if ax_metric.get_legend():
                        ax_metric.get_legend().remove()
                    # Legend for variations if requested
                    if show_legend:
                        legend_handles = []
                        legend_labels = []
                        for v in var_order:
                            mshape = marker_map_plotly_to_mpl.get(m_map.get(v, 'circle'), 'o')
                            handle = plt.Line2D([], [], linestyle='None', marker=mshape,
                                                markerfacecolor=c_map[v], markeredgecolor=c_map[v], markersize=marker_sz,
                                                label=v)
                            legend_handles.append(handle)
                            legend_labels.append(v)
                        loc_map = {
                            'bottom': ('lower center', 0.5, -0.15),
                            'top': ('upper center', 0.5, 1.15),
                            'left': ('center left', -0.1, 0.5),
                            'right': ('center right', 1.1, 0.5),
                        }
                        loc, xoff, yoff = loc_map.get(legend_position, ('lower center', 0.5, -0.15))
                        ax_metric.legend(
                            handles=legend_handles,
                            labels=legend_labels,
                            loc=loc,
                            bbox_to_anchor=(xoff, yoff),
                            ncol=max(1, int(len(legend_labels) / 4) + 1),
                            frameon=False,
                            fontsize=font_size
                        )
                    fig_metric.tight_layout()
                    st.pyplot(fig_metric)
                    buf = io.BytesIO()
                    fig_metric.savefig(buf, format=download_fmt, dpi=300)
                    st.download_button(label=f"Download {metric} ({s}) box plot", data=buf.getvalue(),
                                       file_name=f"{export_base}_{metric}_{s}_box_plot.{download_fmt}", mime=f"image/{download_fmt}")
                    plt.close(fig_metric)
        # Individual Variation Analysis
        if selected_variations:
            st.subheader("Individual Variation Analysis")
            indiv_var = st.selectbox("Select Individual Variation for Analysis", options=selected_variations, index=0)
            if indiv_var:
                var_df = data[data["Variation"] == indiv_var]
                if var_df.empty:
                    st.info("No data available for the selected variation.")
                else:
                    metric_tabs = st.tabs([m for m in selected_metrics])
                    for (tab, metric_name) in zip(metric_tabs, selected_metrics):
                        with tab:
                            col_name = metric_map[metric_name]
                            if col_name not in var_df.columns or var_df[col_name].dropna().empty:
                                st.write("No data for this metric.")
                                continue
                            fig_var, ax_var = plt.subplots(figsize=(plot_width, plot_height))
                            sns.boxplot(
                                x="Scan",
                                y=col_name,
                                data=var_df,
                                ax=ax_var,
                                color=c_map.get(indiv_var, base_palette[0]),
                                width=box_width,
                                linewidth=(outline_px if show_boxes else 0),
                                flierprops=flierprops,
                                showmeans=True,
                                meanprops=meanprops
                            )
                            title_text = f"{metric_name} distribution for {indiv_var}"
                            ax_var.set_title(
                                title_text,
                                fontsize=font_size + 2,
                                fontname=font_family,
                                fontweight=('bold' if bold_title else 'normal')
                            )
                            xlabel_text = "" if not show_x_label_global else "Scan"
                            ylabel_text = "" if not show_y_label_global else custom_labels.get(metric_name, metric_name)
                            ax_var.set_xlabel(
                                xlabel_text,
                                fontsize=font_size,
                                fontname=font_family,
                                fontstyle=('italic' if italic_axes else 'normal')
                            )
                            ax_var.set_ylabel(
                                ylabel_text,
                                fontsize=font_size,
                                fontname=font_family,
                                fontstyle=('italic' if italic_axes else 'normal')
                            )
                            # Apply axis limits if provided
                            if axis_limits.get(col_name):
                                ax_var.set_ylim(axis_limits[col_name])
                            # Adjust x-ticks to show only scan names in selected order
                            scan_list = [s for s in selected_scans if not var_df[var_df["Scan"] == s].empty]
                            ax_var.set_xticks(range(len(scan_list)))
                            ax_var.set_xticklabels(scan_list, rotation=x_axis_rotation)
                            if x_axis_rotation:
                                for tick in ax_var.get_xticklabels():
                                    tick.set_ha('right')
                            ax_var.tick_params(axis='both', labelsize=font_size)
                            for tick in ax_var.get_xticklabels() + ax_var.get_yticklabels():
                                tick.set_fontname(font_family)
                            # Set box opacity (single colour for the variation).  Box patches are in ax_var.patches.
                            for patch in ax_var.patches:
                                colour = c_map.get(indiv_var, base_palette[0])
                                rgba = mcolors.to_rgba(colour, alpha=box_opacity)
                                patch.set_facecolor(rgba)
                            # Overlay points for each scan if chosen
                            if overlay_style != "None":
                                for s in scan_list:
                                    sub_s = var_df[var_df["Scan"] == s]
                                    if sub_s.empty:
                                        continue
                                    x_pos = scan_list.index(s)
                                    y_vals = sub_s[col_name].tolist()
                                    jitter_range = jitter_spread * box_width / 2.0
                                    jitter_offsets = np.random.uniform(-jitter_range, jitter_range, size=len(y_vals))
                                    x_jittered = [x_pos + j for j in jitter_offsets]
                                    if overlay_style == "Scatter":
                                        mshape = 'o'
                                    else:
                                        mshape = marker_map_plotly_to_mpl.get(m_map.get(indiv_var, 'circle'), 'o')
                                    ax_var.scatter(
                                        x_jittered,
                                        y_vals,
                                        s=marker_sz ** 2,
                                        marker=mshape,
                                        color=c_map.get(indiv_var, base_palette[0]),
                                        alpha=marker_opacity,
                                        edgecolors='none'
                                    )
                            # Remove default legend
                            if ax_var.get_legend():
                                ax_var.get_legend().remove()
                            # Show legend if requested (only one variation in this case)
                            if show_legend:
                                mshape = marker_map_plotly_to_mpl.get(m_map.get(indiv_var, 'circle'), 'o')
                                handle = plt.Line2D([], [], linestyle='None', marker=mshape,
                                                    markerfacecolor=c_map.get(indiv_var, base_palette[0]),
                                                    markeredgecolor=c_map.get(indiv_var, base_palette[0]),
                                                    markersize=marker_sz,
                                                    label=indiv_var)
                                loc_map = {
                                    'bottom': ('lower center', 0.5, -0.15),
                                    'top': ('upper center', 0.5, 1.15),
                                    'left': ('center left', -0.1, 0.5),
                                    'right': ('center right', 1.1, 0.5),
                                }
                                loc, xoff, yoff = loc_map.get(legend_position, ('lower center', 0.5, -0.15))
                                ax_var.legend(
                                    handles=[handle],
                                    labels=[indiv_var],
                                    loc=loc,
                                    bbox_to_anchor=(xoff, yoff),
                                    frameon=False,
                                    fontsize=font_size
                                )
                            fig_var.tight_layout()
                            st.pyplot(fig_var)
                            buf = io.BytesIO()
                            fig_var.savefig(buf, format=download_fmt, dpi=300)
                            st.download_button(
                                label=f"Download {metric_name} plot for {indiv_var}",
                                data=buf.getvalue(),
                                file_name=f"{export_base}_{metric_name}_{indiv_var}_box_plot.{download_fmt}",
                                mime=f"image/{download_fmt}"
                            )
                            plt.close(fig_var)
    elif mode == "IPCE Curve":
        # (IPCE Curve code remains unchanged)
        st.sidebar.header("Upload IPCE Files")
        ipce_files = st.sidebar.file_uploader("IPCE .txt file(s)", type=["txt"], accept_multiple_files=True)
        if not ipce_files:
            st.info("Upload IPCE data files to generate curves.")
            return
        ipce_mode = st.sidebar.radio("Plot mode", ["Overlay all variations", "Separate plots"], index=0)
        labels: List[str] = []
        colours: List[str] = []
        st.sidebar.header("Labels & Colours")
        for i, f in enumerate(ipce_files, start=1):
            default_lbl = extract_variation(f.name)
            lbl = st.sidebar.text_input(f"Label for file {i} ({f.name})", value=default_lbl, key=f"ipce_lbl_{i}")
            col = st.sidebar.color_picker(f"Colour for {lbl}", base_palette[(i - 1) % len(base_palette)], key=f"ipce_col_{i}")
            labels.append(lbl)
            colours.append(col)
        normalize_ipce = st.sidebar.checkbox("Normalize IPCE curves to 100%", value=False)
        normalize_jint = st.sidebar.checkbox("Normalize integrated current", value=False)
        scale_factor = st.sidebar.number_input("Apply scaling factor to integrated current (default = 1)", value=1.0, step=0.1)
        export_fmt = st.sidebar.selectbox("Download format", ["PNG", "SVG"], index=0).lower()
        export_name = st.sidebar.text_input("Export file name base", value="ipce_plot")
        curves = []
        for f in ipce_files:
            df = parse_ipce(f)
            if not df.empty and "J_integrated" in df.columns:
                df["J_integrated"] = df["J_integrated"] * scale_factor
            curves.append(df)
        st.subheader("Parsed IPCE File (Before Plotting)")
        for lbl, df in zip(labels, curves):
            with st.expander(lbl, expanded=False):
                st.dataframe(df)
        if ipce_mode == "Overlay all variations":
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            for i, df in enumerate(curves):
                if df.empty:
                    continue
                ipce_y = df["IPCE"]
                jint_y = df["J_integrated"] if "J_integrated" in df.columns else None
                if normalize_ipce:
                    ipce_y = ipce_y / ipce_y.max() * 100.0
                if jint_y is not None and normalize_jint:
                    jint_y = jint_y / jint_y.max() * 100.0
                fig.add_trace(go.Scatter(x=df["Wavelength"], y=ipce_y, name=f"{labels[i]} IPCE", line=dict(color=colours[i], width=3)), secondary_y=False)
                if jint_y is not None:
                    fig.add_trace(go.Scatter(x=df["Wavelength"], y=jint_y, name=f"{labels[i]} J_int", line=dict(color=colours[i], width=3, dash="dash")), secondary_y=True)
            fig.update_xaxes(title_text="Wavelength (nm)")
            fig.update_yaxes(title_text="IPCE (%)" + (" (norm.)" if normalize_ipce else ""), secondary_y=False)
            fig.update_yaxes(title_text=("J_int (mA/cm²)" if not normalize_jint else "J_int (norm.)"), secondary_y=True)
            fig.update_layout(
                title="IPCE and Integrated Current vs Wavelength",
                template=template_name,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                margin=dict(l=80, r=80, b=80, t=80),
                font=dict(size=font_size_global, family=font_family_global),
                title_font=dict(size=font_size_global + 2, family=font_family_global)
            )
            st.plotly_chart(fig, use_container_width=True)
            img = fig.to_image(format=export_fmt, width=1600, height=900, scale=2)
            st.download_button(label="Download IPCE plot", data=img, file_name=f"{export_name}.{export_fmt}", mime=f"image/{export_fmt}")
        else:
            tabs = st.tabs([lbl for lbl in labels])
            for i, (tab, df) in enumerate(zip(tabs, curves)):
                with tab:
                    if df.empty:
                        st.info("No data parsed.")
                        continue
                    ipce_y = df["IPCE"]
                    jint_y = df["J_integrated"] if "J_integrated" in df.columns else None
                    if normalize_ipce:
                        ipce_y = ipce_y / ipce_y.max() * 100.0
                    if jint_y is not None and normalize_jint:
                        jint_y = jint_y / jint_y.max() * 100.0
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    fig.add_trace(go.Scatter(x=df["Wavelength"], y=ipce_y, name=f"IPCE", line=dict(color=colours[i], width=3)), secondary_y=False)
                    if jint_y is not None:
                        fig.add_trace(go.Scatter(x=df["Wavelength"], y=jint_y, name=f"J_int", line=dict(color=colours[i], width=3, dash="dash")), secondary_y=True)
                    fig.update_xaxes(title_text="Wavelength (nm)")
                    fig.update_yaxes(title_text="IPCE (%)" + (" (norm.)" if normalize_ipce else ""), secondary_y=False)
                    fig.update_yaxes(title_text=("J_int (mA/cm²)" if not normalize_jint else "J_int (norm.)"), secondary_y=True)
                    fig.update_layout(
                        title=f"{labels[i]}: IPCE and Integrated Current",
                        template=template_name,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                        margin=dict(l=80, r=80, b=80, t=80),
                        font=dict(size=font_size_global, family=font_family_global),
                        title_font=dict(size=font_size_global + 2, family=font_family_global)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    img = fig.to_image(format=export_fmt, width=1600, height=900, scale=2)
                    st.download_button(label="Download plot", data=img, file_name=f"{export_name}_{labels[i]}.{export_fmt}", mime=f"image/{export_fmt}")
    elif mode == "JV Curve":
        # (JV Curve code unchanged)
        st.sidebar.header("Upload JV Files")
        jv_files = st.sidebar.file_uploader("JV .txt file(s)", type=["txt"], accept_multiple_files=True)
        if not jv_files:
            st.info("Upload JV files to generate curves.")
            return
        plot_mode = st.sidebar.radio("Plot mode", ["Overlay all variations", "Separate plots"], index=0)
        st.sidebar.header("Curve Options")
        smooth_toggle = st.sidebar.checkbox("Apply smoothing (moving average)", value=False)
        smooth_win = st.sidebar.slider("Smoothing window", 3, 21, 5, step=2)
        normalize_j = st.sidebar.checkbox("Normalize current to 100%", value=False)
        line_px = st.sidebar.slider("Line thickness (px)", 1, 6, 3)
        export_fmt = st.sidebar.selectbox("Download format", ["PNG", "SVG"], index=0).lower()
        export_name = st.sidebar.text_input("Export file name base", value="jv_plot")
        # Labels and colours
        labels: List[str] = []
        colours: List[str] = []
        st.sidebar.header("Labels & Colours")
        for i, f in enumerate(jv_files, start=1):
            default_lbl = extract_variation(f.name)
            lbl = st.sidebar.text_input(f"Label for file {i} ({f.name})", value=default_lbl, key=f"jv_lbl_{i}")
            col = st.sidebar.color_picker(f"Colour for {lbl}", base_palette[(i - 1) % len(base_palette)], key=f"jv_col_{i}")
            labels.append(lbl)
            colours.append(col)
        # Parse all curves and keep a record for preview
        curves: List[pd.DataFrame] = []
        raw_jv_previews: List[Tuple[str, pd.DataFrame]] = []
        for f, lbl in zip(jv_files, labels):
            df, _ = parse_jv(f)
            curves.append(df)
            raw_jv_previews.append((lbl, df.copy()))
        # Show parsed JV files as preview
        st.subheader("Parsed JV Data (Before Plotting)")
        for lbl, df in raw_jv_previews:
            with st.expander(lbl, expanded=False):
                st.dataframe(df)
        if plot_mode == "Overlay all variations":
            fig = go.Figure()
            for i, df in enumerate(curves):
                if df.empty:
                    continue
                if "V_FW" in df.columns and "J_FW" in df.columns:
                    j_fw = df["J_FW"].to_numpy()
                    v_fw = df["V_FW"].to_numpy()
                    if smooth_toggle:
                        j_fw = pd.Series(j_fw).rolling(window=smooth_win, min_periods=1, center=True).mean().to_numpy()
                    if normalize_j and np.max(np.abs(j_fw)) != 0:
                        j_fw = j_fw / np.max(np.abs(j_fw)) * 100.0
                    fig.add_trace(go.Scatter(x=v_fw, y=j_fw, name=f"{labels[i]} FW", line=dict(color=colours[i], width=line_px)))
                if "V_RV" in df.columns and "J_RV" in df.columns:
                    j_rv = df["J_RV"].to_numpy()
                    v_rv = df["V_RV"].to_numpy()
                    if smooth_toggle:
                        j_rv = pd.Series(j_rv).rolling(window=smooth_win, min_periods=1, center=True).mean().to_numpy()
                    if normalize_j and np.max(np.abs(j_rv)) != 0:
                        j_rv = j_rv / np.max(np.abs(j_rv)) * 100.0
                    fig.add_trace(go.Scatter(x=v_rv, y=j_rv, name=f"{labels[i]} RV", line=dict(color=colours[i], width=line_px, dash="dash")))
            fig.update_layout(
                title="JV Curves",
                xaxis_title="Voltage (V)",
                yaxis_title="Current Density (mA/cm²" + (" (norm.)" if normalize_j else ")"),
                template=template_name,
                font=dict(size=font_size_global, family=font_family_global),
                title_font=dict(size=font_size_global + 2, family=font_family_global, color="black"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                margin=dict(l=80, r=80, b=80, t=80),
            )
            st.plotly_chart(fig, use_container_width=True)
            img = fig.to_image(format=export_fmt, width=1600, height=900, scale=2)
            st.download_button(label="Download JV plot", data=img, file_name=f"{export_name}.{export_fmt}", mime=f"image/{export_fmt}")
        else:
            tabs = st.tabs([lbl for lbl in labels])
            for i, (tab, df) in enumerate(zip(tabs, curves)):
                with tab:
                    if df.empty:
                        st.info("No data parsed.")
                        continue
                    fig = go.Figure()
                    if "V_FW" in df.columns and "J_FW" in df.columns:
                        j_fw = df["J_FW"].to_numpy()
                        v_fw = df["V_FW"].to_numpy()
                        if smooth_toggle:
                            j_fw = pd.Series(j_fw).rolling(window=smooth_win, min_periods=1, center=True).mean().to_numpy()
                        if normalize_j and np.max(np.abs(j_fw)) != 0:
                            j_fw = j_fw / np.max(np.abs(j_fw)) * 100.0
                        fig.add_trace(go.Scatter(x=v_fw, y=j_fw, name="FW", line=dict(color=colours[i], width=line_px)))
                    if "V_RV" in df.columns and "J_RV" in df.columns:
                        j_rv = df["J_RV"].to_numpy()
                        v_rv = df["V_RV"].to_numpy()
                        if smooth_toggle:
                            j_rv = pd.Series(j_rv).rolling(window=smooth_win, min_periods=1, center=True).mean().to_numpy()
                        if normalize_j and np.max(np.abs(j_rv)) != 0:
                            j_rv = j_rv / np.max(np.abs(j_rv)) * 100.0
                        fig.add_trace(go.Scatter(x=v_rv, y=j_rv, name="RV", line=dict(color=colours[i], width=line_px, dash="dash")))
                    fig.update_layout(
                        title=f"{labels[i]} JV Curve",
                        xaxis_title="Voltage (V)",
                        yaxis_title="Current Density (mA/cm²" + (" (norm.)" if normalize_j else ")"),
                        template=template_name,
                        font=dict(size=font_size_global, family=font_family_global),
                        title_font=dict(size=font_size_global + 2, family=font_family_global, color="black"),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                        margin=dict(l=80, r=80, b=80, t=80),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    img = fig.to_image(format=export_fmt, width=1600, height=900, scale=2)
                    st.download_button(label="Download JV plot", data=img, file_name=f"{export_name}_{labels[i]}.{export_fmt}", mime=f"image/{export_fmt}")

if __name__ == "__main__":
    main()
