"""Streamlit application for exploring single EEG/EDF recordings."""
from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import streamlit as st
from mne.time_frequency import psd_array_welch

try:  # Optional imports used in the cleaning pipeline
    from autoreject import AutoReject
except Exception:  # pragma: no cover - optional dependency
    AutoReject = None

try:  # Optional dependency for the PREP pipeline
    from pyprep.prep_pipeline import PrepPipeline
except Exception:  # pragma: no cover - optional dependency
    PrepPipeline = None

try:  # Optional dependency for spectrogram computation
    from scipy.signal import spectrogram
except Exception:  # pragma: no cover - optional dependency
    spectrogram = None

DEFAULT_FILE = Path("EDF_Format/Controls/Controls/61-70 data/FA0011KC_1-2.edf")
NUMERIC_KINDS = {"eeg", "seeg", "ecog", "dbs", "misc", "eog", "emg", "ecg", "meg"}
PSWE_EPS = 1e-12

mne.set_log_level("WARNING")
st.set_page_config(page_title="Single Scan EEG Overview", layout="wide")
st.set_option("deprecation.showPyplotGlobalUse", False)


def read_recording(path: Path) -> mne.io.BaseRaw:
    """Read a recording from disk using MNE."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    reader = None
    if ext in {".edf", ".bdf", ".gdf", ".rec", ".eeg"}:
        reader = mne.io.read_raw_edf
    elif ext == ".fif":
        reader = mne.io.read_raw_fif
    elif ext == ".set":
        reader = mne.io.read_raw_eeglab

    if reader is None:
        raise ValueError(f"Unsupported file extension '{ext}'. Supported: EDF, BDF, GDF, REC, EEG, FIF, SET")

    raw = reader(path.as_posix(), preload=True, verbose="ERROR")
    return raw


def load_uploaded_file(upload) -> Tuple[mne.io.BaseRaw, str]:
    """Persist the uploaded file to a temporary location and load it with MNE."""
    suffix = Path(upload.name).suffix
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(upload.getbuffer())
        tmp_path = Path(tmp.name)

    try:
        raw = read_recording(tmp_path)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass

    identifier = f"upload:{upload.name}:{getattr(upload, 'size', tmp_path.stat().st_size)}"
    return raw, identifier


def load_default_file(path: Path) -> Tuple[mne.io.BaseRaw, str]:
    raw = read_recording(path)
    return raw, f"default:{path.as_posix()}"


def get_numeric_picks(raw: mne.io.BaseRaw) -> List[int]:
    """Return indices of numeric channels only."""
    picks: List[int] = []
    for idx, ch in enumerate(raw.info["chs"]):
        kind = ch.get("kind", None)
        ch_type = mne.io.pick.channel_type(raw.info, idx)
        if ch_type and ch_type.lower() in NUMERIC_KINDS:
            picks.append(idx)
            continue

        try:
            data = raw.get_data(picks=[idx], reject_by_annotation=False)
        except Exception:
            continue
        if data.size == 0:
            continue
        if np.issubdtype(data.dtype, np.floating):
            picks.append(idx)
    return picks


def count_spikes(signal: np.ndarray, multiplier: float = 5.0) -> int:
    """Simple spike detector using MAD-based thresholding."""
    if signal.size == 0:
        return 0
    baseline = np.nanmedian(signal)
    deviation = np.abs(signal - baseline)
    mad = np.nanmedian(deviation)
    if not np.isfinite(mad) or mad == 0:
        mad = np.nanstd(signal)
    if not np.isfinite(mad) or mad == 0:
        mad = np.nanmean(deviation) if np.nanmean(deviation) != 0 else 1.0
    threshold = multiplier * mad * 1.4826
    return int(np.sum(deviation > threshold))


def compute_channel_statistics(data: np.ndarray, names: Sequence[str]) -> pd.DataFrame:
    """Compute descriptive statistics for each channel."""
    rows = []
    for idx, name in enumerate(names):
        channel = data[idx]
        rows.append(
            {
                "Channel": name,
                "Minimum": float(np.nanmin(channel)),
                "Maximum": float(np.nanmax(channel)),
                "Mean": float(np.nanmean(channel)),
                "Std": float(np.nanstd(channel)),
                "Var": float(np.nanvar(channel)),
                "Spike Count": count_spikes(channel),
            }
        )
    df = pd.DataFrame(rows)
    return df.set_index("Channel")


def compute_pswe(psd_values: np.ndarray) -> float:
    """Compute the Power Spectral Wavelet Entropy (approximated via normalized PSD entropy)."""
    total_power = float(np.sum(psd_values))
    if total_power <= 0:
        return 0.0
    probabilities = psd_values / (total_power + PSWE_EPS)
    n_bins = len(probabilities)
    if n_bins <= 1:
        return 0.0
    entropy = -np.sum(probabilities * np.log(probabilities + PSWE_EPS))
    return float(entropy / np.log(n_bins))


def plot_raw_signals(times: np.ndarray, data: np.ndarray, names: Sequence[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, max(3.5, min(10, 1.2 * len(names)))))
    if data.size == 0:
        ax.set_visible(False)
        return fig
    scale = np.nanpercentile(np.abs(data), 95)
    if not np.isfinite(scale) or scale == 0:
        scale = 1.0
    for idx, name in enumerate(names):
        offset = idx * scale * 2.0
        ax.plot(times, data[idx] + offset, label=name)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude + offset")
    ax.set_title("Raw channel preview")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_yticks([])
    ax.legend(loc="upper right", ncol=2, fontsize="small")
    fig.tight_layout()
    return fig


def plot_psd(freqs: np.ndarray, psds: np.ndarray, names: Sequence[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    if psds.size == 0:
        ax.set_visible(False)
        return fig
    for idx, name in enumerate(names):
        ax.semilogy(freqs, psds[idx], alpha=0.3, linewidth=1.0)
    mean_psd = np.nanmean(psds, axis=0)
    ax.semilogy(freqs, mean_psd, color="black", linewidth=2.0, label="Average")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (V^2/Hz)")
    ax.set_title("Power Spectral Density")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_spectrogram(
    times: np.ndarray,
    signal: np.ndarray,
    sfreq: float,
    channel_name: str,
) -> Optional[plt.Figure]:
    if spectrogram is None:
        return None
    if signal.size < 16 or not np.isfinite(signal).any():
        return None
    nperseg = int(min(len(signal), max(64, sfreq * 4)))
    if nperseg <= 0:
        return None
    noverlap = nperseg // 2
    freqs, time_points, spec = spectrogram(signal, fs=sfreq, nperseg=nperseg, noverlap=noverlap)
    if spec.size == 0:
        return None
    fig, ax = plt.subplots(figsize=(10, 4))
    mesh = ax.pcolormesh(times[0] + time_points, freqs, 10 * np.log10(spec + PSWE_EPS), shading="gouraud")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Spectrogram – {channel_name}")
    fig.colorbar(mesh, ax=ax, label="Power (dB)")
    fig.tight_layout()
    return fig


def prepare_topomap_raw(raw: mne.io.BaseRaw, picks: Sequence[int]) -> Optional[mne.io.BaseRaw]:
    if len(picks) < 3:
        return None
    topo_raw = raw.copy().pick(picks)
    if topo_raw.get_montage() is None:
        try:
            montage = mne.channels.make_standard_montage("standard_1020")
            topo_raw.set_montage(montage, match_case=False, on_missing="warn")
        except Exception:
            return None
    pos = topo_raw.get_montage()
    if pos is None:
        return None
    if not pos.get_positions()["ch_pos"]:
        return None
    return topo_raw


def plot_topomap(values: np.ndarray, topo_raw: mne.io.BaseRaw, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    mne.viz.plot_topomap(values, topo_raw.info, axes=ax, show=False, cmap="viridis", contours=4)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def apply_autoreject_pipeline(raw: mne.io.BaseRaw) -> Tuple[mne.io.BaseRaw, str]:
    if AutoReject is None:
        raise RuntimeError("autoreject is not installed")
    epochs = mne.make_fixed_length_epochs(raw, duration=2.0, preload=True, reject_by_annotation=False)
    if len(epochs) == 0:
        raise RuntimeError("not enough data for AutoReject epochs")
    ar = AutoReject(random_state=97, n_jobs=1, verbose=False)
    ar.fit(epochs)
    reject_log = ar.get_reject_log(epochs)

    cleaned = raw.copy().load_data()
    sfreq = raw.info["sfreq"]
    epoch_duration = epochs.tmax - epochs.tmin
    for idx, bad in enumerate(reject_log.bad_epochs):
        if not bad:
            continue
        onset = epochs.events[idx, 0] / sfreq + epochs.tmin
        cleaned.annotations.append(onset, epoch_duration, "bad_autoreject")

    bad_channels: List[str] = []
    for channel_list in reject_log.bad_channels:
        bad_channels.extend(channel_list)
    if bad_channels:
        cleaned.info["bads"] = sorted(set(bad_channels))
        cleaned.interpolate_bads(reset_bads=True)
    return cleaned, "AutoReject applied (bad epochs annotated and channels interpolated)."


def apply_prep_pipeline(raw: mne.io.BaseRaw) -> Tuple[mne.io.BaseRaw, str]:
    if PrepPipeline is None:
        raise RuntimeError("pyprep is not installed")
    prep_params = {
        "ref_chs": raw.ch_names,
        "reref_chs": raw.ch_names,
        "line_freqs": np.array([50.0]),
    }
    pipeline = PrepPipeline(raw.copy(), prep_params, montage=None)
    pipeline.fit()
    return pipeline.raw, "PREP pipeline applied."


def run_cleaning_pipeline(raw: mne.io.BaseRaw) -> Tuple[mne.io.BaseRaw, List[str]]:
    log: List[str] = []
    cleaned = raw.copy().load_data()
    try:
        cleaned.notch_filter(50.0)
        log.append("50 Hz notch filter applied.")
    except Exception as exc:
        log.append(f"Notch filter failed: {exc}")
    try:
        cleaned.filter(0.1, 100.0)
        log.append("0.1–100 Hz band-pass filter applied.")
    except Exception as exc:
        log.append(f"Band-pass filter failed: {exc}")

    try:
        cleaned, message = apply_autoreject_pipeline(cleaned)
        log.append(message)
    except Exception as exc:
        log.append(f"AutoReject skipped: {exc}")

    try:
        cleaned, message = apply_prep_pipeline(cleaned)
        log.append(message)
    except Exception as exc:
        log.append(f"PREP pipeline skipped: {exc}")

    return cleaned, log


def format_stats_table(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()
    for column in ["Minimum", "Maximum", "Mean", "Std", "Var"]:
        formatted[column] = formatted[column].map(lambda x: f"{x:.4f}")
    formatted["Spike Count"] = formatted["Spike Count"].astype(int)
    return formatted


def main() -> None:
    st.title("Single Scan EEG/EDF Overview")
    st.markdown(
        """
        Upload an EDF/EEG recording or explore the default control scan. The app highlights numeric channels,
        visualises the raw signal, summarises channel statistics, and offers optional cleaning tools (notch + band-pass
        filtering, AutoReject, and the PREP pipeline). Frequency-domain views, spectrograms, entropy metrics, and
        topographic summaries are included below. All content renders on a single scrollable page using containers
        and expanders.
        """
    )

    with st.container():
        st.subheader("1. Recording selection")
        uploaded = st.file_uploader("Upload EDF/EEG file", type=["edf", "EDF", "eeg", "EEG", "bdf", "BDF", "gdf", "GDF", "fif", "FIF", "set", "SET"])
        default_available = DEFAULT_FILE.exists()
        if default_available:
            st.caption(f"Default file: {DEFAULT_FILE}")
        else:
            st.warning("Default file not found in repository. Upload a file to continue.")
        use_default = st.checkbox("Use default recording when no upload is provided", value=default_available)

    raw: Optional[mne.io.BaseRaw] = None
    file_identifier: Optional[str] = None
    load_error: Optional[str] = None

    if uploaded is not None:
        try:
            raw, file_identifier = load_uploaded_file(uploaded)
        except Exception as exc:  # pragma: no cover - depends on external files
            load_error = f"Unable to load uploaded file: {exc}"
    elif use_default and default_available:
        try:
            raw, file_identifier = load_default_file(DEFAULT_FILE)
        except Exception as exc:
            load_error = f"Unable to load default file: {exc}"

    if load_error:
        st.error(load_error)
        return

    if raw is None:
        st.info("Upload a compatible file or enable the default recording to begin.")
        return

    raw.load_data()

    if st.session_state.get("current_file_id") != file_identifier:
        st.session_state["current_file_id"] = file_identifier
        st.session_state["cleaned_raw"] = None
        st.session_state["cleaning_log"] = []

    cleaned_raw = st.session_state.get("cleaned_raw")
    cleaning_log: List[str] = st.session_state.get("cleaning_log", [])
    active_raw = cleaned_raw if cleaned_raw is not None else raw

    numeric_picks = get_numeric_picks(active_raw)
    if not numeric_picks:
        st.error("No numeric channels were detected in the recording.")
        return

    channel_names = [active_raw.ch_names[idx] for idx in numeric_picks]
    data = active_raw.get_data(picks=numeric_picks)
    sfreq = float(active_raw.info["sfreq"])
    duration = active_raw.n_times / sfreq if sfreq > 0 else 0

    with st.container():
        st.subheader("2. Recording overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Channels", f"{len(channel_names)}")
        col2.metric("Sampling rate (Hz)", f"{sfreq:.2f}")
        col3.metric("Duration (s)", f"{duration:.2f}")
        col4.metric("Data source", "Cleaned" if cleaned_raw is not None else "Raw")

        with st.expander("Channel statistics", expanded=False):
            stats_df = compute_channel_statistics(data, channel_names)
            st.dataframe(format_stats_table(stats_df))

    with st.container():
        st.subheader("3. Raw signal inspection")
        if duration <= 0:
            st.warning("Unable to determine recording duration.")
        else:
            max_time = max(duration, 1.0)
            default_end = min(30.0, max_time)
            if default_end <= 0:
                default_end = max_time
            time_window = st.slider(
                "Select time window (seconds)",
                min_value=0.0,
                max_value=float(max_time),
                value=(0.0, float(default_end)),
                step=0.5,
            )
            start_sample = int(max(0, time_window[0] * sfreq))
            stop_sample = int(min(data.shape[1], max(start_sample + 1, time_window[1] * sfreq)))
            window_times = np.arange(start_sample, stop_sample) / sfreq
            window_data = data[:, start_sample:stop_sample]

            default_channels = channel_names[: min(6, len(channel_names))]
            selected_channels = st.multiselect(
                "Channels to plot",
                channel_names,
                default=default_channels,
            )
            if not selected_channels:
                st.warning("Select at least one channel to visualise raw traces.")
            else:
                indices = [channel_names.index(name) for name in selected_channels]
                fig = plot_raw_signals(window_times, window_data[indices], selected_channels)
                st.pyplot(fig)
                plt.close(fig)

    with st.container():
        st.subheader("4. Cleaning pipeline")
        st.write(
            "Apply a cleaning pipeline (50 Hz notch, 0.1–100 Hz band-pass, AutoReject, PREP). Each step logs whether it succeeded or was skipped."
        )
        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button("Run cleaning pipeline", type="primary"):
                with st.spinner("Running cleaning pipeline..."):
                    cleaned, log = run_cleaning_pipeline(raw)
                    st.session_state["cleaned_raw"] = cleaned
                    st.session_state["cleaning_log"] = log
                    st.experimental_rerun()
        with col_b:
            if st.button("Reset to raw data"):
                st.session_state["cleaned_raw"] = None
                st.session_state["cleaning_log"] = []
                st.experimental_rerun()

        if cleaning_log:
            with st.expander("Cleaning log", expanded=True):
                for entry in cleaning_log:
                    st.write(f"• {entry}")

    active_raw = st.session_state.get("cleaned_raw") or raw
    numeric_picks = get_numeric_picks(active_raw)
    channel_names = [active_raw.ch_names[idx] for idx in numeric_picks]
    data = active_raw.get_data(picks=numeric_picks)
    sfreq = float(active_raw.info["sfreq"])

    psds, freqs = psd_array_welch(data, sfreq=sfreq, fmin=0.1, fmax=100.0, average="mean")
    pswe_values = np.array([compute_pswe(psd) for psd in psds])

    with st.container():
        st.subheader("5. Spectral analysis")
        col1, col2 = st.columns(2)
        with col1:
            fig_psd = plot_psd(freqs, psds, channel_names)
            st.pyplot(fig_psd)
            plt.close(fig_psd)
        with col2:
            summary_df = pd.DataFrame({"Channel": channel_names, "PSWE": pswe_values, "Spike Count": [count_spikes(ch) for ch in data]}).set_index("Channel")
            with st.expander("PSWE and spike summary", expanded=False):
                st.dataframe(summary_df)

        if spectrogram is None:
            st.info("Install SciPy to enable spectrogram visualisations.")
        else:
            first_idx = 0
            if len(channel_names) > 1:
                selection = st.selectbox("Channel for spectrogram", channel_names, index=0)
                first_idx = channel_names.index(selection)
            signal = data[first_idx]
            time_axis = active_raw.times
            fig_spec = plot_spectrogram(time_axis, signal, sfreq, channel_names[first_idx])
            if fig_spec is None:
                st.info("Unable to compute spectrogram for the selected channel.")
            else:
                st.pyplot(fig_spec)
                plt.close(fig_spec)

    with st.container():
        st.subheader("6. Topographic views")
        topo_raw = prepare_topomap_raw(active_raw, numeric_picks)
        if topo_raw is None:
            st.info("Topomap visualisation requires at least three channels with known positions.")
        else:
            spike_counts = np.array([count_spikes(ch) for ch in data])
            fig_spike_topo = plot_topomap(spike_counts, topo_raw, "Spike count topomap")
            st.pyplot(fig_spike_topo)
            plt.close(fig_spike_topo)

            fig_pswe_topo = plot_topomap(pswe_values, topo_raw, "PSWE topomap")
            st.pyplot(fig_pswe_topo)
            plt.close(fig_pswe_topo)


if __name__ == "__main__":
    main()
