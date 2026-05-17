"""
Sonar Array Simulator — Vessel Separation Dataset Generator
============================================================

Purpose
-------
Synthetic, labelled-data generator for a DL model that must detect a *quiet*
vessel masked by the main-lobe or sidelobes of a *loud* vessel after
Delay-and-Sum (DAS) beamforming.

Performance targets this dataset is built to satisfy
----------------------------------------------------
• Detect quiet target when the interferer's sidelobe is ≥10 dB stronger
  → sir_range_db reaches 25 dB; ULA first sidelobe ≈ −13 dB, so at
    SIR=25 dB the sidelobe at the quiet target is ~12 dB above it.
• Separate targets whose intensity difference is 3-5 dB  → covered by the
  low end of sir_range_db (0–5 dB).
• Targets overlapping tightly (hard case ±0.1–3°) forces the model to
  distinguish main-lobe / immediate sidelobe masking on a 100-element array
  (beamwidth ≈ 1.1° at broadside, design freq 1500 Hz).

Pipeline (per sample)
---------------------
1. Pick num_targets from target_counts.
2. Place loudest target at a random AoA.  If num_targets == 2:
   - hard case (70 %) → quiet target forced within ±0.1–3° of loudest
     (directly inside the main lobe / first sidelobes of a 100-el array).
   - easy case (30 %) → quiet target placed ≥ easy_min_sep_deg away.
   - 1-target scenes always get hard_case = False (no masking possible).
3. Calibrate amplitudes from sir_range_db (loudest amplitude = 1.0).
4. Far-field plane-wave propagation: phase-only per-element steering vector.
5. Add ambient noise at the configured SNR.
6. DAS beamform at the quietest target's angle ± steering_error_deg.
   For 1-target scenes pick a random "ghost" angle near the loud target.
7. Welch PSD of the beamformed output → save .npz + .wav + .png + CSV row.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import resample_poly, welch
from tqdm import tqdm


# =============================================================================
# CENTRAL CONFIGURATION (edit here — never hard-code below)
# =============================================================================

@dataclass
class ArrayParams:
    num_elements: int = 16          # N
    spacing: float = 0.5            # d (m), λ/2 at 1500 Hz for c=1500
    sound_speed: float = 1500.0     # c (m/s)
    design_freq_hz: float = 1500.0  # anchor for lobe-geometry calc


@dataclass
class NoiseParams:
    # noise_type is a string key looked up in NOISE_GENERATORS (see below).
    # Built-ins: 'WGN', 'pink', 'brown', 'Colored' (=pink). Add your own by
    # decorating a function with @register_noise("my_name") -- it will then
    # be selectable by name here without any other code changes.
    noise_type: str = "WGN"
    snr_range_db: tuple = (10.0, 30.0)   # SNR of loudest target vs. noise (dB)


@dataclass
class WelchParams:
    window: str = "hann"
    nperseg: int = 1024
    noverlap: int = 768


@dataclass
class SimConfig:
    # -------- core physics --------
    array_params: ArrayParams = field(default_factory=ArrayParams)

    # -------- targets --------
    target_counts: tuple = (1, 2)           # legal scene cardinalities
    sir_range_db: tuple = (0.0, 10.0)       # loud / quiet ratio per quieter tgt
    p_hard_case: float = 0.70               # P(quieter targets land in lobes)
    loud_angle_range_deg: tuple = (-60.0, 60.0)
    easy_min_sep_deg: float = 20.0
    easy_max_sep_deg: float = 60.0
    ghost_min_sep_deg: float = 2.0          # 1-target scene: how close ghost
    ghost_max_sep_deg: float = 20.0
    min_inter_target_sep_deg: float = 0.1   # keep targets distinguishable
    # Hard-case placement window (degrees from loud target, for N=100 array)
    hard_case_max_sep_deg: float = 3.0      # ±0.1–3° forces targets into lobes
    hard_case_min_sep_deg: float = 0.1

    # -------- noise & processing --------
    noise_params: NoiseParams = field(default_factory=NoiseParams)
    welch_params: WelchParams = field(default_factory=WelchParams)
    steering_error_deg: float = 0.5

    # -------- timebase --------
    fs: int = 16000
    duration_s: float = 3.0

    # -------- beam-scan axis (visualisation / PNG only) --------
    scan_angle_min_deg: float = -90.0
    scan_angle_max_deg: float = 90.0
    # 0.2° resolves the ~1.1° beamwidth of a 100-el array; safe to increase
    # to 0.5° for even faster PNG generation at the cost of coarser heatmap.
    scan_angle_step_deg: float = 0.2
    # Number of time samples used for the periodogram scan (PNG only).
    # Larger → better frequency resolution in the heatmap; does NOT affect
    # the Welch PSD saved in the .npz. Memory ∝ N × scan_n_fft.
    scan_n_fft: int = 1024


# =============================================================================
# Noise generator registry  (extension point)
# =============================================================================
#
# To add a new noise color, write a function that takes (rng, shape) and
# returns an ndarray of that shape, then tag it with @register_noise("name").
# The new name is immediately selectable via NoiseParams(noise_type="name").
# RMS calibration is handled by SonarSimulator.make_noise -- your generator
# only needs to produce a finite-variance time series.

NOISE_GENERATORS: dict = {}


def register_noise(name: str):
    """Decorator: register a noise generator under `name`."""
    def deco(fn):
        NOISE_GENERATORS[name] = fn
        NOISE_GENERATORS[name.lower()] = fn
        return fn
    return deco


@register_noise("WGN")
def _wgn(rng: np.random.Generator, shape: tuple) -> np.ndarray:
    """White Gaussian noise (i.i.d. per sample, per element)."""
    return rng.standard_normal(shape)


def _power_law_noise(rng: np.random.Generator, shape: tuple,
                     alpha: float) -> np.ndarray:
    """1/f^alpha colored noise generated via FFT shaping. Per-element i.i.d."""
    n_el, n_t = shape
    n_freqs = n_t // 2 + 1
    freqs = np.arange(n_freqs, dtype=np.float64)
    freqs[0] = 1.0
    scale = 1.0 / freqs ** (alpha / 2.0)
    scale[0] = 0.0
    out = np.empty(shape, dtype=np.float64)
    for k in range(n_el):
        re = rng.standard_normal(n_freqs)
        im = rng.standard_normal(n_freqs)
        spec = (re + 1j * im) * scale
        spec[0] = 0.0
        if n_t % 2 == 0:
            spec[-1] = spec[-1].real
        out[k] = np.fft.irfft(spec, n=n_t)
    return out


@register_noise("pink")
@register_noise("Colored")              # alias
def _pink(rng, shape):  return _power_law_noise(rng, shape, alpha=1.0)


@register_noise("brown")                # also called "red" noise
def _brown(rng, shape): return _power_law_noise(rng, shape, alpha=2.0)


# Example of how to add your own noise type from outside this file:
#
#   from sonar_simulator import register_noise
#   @register_noise("ship_noise")
#   def my_noise(rng, shape):
#       return rng.standard_normal(shape) * 0.5 + ...   # your model
#
#   sim = SonarSimulator(SimConfig(noise_params=NoiseParams(noise_type="ship_noise")))


# =============================================================================
# Simulator
# =============================================================================

class SonarSimulator:
    """
    Far-field ULA simulator. All knobs come from a single :class:`SimConfig`.

    Mathematical core (plane wave from angle θ on element n):

        X_n(f) = A · S(f) · exp(-j 2π f x_n sin(θ) / c)

    where x_n is the centred element position, S(f) is the source rFFT and
    A is the amplitude that sets the SIR.
    """

    # ----------------------------- construction --------------------------------

    def __init__(self, cfg: SimConfig = SimConfig(),
                 rng: Optional[np.random.Generator] = None):
        self.cfg = cfg
        self.rng = rng if rng is not None else np.random.default_rng()

    # ----------------------------- geometry ------------------------------------

    @property
    def element_positions(self) -> np.ndarray:
        n = self.cfg.array_params.num_elements
        return (np.arange(n) - (n - 1) / 2.0) * self.cfg.array_params.spacing

    def steering_vector(self, theta_rad: float, freqs: np.ndarray) -> np.ndarray:
        """Per-element complex64 steering vector, shape (N, F).
        Uses float32 internally to avoid the default complex128 promotion
        that would allocate ~38 MB per call with N=100."""
        x    = self.element_positions.astype(np.float32)[:, None]   # (N,1)
        f    = np.asarray(freqs, dtype=np.float32)[None, :]          # (1,F)
        phase = (np.float32(2.0 * math.pi) *
                 np.float32(math.sin(theta_rad)) * x * f /
                 np.float32(self.cfg.array_params.sound_speed))      # (N,F) float32
        return (np.cos(phase) - 1j * np.sin(phase)).astype(np.complex64)

    def analytical_beam_pattern(self, theta_steer_rad: float,
                                theta_scan_rad: np.ndarray,
                                freq_hz: float) -> np.ndarray:
        """|B(θ)| at one frequency for uniform-weight DAS."""
        x = self.element_positions[:, None]
        sin_diff = (np.sin(theta_scan_rad) - math.sin(theta_steer_rad))[None, :]
        phase = 2.0 * math.pi * freq_hz * x * sin_diff / self.cfg.array_params.sound_speed
        return np.abs(np.exp(1j * phase).mean(axis=0))

    def lobe_offsets_deg(self, theta_loud_deg: float) -> dict:
        """
        Analytical main-edge / 1st sidelobe peak / 2nd sidelobe peak offsets
        (deg, relative to the loudest target) at the design frequency.

        Derived from sin(θ) - sin(θ_loud) = k · λ / (N · d).
        """
        ap = self.cfg.array_params
        lam = ap.sound_speed / ap.design_freq_hz
        sin_l = math.sin(math.radians(theta_loud_deg))

        def offset(k: float, sign: float) -> float:
            sin_t = sin_l + sign * k * lam / (ap.num_elements * ap.spacing)
            if abs(sin_t) > 1.0:
                return float("nan")
            return math.degrees(math.asin(sin_t)) - theta_loud_deg

        return {
            "main_edge_pos":  offset(1.0, +1.0),
            "main_edge_neg":  offset(1.0, -1.0),
            "sidelobe_1_pos": offset(1.5, +1.0),
            "sidelobe_1_neg": offset(1.5, -1.0),
            "sidelobe_2_pos": offset(2.5, +1.0),
            "sidelobe_2_neg": offset(2.5, -1.0),
        }

    # ----------------------------- wav I/O -------------------------------------

    def _list_wavs(self, root: str | Path) -> list[str]:
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(
                f"WAV_DIR does not exist: {root}\n"
                f"  Edit the WAV_DIR line in the __main__ block at the bottom "
                f"of this file to point at the folder that contains your .wav "
                f"recordings, e.g.:\n"
                f"      WAV_DIR = Path(r\"C:\\Users\\akiva\\Desktop\\Final "
                f"Project\\data\\sounds-and-images\")"
            )
        files = [str(p) for p in root.rglob("*.wav")]
        if not files:
            raise FileNotFoundError(
                f"No .wav files found under {root}\n"
                f"  The folder exists but contains no .wav files (recursive "
                f"search). Point WAV_DIR at a folder that has them."
            )
        return files

    def _load_clip(self, path: str) -> np.ndarray:
        sr, x = wavfile.read(path)
        if x.dtype.kind == "i":
            x = x.astype(np.float32) / float(np.iinfo(x.dtype).max)
        else:
            x = x.astype(np.float32)
        if x.ndim > 1:
            x = x.mean(axis=1)

        fs = self.cfg.fs
        if sr != fs:
            g = math.gcd(int(sr), int(fs))
            x = resample_poly(x, fs // g, sr // g).astype(np.float32)

        n_want = int(fs * self.cfg.duration_s)
        if len(x) < n_want:
            x = np.tile(x, int(np.ceil(n_want / max(len(x), 1))))
        start = int(self.rng.integers(0, max(len(x) - n_want, 0) + 1))
        x = x[start:start + n_want]
        rms = float(np.sqrt(np.mean(x ** 2)) + 1e-12)
        return (x / rms).astype(np.float32)

    # ----------------------------- propagation ---------------------------------

    def propagate_source(self, signal: np.ndarray,
                         theta_rad: float, amplitude: float) -> np.ndarray:
        """Far-field plane-wave propagation: phase-only, no 1/r.
        Stays in float32/complex64 throughout to keep peak memory ~4× lower
        than the default float64/complex128 numpy path."""
        n     = signal.shape[0]
        S     = np.fft.rfft(signal.astype(np.float32)).astype(np.complex64)  # (F,)
        freqs = np.fft.rfftfreq(n, 1.0 / self.cfg.fs).astype(np.float32)    # (F,)
        A     = self.steering_vector(theta_rad, freqs)                        # (N,F) complex64
        X     = np.float32(amplitude) * A * S[None, :]                       # (N,F) complex64
        return np.fft.irfft(X.astype(np.complex128), n=n, axis=1).astype(np.float32)

    # ----------------------------- noise ---------------------------------------
    #
    # Noise generation is pluggable. Each generator is a function
    #     fn(rng, shape) -> np.ndarray of shape `shape`
    # registered in NOISE_GENERATORS (see bottom of this section).
    # `make_noise` looks up the generator by name and rescales it to the
    # requested RMS. To add a new noise type, write a function and tag it
    # with @register_noise("name") -- no other code change is needed.

    def make_noise(self, shape: tuple, rms: float) -> np.ndarray:
        """
        Build a (N_elements, T) noise field at the requested RMS using the
        generator named in `self.cfg.noise_params.noise_type`.
        """
        key = self.cfg.noise_params.noise_type
        gen = NOISE_GENERATORS.get(key) or NOISE_GENERATORS.get(key.lower())
        if gen is None:
            raise ValueError(
                f"Unknown noise_type {key!r}. "
                f"Registered: {sorted(NOISE_GENERATORS)}"
            )
        n = gen(self.rng, shape).astype(np.float32)
        # Rescale each element to unit RMS, then scale to the target RMS
        cur = np.sqrt(np.mean(n ** 2, axis=1, keepdims=True)) + 1e-12
        return (n / cur * rms).astype(np.float32)

    # ----------------------------- DAS + Welch ---------------------------------

    def das_beamform(self, mc: np.ndarray, theta_rad: float) -> np.ndarray:
        n     = mc.shape[1]
        X     = np.fft.rfft(mc.astype(np.float32), axis=1).astype(np.complex64)  # (N,F)
        freqs = np.fft.rfftfreq(n, 1.0 / self.cfg.fs).astype(np.float32)
        A     = self.steering_vector(theta_rad, freqs)                            # (N,F) complex64
        Y     = (np.conj(A) * X).mean(axis=0)                                    # (F,) complex64
        return np.fft.irfft(Y.astype(np.complex128), n=n).astype(np.float32)

    def welch_psd(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        wp = self.cfg.welch_params
        f, p = welch(x, fs=self.cfg.fs, window=wp.window,
                     nperseg=wp.nperseg, noverlap=wp.noverlap,
                     detrend=False, scaling="density")
        return f.astype(np.float32), p.astype(np.float32)

    def beam_scan_periodogram(self, mc: np.ndarray,
                              scan_angles_deg: np.ndarray,
                              n_fft: int = 1024,
                              chunk: int = 50,
                              ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fast angle scan → (n_angles, n_freqs) power matrix for PNG heatmaps.

        Uses a periodogram (squared rFFT magnitude) on the first `n_fft`
        samples of the multichannel signal.  Memory cost per chunk:
        (chunk × N × n_fft/2 × 4) bytes — for chunk=50, N=100, n_fft=1024
        that is only ~10 MB, vs. hundreds of MB for a full Welch scan.

        This is NOT used for the saved .npz training data; the proper Welch
        PSD at the steer angle is always written to disk separately.
        """
        n_fft = min(n_fft, mc.shape[1])
        # Use only the first n_fft samples — enough to show the spatial pattern
        X = np.fft.rfft(mc[:, :n_fft].astype(np.float32),
                        axis=1).astype(np.complex64)          # (N, F)
        F = X.shape[1]
        freqs = np.fft.rfftfreq(n_fft, 1.0 / self.cfg.fs).astype(np.float32)

        x_pos = self.element_positions.astype(np.float32)
        c     = np.float32(self.cfg.array_params.sound_speed)
        # base[n,f] = 2π x_n f / c  — precomputed once, shape (N, F)
        base  = (np.float32(2.0 * math.pi) / c *
                 np.outer(x_pos, freqs).astype(np.float32))

        sin_all = np.sin(np.deg2rad(scan_angles_deg)).astype(np.float32)

        rows: list = []
        for start in range(0, len(sin_all), chunk):
            sc    = sin_all[start:start + chunk]               # (k,)
            phase = sc[:, None, None] * base[None, :, :]       # (k, N, F) float32
            A_c   = (np.cos(phase) + 1j * np.sin(phase)).astype(np.complex64)
            Y     = (A_c * X[None, :, :]).mean(axis=1)         # (k, F)
            rows.append((np.abs(Y) ** 2 / n_fft).real.astype(np.float32))

        return freqs, np.vstack(rows)                          # (n_ang, F)

    # ----------------------------- scene sampler -------------------------------

    def _sample_scene(self) -> dict:
        cfg = self.cfg
        num_targets = int(self.rng.choice(cfg.target_counts))

        # Loudest target (rank 0 = loudest, SIR = 0 dB by definition)
        loud_angle = float(self.rng.uniform(*cfg.loud_angle_range_deg))
        angles = [loud_angle]
        sirs   = [0.0]

        # hard_case is ONLY meaningful when there are 2+ targets.
        # A 1-target scene has no second source to mask, so hard_case = False.
        if num_targets == 1:
            hard_case = False
        else:
            hard_case = bool(self.rng.uniform() < cfg.p_hard_case)

        for _ in range(1, num_targets):
            sir_db = float(self.rng.uniform(*cfg.sir_range_db))

            # Try up to 50 times to find a valid angle (avoids coincidence)
            for _try in range(50):
                if hard_case:
                    # CRITICAL: force quiet target within ±hard_case_max_sep_deg
                    # of the loudest target so it lands in the main lobe or
                    # immediate sidelobes of the 100-element array (BW ≈ 1.1°).
                    delta = float(self.rng.uniform(cfg.hard_case_min_sep_deg,
                                                   cfg.hard_case_max_sep_deg))
                else:
                    delta = float(self.rng.uniform(cfg.easy_min_sep_deg,
                                                   cfg.easy_max_sep_deg))

                sign = 1.0 if self.rng.uniform() < 0.5 else -1.0
                cand = float(np.clip(loud_angle + sign * delta, -85.0, 85.0))

                if all(abs(cand - a) >= cfg.min_inter_target_sep_deg
                       for a in angles):
                    break   # valid placement found

            angles.append(cand)
            sirs.append(sir_db)

        # Steer toward the quietest target (highest SIR = most attenuated)
        if num_targets > 1:
            quiet_idx = int(np.argmax(sirs))
            jitter = float(self.rng.uniform(-cfg.steering_error_deg,
                                            cfg.steering_error_deg))
            steer = angles[quiet_idx] + jitter
        else:
            # 1-target: steer to a random "ghost" angle near the loud target
            sep = float(self.rng.uniform(cfg.ghost_min_sep_deg,
                                         cfg.ghost_max_sep_deg))
            sign = 1.0 if self.rng.uniform() < 0.5 else -1.0
            steer = loud_angle + sign * sep
            quiet_idx = -1

        snr_db = float(self.rng.uniform(*cfg.noise_params.snr_range_db))

        return {
            "num_targets": num_targets,
            "hard_case": hard_case,
            "angles_deg": angles,
            "sirs_db":   sirs,
            "quiet_idx": quiet_idx,
            "steer_angle_deg": float(np.clip(steer, -89.0, 89.0)),
            "snr_db": snr_db,
        }

    # ----------------------------- one sample ---------------------------------

    def generate_one(self, wav_pool: list[str], sample_id: int) -> dict:
        scene = self._sample_scene()
        n_t = int(self.cfg.fs * self.cfg.duration_s)

        # Build superposition at the array
        mc = np.zeros((self.cfg.array_params.num_elements, n_t), dtype=np.float32)
        for theta_deg, sir_db in zip(scene["angles_deg"], scene["sirs_db"]):
            clip = self._load_clip(self.rng.choice(wav_pool))
            amp = 10.0 ** (-sir_db / 20.0)      # loud=1.0, quieter<1.0
            mc += self.propagate_source(clip, math.radians(theta_deg), amp)

        # Ambient noise relative to the LOUDEST target's per-element RMS
        ref_rms = float(np.sqrt(np.mean(mc[0] ** 2)) + 1e-12)
        noise_rms = ref_rms * 10.0 ** (-scene["snr_db"] / 20.0)
        mc += self.make_noise(mc.shape, noise_rms)

        # Beamform + Welch at the steer angle
        y = self.das_beamform(mc, math.radians(scene["steer_angle_deg"]))
        f_steer, psd_steer = self.welch_psd(y)

        # Fast periodogram scan — only used for the PNG heatmap, NOT saved to .npz
        scan_angles = np.arange(self.cfg.scan_angle_min_deg,
                                self.cfg.scan_angle_max_deg
                                + self.cfg.scan_angle_step_deg,
                                self.cfg.scan_angle_step_deg,
                                dtype=np.float32)
        f_scan, psd_scan = self.beam_scan_periodogram(
            mc, scan_angles, n_fft=self.cfg.scan_n_fft)

        return {
            "scene":         scene,
            "sample_id":     sample_id,
            "beamformed":    y,
            "freqs":         f_steer,
            "psd_at_steer":  psd_steer,
            "scan_angles_deg": scan_angles,
            "scan_freqs":    f_scan,
            "psd_scan":      psd_scan,
        }

    # ----------------------------- plotting -----------------------------------

    @staticmethod
    def _target_color(rank: int) -> str:
        """Color by loudness rank. 0=loudest -> red. Quieter -> cooler colors."""
        palette = ["red", "darkorange", "gold", "limegreen", "cyan", "magenta"]
        return palette[min(rank, len(palette) - 1)]

    def plot_sample(self, sample: dict, fig=None, show: bool = False):
        """
        Two-panel figure:
          top    — 1-D Welch PSD at the steer angle (frequency view)
          bottom — Welch PSD across angles, with vertical dashed lines at
                   true target bearings (red=loudest, warmer colors for
                   progressively quieter targets) and a white dotted line
                   at the steer angle.
        """
        sc = sample["scene"]
        if fig is None:
            fig = plt.figure(figsize=(11, 8), layout="constrained")
        gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.8], hspace=0.32)

        # --- top: 1D PSD at steer ---
        ax_top = fig.add_subplot(gs[0])
        psd_db = 10.0 * np.log10(sample["psd_at_steer"] + 1e-20)
        ax_top.plot(sample["freqs"], psd_db, color="navy", linewidth=1.0)
        ax_top.set_xlabel("Frequency [Hz]")
        ax_top.set_ylabel("PSD [dB / Hz]")
        ax_top.set_title(
            f"Sample {sample['sample_id']}  —  "
            f"{sc['num_targets']}-target"
            f"{', hard' if sc['hard_case'] else ', easy'}  "
            f"|  steer = {sc['steer_angle_deg']:+.2f}°  "
            f"|  SNR = {sc['snr_db']:.1f} dB  "
            f"|  noise = {self.cfg.noise_params.noise_type}"
        )
        ax_top.grid(alpha=0.3)

        # --- bottom: angle × frequency heatmap ---
        ax = fig.add_subplot(gs[1])
        angles = sample["scan_angles_deg"]
        freqs = sample["scan_freqs"]
        psd2d_db = 10.0 * np.log10(sample["psd_scan"] + 1e-20)
        vmax = float(psd2d_db.max())
        im = ax.pcolormesh(angles, freqs, psd2d_db.T, shading="auto",
                           cmap="viridis", vmin=vmax - 60.0, vmax=vmax)
        ax.set_xlabel("Steering angle [deg]")
        ax.set_ylabel("Frequency [Hz]")
        plt.colorbar(im, ax=ax, label="PSD [dB]")

        # Target vertical lines
        for rank, (theta, sir) in enumerate(zip(sc["angles_deg"], sc["sirs_db"])):
            label = (f"target #{rank} @ {theta:+.1f}°  "
                     + ("(loudest)" if rank == 0 else f"(SIR={sir:.1f} dB)"))
            ax.axvline(theta, color=self._target_color(rank),
                       linestyle="--", linewidth=2.0, label=label)
        ax.axvline(sc["steer_angle_deg"], color="white", linestyle=":",
                   linewidth=1.5,
                   label=f"steer @ {sc['steer_angle_deg']:+.2f}°")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

        if show:
            plt.show()
        return fig

    # ----------------------------- batch loop ----------------------------------

    def run_batch(self, n_samples: int, wav_dir: str | Path,
                  out_dir: str | Path, progress: bool = True) -> Path:
        wav_pool = self._list_wavs(wav_dir)
        out_dir = Path(out_dir)
        (out_dir / "psd").mkdir(parents=True, exist_ok=True)
        (out_dir / "wav").mkdir(parents=True, exist_ok=True)
        (out_dir / "plots").mkdir(parents=True, exist_ok=True)

        # Fixed-column schema -- pad with NaN up to max(target_counts)
        max_tg = int(max(self.cfg.target_counts))
        fieldnames = ["sample_id", "num_targets"]
        for k in range(max_tg):
            fieldnames += [f"target_{k}_angle_deg", f"target_{k}_sir_db"]
        fieldnames += [
            "steer_angle_deg", "snr_db", "hard_case", "noise_type",
            "psd_path", "wav_path", "plot_path",
        ]

        csv_path = out_dir / "labels.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()

            iterator = range(n_samples)
            if progress:
                iterator = tqdm(iterator, desc="Generating samples", unit="sample")

            for i in iterator:
                ex = self.generate_one(wav_pool, sample_id=i)
                stem = f"sample_{i:06d}"

                # ---- npz ----
                psd_path = out_dir / "psd" / f"{stem}.npz"
                np.savez_compressed(
                    psd_path,
                    freqs=ex["freqs"],
                    psd_at_steer=ex["psd_at_steer"],
                    scan_angles_deg=ex["scan_angles_deg"],
                    scan_freqs=ex["scan_freqs"],
                    psd_scan=ex["psd_scan"],
                )

                # ---- wav ----
                wav_path = out_dir / "wav" / f"{stem}.wav"
                self._save_wav(wav_path, ex["beamformed"])

                # ---- png ----
                plot_path = out_dir / "plots" / f"{stem}.png"
                fig = self.plot_sample(ex)
                fig.savefig(plot_path, dpi=120)
                plt.close(fig)

                # ---- csv row ----
                sc = ex["scene"]
                row = {
                    "sample_id": i,
                    "num_targets": sc["num_targets"],
                    "steer_angle_deg": sc["steer_angle_deg"],
                    "snr_db": sc["snr_db"],
                    "hard_case": sc["hard_case"],
                    "noise_type": self.cfg.noise_params.noise_type,
                    "psd_path":  str(psd_path.relative_to(out_dir)),
                    "wav_path":  str(wav_path.relative_to(out_dir)),
                    "plot_path": str(plot_path.relative_to(out_dir)),
                }
                for k in range(max_tg):
                    if k < sc["num_targets"]:
                        row[f"target_{k}_angle_deg"] = sc["angles_deg"][k]
                        row[f"target_{k}_sir_db"]    = sc["sirs_db"][k]
                    else:
                        row[f"target_{k}_angle_deg"] = float("nan")
                        row[f"target_{k}_sir_db"]    = float("nan")
                writer.writerow(row)

        # Dump full config snapshot for reproducibility
        with open(out_dir / "config.json", "w", encoding="utf-8") as fh:
            json.dump(_cfg_to_dict(self.cfg), fh, indent=2)
        return csv_path

    # ----------------------------- helpers ------------------------------------

    def _save_wav(self, path: Path, x: np.ndarray) -> None:
        peak = float(np.max(np.abs(x)) + 1e-12)
        x16 = (np.clip(x / peak * 0.95, -1.0, 1.0) * 32767.0).astype(np.int16)
        wavfile.write(str(path), self.cfg.fs, x16)


def _cfg_to_dict(cfg: SimConfig) -> dict:
    """asdict() but tuples kept as lists for JSON-serialisability."""
    return json.loads(json.dumps(asdict(cfg), default=list))


# =============================================================================
# DEMO: 3 test samples, displayed with plt.show()
# =============================================================================

if __name__ == "__main__":
    # =========================================================================
    # >>>   USER-EDITABLE PARAMETERS   <<<
    # =========================================================================
    # This is the only place you need to touch to run a new experiment.
    # Everything below the box is library code -- it does not need editing.
    # -------------------------------------------------------------------------

    # ---- Dataset size & I/O paths -------------------------------------------
    N_SAMPLES   = 3                                         # how many to make
    WAV_DIR     = Path(__file__).resolve().parent / "data" / "sounds-and-images"
    OUTPUT_DIR  = Path(__file__).resolve().parent / "synth_dataset"
    RANDOM_SEED = 11                                        # reproducibility
    SHOW_PLOTS  = True                                      # plt.show() each one

    # ---- All physics / generation knobs (central config) --------------------
    CFG = SimConfig(
        array_params  = ArrayParams(
            num_elements   = 100,       # 100-el ULA → BW ≈ 1.1° at broadside
            spacing        = 0.5,       # λ/2 at design_freq for c=1500
            sound_speed    = 1500.0,
            design_freq_hz = 1500.0,
        ),
        target_counts = (1, 2),         # 1-target (negative) or 2-target (positive)
        sir_range_db  = (0.0, 25.0),    # up to 25 dB covers sidelobe-10-dB scenario
        p_hard_case   = 0.70,           # 70% of 2-target scenes: tight placement
        hard_case_min_sep_deg = 0.1,    # quiet target ≥ 0.1° from loud
        hard_case_max_sep_deg = 3.0,    # quiet target ≤ 3° from loud (in lobes)
        easy_min_sep_deg = 20.0,        # easy 30%: targets well separated
        easy_max_sep_deg = 60.0,
        min_inter_target_sep_deg = 0.1,
        noise_params  = NoiseParams(
            noise_type    = "WGN",      # switch to "pink" or "brown" anytime
            snr_range_db  = (20.0, 40.0),
        ),
        welch_params  = WelchParams(
            window   = "hann",
            nperseg  = 1024,
            noverlap = 768,
        ),
        steering_error_deg = 0.5,
        fs                 = 16000,
        duration_s         = 3.0,
        scan_angle_step_deg = 0.2,   # 0.2° resolves the ~1.1° beamwidth of N=100
        scan_n_fft          = 1024,  # periodogram FFT size for PNG heatmap
    )
    # =========================================================================
    # >>>   END USER-EDITABLE PARAMETERS   <<<
    # =========================================================================

    # Fall back to the flat data folder if the nested one isn't present
    if not WAV_DIR.exists():
        WAV_DIR = WAV_DIR.parent

    sim = SonarSimulator(cfg=CFG, rng=np.random.default_rng(seed=RANDOM_SEED))
    csv_path = sim.run_batch(
        n_samples = N_SAMPLES,
        wav_dir   = WAV_DIR,
        out_dir   = OUTPUT_DIR,
    )
    print(f"\n[OK] Wrote {N_SAMPLES} samples + labels to: {csv_path}")
    print(f"      .npz  -> {OUTPUT_DIR / 'psd'}")
    print(f"      .wav  -> {OUTPUT_DIR / 'wav'}")
    print(f"      .png  -> {OUTPUT_DIR / 'plots'}")

    if SHOW_PLOTS:
        # Display all saved PNGs in one grid window — no re-computation.
        plot_files = sorted((OUTPUT_DIR / "plots").glob("*.png"))
        n = len(plot_files)
        cols = min(n, 3)
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols,
                                 figsize=(cols * 8, rows * 7))
        axes = np.array(axes).flatten()
        for ax, png in zip(axes, plot_files):
            ax.imshow(plt.imread(png))
            ax.axis("off")
        for ax in axes[len(plot_files):]:   # hide unused cells
            ax.set_visible(False)
        fig.tight_layout()
        plt.show()
