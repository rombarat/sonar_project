"""
Sonar Array Simulator for Deep-Learning Dataset Generation
==========================================================

Goal
----
Generate a large, labelled synthetic dataset that captures the classical
"sidelobe-masking" problem: a quiet target sitting in the sidelobe of a loud
target after beamforming.

Pipeline (per sample)
---------------------
1. Pick 1 or 2 source signals from a directory of clean recordings (.wav).
2. Place each source at a random Angle-of-Arrival (AoA) and range.
3. Propagate each source to every hydrophone of a Uniform Linear Array
   (ULA) using a frequency-domain steering vector and spherical-spread
   amplitude loss.
4. If 2 targets: one is "loud" (high SPL), one is "quiet" (heavily
   attenuated), and the angular separation is constrained so that the
   quiet target falls inside / near a sidelobe of the loud target.
5. Add ambient acoustic noise.
6. Steer a Delay-and-Sum beamformer toward the expected angle of the
   quiet target -> output deliberately contains sidelobe leakage from
   the loud target.
7. Save the beamformer output (and optionally the raw multi-channel
   recording and its STFT) plus a labels.csv row.

Conventions
-----------
- Angles are measured from broadside (0 rad = perpendicular to the array).
- Positive angle => positive x-direction along the array.
- All processing is done in the frequency domain via rFFT so that the
  per-element delay is applied as an exact phase shift rather than an
  integer sample shift.
- Underwater sound speed default c = 1500 m/s.
"""

from __future__ import annotations

import csv
import os
import glob
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, resample_poly


# ---------------------------------------------------------------------------
# Configuration containers
# ---------------------------------------------------------------------------

@dataclass
class ArrayConfig:
    num_elements: int = 16          # N hydrophones
    element_spacing: float = 0.5    # d in metres (λ/2 at design_freq -> ~1500 Hz)
    sound_speed: float = 1500.0     # c in m/s (underwater)


@dataclass
class ScenarioConfig:
    fs: int = 16000                 # processing sample rate (Hz)
    duration_s: float = 4.0         # clip length per sample
    p_two_targets: float = 0.5      # probability scene has 2 targets
    # Loud-target placement
    loud_distance_range: tuple = (200.0, 800.0)     # metres
    loud_angle_range_deg: tuple = (-60.0, 60.0)     # off broadside
    # Quiet-target placement (relative to loud)
    quiet_distance_range: tuple = (800.0, 3000.0)   # metres
    quiet_min_angle_sep_deg: float = 4.0            # avoid main-lobe overlap
    quiet_max_angle_sep_deg: float = 25.0           # stay in near sidelobes
    # SPL gap between loud and quiet sources at the array (dB)
    spl_gap_db_range: tuple = (15.0, 35.0)
    # Ambient noise (relative to LOUD source level at the array)
    noise_db_below_loud_range: tuple = (20.0, 40.0)
    # Whether the beam-steer is *exactly* at the quiet target or jittered
    steer_jitter_deg: float = 1.0


# ---------------------------------------------------------------------------
# Core simulator
# ---------------------------------------------------------------------------

class SonarSimulator:
    """
    Object-oriented batch simulator for a ULA receiver.

    The simulator works internally in the rFFT domain. For a signal s(t)
    with rFFT S(f), the contribution of a far-field plane wave at angle
    theta to the n-th element is

        X_n(f) = (A / r) * S(f) * exp(-j 2π f τ_n(theta))

    where τ_n(theta) = n * d * sin(theta) / c is the relative delay
    between element 0 and element n, A is a source-level scaler, and r
    is the slant range used for spherical-spread loss (1/r amplitude).
    """

    # ------------------------- construction --------------------------------

    def __init__(
        self,
        array_cfg: ArrayConfig = ArrayConfig(),
        scenario_cfg: ScenarioConfig = ScenarioConfig(),
        rng: Optional[np.random.Generator] = None,
    ):
        self.array = array_cfg
        self.scene = scenario_cfg
        self.rng = rng if rng is not None else np.random.default_rng()

    # ------------------------- array geometry ------------------------------

    @property
    def element_positions(self) -> np.ndarray:
        """1-D array of x-positions for each element, centred on the array."""
        n = self.array.num_elements
        return (np.arange(n) - (n - 1) / 2.0) * self.array.element_spacing

    def steering_vector(self, theta_rad: float, freqs: np.ndarray) -> np.ndarray:
        """
        Frequency-dependent steering vector for a plane wave arriving from
        angle theta.

        Returns array of shape (N_elements, N_freqs), complex.
        Element n has phase exp(-j 2π f x_n sin(theta) / c) where x_n is
        the element x-position relative to the array centre.
        """
        x = self.element_positions[:, None]          # (N, 1)
        f = freqs[None, :]                            # (1, F)
        tau = x * math.sin(theta_rad) / self.array.sound_speed
        return np.exp(-1j * 2.0 * math.pi * f * tau)

    # ------------------------- beam pattern --------------------------------

    def analytical_beam_pattern(
        self,
        theta_steer_rad: float,
        theta_scan_rad: np.ndarray,
        freq_hz: float,
    ) -> np.ndarray:
        """
        Analytical (narrowband) DAS beam pattern of the ULA at one
        frequency: returns |B(theta)| with shape == theta_scan_rad.shape.

            B(theta) = (1/N) * sum_n exp(j 2π f x_n (sin theta - sin theta_s) / c)
        """
        x = self.element_positions[:, None]                    # (N,1)
        sin_diff = (np.sin(theta_scan_rad) - math.sin(theta_steer_rad))[None, :]
        phase = 2.0 * math.pi * freq_hz * x * sin_diff / self.array.sound_speed
        B = np.exp(1j * phase).mean(axis=0)
        return np.abs(B)

    # ------------------------- wav loading ---------------------------------

    def _list_wavs(self, root: str | Path) -> list[str]:
        """Recursively gather every .wav under root."""
        root = Path(root)
        files = [str(p) for p in root.rglob("*.wav")]
        if not files:
            raise FileNotFoundError(f"No .wav files found under {root}")
        return files

    def _load_clip(self, path: str) -> np.ndarray:
        """
        Load a wav file, downmix to mono, resample to self.scene.fs and
        return a 1-D float32 array of length self.scene.fs * duration_s.
        If the file is shorter than the requested duration we wrap-around.
        """
        sr, x = wavfile.read(path)
        if x.dtype.kind == "i":
            max_val = float(np.iinfo(x.dtype).max)
            x = x.astype(np.float32) / max_val
        else:
            x = x.astype(np.float32)

        if x.ndim > 1:
            x = x.mean(axis=1)

        target_fs = self.scene.fs
        if sr != target_fs:
            g = math.gcd(int(sr), int(target_fs))
            x = resample_poly(x, target_fs // g, sr // g).astype(np.float32)

        n_want = int(target_fs * self.scene.duration_s)
        if len(x) < n_want:
            reps = int(np.ceil(n_want / max(len(x), 1)))
            x = np.tile(x, reps)
        # Random offset so we don't always grab the first chunk
        start_max = len(x) - n_want
        start = int(self.rng.integers(0, start_max + 1)) if start_max > 0 else 0
        x = x[start:start + n_want]

        # Normalise to unit RMS so SPL scaling later is well-defined
        rms = float(np.sqrt(np.mean(x ** 2)) + 1e-12)
        return (x / rms).astype(np.float32)

    # ------------------------- propagation ---------------------------------

    def propagate_source(
        self,
        signal: np.ndarray,
        theta_rad: float,
        distance: float,
        amplitude: float,
    ) -> np.ndarray:
        """
        Produce the (N_elements, N_samples) complex-summed real-valued
        multichannel recording for a single source.

        Frequency domain:
            X_n(f) = amplitude / max(distance,1) * S(f) * a_n(theta, f)
        """
        n_samples = signal.shape[0]
        S = np.fft.rfft(signal)                       # (F,)
        freqs = np.fft.rfftfreq(n_samples, 1.0 / self.scene.fs)
        A = self.steering_vector(theta_rad, freqs)    # (N, F)
        spread = amplitude / max(distance, 1.0)
        X = spread * A * S[None, :]                   # (N, F)
        x = np.fft.irfft(X, n=n_samples, axis=1)      # (N, T)
        return x.astype(np.float32)

    # ------------------------- noise ---------------------------------------

    def add_ambient_noise(
        self,
        multichannel: np.ndarray,
        noise_rms: float,
    ) -> np.ndarray:
        """
        Additive Gaussian noise that is independent across elements
        (sensor-noise model). For a more realistic isotropic ambient
        field one would use a spatially-correlated noise matrix; for
        first-pass training data, white spatial noise is the usual
        choice.
        """
        n = self.rng.standard_normal(multichannel.shape).astype(np.float32)
        return multichannel + noise_rms * n

    # ------------------------- beamforming ---------------------------------

    def das_beamform(
        self,
        multichannel: np.ndarray,
        theta_rad: float,
    ) -> np.ndarray:
        """
        Delay-and-Sum beamformer toward theta_rad.

        Applies the conjugate steering vector (i.e. compensating the
        per-element phase delays) in the frequency domain, then averages.
        The output is a single-channel real-valued time series of the
        same length as the input.
        """
        n_samples = multichannel.shape[1]
        X = np.fft.rfft(multichannel, axis=1)                  # (N, F)
        freqs = np.fft.rfftfreq(n_samples, 1.0 / self.scene.fs)
        A = self.steering_vector(theta_rad, freqs)             # (N, F)
        # Conjugate (= align) and sum across elements
        Y = (np.conj(A) * X).mean(axis=0)                      # (F,)
        y = np.fft.irfft(Y, n=n_samples)
        return y.astype(np.float32)

    # ------------------------- scenario sampling ---------------------------

    def _sample_geometry(self) -> dict:
        """Draw a random scene description."""
        sc = self.scene
        # Loud target
        loud_angle = float(self.rng.uniform(*sc.loud_angle_range_deg))
        loud_dist  = float(self.rng.uniform(*sc.loud_distance_range))

        two_targets = bool(self.rng.uniform() < sc.p_two_targets)
        quiet_angle = math.nan
        quiet_dist  = math.nan
        spl_gap_db  = math.nan
        if two_targets:
            sep = float(self.rng.uniform(sc.quiet_min_angle_sep_deg,
                                         sc.quiet_max_angle_sep_deg))
            sign = 1.0 if self.rng.uniform() < 0.5 else -1.0
            quiet_angle = loud_angle + sign * sep
            # keep inside the global angle window
            quiet_angle = float(np.clip(quiet_angle, -85.0, 85.0))
            quiet_dist  = float(self.rng.uniform(*sc.quiet_distance_range))
            spl_gap_db  = float(self.rng.uniform(*sc.spl_gap_db_range))

        noise_db = float(self.rng.uniform(*sc.noise_db_below_loud_range))

        # Where do we steer the beamformer?
        # If a quiet target exists, steer at it (with small jitter).
        # If not, steer somewhere away from the loud target to mimic the
        # operator hunting for a possible faint contact.
        if two_targets:
            steer_deg = quiet_angle + float(
                self.rng.uniform(-sc.steer_jitter_deg, sc.steer_jitter_deg)
            )
        else:
            offset = float(self.rng.uniform(sc.quiet_min_angle_sep_deg,
                                            sc.quiet_max_angle_sep_deg))
            sign = 1.0 if self.rng.uniform() < 0.5 else -1.0
            steer_deg = loud_angle + sign * offset

        return {
            "two_targets": two_targets,
            "loud_angle_deg": loud_angle,
            "loud_dist_m": loud_dist,
            "quiet_angle_deg": quiet_angle,
            "quiet_dist_m": quiet_dist,
            "spl_gap_db": spl_gap_db,
            "noise_db_below_loud": noise_db,
            "steer_angle_deg": steer_deg,
        }

    # ------------------------- single sample -------------------------------

    def generate_one(
        self,
        wav_pool: list[str],
        sample_id: int,
    ) -> dict:
        """
        Build one labelled example. Returns a dictionary with:
            'beamformed'    : 1-D float32 array (DAS output)
            'multichannel'  : (N, T) float32 array (raw array signal)
            'label'         : dict of ground-truth fields
        """
        geom = self._sample_geometry()

        # ----- loud source -----
        loud_clip = self._load_clip(self.rng.choice(wav_pool))
        loud_amp  = 1.0
        loud_mc   = self.propagate_source(
            loud_clip,
            math.radians(geom["loud_angle_deg"]),
            geom["loud_dist_m"],
            loud_amp,
        )

        # ----- quiet source (optional) -----
        if geom["two_targets"]:
            quiet_clip = self._load_clip(self.rng.choice(wav_pool))
            quiet_amp = loud_amp * (10.0 ** (-geom["spl_gap_db"] / 20.0))
            quiet_mc  = self.propagate_source(
                quiet_clip,
                math.radians(geom["quiet_angle_deg"]),
                geom["quiet_dist_m"],
                quiet_amp,
            )
            mc = loud_mc + quiet_mc
        else:
            mc = loud_mc

        # ----- ambient noise -----
        # Reference: rms of one channel of the loud source at the array
        ref_rms = float(np.sqrt(np.mean(loud_mc[0] ** 2)) + 1e-12)
        noise_rms = ref_rms * (10.0 ** (-geom["noise_db_below_loud"] / 20.0))
        mc = self.add_ambient_noise(mc, noise_rms)

        # ----- beamform toward the (possibly) quiet target -----
        y = self.das_beamform(mc, math.radians(geom["steer_angle_deg"]))

        label = {
            "sample_id": sample_id,
            "num_targets": 2 if geom["two_targets"] else 1,
            "angle_loud_target_deg": geom["loud_angle_deg"],
            "angle_quiet_target_deg": geom["quiet_angle_deg"],
            "distance_loud_m": geom["loud_dist_m"],
            "distance_quiet_m": geom["quiet_dist_m"],
            "SNR_difference_db": geom["spl_gap_db"],
            "steer_angle_deg": geom["steer_angle_deg"],
            "noise_db_below_loud": geom["noise_db_below_loud"],
        }
        return {"beamformed": y, "multichannel": mc, "label": label}

    # ------------------------- batch loop ----------------------------------

    def run_batch(
        self,
        n_samples: int,
        wav_dir: str | Path,
        out_dir: str | Path,
        save_format: str = "stft",       # 'stft' | 'wav' | 'both'
        save_multichannel: bool = False, # also dump the raw (N,T) array
        stft_nperseg: int = 512,
        stft_noverlap: int = 384,
    ) -> Path:
        """
        Generate `n_samples` scenes and write them to `out_dir` along with
        a labels.csv file. Returns the path to the CSV.
        """
        wav_pool = self._list_wavs(wav_dir)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "beamformed").mkdir(exist_ok=True)
        if save_multichannel:
            (out_dir / "multichannel").mkdir(exist_ok=True)
        if save_format in ("stft", "both"):
            (out_dir / "stft").mkdir(exist_ok=True)

        csv_path = out_dir / "labels.csv"
        fieldnames = [
            "sample_id", "num_targets",
            "angle_loud_target_deg", "angle_quiet_target_deg",
            "distance_loud_m", "distance_quiet_m",
            "SNR_difference_db", "steer_angle_deg", "noise_db_below_loud",
            "beamformed_path", "stft_path", "multichannel_path",
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for i in range(n_samples):
                ex = self.generate_one(wav_pool, sample_id=i)
                row = dict(ex["label"])
                row.update({"beamformed_path": "", "stft_path": "",
                            "multichannel_path": ""})

                stem = f"sample_{i:06d}"

                if save_format in ("wav", "both"):
                    bf_path = out_dir / "beamformed" / f"{stem}.wav"
                    self._save_wav(bf_path, ex["beamformed"])
                    row["beamformed_path"] = str(bf_path.relative_to(out_dir))
                else:
                    # Always save the beamformer time series in some form;
                    # default to .npy so DL pipelines can mmap.
                    bf_path = out_dir / "beamformed" / f"{stem}.npy"
                    np.save(bf_path, ex["beamformed"])
                    row["beamformed_path"] = str(bf_path.relative_to(out_dir))

                if save_format in ("stft", "both"):
                    f_axis, t_axis, Z = stft(
                        ex["beamformed"],
                        fs=self.scene.fs,
                        nperseg=stft_nperseg,
                        noverlap=stft_noverlap,
                    )
                    sp_path = out_dir / "stft" / f"{stem}.npz"
                    # Save magnitude (log-spec friendly later) + complex
                    np.savez_compressed(
                        sp_path,
                        f=f_axis.astype(np.float32),
                        t=t_axis.astype(np.float32),
                        Z=Z.astype(np.complex64),
                    )
                    row["stft_path"] = str(sp_path.relative_to(out_dir))

                if save_multichannel:
                    mc_path = out_dir / "multichannel" / f"{stem}.npy"
                    np.save(mc_path, ex["multichannel"])
                    row["multichannel_path"] = str(mc_path.relative_to(out_dir))

                writer.writerow(row)

        # also dump simulator config for reproducibility
        with open(out_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(
                {"array": asdict(self.array), "scene": asdict(self.scene)},
                f, indent=2,
            )
        return csv_path

    # ------------------------- helpers -------------------------------------

    def _save_wav(self, path: Path, x: np.ndarray) -> None:
        """Write a float32 mono wav, peak-normalised to 0.95."""
        peak = float(np.max(np.abs(x)) + 1e-12)
        x16 = np.clip(x / peak * 0.95, -1.0, 1.0)
        x16 = (x16 * 32767.0).astype(np.int16)
        wavfile.write(str(path), self.scene.fs, x16)


# ---------------------------------------------------------------------------
# Usage example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Default: data lives next to this file under data/sounds-and-images
    here = Path(__file__).resolve().parent
    wav_dir = here / "data" / "sounds-and-images"
    if not wav_dir.exists():
        # fall back to the flat 'data' folder if user only has the top-level wavs
        wav_dir = here / "data"

    out_dir = here / "synth_dataset"

    sim = SonarSimulator(
        array_cfg=ArrayConfig(num_elements=16, element_spacing=0.5,
                              sound_speed=1500.0),
        scenario_cfg=ScenarioConfig(fs=16000, duration_s=3.0,
                                    p_two_targets=0.7),
        rng=np.random.default_rng(seed=0),
    )

    csv_path = sim.run_batch(
        n_samples=5,
        wav_dir=wav_dir,
        out_dir=out_dir,
        save_format="both",        # write both .npy beamformer + STFT
        save_multichannel=True,    # keep the raw 16-channel arrays too
    )
    print(f"Wrote 5 samples + labels to: {csv_path}")
