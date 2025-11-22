"""
Advanced Audio Equalizer with Spatial Audio Simulation
Part of the AirPods-inspired ANC enhancement project

Features:
- Multi-band parametric equalizer (7 bands)
- Presets (Bass Boost, Vocal Clarity, Balanced, etc.)
- Spatial audio simulation (3D positioning)
- Real-time processing
"""

import numpy as np
from scipy.signal import butter, lfilter, filtfilt
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class EqualizerBand:
    """Single equalizer band configuration"""
    frequency: float      # Center frequency (Hz)
    gain: float          # Gain in dB (-12 to +12)
    q_factor: float = 1.0  # Quality factor (bandwidth)


class AudioEqualizer:
    """
    Multi-band parametric equalizer
    7 bands covering the full audible spectrum
    """

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

        # Define frequency bands
        self.bands = {
            'sub_bass': EqualizerBand(frequency=40, gain=0.0, q_factor=0.7),
            'bass': EqualizerBand(frequency=150, gain=0.0, q_factor=0.7),
            'low_mid': EqualizerBand(frequency=400, gain=0.0, q_factor=1.0),
            'mid': EqualizerBand(frequency=1000, gain=0.0, q_factor=1.0),
            'high_mid': EqualizerBand(frequency=3000, gain=0.0, q_factor=1.0),
            'presence': EqualizerBand(frequency=5000, gain=0.0, q_factor=1.0),
            'brilliance': EqualizerBand(frequency=10000, gain=0.0, q_factor=0.7),
        }

        self.presets = {
            'flat': self._create_flat_preset(),
            'bass_boost': self._create_bass_boost_preset(),
            'vocal_clarity': self._create_vocal_clarity_preset(),
            'treble_boost': self._create_treble_boost_preset(),
            'balanced': self._create_balanced_preset(),
            'podcast': self._create_podcast_preset(),
            'music': self._create_music_preset(),
            'classical': self._create_classical_preset(),
        }

    def _create_flat_preset(self) -> Dict[str, float]:
        """Flat response - no adjustment"""
        return {band: 0.0 for band in self.bands.keys()}

    def _create_bass_boost_preset(self) -> Dict[str, float]:
        """Enhanced bass response"""
        return {
            'sub_bass': 6.0,
            'bass': 4.0,
            'low_mid': 2.0,
            'mid': 0.0,
            'high_mid': -1.0,
            'presence': 0.0,
            'brilliance': 1.0,
        }

    def _create_vocal_clarity_preset(self) -> Dict[str, float]:
        """Optimized for speech/vocals"""
        return {
            'sub_bass': -3.0,
            'bass': -2.0,
            'low_mid': 1.0,
            'mid': 4.0,
            'high_mid': 5.0,
            'presence': 3.0,
            'brilliance': 0.0,
        }

    def _create_treble_boost_preset(self) -> Dict[str, float]:
        """Enhanced high frequencies"""
        return {
            'sub_bass': -2.0,
            'bass': 0.0,
            'low_mid': 1.0,
            'mid': 2.0,
            'high_mid': 4.0,
            'presence': 6.0,
            'brilliance': 5.0,
        }

    def _create_balanced_preset(self) -> Dict[str, float]:
        """Balanced V-shape for pleasant listening"""
        return {
            'sub_bass': 3.0,
            'bass': 2.0,
            'low_mid': 0.0,
            'mid': -1.0,
            'high_mid': 1.0,
            'presence': 2.0,
            'brilliance': 3.0,
        }

    def _create_podcast_preset(self) -> Dict[str, float]:
        """Optimized for podcast/audiobook listening"""
        return {
            'sub_bass': -4.0,
            'bass': -2.0,
            'low_mid': 2.0,
            'mid': 5.0,
            'high_mid': 4.0,
            'presence': 2.0,
            'brilliance': -1.0,
        }

    def _create_music_preset(self) -> Dict[str, float]:
        """General music listening"""
        return {
            'sub_bass': 2.0,
            'bass': 3.0,
            'low_mid': 1.0,
            'mid': 0.0,
            'high_mid': 1.0,
            'presence': 2.0,
            'brilliance': 3.0,
        }

    def _create_classical_preset(self) -> Dict[str, float]:
        """Classical music - natural reproduction"""
        return {
            'sub_bass': 1.0,
            'bass': 1.0,
            'low_mid': 0.0,
            'mid': 0.0,
            'high_mid': 1.0,
            'presence': 2.0,
            'brilliance': 2.0,
        }

    def apply_preset(self, preset_name: str):
        """Apply a named preset"""
        if preset_name not in self.presets:
            logger.warning(f"Unknown preset: {preset_name}")
            return

        preset = self.presets[preset_name]
        for band_name, gain in preset.items():
            if band_name in self.bands:
                self.bands[band_name].gain = gain

        logger.info(f"Applied preset: {preset_name}")

    def set_band_gain(self, band_name: str, gain_db: float):
        """Set gain for a specific band"""
        if band_name not in self.bands:
            logger.warning(f"Unknown band: {band_name}")
            return

        # Clamp gain to reasonable range
        gain_db = np.clip(gain_db, -12.0, 12.0)
        self.bands[band_name].gain = gain_db

    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply equalization to audio

        Args:
            audio: Input audio (mono or stereo)

        Returns:
            Equalized audio
        """
        is_stereo = len(audio.shape) > 1 and audio.shape[1] == 2

        if is_stereo:
            # Process each channel
            left = self._process_channel(audio[:, 0])
            right = self._process_channel(audio[:, 1])
            return np.column_stack([left, right])
        else:
            return self._process_channel(audio)

    def _process_channel(self, audio: np.ndarray) -> np.ndarray:
        """Process single channel through all EQ bands"""
        result = np.zeros_like(audio)

        for band_name, band in self.bands.items():
            if abs(band.gain) < 0.1:  # Skip if gain is negligible
                continue

            # Design peaking filter for this band
            filtered = self._apply_peaking_filter(audio, band)

            # Add to result
            result += filtered

        # Add original signal (dry)
        result += audio

        # Prevent clipping
        max_val = np.max(np.abs(result))
        if max_val > 1.0:
            result = result / max_val

        return result

    def _apply_peaking_filter(self, audio: np.ndarray, band: EqualizerBand) -> np.ndarray:
        """
        Apply peaking (bell) filter for a specific band

        This is a second-order IIR filter with adjustable center frequency,
        gain, and Q factor (bandwidth)
        """
        # Convert gain from dB to linear
        A = 10 ** (band.gain / 40)  # Divide by 40 for peaking filter

        # Normalized frequency
        omega = 2 * np.pi * band.frequency / self.sample_rate
        cos_omega = np.cos(omega)
        sin_omega = np.sin(omega)
        alpha = sin_omega / (2 * band.q_factor)

        # Peaking filter coefficients (RBJ Audio EQ Cookbook)
        b0 = 1 + alpha * A
        b1 = -2 * cos_omega
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cos_omega
        a2 = 1 - alpha / A

        # Normalize
        b = np.array([b0, b1, b2]) / a0
        a = np.array([1, a1 / a0, a2 / a0])

        # Apply filter
        filtered = lfilter(b, a, audio)

        return filtered


class SpatialAudioSimulator:
    """
    Spatial Audio Simulation - Creates 3D audio positioning effect
    Inspired by AirPods Pro spatial audio feature
    """

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

        # Head-related transfer function (HRTF) simulation
        # These are simplified ITD (Interaural Time Difference) values
        self.max_itd_samples = int(0.0007 * sample_rate)  # ~0.7ms max ITD

        # Room simulation parameters
        self.room_size = 0.5  # 0-1 (small to large)
        self.reverb_amount = 0.3  # 0-1

    def process_stereo(
        self,
        audio: np.ndarray,
        azimuth: float = 0.0,
        elevation: float = 0.0,
        distance: float = 1.0
    ) -> np.ndarray:
        """
        Apply spatial audio effects to stereo audio

        Args:
            audio: Input stereo audio (N, 2)
            azimuth: Horizontal angle in degrees (-180 to 180, 0 = front)
            elevation: Vertical angle in degrees (-90 to 90, 0 = level)
            distance: Distance factor (0.5 to 3.0, 1.0 = normal)

        Returns:
            Spatialized stereo audio
        """
        if len(audio.shape) == 1:
            # Convert mono to stereo
            audio = np.column_stack([audio, audio])

        # Normalize angles
        azimuth = np.clip(azimuth, -180, 180)
        elevation = np.clip(elevation, -90, 90)
        distance = np.clip(distance, 0.5, 3.0)

        # Apply HRTF simulation
        spatialized = self._apply_hrtf(audio, azimuth, elevation)

        # Apply distance attenuation
        spatialized = spatialized / distance

        # Add room reverberation
        if self.reverb_amount > 0:
            spatialized = self._apply_reverb(spatialized)

        # Prevent clipping
        max_val = np.max(np.abs(spatialized))
        if max_val > 1.0:
            spatialized = spatialized / max_val

        return spatialized

    def _apply_hrtf(self, audio: np.ndarray, azimuth: float, elevation: float) -> np.ndarray:
        """
        Apply simplified Head-Related Transfer Function

        This simulates how our ears perceive sound from different directions
        using interaural time difference (ITD) and interaural level difference (ILD)
        """
        left = audio[:, 0]
        right = audio[:, 1]

        # Calculate ITD (Interaural Time Difference)
        # Sound from the right arrives at right ear first
        azimuth_rad = np.radians(azimuth)
        itd_samples = int(self.max_itd_samples * np.sin(azimuth_rad))

        # Apply time delay
        if itd_samples > 0:  # Sound from right
            left = np.pad(left, (itd_samples, 0))[:-itd_samples]
        elif itd_samples < 0:  # Sound from left
            right = np.pad(right, (-itd_samples, 0))[:itd_samples]

        # Calculate ILD (Interaural Level Difference)
        # Sound from one side is louder in the near ear
        ild_db = 10 * np.sin(azimuth_rad)  # Up to ±10 dB difference
        left_gain = 10 ** ((-ild_db) / 20)
        right_gain = 10 ** ((ild_db) / 20)

        # Apply elevation effect (simplified - reduces high frequencies for below)
        if elevation < 0:
            # Sound from below - reduce high frequencies
            elevation_factor = 1.0 + (elevation / 90.0) * 0.3
            left = self._apply_lowpass(left, cutoff=8000 * elevation_factor)
            right = self._apply_lowpass(right, cutoff=8000 * elevation_factor)

        left = left * left_gain
        right = right * right_gain

        return np.column_stack([left, right])

    def _apply_lowpass(self, audio: np.ndarray, cutoff: float) -> np.ndarray:
        """Apply lowpass filter"""
        nyquist = self.sample_rate / 2
        normalized_cutoff = cutoff / nyquist
        normalized_cutoff = np.clip(normalized_cutoff, 0.01, 0.99)

        b, a = butter(4, normalized_cutoff, btype='low')
        return filtfilt(b, a, audio)

    def _apply_reverb(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply simple reverb effect
        Uses multiple delayed copies with decay
        """
        # Reverb parameters based on room size
        delays = [int(0.03 * self.sample_rate * self.room_size),
                  int(0.05 * self.sample_rate * self.room_size),
                  int(0.07 * self.sample_rate * self.room_size),
                  int(0.09 * self.sample_rate * self.room_size)]

        decays = [0.6, 0.4, 0.3, 0.2]

        reverb = np.zeros_like(audio)

        for delay, decay in zip(delays, decays):
            # Create delayed version
            delayed = np.pad(audio, ((delay, 0), (0, 0)))[:-delay] * decay * self.reverb_amount
            reverb += delayed

        # Mix dry and wet
        wet_mix = 0.3 * self.reverb_amount
        result = audio * (1 - wet_mix) + reverb * wet_mix

        return result


# Demonstration
if __name__ == "__main__":
    print("=" * 70)
    print("  ADVANCED AUDIO EQUALIZER & SPATIAL AUDIO")
    print("=" * 70)

    # Create equalizer
    eq = AudioEqualizer(sample_rate=44100)

    print("\nAvailable EQ Presets:")
    for i, preset_name in enumerate(eq.presets.keys(), 1):
        print(f"  {i}. {preset_name}")

    print("\nAvailable Bands:")
    for band_name, band in eq.bands.items():
        print(f"  - {band_name}: {band.frequency} Hz")

    print("\nExample: Applying 'bass_boost' preset...")
    eq.apply_preset('bass_boost')

    print("\nBand settings after preset:")
    for band_name, band in eq.bands.items():
        print(f"  {band_name}: {band.gain:+.1f} dB")

    # Create spatial audio simulator
    spatial = SpatialAudioSimulator(sample_rate=44100)

    print("\n" + "=" * 70)
    print("  SPATIAL AUDIO SIMULATION")
    print("=" * 70)
    print("\nParameters:")
    print("  - Azimuth: -180° to +180° (left to right)")
    print("  - Elevation: -90° to +90° (below to above)")
    print("  - Distance: 0.5 to 3.0 (close to far)")
    print("\nEffects:")
    print("  ✓ Interaural Time Difference (ITD)")
    print("  ✓ Interaural Level Difference (ILD)")
    print("  ✓ Distance attenuation")
    print("  ✓ Room reverberation")

    print("\n✓ Equalizer and Spatial Audio systems ready!")
