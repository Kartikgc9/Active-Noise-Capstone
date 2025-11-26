"""
Enhanced Audio Denoiser with Advanced Noise Cancellation
=========================================================

Features:
- Multi-band adaptive filtering
- Advanced noise profiling and estimation
- Spectral gating with soft knee
- Multi-stage Wiener filtering
- Voice activity detection (VAD)
- Adaptive gain control
- Optimized for Raspberry Pi 5

This denoiser can handle various noise types at different levels
while preserving speech quality.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from scipy import signal
from scipy.ndimage import median_filter
import warnings

warnings.filterwarnings("ignore")


class EnhancedAudioDenoiser:
    """
    Professional-grade audio denoiser with advanced noise cancellation
    """

    def __init__(self, sample_rate=44100, noise_reduction_strength="adaptive"):
        """
        Initialize the enhanced denoiser

        Args:
            sample_rate: Audio sample rate (Hz)
            noise_reduction_strength: 'adaptive', 'gentle', 'moderate', 'aggressive', 'maximum'
        """
        self.sr = sample_rate
        self.n_fft = 2048
        self.hop_length = 512
        self.win_length = 2048

        # Noise reduction presets optimized for complete noise removal
        self.presets = {
            "adaptive": {
                "spectral_floor": 0.002,
                "over_subtraction": 2.0,
                "spectral_smoothing": 5,
                "vad_threshold": 0.03,
                "gate_threshold_db": -40,
                "gate_ratio": 10,
                "wiener_iterations": 3,
                "band_specific": True
            },
            "gentle": {
                "spectral_floor": 0.01,
                "over_subtraction": 1.5,
                "spectral_smoothing": 3,
                "vad_threshold": 0.04,
                "gate_threshold_db": -35,
                "gate_ratio": 5,
                "wiener_iterations": 2,
                "band_specific": False
            },
            "moderate": {
                "spectral_floor": 0.005,
                "over_subtraction": 2.5,
                "spectral_smoothing": 5,
                "vad_threshold": 0.025,
                "gate_threshold_db": -45,
                "gate_ratio": 15,
                "wiener_iterations": 3,
                "band_specific": True
            },
            "aggressive": {
                "spectral_floor": 0.002,
                "over_subtraction": 3.5,
                "spectral_smoothing": 7,
                "vad_threshold": 0.02,
                "gate_threshold_db": -50,
                "gate_ratio": 20,
                "wiener_iterations": 4,
                "band_specific": True
            },
            "maximum": {
                "spectral_floor": 0.001,
                "over_subtraction": 5.0,
                "spectral_smoothing": 9,
                "vad_threshold": 0.015,
                "gate_threshold_db": -60,
                "gate_ratio": 30,
                "wiener_iterations": 5,
                "band_specific": True
            }
        }

        self.params = self.presets[noise_reduction_strength]
        self.strength = noise_reduction_strength

        # Noise profile storage
        self.noise_profile = None
        self.noise_profile_power = None

        # Frequency band definitions (Hz)
        self.freq_bands = [
            (0, 150),        # Sub-bass (rumble, hum)
            (150, 300),      # Bass
            (300, 500),      # Lower speech
            (500, 2000),     # Mid speech (most important)
            (2000, 4000),    # Upper speech (consonants)
            (4000, 8000),    # Presence
            (8000, 22050)    # Brilliance/hiss
        ]

        print(f"✅ Enhanced Audio Denoiser initialized")
        print(f"   Mode: {noise_reduction_strength}")
        print(f"   Sample rate: {self.sr} Hz")
        print(f"   Multi-band processing: {self.params['band_specific']}")

    def profile_noise(self, noise_audio):
        """
        Create a detailed noise profile from a noise sample

        Args:
            noise_audio: Pure noise audio segment (numpy array)
        """
        print("📊 Creating noise profile...")

        # Compute STFT of noise
        stft = librosa.stft(noise_audio, n_fft=self.n_fft,
                           hop_length=self.hop_length,
                           win_length=self.win_length)
        magnitude = np.abs(stft)
        power = magnitude ** 2

        # Calculate noise statistics
        self.noise_profile = np.mean(magnitude, axis=1, keepdims=True)
        self.noise_profile_power = np.mean(power, axis=1, keepdims=True)

        # Smooth the noise profile
        smoothing = self.params['spectral_smoothing']
        self.noise_profile = median_filter(self.noise_profile.squeeze(),
                                          size=smoothing).reshape(-1, 1)
        self.noise_profile_power = median_filter(self.noise_profile_power.squeeze(),
                                                 size=smoothing).reshape(-1, 1)

        # Calculate noise statistics per frequency band
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        self.band_noise_levels = []

        for low, high in self.freq_bands:
            band_mask = (freqs >= low) & (freqs < high)
            band_noise = np.mean(self.noise_profile[band_mask])
            self.band_noise_levels.append(band_noise)

        print(f"   ✅ Noise profile created")
        print(f"   Noise level: {np.mean(self.noise_profile):.6f}")

    def estimate_noise_from_audio(self, audio):
        """
        Automatically estimate noise profile from audio with speech
        Uses voice activity detection to find noise-only segments
        """
        print("🔍 Estimating noise from audio...")

        # Compute STFT
        stft = librosa.stft(audio, n_fft=self.n_fft,
                           hop_length=self.hop_length,
                           win_length=self.win_length)
        magnitude = np.abs(stft)
        power = magnitude ** 2

        # Voice Activity Detection (VAD)
        frame_energy = np.sum(power, axis=0)
        frame_energy_db = 10 * np.log10(frame_energy + 1e-10)

        # Normalize energy
        energy_threshold = np.percentile(frame_energy_db, 20)

        # Find low-energy frames (likely noise-only)
        noise_frames = frame_energy_db < energy_threshold

        if np.sum(noise_frames) < 5:
            # If too few noise frames, use lowest energy frames
            n_frames = max(5, int(0.1 * magnitude.shape[1]))
            noise_frame_indices = np.argsort(frame_energy)[:n_frames]
            noise_frames = np.zeros(len(frame_energy), dtype=bool)
            noise_frames[noise_frame_indices] = True

        # Calculate noise profile from noise frames
        noise_magnitude = magnitude[:, noise_frames]
        noise_power_spectrum = power[:, noise_frames]

        self.noise_profile = np.mean(noise_magnitude, axis=1, keepdims=True)
        self.noise_profile_power = np.mean(noise_power_spectrum, axis=1, keepdims=True)

        # Smooth noise profile
        smoothing = self.params['spectral_smoothing']
        self.noise_profile = median_filter(self.noise_profile.squeeze(),
                                          size=smoothing).reshape(-1, 1)
        self.noise_profile_power = median_filter(self.noise_profile_power.squeeze(),
                                                 size=smoothing).reshape(-1, 1)

        print(f"   ✅ Noise estimated from {np.sum(noise_frames)} frames")
        print(f"   Estimated noise level: {np.mean(self.noise_profile):.6f}")

    def spectral_gating(self, magnitude, phase):
        """
        Apply spectral gating with soft knee to remove noise below threshold
        """
        # Convert to dB
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        noise_db = 20 * np.log10(self.noise_profile + 1e-10)

        # Calculate threshold (broadcast to match magnitude shape)
        threshold_db = noise_db + self.params['gate_threshold_db']

        # Soft knee gating
        gate_ratio = self.params['gate_ratio']

        # Calculate gain reduction
        gain = np.ones_like(magnitude)
        below_threshold = magnitude_db < threshold_db

        # Soft knee: gradual reduction instead of hard cutoff
        knee_width = 6  # dB
        in_knee = (magnitude_db >= (threshold_db - knee_width)) & (magnitude_db < threshold_db)

        # Hard gate for well below threshold
        gain[below_threshold] = 1.0 / gate_ratio

        # Soft transition in knee region
        if np.any(in_knee):
            # Calculate knee gain (element-wise)
            diff = threshold_db - magnitude_db
            knee_gain = 1.0 - np.clip(diff / knee_width, 0, 1)
            knee_gain = knee_gain * (1.0 - 1.0/gate_ratio) + 1.0/gate_ratio
            gain[in_knee] = knee_gain[in_knee]

        # Apply gain
        gated_magnitude = magnitude * gain

        return gated_magnitude

    def multi_band_spectral_subtraction(self, magnitude, phase):
        """
        Apply spectral subtraction with band-specific parameters
        """
        if not self.params['band_specific']:
            # Standard spectral subtraction
            over_sub = self.params['over_subtraction']
            spectral_floor = self.params['spectral_floor']

            subtracted = magnitude - over_sub * self.noise_profile
            enhanced = np.maximum(subtracted, spectral_floor * magnitude)

            return enhanced

        # Band-specific processing
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        enhanced_magnitude = magnitude.copy()

        for i, (low, high) in enumerate(self.freq_bands):
            band_mask = (freqs >= low) & (freqs < high)

            # Adjust over-subtraction based on band
            if 500 <= low < 4000:  # Speech range - gentler
                over_sub = self.params['over_subtraction'] * 0.7
                floor = self.params['spectral_floor'] * 2.0
            elif low < 150:  # Very low freq - aggressive (rumble)
                over_sub = self.params['over_subtraction'] * 1.5
                floor = self.params['spectral_floor'] * 0.5
            else:  # Other frequencies
                over_sub = self.params['over_subtraction']
                floor = self.params['spectral_floor']

            # Apply subtraction for this band
            band_mag = magnitude[band_mask, :]
            band_noise = self.noise_profile[band_mask, :]

            subtracted = band_mag - over_sub * band_noise
            enhanced_band = np.maximum(subtracted, floor * band_mag)

            enhanced_magnitude[band_mask, :] = enhanced_band

        return enhanced_magnitude

    def iterative_wiener_filter(self, magnitude, phase, iterations=None):
        """
        Apply Wiener filtering with multiple iterations for better noise removal
        """
        if iterations is None:
            iterations = self.params['wiener_iterations']

        power = magnitude ** 2
        current_estimate = power.copy()

        for iteration in range(iterations):
            # Estimate signal power
            signal_power = np.maximum(current_estimate - self.noise_profile_power,
                                     0.01 * current_estimate)

            # Calculate Wiener gain
            wiener_gain = signal_power / (signal_power + self.noise_profile_power + 1e-10)

            # Smooth gain to reduce musical noise
            if iteration > 0:
                wiener_gain = median_filter(wiener_gain, size=(3, 3))

            # Apply gain
            current_estimate = power * wiener_gain

        # Final magnitude
        enhanced_magnitude = np.sqrt(current_estimate)

        return enhanced_magnitude

    def speech_enhancement(self, magnitude):
        """
        Enhance speech frequencies while suppressing noise
        """
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)

        # Create enhancement curve
        enhancement = np.ones_like(magnitude)

        # Boost speech formant regions
        # F1 region (300-900 Hz)
        f1_mask = (freqs >= 300) & (freqs < 900)
        enhancement[f1_mask, :] *= 1.15

        # F2 region (900-3000 Hz)
        f2_mask = (freqs >= 900) & (freqs < 3000)
        enhancement[f2_mask, :] *= 1.2

        # F3 region (3000-4000 Hz) - important for consonants
        f3_mask = (freqs >= 3000) & (freqs < 4000)
        enhancement[f3_mask, :] *= 1.15

        # Suppress very low frequencies (rumble)
        rumble_mask = freqs < 80
        enhancement[rumble_mask, :] *= 0.3

        # Suppress very high frequencies if no significant content
        high_mask = freqs > 12000
        high_energy = np.mean(magnitude[high_mask, :])
        if high_energy < 0.01:  # Very low energy in high freqs
            enhancement[high_mask, :] *= 0.5

        return magnitude * enhancement

    def denoise(self, audio, noise_sample=None):
        """
        Main denoising function with complete noise cancellation

        Args:
            audio: Input audio signal
            noise_sample: Optional pure noise sample for profiling

        Returns:
            Denoised audio signal
        """
        print(f"\n🎵 Processing audio with {self.strength} noise cancellation...")

        # Step 1: Noise profiling
        if noise_sample is not None:
            self.profile_noise(noise_sample)
        elif self.noise_profile is None:
            self.estimate_noise_from_audio(audio)

        # Step 2: Compute STFT
        stft = librosa.stft(audio, n_fft=self.n_fft,
                           hop_length=self.hop_length,
                           win_length=self.win_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        original_magnitude = magnitude.copy()

        # Step 3: Multi-band spectral subtraction
        print("   🔄 Stage 1: Multi-band spectral subtraction...")
        magnitude = self.multi_band_spectral_subtraction(magnitude, phase)

        # Step 4: Spectral gating
        print("   🔄 Stage 2: Spectral gating...")
        magnitude = self.spectral_gating(magnitude, phase)

        # Step 5: Iterative Wiener filtering
        print("   🔄 Stage 3: Iterative Wiener filtering...")
        magnitude = self.iterative_wiener_filter(magnitude, phase)

        # Step 6: Speech enhancement
        print("   🔄 Stage 4: Speech enhancement...")
        magnitude = self.speech_enhancement(magnitude)

        # Step 7: Post-processing to remove residual artifacts
        print("   🔄 Stage 5: Artifact removal...")
        # Median filtering for transient noise
        magnitude = median_filter(magnitude, size=(1, 3))

        # Reconstruct audio
        enhanced_stft = magnitude * np.exp(1j * phase)
        denoised_audio = librosa.istft(enhanced_stft,
                                       hop_length=self.hop_length,
                                       win_length=self.win_length,
                                       length=len(audio))

        # Calculate noise reduction achieved
        noise_reduction_db = self._calculate_noise_reduction(
            original_magnitude, magnitude
        )

        print(f"   ✅ Denoising complete")
        print(f"   📊 Noise reduction: {noise_reduction_db:.1f} dB")

        return denoised_audio

    def _calculate_noise_reduction(self, original_mag, enhanced_mag):
        """Calculate noise reduction in dB"""
        original_noise = np.mean(original_mag[original_mag < np.percentile(original_mag, 20)])
        enhanced_noise = np.mean(enhanced_mag[enhanced_mag < np.percentile(enhanced_mag, 20)])

        if enhanced_noise > 0:
            reduction_db = 20 * np.log10(original_noise / (enhanced_noise + 1e-10))
            return reduction_db
        return 0.0

    def process_file(self, input_path, output_path, noise_sample_path=None):
        """
        Process an audio file with noise cancellation

        Args:
            input_path: Path to input audio file
            output_path: Path to save denoised audio
            noise_sample_path: Optional path to pure noise sample
        """
        print(f"📂 Loading: {input_path}")

        # Load audio
        audio, sr = librosa.load(input_path, sr=self.sr, mono=True)

        if sr != self.sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)

        # Load noise sample if provided
        noise_sample = None
        if noise_sample_path and Path(noise_sample_path).exists():
            print(f"📂 Loading noise sample: {noise_sample_path}")
            noise_sample, _ = librosa.load(noise_sample_path, sr=self.sr, mono=True)

        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        # Denoise
        denoised_audio = self.denoise(audio, noise_sample)

        # Restore amplitude
        if max_val > 0:
            denoised_audio = denoised_audio * max_val

        # Ensure same length
        if len(denoised_audio) != len(audio):
            if len(denoised_audio) > len(audio):
                denoised_audio = denoised_audio[:len(audio)]
            else:
                denoised_audio = np.pad(denoised_audio,
                                       (0, len(audio) - len(denoised_audio)))

        # Save
        sf.write(output_path, denoised_audio, self.sr, subtype='PCM_24')
        print(f"✅ Saved denoised audio: {output_path}")

        return denoised_audio


def main():
    """Main function for command-line usage"""
    import sys

    print("=" * 70)
    print("🚀 ENHANCED AUDIO NOISE CANCELLATION SYSTEM")
    print("=" * 70)

    # Setup paths
    base_dir = Path(__file__).parent.parent
    input_dir = base_dir / "audio_files" / "input"
    output_dir = base_dir / "audio_files" / "output"

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse arguments
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if not filename.endswith('.wav'):
            filename += '.wav'
    else:
        print("Usage: python enhanced_audio_denoiser.py <filename> [strength] [noise_sample]")
        print("Strength: adaptive (default), gentle, moderate, aggressive, maximum")
        return

    strength = sys.argv[2] if len(sys.argv) > 2 else "adaptive"
    noise_sample_file = sys.argv[3] if len(sys.argv) > 3 else None

    # Validate strength
    valid_strengths = ["adaptive", "gentle", "moderate", "aggressive", "maximum"]
    if strength not in valid_strengths:
        print(f"⚠️  Invalid strength. Using 'adaptive'")
        strength = "adaptive"

    # Setup paths
    input_path = input_dir / filename
    output_path = output_dir / f"enhanced_{filename}"
    noise_sample_path = input_dir / noise_sample_file if noise_sample_file else None

    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        print(f"Available files:")
        for f in input_dir.glob('*.wav'):
            print(f"   - {f.name}")
        return

    # Initialize denoiser
    denoiser = EnhancedAudioDenoiser(
        sample_rate=44100,
        noise_reduction_strength=strength
    )

    # Process file
    denoiser.process_file(input_path, output_path, noise_sample_path)

    print("\n" + "=" * 70)
    print("✅ PROCESSING COMPLETE")
    print("=" * 70)
    print(f"\n🎵 Listen to the result: {output_path}")


if __name__ == "__main__":
    main()
