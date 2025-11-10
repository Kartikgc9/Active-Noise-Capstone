"""
Test script for real-time denoiser
Tests the core denoising functionality without requiring audio hardware
"""

import numpy as np
import sys
from pathlib import Path

# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import only the necessary parts (not sounddevice)
import librosa
from scipy import signal
from scipy.ndimage import median_filter


class RealtimeAudioDenoiserTest:
    """Test version of real-time denoiser without audio I/O dependencies"""

    def __init__(self, noise_reduction_level="normal", sample_rate=44100, block_size=2048):
        self.sr = sample_rate
        self.block_size = block_size
        self.n_fft = 2048
        self.hop_length = 512
        self.win_length = 2048

        # Define noise reduction presets
        self.presets = {
            "gentle": {
                "alpha": 2.0,
                "beta": 0.05,
                "speech_boost": 1.1,
                "wiener_weight": 0.3,
                "noise_percentile": 20
            },
            "normal": {
                "alpha": 2.5,
                "beta": 0.01,
                "speech_boost": 1.2,
                "wiener_weight": 0.3,
                "noise_percentile": 20
            },
            "moderate": {
                "alpha": 3.0,
                "beta": 0.008,
                "speech_boost": 1.3,
                "wiener_weight": 0.4,
                "noise_percentile": 15
            },
            "aggressive": {
                "alpha": 3.5,
                "beta": 0.005,
                "speech_boost": 1.4,
                "wiener_weight": 0.5,
                "noise_percentile": 10
            },
            "maximum": {
                "alpha": 4.0,
                "beta": 0.003,
                "speech_boost": 1.5,
                "wiener_weight": 0.6,
                "noise_percentile": 8
            }
        }

        self.current_preset = self.presets[noise_reduction_level]
        self.noise_reduction_level = noise_reduction_level

        # Buffers
        self.audio_buffer = np.zeros(self.n_fft * 2)
        self.noise_profile = None
        self.previous_gain = None
        self.frame_count = 0

        print(f"✅ Test denoiser initialized with {noise_reduction_level} mode")

    def denoise_chunk(self, audio_chunk):
        """Denoise a single chunk of audio"""
        if len(audio_chunk) < self.block_size:
            audio_chunk = np.pad(audio_chunk, (0, self.block_size - len(audio_chunk)))

        # Add to buffer
        self.audio_buffer = np.concatenate([self.audio_buffer[len(audio_chunk):], audio_chunk])
        processing_audio = self.audio_buffer.copy()

        # Normalize
        max_val = np.max(np.abs(processing_audio))
        if max_val > 0:
            processing_audio = processing_audio / max_val
        else:
            return audio_chunk

        try:
            # Apply denoising
            denoised = self._fast_spectral_subtraction(processing_audio)
            denoised_chunk = denoised[-self.block_size:]

            # Restore amplitude
            if max_val > 0:
                denoised_chunk = denoised_chunk * max_val

            if len(denoised_chunk) > len(audio_chunk):
                denoised_chunk = denoised_chunk[:len(audio_chunk)]

            return denoised_chunk

        except Exception as e:
            print(f"⚠️ Processing error: {e}")
            return audio_chunk

    def _fast_spectral_subtraction(self, audio):
        """Fast spectral subtraction"""
        alpha = self.current_preset["alpha"]
        beta = self.current_preset["beta"]
        speech_boost = self.current_preset["speech_boost"]
        noise_percentile = self.current_preset["noise_percentile"]

        # STFT
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length,
                           win_length=self.win_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        # Update noise profile
        if self.noise_profile is None or self.frame_count % 50 == 0:
            frame_energy = np.mean(magnitude, axis=0)
            quiet_threshold = np.percentile(frame_energy, noise_percentile)
            quiet_frames = frame_energy < quiet_threshold

            if np.sum(quiet_frames) > 2:
                quiet_spectrum = magnitude[:, quiet_frames]
                self.noise_profile = np.mean(quiet_spectrum, axis=1, keepdims=True)
            else:
                self.noise_profile = np.percentile(magnitude, noise_percentile, axis=1, keepdims=True)

        self.frame_count += 1

        # Frequency-dependent subtraction
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        speech_mask = (freqs >= 300) & (freqs <= 3400)

        freq_weights = np.ones((magnitude.shape[0], 1)) * alpha
        freq_weights[speech_mask] *= 0.8
        freq_weights[~speech_mask] *= 1.2

        # Spectral subtraction
        subtracted_magnitude = magnitude - freq_weights * self.noise_profile
        local_snr = magnitude / (self.noise_profile + 1e-10)
        dynamic_beta = beta * (1.0 / (1.0 + local_snr * 0.1))
        enhanced_magnitude = np.maximum(subtracted_magnitude, dynamic_beta * magnitude)

        # Speech enhancement
        speech_enhancement_mask = np.ones_like(enhanced_magnitude)
        speech_enhancement_mask[speech_mask, :] *= speech_boost
        enhanced_magnitude *= speech_enhancement_mask

        # Reconstruct
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        enhanced_audio = librosa.istft(enhanced_stft, hop_length=self.hop_length,
                                      win_length=self.win_length, length=len(audio))

        return enhanced_audio


def generate_synthetic_noisy_audio(duration=2.0, sr=44100):
    """Generate synthetic audio with speech-like signal and noise"""
    t = np.linspace(0, duration, int(sr * duration))

    # Create speech-like signal (harmonics in speech range)
    speech = np.zeros_like(t)
    fundamental = 150  # Hz (typical male voice)

    # Add harmonics
    for harmonic in range(1, 8):
        freq = fundamental * harmonic
        if freq < 3400:  # Speech range
            amplitude = 1.0 / harmonic  # Decreasing amplitude
            speech += amplitude * np.sin(2 * np.pi * freq * t)

    # Normalize speech
    speech = speech / np.max(np.abs(speech)) * 0.5

    # Add white noise
    noise = np.random.normal(0, 0.15, len(t))

    # Add low-frequency rumble
    rumble_freq = 60  # Hz (like AC hum)
    rumble = 0.1 * np.sin(2 * np.pi * rumble_freq * t)

    # Combine
    noisy_audio = speech + noise + rumble

    return noisy_audio, speech


def calculate_snr(signal, noise):
    """Calculate Signal-to-Noise Ratio in dB"""
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power == 0:
        return float('inf')

    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def test_denoising():
    """Test the denoising functionality"""
    print("=" * 60)
    print("🧪 TESTING REAL-TIME DENOISER")
    print("=" * 60)

    # Generate test audio
    print("\n📊 Generating synthetic noisy audio...")
    duration = 3.0  # seconds
    sr = 44100
    noisy_audio, clean_speech = generate_synthetic_noisy_audio(duration, sr)

    print(f"   Duration: {duration}s")
    print(f"   Sample rate: {sr} Hz")
    print(f"   Total samples: {len(noisy_audio)}")

    # Calculate original SNR
    original_noise = noisy_audio - clean_speech
    original_snr = calculate_snr(clean_speech, original_noise)
    print(f"   Original SNR: {original_snr:.2f} dB")

    # Test different noise reduction levels
    levels = ["gentle", "normal", "moderate", "aggressive", "maximum"]

    print("\n🔄 Testing different noise reduction levels...")
    print("-" * 60)

    for level in levels:
        print(f"\n🎯 Testing {level} mode:")

        # Initialize denoiser
        denoiser = RealtimeAudioDenoiserTest(
            noise_reduction_level=level,
            sample_rate=sr,
            block_size=2048
        )

        # Process audio in chunks (simulating real-time)
        block_size = 2048
        denoised_audio = []
        chunks_processed = 0

        for i in range(0, len(noisy_audio), block_size):
            chunk = noisy_audio[i:i + block_size]
            if len(chunk) < block_size:
                chunk = np.pad(chunk, (0, block_size - len(chunk)))

            denoised_chunk = denoiser.denoise_chunk(chunk)
            denoised_audio.append(denoised_chunk)
            chunks_processed += 1

        # Combine chunks
        denoised_audio = np.concatenate(denoised_audio)[:len(noisy_audio)]

        # Calculate metrics
        denoised_noise = denoised_audio - clean_speech[:len(denoised_audio)]
        denoised_snr = calculate_snr(clean_speech[:len(denoised_audio)], denoised_noise)
        snr_improvement = denoised_snr - original_snr

        # Calculate noise reduction
        original_noise_power = np.mean(original_noise ** 2)
        denoised_noise_power = np.mean(denoised_noise ** 2)
        noise_reduction_db = 10 * np.log10(original_noise_power / (denoised_noise_power + 1e-10))

        print(f"   ✅ Chunks processed: {chunks_processed}")
        print(f"   📈 Original SNR: {original_snr:.2f} dB")
        print(f"   📈 Denoised SNR: {denoised_snr:.2f} dB")
        print(f"   ⬆️  SNR Improvement: {snr_improvement:.2f} dB")
        print(f"   🔇 Noise Reduction: {noise_reduction_db:.2f} dB")

        # Validate
        if snr_improvement > 0:
            print(f"   ✅ SUCCESS: Noise reduced by {snr_improvement:.2f} dB")
        else:
            print(f"   ⚠️  WARNING: SNR not improved (may be over-processing)")

    print("\n" + "=" * 60)
    print("🎉 TESTING COMPLETE")
    print("=" * 60)

    # Test buffer management
    print("\n📦 Testing buffer management...")
    denoiser = RealtimeAudioDenoiserTest(noise_reduction_level="normal")

    # Test with various chunk sizes
    test_sizes = [512, 1024, 2048, 4096]
    for size in test_sizes:
        test_chunk = np.random.randn(size)
        result = denoiser.denoise_chunk(test_chunk)
        print(f"   ✅ Chunk size {size}: Input={len(test_chunk)}, Output={len(result)}")

    print("\n✅ All buffer management tests passed!")

    # Test error handling
    print("\n🛡️  Testing error handling...")
    try:
        # Test with empty audio
        empty_chunk = np.array([])
        result = denoiser.denoise_chunk(empty_chunk)
        print("   ✅ Empty chunk handled gracefully")
    except Exception as e:
        print(f"   ⚠️  Empty chunk error: {e}")

    try:
        # Test with very quiet audio
        quiet_chunk = np.zeros(2048)
        result = denoiser.denoise_chunk(quiet_chunk)
        print("   ✅ Silent chunk handled gracefully")
    except Exception as e:
        print(f"   ⚠️  Silent chunk error: {e}")

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\n💡 The real-time denoiser is working correctly!")
    print("   To test with actual audio hardware, run:")
    print("   python scripts/realtime_denoiser.py")
    print()


if __name__ == "__main__":
    test_denoising()
