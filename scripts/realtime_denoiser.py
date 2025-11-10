"""
Real-Time Audio Denoiser
========================

Continuously captures audio from microphone, denoises it in real-time,
and plays back the denoised audio through speakers with minimal latency.

Usage:
    python scripts/realtime_denoiser.py [noise_reduction_level]

Noise reduction levels: gentle, normal, moderate, aggressive, maximum

Controls:
    Press Ctrl+C to stop the real-time denoising
"""

import numpy as np
import sounddevice as sd
import librosa
from scipy import signal
from scipy.ndimage import median_filter
from pathlib import Path
import queue
import threading
import time
import warnings
import sys

# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

warnings.filterwarnings("ignore")


class RealtimeAudioDenoiser:
    """
    Real-time audio denoiser with streaming capability
    """

    def __init__(self, noise_reduction_level="normal", sample_rate=44100,
                 block_size=2048, latency='low'):
        """
        Initialize real-time denoiser

        Args:
            noise_reduction_level: One of gentle, normal, moderate, aggressive, maximum
            sample_rate: Audio sample rate in Hz
            block_size: Size of audio chunks to process (smaller = lower latency, more CPU)
            latency: 'low' or 'high' - trade-off between latency and stability
        """
        self.sr = sample_rate
        self.block_size = block_size
        self.n_fft = 2048
        self.hop_length = 512
        self.win_length = 2048
        self.latency = latency

        # Define noise reduction presets (more aggressive for real-time use)
        self.presets = {
            "gentle": {
                "alpha": 2.5,
                "beta": 0.02,
                "speech_boost": 1.2,
                "wiener_weight": 0.4,
                "noise_percentile": 20,
                "low_freq_boost": 1.3  # Extra reduction for low-freq noise (fans)
            },
            "normal": {
                "alpha": 3.5,
                "beta": 0.008,
                "speech_boost": 1.3,
                "wiener_weight": 0.5,
                "noise_percentile": 15,
                "low_freq_boost": 1.5
            },
            "moderate": {
                "alpha": 4.5,
                "beta": 0.005,
                "speech_boost": 1.4,
                "wiener_weight": 0.6,
                "noise_percentile": 12,
                "low_freq_boost": 1.7
            },
            "aggressive": {
                "alpha": 5.5,
                "beta": 0.003,
                "speech_boost": 1.5,
                "wiener_weight": 0.7,
                "noise_percentile": 8,
                "low_freq_boost": 2.0
            },
            "maximum": {
                "alpha": 6.5,
                "beta": 0.001,
                "speech_boost": 1.6,
                "wiener_weight": 0.8,
                "noise_percentile": 5,
                "low_freq_boost": 2.5
            }
        }

        self.current_preset = self.presets[noise_reduction_level]
        self.noise_reduction_level = noise_reduction_level

        # Buffers for maintaining context across chunks
        self.audio_buffer = np.zeros(self.n_fft * 2)
        self.noise_profile = None
        self.previous_gain = None
        self.frame_count = 0

        # Noise profiling phase
        self.noise_calibration_chunks = []
        self.calibration_duration = 3.0  # seconds
        self.calibration_complete = False
        self.required_calibration_chunks = int(self.calibration_duration * self.sr / self.block_size)

        # Threading components
        self.input_queue = queue.Queue(maxsize=10)
        self.output_queue = queue.Queue(maxsize=10)
        self.is_running = False
        self.processing_thread = None

        # Statistics
        self.processed_chunks = 0
        self.dropped_chunks = 0

        print(f"🎯 Real-time Audio Denoiser initialized")
        print(f"   Mode: {noise_reduction_level}")
        print(f"   Sample rate: {self.sr} Hz")
        print(f"   Block size: {self.block_size} samples (~{self.block_size/self.sr*1000:.1f}ms)")
        print(f"   Latency mode: {self.latency}")

    def denoise_chunk(self, audio_chunk):
        """
        Denoise a single chunk of audio using fast spectral processing

        Args:
            audio_chunk: numpy array of audio samples

        Returns:
            Denoised audio chunk
        """
        # Ensure we have enough samples
        if len(audio_chunk) < self.block_size:
            audio_chunk = np.pad(audio_chunk, (0, self.block_size - len(audio_chunk)))

        # Noise profiling phase - collect background noise samples
        if not self.calibration_complete:
            self.noise_calibration_chunks.append(audio_chunk.copy())

            if len(self.noise_calibration_chunks) >= self.required_calibration_chunks:
                # Build noise profile from calibration data
                self._build_noise_profile_from_calibration()
                self.calibration_complete = True
                print("✅ Noise profiling complete - starting denoising")

            # During calibration, pass through audio unchanged
            return audio_chunk

        # Add to buffer for context
        self.audio_buffer = np.concatenate([self.audio_buffer[len(audio_chunk):], audio_chunk])

        # Use the full buffer for processing to maintain context
        processing_audio = self.audio_buffer.copy()

        # Normalize
        max_val = np.max(np.abs(processing_audio))
        if max_val > 0:
            processing_audio = processing_audio / max_val
        else:
            return audio_chunk

        try:
            # Apply combined denoising (spectral subtraction + Wiener filtering)
            wiener_weight = self.current_preset["wiener_weight"]

            # Apply spectral subtraction
            spec_sub = self._fast_spectral_subtraction(processing_audio)

            # Apply Wiener filtering
            wiener = self._wiener_filter(processing_audio)

            # Blend both methods for better results
            denoised = (1 - wiener_weight) * spec_sub + wiener_weight * wiener

            # Extract the current chunk from the processed audio
            denoised_chunk = denoised[-self.block_size:]

            # Restore amplitude
            if max_val > 0:
                denoised_chunk = denoised_chunk * max_val

            # Ensure same length as input
            if len(denoised_chunk) > len(audio_chunk):
                denoised_chunk = denoised_chunk[:len(audio_chunk)]

            return denoised_chunk

        except Exception as e:
            print(f"⚠️ Processing error: {e}")
            return audio_chunk

    def _build_noise_profile_from_calibration(self):
        """
        Build accurate noise profile from calibration chunks
        """
        print("🔍 Building noise profile from background noise sample...")

        # Concatenate all calibration chunks
        calibration_audio = np.concatenate(self.noise_calibration_chunks)

        # Compute STFT of calibration audio
        stft = librosa.stft(calibration_audio, n_fft=self.n_fft,
                           hop_length=self.hop_length, win_length=self.win_length)
        magnitude = np.abs(stft)

        # Take the average magnitude across all frames as noise profile
        # This gives us a very accurate estimate since it's pure background noise
        self.noise_profile = np.mean(magnitude, axis=1, keepdims=True)

        print(f"   📊 Captured {len(self.noise_calibration_chunks)} chunks ({self.calibration_duration}s)")
        print(f"   📊 Noise profile shape: {self.noise_profile.shape}")
        print(f"   📊 Average noise magnitude: {np.mean(self.noise_profile):.6f}")

    def _wiener_filter(self, audio):
        """
        Wiener filtering for noise reduction
        """
        wiener_weight = self.current_preset["wiener_weight"]
        noise_percentile = self.current_preset["noise_percentile"]

        # Compute STFT
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length,
                           win_length=self.win_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        power = magnitude ** 2

        # Estimate noise power from quietest frames
        frame_energy = np.mean(power, axis=0)
        noise_threshold = np.percentile(frame_energy, noise_percentile)
        noise_frames = frame_energy < noise_threshold

        if np.sum(noise_frames) > 2:
            noise_power = np.mean(power[:, noise_frames], axis=1, keepdims=True)
        else:
            # Use minimum statistics if no quiet frames found
            noise_power = np.percentile(power, 5, axis=1, keepdims=True)

        # Wiener filter gain
        signal_power = np.maximum(power - noise_power, 0.01 * power)
        wiener_gain = signal_power / (signal_power + noise_power + 1e-10)

        # Apply gain smoothing to reduce musical noise
        if self.previous_gain is not None:
            wiener_gain = 0.7 * self.previous_gain + 0.3 * wiener_gain
        self.previous_gain = wiener_gain.copy()

        # Apply filter
        filtered_magnitude = magnitude * wiener_gain

        # Reconstruct
        filtered_stft = filtered_magnitude * np.exp(1j * phase)
        filtered_audio = librosa.istft(filtered_stft, hop_length=self.hop_length,
                                      win_length=self.win_length, length=len(audio))

        return filtered_audio

    def _fast_spectral_subtraction(self, audio):
        """
        Fast spectral subtraction optimized for real-time processing
        """
        alpha = self.current_preset["alpha"]
        beta = self.current_preset["beta"]
        speech_boost = self.current_preset["speech_boost"]
        noise_percentile = self.current_preset["noise_percentile"]

        # Compute STFT
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length,
                           win_length=self.win_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        # Improved adaptive noise profile estimation
        frame_energy = np.mean(magnitude, axis=0)

        # Identify quiet/noise-only frames using energy-based detection
        mean_energy = np.mean(frame_energy)
        std_energy = np.std(frame_energy)
        quiet_threshold = mean_energy - 0.5 * std_energy  # Frames below this are likely noise-only
        quiet_frames = frame_energy < quiet_threshold

        # Initialize or update noise profile
        if self.noise_profile is None:
            # Initial estimation using minimum statistics
            if np.sum(quiet_frames) > 2:
                quiet_spectrum = magnitude[:, quiet_frames]
                self.noise_profile = np.mean(quiet_spectrum, axis=1, keepdims=True)
            else:
                # Use percentile-based estimation as fallback
                self.noise_profile = np.percentile(magnitude, 5, axis=1, keepdims=True)
        else:
            # Continuous adaptive update using smoothing
            if np.sum(quiet_frames) > 1:
                current_noise = np.mean(magnitude[:, quiet_frames], axis=1, keepdims=True)
                # Smooth update: blend old and new estimates
                self.noise_profile = 0.9 * self.noise_profile + 0.1 * current_noise

        self.frame_count += 1

        # Frequency-dependent subtraction
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        speech_mask = (freqs >= 300) & (freqs <= 3400)
        low_freq_mask = (freqs >= 20) & (freqs <= 500)  # Fan/hum noise range

        freq_weights = np.ones((magnitude.shape[0], 1)) * alpha
        freq_weights[speech_mask] *= 0.8  # Gentler in speech range
        freq_weights[low_freq_mask] *= self.current_preset["low_freq_boost"]  # Extra aggressive for fans/hum
        freq_weights[~speech_mask & ~low_freq_mask] *= 1.2  # More aggressive elsewhere

        # Spectral subtraction
        subtracted_magnitude = magnitude - freq_weights * self.noise_profile

        # Dynamic spectral floor
        local_snr = magnitude / (self.noise_profile + 1e-10)
        dynamic_beta = beta * (1.0 / (1.0 + local_snr * 0.1))

        enhanced_magnitude = np.maximum(subtracted_magnitude, dynamic_beta * magnitude)

        # Speech enhancement
        speech_enhancement_mask = np.ones_like(enhanced_magnitude)
        speech_enhancement_mask[speech_mask, :] *= speech_boost
        enhanced_magnitude *= speech_enhancement_mask

        # Reconstruct audio
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        enhanced_audio = librosa.istft(enhanced_stft, hop_length=self.hop_length,
                                      win_length=self.win_length, length=len(audio))

        return enhanced_audio

    def audio_callback(self, indata, outdata, frames, time_info, status):
        """
        Callback function for audio stream processing
        This is called by the audio system for each block of audio
        """
        if status:
            print(f"⚠️ Stream status: {status}")

        try:
            # Get input audio (copy to avoid issues)
            input_audio = indata[:, 0].copy()

            # Try to put in input queue (non-blocking)
            try:
                self.input_queue.put_nowait(input_audio)
            except queue.Full:
                self.dropped_chunks += 1
                if self.dropped_chunks % 10 == 0:
                    print(f"⚠️ Dropped {self.dropped_chunks} chunks (processing too slow)")

            # Try to get processed audio (non-blocking)
            try:
                processed_audio = self.output_queue.get_nowait()
                outdata[:, 0] = processed_audio[:frames]
                self.processed_chunks += 1
            except queue.Empty:
                # No processed audio available, output silence
                outdata[:, 0] = 0

        except Exception as e:
            print(f"❌ Callback error: {e}")
            outdata[:, 0] = 0

    def processing_worker(self):
        """
        Worker thread that processes audio chunks from input queue
        and puts results in output queue
        """
        print("🔄 Processing worker started")

        while self.is_running:
            try:
                # Get audio chunk from input queue (with timeout)
                audio_chunk = self.input_queue.get(timeout=0.1)

                # Denoise the chunk
                denoised_chunk = self.denoise_chunk(audio_chunk)

                # Put in output queue (non-blocking)
                try:
                    self.output_queue.put_nowait(denoised_chunk)
                except queue.Full:
                    # If output queue is full, remove oldest item
                    try:
                        self.output_queue.get_nowait()
                        self.output_queue.put_nowait(denoised_chunk)
                    except:
                        pass

            except queue.Empty:
                # No audio to process, continue
                continue
            except Exception as e:
                print(f"❌ Processing worker error: {e}")
                continue

        print("🛑 Processing worker stopped")

    def start(self):
        """Start real-time audio processing"""
        print("\n🎤 Starting real-time audio denoising...")
        print("🔊 Audio will be captured, denoised, and played back automatically")
        print("⏸️  Press Ctrl+C to stop\n")

        self.is_running = True

        # Start processing thread
        self.processing_thread = threading.Thread(target=self.processing_worker)
        self.processing_thread.start()

        # Give processing thread a moment to start
        time.sleep(0.1)

        try:
            # Determine latency setting
            latency_val = 'low' if self.latency == 'low' else 'high'

            # Create audio stream with callback
            with sd.Stream(
                samplerate=self.sr,
                blocksize=self.block_size,
                channels=1,
                dtype='float32',
                latency=latency_val,
                callback=self.audio_callback
            ):
                print("✅ Real-time denoising active!")
                print(f"📊 Latency: ~{self.block_size/self.sr*1000*2:.1f}ms (buffer + processing)")
                print("\nListening... Speak into your microphone")
                print("You should hear denoised audio through your speakers/headphones\n")

                # Keep running until interrupted
                while self.is_running:
                    time.sleep(0.5)
                    # Print stats every 10 seconds
                    if self.processed_chunks % (self.sr // self.block_size * 10) == 0 and self.processed_chunks > 0:
                        print(f"📊 Processed {self.processed_chunks} chunks | Dropped: {self.dropped_chunks}")

        except KeyboardInterrupt:
            print("\n⏸️  Stopping...")
        except Exception as e:
            print(f"\n❌ Stream error: {e}")
        finally:
            self.stop()

    def stop(self):
        """Stop real-time audio processing"""
        self.is_running = False

        # Wait for processing thread to finish
        if self.processing_thread is not None:
            self.processing_thread.join(timeout=2.0)

        print(f"\n✅ Stopped. Total chunks processed: {self.processed_chunks}")
        print(f"   Dropped chunks: {self.dropped_chunks}")


def main():
    """Main function to run real-time denoiser"""
    import sys

    print("🚀 REAL-TIME AUDIO DENOISING SYSTEM")
    print("=" * 60)

    # Parse command line arguments
    reduction_level = "normal"  # Default

    if len(sys.argv) > 1 and sys.argv[1] in ["gentle", "normal", "moderate", "aggressive", "maximum"]:
        reduction_level = sys.argv[1]

    print(f"\n💡 Usage: python realtime_denoiser.py [noise_reduction_level]")
    print(f"   Levels: gentle, normal, moderate, aggressive, maximum")
    print(f"   Current: {reduction_level}\n")

    # Check available audio devices
    print("🎵 Available audio devices:")
    print(sd.query_devices())
    print()

    try:
        # Initialize and start real-time denoiser
        denoiser = RealtimeAudioDenoiser(
            noise_reduction_level=reduction_level,
            sample_rate=44100,
            block_size=2048,  # ~46ms latency per block
            latency='low'
        )

        print("\n" + "=" * 60)
        print("🔇 NOISE PROFILING PHASE")
        print("=" * 60)
        print(f"⚠️  Please stay QUIET for {denoiser.calibration_duration} seconds...")
        print("   (The system needs to capture background noise only)")
        print("   - Fan noise, AC, birds, etc. is OK")
        print("   - Do NOT speak or make sounds")
        print("=" * 60)

        denoiser.start()

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
