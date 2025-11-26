"""
Active Noise Cancellation (ANC) System - Fixed Version
======================================================

Fixed version with reduced distortions and better audio quality.

Key Fixes:
- Less aggressive parameters to prevent over-processing
- Better normalization to prevent amplitude distortions
- Clipping protection
- Smoother gain transitions
- Conservative noise reduction

Usage:
    python scripts/anc_system_fixed.py [noise_reduction_level]
"""

import numpy as np
import sounddevice as sd
import librosa
from scipy import signal
import queue
import threading
import time
import warnings
import sys
from collections import deque

# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

warnings.filterwarnings("ignore")


class ANCSystemFixed:
    """
    Fixed Active Noise Cancellation System with reduced distortions
    """

    def __init__(self, noise_reduction_level="gentle", sample_rate=44100,
                 block_size=2048, output_delay_chunks=3):
        """
        Initialize ANC system with conservative parameters

        Args:
            noise_reduction_level: gentle, normal, moderate, aggressive, maximum
            sample_rate: Audio sample rate in Hz
            block_size: Size of audio chunks to process
            output_delay_chunks: Number of chunks to delay output (prevents echo)
        """
        self.sr = sample_rate
        self.block_size = block_size
        self.n_fft = 2048
        self.hop_length = 512
        self.win_length = 2048
        self.output_delay_chunks = output_delay_chunks

        # FIXED: Much more conservative parameters to prevent distortions
        self.presets = {
            "gentle": {
                "alpha": 1.5,          # Reduced from 2.5
                "beta": 0.05,          # Increased spectral floor
                "speech_boost": 1.0,   # No boost to prevent distortion
                "wiener_weight": 0.3,  # Less Wiener filtering
                "noise_percentile": 30, # More conservative
                "low_freq_boost": 1.0  # No extra boost initially
            },
            "normal": {
                "alpha": 2.0,          # Reduced from 3.5
                "beta": 0.03,
                "speech_boost": 1.0,
                "wiener_weight": 0.4,
                "noise_percentile": 25,
                "low_freq_boost": 1.2
            },
            "moderate": {
                "alpha": 2.5,          # Reduced from 4.5
                "beta": 0.02,
                "speech_boost": 1.1,
                "wiener_weight": 0.5,
                "noise_percentile": 20,
                "low_freq_boost": 1.3
            },
            "aggressive": {
                "alpha": 3.0,          # Reduced from 5.5
                "beta": 0.01,
                "speech_boost": 1.1,
                "wiener_weight": 0.5,
                "noise_percentile": 15,
                "low_freq_boost": 1.5
            },
            "maximum": {
                "alpha": 3.5,          # Reduced from 6.5
                "beta": 0.008,
                "speech_boost": 1.2,
                "wiener_weight": 0.6,
                "noise_percentile": 10,
                "low_freq_boost": 1.7
            }
        }

        self.current_preset = self.presets[noise_reduction_level]
        self.noise_reduction_level = noise_reduction_level

        # Buffers
        self.audio_buffer = np.zeros(self.n_fft * 2)
        self.noise_profile = None
        self.previous_gain = None
        self.frame_count = 0

        # FIXED: Track RMS for better normalization
        self.rms_history = deque(maxlen=10)

        # Delay buffer for echo prevention
        self.delay_buffer = deque(maxlen=output_delay_chunks)
        for _ in range(output_delay_chunks):
            self.delay_buffer.append(np.zeros(block_size))

        # Threading components
        self.input_queue = queue.Queue(maxsize=10)
        self.output_queue = queue.Queue(maxsize=10)
        self.is_running = False
        self.processing_thread = None

        # Statistics
        self.processed_chunks = 0
        self.dropped_chunks = 0

        delay_ms = (output_delay_chunks * block_size / sample_rate) * 1000

        print(f"\n{'='*60}")
        print(f"Active Noise Cancellation System (FIXED)")
        print(f"{'='*60}")
        print(f"Mode: {noise_reduction_level}")
        print(f"Sample rate: {self.sr} Hz")
        print(f"Block size: {self.block_size} samples (~{self.block_size/self.sr*1000:.1f}ms)")
        print(f"Output delay: {output_delay_chunks} chunks (~{delay_ms:.1f}ms)")
        print(f"Improvements: Reduced distortions, better audio quality")
        print(f"{'='*60}\n")

    def denoise_chunk(self, audio_chunk):
        """
        Denoise a single chunk with improved handling to reduce distortions

        Args:
            audio_chunk: numpy array of audio samples

        Returns:
            Denoised audio chunk
        """
        # Ensure we have enough samples
        if len(audio_chunk) < self.block_size:
            audio_chunk = np.pad(audio_chunk, (0, self.block_size - len(audio_chunk)))

        # FIXED: Calculate RMS for better normalization
        rms = np.sqrt(np.mean(audio_chunk ** 2))
        if rms > 0:
            self.rms_history.append(rms)

        # Add to buffer for context
        self.audio_buffer = np.concatenate([self.audio_buffer[len(audio_chunk):], audio_chunk])
        processing_audio = self.audio_buffer.copy()

        # FIXED: Better normalization using RMS instead of max
        if len(self.rms_history) > 0:
            target_rms = np.median(list(self.rms_history))
            current_rms = np.sqrt(np.mean(processing_audio ** 2))
            if current_rms > 0 and target_rms > 0:
                # Gentle normalization
                norm_factor = np.clip(target_rms / current_rms, 0.5, 2.0)
                processing_audio = processing_audio * norm_factor

        # Additional clipping protection
        max_val = np.max(np.abs(processing_audio))
        if max_val > 1.0:
            processing_audio = processing_audio / max_val

        try:
            # FIXED: Use only spectral subtraction for less distortion
            # Wiener filtering can introduce artifacts
            denoised = self._fast_spectral_subtraction(processing_audio)

            # Extract current chunk
            denoised_chunk = denoised[-self.block_size:]

            # FIXED: Soft clipping to prevent hard distortions
            denoised_chunk = np.tanh(denoised_chunk * 0.9) / 0.9

            # FIXED: Apply gentle low-pass filter to remove high-frequency artifacts
            denoised_chunk = self._apply_smoothing(denoised_chunk)

            # Ensure same length as input
            if len(denoised_chunk) > len(audio_chunk):
                denoised_chunk = denoised_chunk[:len(audio_chunk)]

            # FIXED: Final clipping protection
            denoised_chunk = np.clip(denoised_chunk, -0.95, 0.95)

            return denoised_chunk

        except Exception as e:
            print(f"Warning: Processing error: {e}")
            return audio_chunk

    def _apply_smoothing(self, audio):
        """Apply gentle smoothing to reduce artifacts"""
        # Simple moving average for smoothing
        window_size = 5
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(audio, kernel, mode='same')
        # Blend original and smoothed (90% original, 10% smoothed)
        return 0.9 * audio + 0.1 * smoothed

    def _fast_spectral_subtraction(self, audio):
        """
        Fixed spectral subtraction with conservative parameters
        """
        alpha = self.current_preset["alpha"]
        beta = self.current_preset["beta"]
        noise_percentile = self.current_preset["noise_percentile"]

        # Compute STFT
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length,
                           win_length=self.win_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        # FIXED: More conservative noise estimation
        frame_energy = np.mean(magnitude, axis=0)

        # Use percentile instead of mean-std for more stable estimation
        quiet_threshold = np.percentile(frame_energy, noise_percentile)
        quiet_frames = frame_energy < quiet_threshold

        # Initialize or update noise profile
        if self.noise_profile is None:
            if np.sum(quiet_frames) > 2:
                quiet_spectrum = magnitude[:, quiet_frames]
                self.noise_profile = np.mean(quiet_spectrum, axis=1, keepdims=True)
            else:
                self.noise_profile = np.percentile(magnitude, 10, axis=1, keepdims=True)
        else:
            # FIXED: Much slower adaptation to prevent tracking speech as noise
            if np.sum(quiet_frames) > 1:
                current_noise = np.mean(magnitude[:, quiet_frames], axis=1, keepdims=True)
                # Changed from 0.9/0.1 to 0.95/0.05 for slower adaptation
                self.noise_profile = 0.95 * self.noise_profile + 0.05 * current_noise

        self.frame_count += 1

        # FIXED: More conservative frequency-dependent subtraction
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        speech_mask = (freqs >= 200) & (freqs <= 4000)  # Wider speech range
        low_freq_mask = (freqs >= 20) & (freqs <= 300)   # Adjusted

        freq_weights = np.ones((magnitude.shape[0], 1)) * alpha
        freq_weights[speech_mask] *= 0.5  # Much gentler in speech range
        freq_weights[low_freq_mask] *= self.current_preset["low_freq_boost"]
        freq_weights[~speech_mask & ~low_freq_mask] *= 1.0  # No extra boost

        # Spectral subtraction
        subtracted_magnitude = magnitude - freq_weights * self.noise_profile

        # FIXED: Higher spectral floor to prevent over-subtraction
        enhanced_magnitude = np.maximum(subtracted_magnitude, beta * magnitude)

        # NO speech boost to prevent distortions

        # Reconstruct audio
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        enhanced_audio = librosa.istft(enhanced_stft, hop_length=self.hop_length,
                                      win_length=self.win_length, length=len(audio))

        return enhanced_audio

    def audio_callback(self, indata, outdata, frames, time_info, status):
        """Callback function for audio stream processing with delay buffer"""
        if status:
            print(f"Warning: Stream status: {status}")

        try:
            # Get input audio
            input_audio = indata[:, 0].copy()

            # Put in input queue (non-blocking)
            try:
                self.input_queue.put_nowait(input_audio)
            except queue.Full:
                self.dropped_chunks += 1

            # Try to get processed audio from output queue
            try:
                processed_audio = self.output_queue.get_nowait()

                # Add to delay buffer
                self.delay_buffer.append(processed_audio)

                # Get delayed audio from buffer (oldest chunk)
                delayed_audio = self.delay_buffer[0]

                outdata[:, 0] = delayed_audio[:frames]
                self.processed_chunks += 1

            except queue.Empty:
                # No processed audio available, output silence
                outdata[:, 0] = 0

        except Exception as e:
            print(f"Error in callback: {e}")
            outdata[:, 0] = 0

    def processing_worker(self):
        """Worker thread that processes audio chunks"""
        print("Processing worker started")

        while self.is_running:
            try:
                # Get audio chunk from input queue
                audio_chunk = self.input_queue.get(timeout=0.1)

                # Denoise the chunk
                denoised_chunk = self.denoise_chunk(audio_chunk)

                # Put in output queue
                try:
                    self.output_queue.put_nowait(denoised_chunk)
                except queue.Full:
                    # Remove oldest and add new
                    try:
                        self.output_queue.get_nowait()
                        self.output_queue.put_nowait(denoised_chunk)
                    except:
                        pass

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in processing worker: {e}")
                continue

        print("Processing worker stopped")

    def start(self):
        """Start real-time ANC system"""
        print("Starting Active Noise Cancellation (FIXED VERSION)...")
        print("="*60)
        print("Status: READY")
        print("="*60)
        print()
        print("Improvements:")
        print("  - Reduced distortions and artifacts")
        print("  - Better audio quality")
        print("  - More natural sound")
        print()
        print("Instructions:")
        print("  1. Speak into your microphone")
        print("  2. Denoised audio will play through your output device")
        print(f"  3. There is a {self.output_delay_chunks}-chunk delay to prevent echo")
        print("  4. Press Ctrl+C to stop")
        print()
        print("="*60)
        print()

        self.is_running = True

        # Start processing thread
        self.processing_thread = threading.Thread(target=self.processing_worker)
        self.processing_thread.start()

        time.sleep(0.1)

        try:
            # Create audio stream
            with sd.Stream(
                samplerate=self.sr,
                blocksize=self.block_size,
                channels=1,
                dtype='float32',
                latency='low',
                callback=self.audio_callback
            ):
                print("ANC System ACTIVE!")
                print(f"Latency: ~{(self.block_size * self.output_delay_chunks)/self.sr*1000:.1f}ms")
                print()
                print("Listening... Speak now")
                print("Audio should be clearer with less distortion")
                print()

                # Keep running
                while self.is_running:
                    time.sleep(1)
                    # Print stats every 30 seconds
                    if self.processed_chunks % (self.sr // self.block_size * 30) == 0 and self.processed_chunks > 0:
                        print(f"Stats: {self.processed_chunks} chunks processed | Dropped: {self.dropped_chunks}")

        except KeyboardInterrupt:
            print("\nStopping...")
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()

    def stop(self):
        """Stop ANC system"""
        self.is_running = False

        if self.processing_thread is not None:
            self.processing_thread.join(timeout=2.0)

        print(f"\nStopped. Total chunks processed: {self.processed_chunks}")
        print(f"Dropped chunks: {self.dropped_chunks}")


def main():
    """Main function"""
    import sys

    print("\n" + "="*60)
    print("ACTIVE NOISE CANCELLATION SYSTEM - FIXED VERSION")
    print("="*60)
    print("\nImprovements:")
    print("  - Reduced distortions and anomalies")
    print("  - Better normalization")
    print("  - Clipping protection")
    print("  - Smoother audio output")
    print()

    # Parse command line arguments
    reduction_level = "gentle"  # Default

    if len(sys.argv) > 1 and sys.argv[1] in ["gentle", "normal", "moderate", "aggressive", "maximum"]:
        reduction_level = sys.argv[1]

    print(f"Usage: python scripts/anc_system_fixed.py [noise_reduction_level]")
    print(f"Levels: gentle, normal, moderate, aggressive, maximum")
    print(f"Current: {reduction_level}")
    print()

    # Show available audio devices
    print("Available audio devices:")
    print("-" * 60)
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        print(f"  {i}: {dev['name']}")
    print()

    try:
        # Initialize and start ANC system
        anc = ANCSystemFixed(
            noise_reduction_level=reduction_level,
            sample_rate=44100,
            block_size=2048,
            output_delay_chunks=3
        )

        anc.start()

    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
