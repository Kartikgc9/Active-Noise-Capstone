"""
Active Noise Cancellation (ANC) System - Main File
===================================================

Real-time audio denoising system with echo prevention.
Captures microphone input (speech + noise), denoises, and plays with delay buffer.

Usage:
    python scripts/anc_system.py [noise_reduction_level]

Noise reduction levels: gentle, normal, moderate, aggressive, maximum
Default: gentle (best performance based on testing)

Press Ctrl+C to stop
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


class ANCSystem:
    """
    Active Noise Cancellation System with echo prevention
    """

    def __init__(self, noise_reduction_level="gentle", sample_rate=44100,
                 block_size=2048, output_delay_chunks=3):
        """
        Initialize ANC system

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

        # Noise reduction presets (based on test results - gentle performs best)
        self.presets = {
            "gentle": {
                "alpha": 2.5,
                "beta": 0.02,
                "speech_boost": 1.2,
                "wiener_weight": 0.4,
                "noise_percentile": 20,
                "low_freq_boost": 1.3
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

        # Buffers
        self.audio_buffer = np.zeros(self.n_fft * 2)
        self.noise_profile = None
        self.previous_gain = None
        self.frame_count = 0

        # Delay buffer for echo prevention (stores processed chunks)
        self.delay_buffer = deque(maxlen=output_delay_chunks)
        # Pre-fill delay buffer with silence
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
        print(f"Active Noise Cancellation System")
        print(f"{'='*60}")
        print(f"Mode: {noise_reduction_level}")
        print(f"Sample rate: {self.sr} Hz")
        print(f"Block size: {self.block_size} samples (~{self.block_size/self.sr*1000:.1f}ms)")
        print(f"Output delay: {output_delay_chunks} chunks (~{delay_ms:.1f}ms)")
        print(f"{'='*60}\n")

    def denoise_chunk(self, audio_chunk):
        """
        Denoise a single chunk of audio

        Args:
            audio_chunk: numpy array of audio samples

        Returns:
            Denoised audio chunk
        """
        # Ensure we have enough samples
        if len(audio_chunk) < self.block_size:
            audio_chunk = np.pad(audio_chunk, (0, self.block_size - len(audio_chunk)))

        # Add to buffer for context
        self.audio_buffer = np.concatenate([self.audio_buffer[len(audio_chunk):], audio_chunk])

        # Use the full buffer for processing
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

            # Blend both methods
            denoised = (1 - wiener_weight) * spec_sub + wiener_weight * wiener

            # Extract current chunk
            denoised_chunk = denoised[-self.block_size:]

            # Restore amplitude
            if max_val > 0:
                denoised_chunk = denoised_chunk * max_val

            # Ensure same length as input
            if len(denoised_chunk) > len(audio_chunk):
                denoised_chunk = denoised_chunk[:len(audio_chunk)]

            return denoised_chunk

        except Exception as e:
            print(f"Warning: Processing error: {e}")
            return audio_chunk

    def _wiener_filter(self, audio):
        """Wiener filtering for noise reduction"""
        noise_percentile = self.current_preset["noise_percentile"]

        # Compute STFT
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length,
                           win_length=self.win_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        power = magnitude ** 2

        # Estimate noise power from quietest frames (adaptive)
        frame_energy = np.mean(power, axis=0)
        noise_threshold = np.percentile(frame_energy, noise_percentile)
        noise_frames = frame_energy < noise_threshold

        if np.sum(noise_frames) > 2:
            noise_power = np.mean(power[:, noise_frames], axis=1, keepdims=True)
        else:
            noise_power = np.percentile(power, 5, axis=1, keepdims=True)

        # Wiener filter gain
        signal_power = np.maximum(power - noise_power, 0.01 * power)
        wiener_gain = signal_power / (signal_power + noise_power + 1e-10)

        # Apply gain smoothing
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
        """Fast spectral subtraction with adaptive noise estimation"""
        alpha = self.current_preset["alpha"]
        beta = self.current_preset["beta"]
        speech_boost = self.current_preset["speech_boost"]

        # Compute STFT
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length,
                           win_length=self.win_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        # Adaptive noise profile estimation
        frame_energy = np.mean(magnitude, axis=0)
        mean_energy = np.mean(frame_energy)
        std_energy = np.std(frame_energy)
        quiet_threshold = mean_energy - 0.5 * std_energy
        quiet_frames = frame_energy < quiet_threshold

        # Initialize or update noise profile adaptively
        if self.noise_profile is None:
            if np.sum(quiet_frames) > 2:
                quiet_spectrum = magnitude[:, quiet_frames]
                self.noise_profile = np.mean(quiet_spectrum, axis=1, keepdims=True)
            else:
                self.noise_profile = np.percentile(magnitude, 5, axis=1, keepdims=True)
        else:
            # Continuous adaptive update
            if np.sum(quiet_frames) > 1:
                current_noise = np.mean(magnitude[:, quiet_frames], axis=1, keepdims=True)
                self.noise_profile = 0.9 * self.noise_profile + 0.1 * current_noise

        self.frame_count += 1

        # Frequency-dependent subtraction
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        speech_mask = (freqs >= 300) & (freqs <= 3400)
        low_freq_mask = (freqs >= 20) & (freqs <= 500)

        freq_weights = np.ones((magnitude.shape[0], 1)) * alpha
        freq_weights[speech_mask] *= 0.8  # Gentler in speech range
        freq_weights[low_freq_mask] *= self.current_preset["low_freq_boost"]  # Aggressive for low-freq
        freq_weights[~speech_mask & ~low_freq_mask] *= 1.2

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
        print("Starting Active Noise Cancellation...")
        print("="*60)
        print("Status: READY")
        print("="*60)
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
                print("You should hear denoised audio through your output device")
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
    print("ACTIVE NOISE CANCELLATION SYSTEM")
    print("="*60)
    print()

    # Parse command line arguments
    reduction_level = "gentle"  # Default (best performer based on testing)

    if len(sys.argv) > 1 and sys.argv[1] in ["gentle", "normal", "moderate", "aggressive", "maximum"]:
        reduction_level = sys.argv[1]

    print(f"Usage: python scripts/anc_system.py [noise_reduction_level]")
    print(f"Levels: gentle, normal, moderate, aggressive, maximum")
    print(f"Current: {reduction_level} (recommended: gentle)")
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
        anc = ANCSystem(
            noise_reduction_level=reduction_level,
            sample_rate=44100,
            block_size=2048,
            output_delay_chunks=3  # ~140ms delay to prevent echo
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
