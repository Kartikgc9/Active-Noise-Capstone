"""
Real-Time Noise Cancellation System
====================================

Complete system that:
1. Captures audio from microphone
2. Profiles noise (initial calibration)
3. Denoises audio in real-time
4. Plays back clean speech through speakers

Optimized for Raspberry Pi 5 with low latency and efficient processing.
"""

import numpy as np
import sounddevice as sd
import librosa
from scipy import signal
from scipy.ndimage import median_filter
import queue
import threading
import time
from pathlib import Path
import argparse


class RealtimeNoiseCanceller:
    """
    Real-time noise cancellation with microphone capture and speaker playback
    """

    def __init__(self, sample_rate=16000, block_size=1024,
                 noise_reduction="adaptive", device_input=None, device_output=None):
        """
        Initialize real-time noise canceller

        Args:
            sample_rate: Audio sample rate (16kHz recommended for Pi5, 44.1kHz for desktop)
            block_size: Processing block size (smaller = lower latency, higher CPU)
            noise_reduction: 'adaptive', 'moderate', 'aggressive'
            device_input: Input device ID (None = default microphone)
            device_output: Output device ID (None = default speakers)
        """
        self.sr = sample_rate
        self.block_size = block_size
        self.n_fft = 1024  # Smaller FFT for lower latency
        self.hop_length = 256
        self.win_length = 1024

        # Device configuration
        self.device_input = device_input
        self.device_output = device_output

        # Noise reduction parameters optimized for real-time
        self.presets = {
            "adaptive": {
                "spectral_floor": 0.002,
                "over_subtraction": 2.0,
                "gate_threshold_db": -40,
                "gate_ratio": 8,
                "smoothing": 3
            },
            "moderate": {
                "spectral_floor": 0.005,
                "over_subtraction": 2.5,
                "gate_threshold_db": -45,
                "gate_ratio": 12,
                "smoothing": 5
            },
            "aggressive": {
                "spectral_floor": 0.001,
                "over_subtraction": 3.5,
                "gate_threshold_db": -50,
                "gate_ratio": 20,
                "smoothing": 7
            }
        }

        self.params = self.presets[noise_reduction]
        self.noise_reduction = noise_reduction

        # Noise profile
        self.noise_profile = None
        self.noise_profile_power = None
        self.noise_calibrated = False

        # Audio buffers
        self.audio_buffer = np.zeros(self.n_fft * 2)
        self.input_queue = queue.Queue(maxsize=20)
        self.output_queue = queue.Queue(maxsize=20)

        # Thread control
        self.is_running = False
        self.is_calibrating = False
        self.processing_thread = None

        # Statistics
        self.processed_blocks = 0
        self.dropped_blocks = 0

        # Frequency bands for speech (Hz)
        self.speech_low = 300
        self.speech_high = 3400

        print(f"✅ Real-Time Noise Canceller initialized")
        print(f"   Sample rate: {self.sr} Hz")
        print(f"   Block size: {block_size} samples (~{block_size/self.sr*1000:.1f}ms)")
        print(f"   Mode: {noise_reduction}")
        print(f"   Expected latency: ~{block_size/self.sr*2000:.1f}ms")

    def list_audio_devices(self):
        """List available audio input and output devices"""
        print("\n🎵 Available Audio Devices:")
        print("=" * 70)
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            dev_type = []
            if dev['max_input_channels'] > 0:
                dev_type.append("INPUT")
            if dev['max_output_channels'] > 0:
                dev_type.append("OUTPUT")
            print(f"{i}: {dev['name']}")
            print(f"   Type: {' + '.join(dev_type)}")
            print(f"   Channels: In={dev['max_input_channels']}, Out={dev['max_output_channels']}")
            print()

    def calibrate_noise(self, duration=3.0):
        """
        Calibrate noise profile by recording silence/ambient noise

        Args:
            duration: Duration to record noise (seconds)
        """
        print(f"\n🎤 NOISE CALIBRATION")
        print("=" * 70)
        print(f"Please remain SILENT for {duration} seconds...")
        print("Recording ambient noise for calibration...\n")

        self.is_calibrating = True

        # Record noise
        noise_samples = []

        def noise_callback(indata, frames, time_info, status):
            if status:
                print(f"⚠️  Status: {status}")
            noise_samples.append(indata[:, 0].copy())

        try:
            with sd.InputStream(samplerate=self.sr,
                              blocksize=self.block_size,
                              channels=1,
                              dtype='float32',
                              device=self.device_input,
                              callback=noise_callback):
                # Show countdown
                for i in range(int(duration)):
                    print(f"   Recording noise... {duration - i:.0f}s remaining")
                    time.sleep(1)

                time.sleep(duration % 1)

        except Exception as e:
            print(f"❌ Calibration error: {e}")
            self.is_calibrating = False
            return False

        # Concatenate noise samples
        noise_audio = np.concatenate(noise_samples)

        # Create noise profile
        stft = librosa.stft(noise_audio, n_fft=self.n_fft,
                           hop_length=self.hop_length,
                           win_length=self.win_length)
        magnitude = np.abs(stft)
        power = magnitude ** 2

        self.noise_profile = np.mean(magnitude, axis=1, keepdims=True)
        self.noise_profile_power = np.mean(power, axis=1, keepdims=True)

        # Smooth noise profile
        smoothing = self.params['smoothing']
        self.noise_profile = median_filter(self.noise_profile.squeeze(),
                                          size=smoothing).reshape(-1, 1)
        self.noise_profile_power = median_filter(self.noise_profile_power.squeeze(),
                                                 size=smoothing).reshape(-1, 1)

        self.noise_calibrated = True
        self.is_calibrating = False

        noise_level_db = 20 * np.log10(np.mean(self.noise_profile) + 1e-10)

        print(f"\n✅ Calibration complete!")
        print(f"   Noise level: {noise_level_db:.1f} dB")
        print("   You can now speak normally. The system will cancel noise.\n")

        return True

    def denoise_block(self, audio_block):
        """
        Denoise a single block of audio in real-time

        Args:
            audio_block: Audio block to denoise

        Returns:
            Denoised audio block
        """
        if not self.noise_calibrated or self.noise_profile is None:
            return audio_block

        # Ensure correct size
        if len(audio_block) < self.block_size:
            audio_block = np.pad(audio_block, (0, self.block_size - len(audio_block)))

        # Add to buffer for context
        self.audio_buffer = np.concatenate([
            self.audio_buffer[len(audio_block):],
            audio_block
        ])

        # Process the buffer
        processing_audio = self.audio_buffer.copy()

        # Normalize
        max_val = np.max(np.abs(processing_audio))
        if max_val < 1e-6:
            return audio_block  # Silent audio

        processing_audio = processing_audio / max_val

        try:
            # Compute STFT
            stft = librosa.stft(processing_audio, n_fft=self.n_fft,
                               hop_length=self.hop_length,
                               win_length=self.win_length)
            magnitude = np.abs(stft)
            phase = np.angle(stft)

            # Multi-stage denoising
            # Stage 1: Spectral subtraction
            magnitude = self._spectral_subtraction(magnitude)

            # Stage 2: Spectral gating
            magnitude = self._spectral_gating(magnitude)

            # Stage 3: Speech enhancement
            magnitude = self._speech_enhancement(magnitude)

            # Reconstruct
            enhanced_stft = magnitude * np.exp(1j * phase)
            denoised_audio = librosa.istft(enhanced_stft,
                                          hop_length=self.hop_length,
                                          win_length=self.win_length,
                                          length=len(processing_audio))

            # Extract the current block
            denoised_block = denoised_audio[-self.block_size:]

            # Restore amplitude
            if max_val > 0:
                denoised_block = denoised_block * max_val

            # Ensure same length as input
            if len(denoised_block) > len(audio_block):
                denoised_block = denoised_block[:len(audio_block)]

            return denoised_block

        except Exception as e:
            print(f"⚠️  Denoising error: {e}")
            return audio_block

    def _spectral_subtraction(self, magnitude):
        """Fast spectral subtraction for real-time processing"""
        over_sub = self.params['over_subtraction']
        floor = self.params['spectral_floor']

        # Get frequency bands
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        speech_mask = (freqs >= self.speech_low) & (freqs <= self.speech_high)

        # Adjust subtraction based on frequency
        subtraction_factor = np.ones((magnitude.shape[0], 1)) * over_sub
        subtraction_factor[speech_mask] *= 0.7  # Gentler on speech

        # Subtract noise
        subtracted = magnitude - subtraction_factor * self.noise_profile
        enhanced = np.maximum(subtracted, floor * magnitude)

        return enhanced

    def _spectral_gating(self, magnitude):
        """Apply spectral gating to remove low-level noise"""
        # Convert to dB
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        noise_db = 20 * np.log10(self.noise_profile + 1e-10)

        # Calculate threshold
        threshold_db = noise_db + self.params['gate_threshold_db']

        # Calculate gain
        gate_ratio = self.params['gate_ratio']
        gain = np.ones_like(magnitude)

        below_threshold = magnitude_db < threshold_db
        gain[below_threshold] = 1.0 / gate_ratio

        # Apply gain
        gated = magnitude * gain

        return gated

    def _speech_enhancement(self, magnitude):
        """Enhance speech frequencies"""
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)

        # Create enhancement curve
        enhancement = np.ones_like(magnitude)

        # Boost speech range
        speech_mask = (freqs >= 300) & (freqs <= 3400)
        enhancement[speech_mask, :] *= 1.2

        # Suppress rumble
        rumble_mask = freqs < 80
        enhancement[rumble_mask, :] *= 0.2

        return magnitude * enhancement

    def audio_callback(self, indata, outdata, frames, time_info, status):
        """
        Callback for real-time audio processing
        """
        if status:
            print(f"⚠️  Stream status: {status}")

        try:
            # Get input audio
            input_audio = indata[:, 0].copy()

            # Try to put in queue
            try:
                self.input_queue.put_nowait(input_audio)
            except queue.Full:
                self.dropped_blocks += 1
                if self.dropped_blocks % 50 == 0:
                    print(f"⚠️  Dropped {self.dropped_blocks} blocks (CPU overloaded)")

            # Try to get processed audio
            try:
                processed_audio = self.output_queue.get_nowait()
                outdata[:, 0] = processed_audio[:frames]
                self.processed_blocks += 1
            except queue.Empty:
                # No processed audio, output silence
                outdata[:, 0] = 0

        except Exception as e:
            print(f"❌ Callback error: {e}")
            outdata[:, 0] = 0

    def processing_worker(self):
        """
        Worker thread that processes audio blocks
        """
        print("🔄 Processing worker started\n")

        while self.is_running:
            try:
                # Get audio block
                audio_block = self.input_queue.get(timeout=0.1)

                # Denoise
                denoised_block = self.denoise_block(audio_block)

                # Put in output queue
                try:
                    self.output_queue.put_nowait(denoised_block)
                except queue.Full:
                    # Remove oldest and add new
                    try:
                        self.output_queue.get_nowait()
                        self.output_queue.put_nowait(denoised_block)
                    except:
                        pass

            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Worker error: {e}")
                continue

        print("🛑 Processing worker stopped")

    def start(self):
        """Start real-time noise cancellation"""
        print("\n🎤 STARTING REAL-TIME NOISE CANCELLATION")
        print("=" * 70)

        # Calibrate noise if not done
        if not self.noise_calibrated:
            success = self.calibrate_noise(duration=3.0)
            if not success:
                print("❌ Calibration failed. Cannot start.")
                return

        self.is_running = True

        # Start processing thread
        self.processing_thread = threading.Thread(target=self.processing_worker)
        self.processing_thread.start()

        # Give thread time to start
        time.sleep(0.1)

        try:
            # Start audio stream
            with sd.Stream(samplerate=self.sr,
                          blocksize=self.block_size,
                          channels=1,
                          dtype='float32',
                          device=(self.device_input, self.device_output),
                          latency='low',
                          callback=self.audio_callback):

                print("\n✅ NOISE CANCELLATION ACTIVE!")
                print("=" * 70)
                print("🎤 Speak into the microphone")
                print("🔊 Denoised audio is playing through speakers")
                print("⏸️  Press Ctrl+C to stop\n")

                # Keep running
                while self.is_running:
                    time.sleep(1.0)

                    # Print stats every 10 seconds
                    if self.processed_blocks % (self.sr // self.block_size * 10) == 0 and self.processed_blocks > 0:
                        print(f"📊 Processed: {self.processed_blocks} blocks | Dropped: {self.dropped_blocks}")

        except KeyboardInterrupt:
            print("\n\n⏸️  Stopping noise cancellation...")
        except Exception as e:
            print(f"\n❌ Stream error: {e}")
        finally:
            self.stop()

    def stop(self):
        """Stop real-time noise cancellation"""
        self.is_running = False

        # Wait for processing thread
        if self.processing_thread is not None:
            self.processing_thread.join(timeout=2.0)

        print(f"\n✅ Stopped")
        print(f"   Total blocks processed: {self.processed_blocks}")
        print(f"   Dropped blocks: {self.dropped_blocks}")
        if self.processed_blocks > 0:
            drop_rate = (self.dropped_blocks / (self.processed_blocks + self.dropped_blocks)) * 100
            print(f"   Drop rate: {drop_rate:.2f}%")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Real-Time Noise Cancellation System'
    )
    parser.add_argument('--sample-rate', type=int, default=16000,
                       help='Sample rate (16000 for Pi5, 44100 for desktop)')
    parser.add_argument('--block-size', type=int, default=1024,
                       help='Processing block size (smaller = lower latency)')
    parser.add_argument('--mode', choices=['adaptive', 'moderate', 'aggressive'],
                       default='adaptive',
                       help='Noise reduction mode')
    parser.add_argument('--input-device', type=int, default=None,
                       help='Input device ID (use --list to see devices)')
    parser.add_argument('--output-device', type=int, default=None,
                       help='Output device ID')
    parser.add_argument('--list', action='store_true',
                       help='List available audio devices and exit')
    parser.add_argument('--calibration-time', type=float, default=3.0,
                       help='Noise calibration duration (seconds)')

    args = parser.parse_args()

    print("=" * 70)
    print("🚀 REAL-TIME NOISE CANCELLATION SYSTEM")
    print("=" * 70)
    print()

    # List devices if requested
    if args.list:
        canceller = RealtimeNoiseCanceller()
        canceller.list_audio_devices()
        return

    # Initialize noise canceller
    canceller = RealtimeNoiseCanceller(
        sample_rate=args.sample_rate,
        block_size=args.block_size,
        noise_reduction=args.mode,
        device_input=args.input_device,
        device_output=args.output_device
    )

    # Start
    try:
        canceller.start()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
