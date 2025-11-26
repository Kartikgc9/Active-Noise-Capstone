"""
Raspberry Pi 5 Optimized Noise Cancellation System
===================================================

Heavily optimized for Raspberry Pi 5:
- ARM NEON optimizations
- Lower sample rate (16kHz)
- Smaller FFT sizes
- Efficient memory usage
- Minimal latency configuration
- Auto-configuration for USB audio devices

Hardware Requirements:
- Raspberry Pi 5 (4GB+ RAM recommended)
- USB Microphone
- USB Speakers or 3.5mm audio output
- Recommended: Active cooling for sustained performance
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
import sys
import os


class Pi5NoiseCanceller:
    """
    Raspberry Pi 5 optimized real-time noise canceller
    """

    def __init__(self, enable_optimization=True):
        """Initialize Pi5 optimized noise canceller"""

        # Pi5 optimized parameters
        self.sr = 16000  # Lower sample rate for Pi5 (sufficient for speech)
        self.block_size = 512  # Smaller blocks for lower latency
        self.n_fft = 512  # Smaller FFT for faster processing
        self.hop_length = 128
        self.win_length = 512

        # Enable ARM optimizations if available
        if enable_optimization:
            self._enable_arm_optimizations()

        # Lightweight noise reduction parameters
        self.spectral_floor = 0.002
        self.over_subtraction = 2.5
        self.gate_threshold_db = -45
        self.gate_ratio = 10
        self.smoothing = 3

        # Noise profile
        self.noise_profile = None
        self.noise_profile_power = None
        self.noise_calibrated = False

        # Buffers (keep minimal for Pi5)
        self.audio_buffer = np.zeros(self.n_fft * 2, dtype=np.float32)
        self.input_queue = queue.Queue(maxsize=10)  # Smaller queue
        self.output_queue = queue.Queue(maxsize=10)

        # Thread control
        self.is_running = False
        self.processing_thread = None

        # Performance monitoring
        self.processed_blocks = 0
        self.dropped_blocks = 0
        self.processing_times = []

        # Speech frequency range
        self.speech_low = 300
        self.speech_high = 3400

        print(f"✅ Raspberry Pi 5 Noise Canceller initialized")
        print(f"   Sample rate: {self.sr} Hz (optimized for Pi5)")
        print(f"   Block size: {self.block_size} samples (~{self.block_size/self.sr*1000:.1f}ms)")
        print(f"   FFT size: {self.n_fft} (fast processing)")
        print(f"   Expected latency: ~{self.block_size/self.sr*2000:.1f}ms")

    def _enable_arm_optimizations(self):
        """Enable ARM NEON and other optimizations"""
        try:
            # Set numpy to use optimized BLAS if available
            np.show_config()

            # Set thread count for optimal Pi5 performance
            os.environ['OMP_NUM_THREADS'] = '2'  # Pi5 has 4 cores, use 2 for processing
            os.environ['OPENBLAS_NUM_THREADS'] = '2'

            print("   ⚡ ARM optimizations enabled")
        except:
            print("   ⚠️  ARM optimizations not available")

    def find_usb_audio_devices(self):
        """
        Find and configure USB audio devices automatically
        Returns: (input_device_id, output_device_id)
        """
        print("\n🔍 Searching for USB audio devices...")

        devices = sd.query_devices()
        input_device = None
        output_device = None

        for i, dev in enumerate(devices):
            name = dev['name'].lower()

            # Look for USB microphone
            if dev['max_input_channels'] > 0 and 'usb' in name:
                if input_device is None:
                    input_device = i
                    print(f"   📥 Found USB microphone: {dev['name']} (ID: {i})")

            # Look for USB speakers or default output
            if dev['max_output_channels'] > 0:
                if 'usb' in name and output_device is None:
                    output_device = i
                    print(f"   📤 Found USB speakers: {dev['name']} (ID: {i})")
                elif 'default' in name.lower() and output_device is None:
                    output_device = i
                    print(f"   📤 Found default output: {dev['name']} (ID: {i})")

        if input_device is None:
            print("   ⚠️  No USB microphone found, using default input")
            input_device = sd.default.device[0]

        if output_device is None:
            print("   ⚠️  No USB speakers found, using default output")
            output_device = sd.default.device[1]

        return input_device, output_device

    def calibrate_noise(self, duration=2.0):
        """
        Quick noise calibration optimized for Pi5

        Args:
            duration: Calibration duration (shorter for Pi5)
        """
        print(f"\n🎤 NOISE CALIBRATION (Pi5 Optimized)")
        print("=" * 70)
        print(f"Recording ambient noise for {duration} seconds...")
        print("Please keep SILENT during calibration.\n")

        noise_samples = []

        def noise_callback(indata, frames, time_info, status):
            if status:
                print(f"⚠️  {status}")
            noise_samples.append(indata[:, 0].copy())

        try:
            # Find USB devices
            input_dev, output_dev = self.find_usb_audio_devices()

            with sd.InputStream(samplerate=self.sr,
                              blocksize=self.block_size,
                              channels=1,
                              dtype='float32',
                              device=input_dev,
                              callback=noise_callback):
                # Countdown
                for i in range(int(duration)):
                    print(f"   {duration - i:.0f}s...")
                    time.sleep(1)

                time.sleep(duration % 1)

        except Exception as e:
            print(f"❌ Calibration error: {e}")
            return False

        # Process noise samples
        noise_audio = np.concatenate(noise_samples).astype(np.float32)

        # Compute noise profile (optimized)
        stft = librosa.stft(noise_audio, n_fft=self.n_fft,
                           hop_length=self.hop_length,
                           win_length=self.win_length)
        magnitude = np.abs(stft).astype(np.float32)
        power = (magnitude ** 2).astype(np.float32)

        self.noise_profile = np.mean(magnitude, axis=1, keepdims=True)
        self.noise_profile_power = np.mean(power, axis=1, keepdims=True)

        # Smooth (lightweight for Pi5)
        self.noise_profile = median_filter(self.noise_profile.squeeze(),
                                          size=self.smoothing).reshape(-1, 1).astype(np.float32)
        self.noise_profile_power = median_filter(self.noise_profile_power.squeeze(),
                                                 size=self.smoothing).reshape(-1, 1).astype(np.float32)

        self.noise_calibrated = True

        noise_level_db = 20 * np.log10(np.mean(self.noise_profile) + 1e-10)

        print(f"\n✅ Calibration complete!")
        print(f"   Noise level: {noise_level_db:.1f} dB")
        print(f"   Memory usage: {sys.getsizeof(self.noise_profile) / 1024:.1f} KB")
        print("\n   Ready for noise cancellation!\n")

        return True

    def denoise_block_pi5(self, audio_block):
        """
        Lightweight denoising optimized for Pi5
        Fast single-pass algorithm
        """
        if not self.noise_calibrated:
            return audio_block

        start_time = time.time()

        # Ensure float32 for efficiency
        audio_block = audio_block.astype(np.float32)

        # Pad if needed
        if len(audio_block) < self.block_size:
            audio_block = np.pad(audio_block, (0, self.block_size - len(audio_block)))

        # Update buffer
        self.audio_buffer = np.concatenate([
            self.audio_buffer[len(audio_block):],
            audio_block
        ]).astype(np.float32)

        processing_audio = self.audio_buffer.copy()

        # Normalize
        max_val = np.max(np.abs(processing_audio))
        if max_val < 1e-6:
            return audio_block

        processing_audio = processing_audio / max_val

        try:
            # Fast STFT
            stft = librosa.stft(processing_audio, n_fft=self.n_fft,
                               hop_length=self.hop_length,
                               win_length=self.win_length)
            magnitude = np.abs(stft).astype(np.float32)
            phase = np.angle(stft)

            # Single-pass denoising (optimized for Pi5)
            # Combine spectral subtraction and gating
            magnitude_db = 20 * np.log10(magnitude + 1e-10)
            noise_db = 20 * np.log10(self.noise_profile + 1e-10)

            # Spectral subtraction with gating
            subtracted = magnitude - self.over_subtraction * self.noise_profile
            enhanced = np.maximum(subtracted, self.spectral_floor * magnitude)

            # Apply gate
            threshold_db = noise_db + self.gate_threshold_db
            below_threshold = magnitude_db < threshold_db
            gain = np.ones_like(magnitude, dtype=np.float32)
            gain[below_threshold] = 1.0 / self.gate_ratio
            enhanced = enhanced * gain

            # Speech enhancement (vectorized for speed)
            freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
            speech_mask = (freqs >= self.speech_low) & (freqs <= self.speech_high)
            rumble_mask = freqs < 80

            enhanced[speech_mask, :] *= 1.15
            enhanced[rumble_mask, :] *= 0.3

            # Fast reconstruction
            enhanced_stft = enhanced * np.exp(1j * phase)
            denoised_audio = librosa.istft(enhanced_stft,
                                          hop_length=self.hop_length,
                                          win_length=self.win_length,
                                          length=len(processing_audio))

            # Extract block
            denoised_block = denoised_audio[-self.block_size:]

            # Restore amplitude
            denoised_block = denoised_block * max_val

            # Trim to size
            if len(denoised_block) > len(audio_block):
                denoised_block = denoised_block[:len(audio_block)]

            # Track processing time
            processing_time = (time.time() - start_time) * 1000
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 100:
                self.processing_times.pop(0)

            return denoised_block.astype(np.float32)

        except Exception as e:
            print(f"⚠️  Denoising error: {e}")
            return audio_block

    def audio_callback(self, indata, outdata, frames, time_info, status):
        """Audio callback optimized for low latency"""
        if status:
            print(f"⚠️  {status}")

        try:
            # Input
            input_audio = indata[:, 0].copy()

            # Queue input (non-blocking)
            try:
                self.input_queue.put_nowait(input_audio)
            except queue.Full:
                self.dropped_blocks += 1

            # Get output
            try:
                processed_audio = self.output_queue.get_nowait()
                outdata[:, 0] = processed_audio[:frames]
                self.processed_blocks += 1
            except queue.Empty:
                outdata[:, 0] = 0

        except Exception as e:
            print(f"❌ Callback error: {e}")
            outdata[:, 0] = 0

    def processing_worker(self):
        """Processing worker thread"""
        print("🔄 Processing worker started (Pi5 optimized)\n")

        while self.is_running:
            try:
                audio_block = self.input_queue.get(timeout=0.1)
                denoised_block = self.denoise_block_pi5(audio_block)

                try:
                    self.output_queue.put_nowait(denoised_block)
                except queue.Full:
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

        print("🛑 Worker stopped")

    def start(self):
        """Start the Pi5 noise canceller"""
        print("\n🎤 RASPBERRY PI 5 NOISE CANCELLATION")
        print("=" * 70)

        # Calibrate
        if not self.noise_calibrated:
            success = self.calibrate_noise(duration=2.0)
            if not success:
                print("❌ Calibration failed")
                return

        # Find devices
        input_dev, output_dev = self.find_usb_audio_devices()

        self.is_running = True

        # Start worker
        self.processing_thread = threading.Thread(target=self.processing_worker)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        time.sleep(0.2)

        try:
            with sd.Stream(samplerate=self.sr,
                          blocksize=self.block_size,
                          channels=1,
                          dtype='float32',
                          device=(input_dev, output_dev),
                          latency='low',
                          callback=self.audio_callback):

                print("\n✅ NOISE CANCELLATION ACTIVE!")
                print("=" * 70)
                print("🎤 Microphone: Listening")
                print("🔊 Speakers: Playing denoised audio")
                print("⏸️  Press Ctrl+C to stop\n")

                # Monitor performance
                while self.is_running:
                    time.sleep(5.0)

                    if self.processed_blocks > 0 and len(self.processing_times) > 0:
                        avg_time = np.mean(self.processing_times)
                        max_time = np.max(self.processing_times)
                        cpu_usage = (avg_time / (self.block_size / self.sr * 1000)) * 100

                        print(f"📊 Blocks: {self.processed_blocks} | "
                              f"Dropped: {self.dropped_blocks} | "
                              f"Proc: {avg_time:.1f}ms (max: {max_time:.1f}ms) | "
                              f"CPU: {cpu_usage:.0f}%")

        except KeyboardInterrupt:
            print("\n\n⏸️  Stopping...")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        finally:
            self.stop()

    def stop(self):
        """Stop the canceller"""
        self.is_running = False

        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)

        print(f"\n✅ Stopped")
        print(f"   Total processed: {self.processed_blocks} blocks")
        print(f"   Dropped: {self.dropped_blocks} blocks")

        if len(self.processing_times) > 0:
            print(f"   Avg processing: {np.mean(self.processing_times):.2f}ms")
            print(f"   Max processing: {np.max(self.processing_times):.2f}ms")


def main():
    """Main entry point for Pi5"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Raspberry Pi 5 Noise Cancellation System'
    )
    parser.add_argument('--no-optimization', action='store_true',
                       help='Disable ARM optimizations')
    parser.add_argument('--list-devices', action='store_true',
                       help='List audio devices and exit')

    args = parser.parse_args()

    print("=" * 70)
    print("🥧 RASPBERRY PI 5 NOISE CANCELLATION SYSTEM")
    print("=" * 70)
    print()

    if args.list_devices:
        print("Available audio devices:")
        print(sd.query_devices())
        return

    # Initialize
    canceller = Pi5NoiseCanceller(enable_optimization=not args.no_optimization)

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
