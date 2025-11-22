"""
Comprehensive Testing Suite for Advanced ANC System
Tests all AirPods-inspired features
"""

import numpy as np
import soundfile as sf
import os
import sys
import time
from typing import List, Dict, Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_anc_system import (
    MultiModeANCSystem,
    AudioMode,
    VoiceActivityDetector,
    AdaptiveTransparencyProcessor,
    HearingAidProcessor,
    TransparencyConfig,
    HearingAidConfig
)
from audio_equalizer import AudioEqualizer, SpatialAudioSimulator


class ANCSystemTester:
    """Comprehensive test suite for advanced ANC features"""

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.test_results = []

    def generate_test_signals(self) -> Dict[str, np.ndarray]:
        """Generate various test signals"""
        duration = 3.0  # seconds
        t = np.linspace(0, duration, int(self.sample_rate * duration))

        signals = {}

        # 1. Pure sine wave (440 Hz - A4 note)
        signals['sine_440hz'] = np.sin(2 * np.pi * 440 * t)

        # 2. White noise
        signals['white_noise'] = np.random.normal(0, 0.1, len(t))

        # 3. Pink noise (1/f noise - more natural)
        white = np.random.normal(0, 1, len(t))
        pink = np.cumsum(white)
        pink = pink / np.max(np.abs(pink)) * 0.1
        signals['pink_noise'] = pink

        # 4. Speech simulation (fundamental + harmonics)
        f0 = 150  # Fundamental frequency
        speech = (
            np.sin(2 * np.pi * f0 * t) +
            0.5 * np.sin(2 * np.pi * 2 * f0 * t) +
            0.3 * np.sin(2 * np.pi * 3 * f0 * t) +
            0.2 * np.sin(2 * np.pi * 4 * f0 * t)
        )
        # Add envelope (amplitude modulation)
        envelope = np.sin(2 * np.pi * 5 * t) * 0.5 + 0.5
        signals['speech_sim'] = speech * envelope * 0.3

        # 5. Low-frequency hum (fan/AC noise)
        signals['low_freq_hum'] = (
            np.sin(2 * np.pi * 50 * t) +  # 50 Hz
            0.5 * np.sin(2 * np.pi * 60 * t)  # 60 Hz
        ) * 0.2

        # 6. Mixed signal (speech + noise)
        signals['speech_plus_noise'] = signals['speech_sim'] + signals['white_noise'] * 0.5

        # 7. Music simulation (multiple frequencies)
        signals['music_sim'] = (
            np.sin(2 * np.pi * 262 * t) +  # C4
            np.sin(2 * np.pi * 330 * t) +  # E4
            np.sin(2 * np.pi * 392 * t)    # G4
        ) / 3 * 0.3

        return signals

    def test_voice_activity_detection(self) -> Dict[str, any]:
        """Test Voice Activity Detector"""
        print("\n" + "=" * 70)
        print("TEST 1: Voice Activity Detection")
        print("=" * 70)

        vad = VoiceActivityDetector(self.sample_rate)
        signals = self.generate_test_signals()

        results = {}

        for signal_name, signal in signals.items():
            is_voice, confidence = vad.detect_voice(signal)
            results[signal_name] = {
                'is_voice': is_voice,
                'confidence': confidence
            }
            print(f"  {signal_name:20s}: Voice={is_voice:5s}  Confidence={confidence:.3f}")

        # Expected results
        expected_voice = ['speech_sim', 'speech_plus_noise']
        expected_non_voice = ['white_noise', 'pink_noise', 'low_freq_hum']

        # Validate
        correct = 0
        total = len(expected_voice) + len(expected_non_voice)

        for signal_name in expected_voice:
            if results[signal_name]['is_voice']:
                correct += 1

        for signal_name in expected_non_voice:
            if not results[signal_name]['is_voice']:
                correct += 1

        accuracy = correct / total * 100

        print(f"\n  ✓ Accuracy: {accuracy:.1f}% ({correct}/{total} correct)")

        return {
            'test_name': 'Voice Activity Detection',
            'accuracy': accuracy,
            'passed': accuracy >= 60,  # At least 60% accuracy
            'details': results
        }

    def test_transparency_mode(self) -> Dict[str, any]:
        """Test Adaptive Transparency Mode"""
        print("\n" + "=" * 70)
        print("TEST 2: Adaptive Transparency Mode")
        print("=" * 70)

        # Test with different configurations
        configs = {
            'default': TransparencyConfig(),
            'high_amplification': TransparencyConfig(amplification=2.0),
            'conversation_focus': TransparencyConfig(
                conversation_boost=2.0,
                ambient_reduction=0.8
            ),
            'balanced': TransparencyConfig(
                amplification=1.5,
                conversation_boost=1.5,
                ambient_reduction=0.5
            )
        }

        signals = self.generate_test_signals()
        test_signal = signals['speech_plus_noise']

        results = {}

        for config_name, config in configs.items():
            processor = AdaptiveTransparencyProcessor(self.sample_rate, config)

            start_time = time.time()
            processed = processor.process(test_signal)
            processing_time = (time.time() - start_time) * 1000  # ms

            # Calculate metrics
            output_rms = np.sqrt(np.mean(processed ** 2))
            input_rms = np.sqrt(np.mean(test_signal ** 2))
            amplification_actual = output_rms / input_rms

            results[config_name] = {
                'processing_time_ms': processing_time,
                'amplification_actual': amplification_actual,
                'output_rms': output_rms,
                'clipping': np.any(np.abs(processed) > 1.0)
            }

            print(f"\n  Config: {config_name}")
            print(f"    Processing time: {processing_time:.2f} ms")
            print(f"    Amplification: {amplification_actual:.2f}x")
            print(f"    Clipping: {results[config_name]['clipping']}")

        # Validation: No clipping, reasonable processing time
        all_valid = all(
            not r['clipping'] and r['processing_time_ms'] < 500
            for r in results.values()
        )

        print(f"\n  ✓ All tests passed: {all_valid}")

        return {
            'test_name': 'Transparency Mode',
            'passed': all_valid,
            'details': results
        }

    def test_hearing_aid_processing(self) -> Dict[str, any]:
        """Test Hearing Aid Functionality"""
        print("\n" + "=" * 70)
        print("TEST 3: Hearing Aid Processing")
        print("=" * 70)

        # Create stereo test signal
        signals = self.generate_test_signals()
        mono_signal = signals['speech_plus_noise']
        stereo_signal = np.column_stack([mono_signal, mono_signal])

        # Test configurations
        configs = {
            'balanced': HearingAidConfig(),
            'left_boost': HearingAidConfig(left_amplification=1.5, right_amplification=1.0),
            'right_boost': HearingAidConfig(left_amplification=1.0, right_amplification=1.5),
            'high_freq_boost': HearingAidConfig(
                frequency_shaping={
                    'low': 0.8,
                    'mid': 1.0,
                    'high': 1.5,
                    'ultra': 1.3
                }
            )
        }

        results = {}

        for config_name, config in configs.items():
            processor = HearingAidProcessor(self.sample_rate, config)

            start_time = time.time()
            processed = processor.process(stereo_signal)
            processing_time = (time.time() - start_time) * 1000

            # Analyze channels
            left_rms = np.sqrt(np.mean(processed[:, 0] ** 2))
            right_rms = np.sqrt(np.mean(processed[:, 1] ** 2))
            balance_ratio = left_rms / (right_rms + 1e-8)

            results[config_name] = {
                'processing_time_ms': processing_time,
                'left_rms': left_rms,
                'right_rms': right_rms,
                'balance_ratio': balance_ratio,
                'clipping': np.any(np.abs(processed) > 1.0)
            }

            print(f"\n  Config: {config_name}")
            print(f"    Processing time: {processing_time:.2f} ms")
            print(f"    L/R balance: {balance_ratio:.2f}")
            print(f"    Clipping: {results[config_name]['clipping']}")

        all_valid = all(
            not r['clipping'] and r['processing_time_ms'] < 500
            for r in results.values()
        )

        print(f"\n  ✓ All tests passed: {all_valid}")

        return {
            'test_name': 'Hearing Aid Processing',
            'passed': all_valid,
            'details': results
        }

    def test_equalizer(self) -> Dict[str, any]:
        """Test Audio Equalizer"""
        print("\n" + "=" * 70)
        print("TEST 4: Audio Equalizer")
        print("=" * 70)

        eq = AudioEqualizer(self.sample_rate)
        signals = self.generate_test_signals()
        test_signal = signals['music_sim']

        # Test all presets
        results = {}

        for preset_name in eq.presets.keys():
            eq.apply_preset(preset_name)

            start_time = time.time()
            processed = eq.process(test_signal)
            processing_time = (time.time() - start_time) * 1000

            # Calculate spectral changes
            input_spectrum = np.abs(np.fft.rfft(test_signal))
            output_spectrum = np.abs(np.fft.rfft(processed))

            spectral_change = np.mean(np.abs(output_spectrum - input_spectrum))

            results[preset_name] = {
                'processing_time_ms': processing_time,
                'spectral_change': spectral_change,
                'clipping': np.any(np.abs(processed) > 1.0)
            }

            print(f"\n  Preset: {preset_name:15s}  Time: {processing_time:6.2f} ms  "
                  f"Spectral Δ: {spectral_change:.2e}  Clipping: {results[preset_name]['clipping']}")

        all_valid = all(
            not r['clipping'] and r['processing_time_ms'] < 1000
            for r in results.values()
        )

        print(f"\n  ✓ All presets tested: {all_valid}")

        return {
            'test_name': 'Equalizer',
            'passed': all_valid,
            'details': results
        }

    def test_spatial_audio(self) -> Dict[str, any]:
        """Test Spatial Audio Simulation"""
        print("\n" + "=" * 70)
        print("TEST 5: Spatial Audio Simulation")
        print("=" * 70)

        spatial = SpatialAudioSimulator(self.sample_rate)
        signals = self.generate_test_signals()

        # Create stereo test signal
        mono_signal = signals['music_sim']
        stereo_signal = np.column_stack([mono_signal, mono_signal])

        # Test different positions
        positions = [
            ('center', 0, 0, 1.0),
            ('left', -90, 0, 1.0),
            ('right', 90, 0, 1.0),
            ('front_left', -45, 0, 1.0),
            ('front_right', 45, 0, 1.0),
            ('above', 0, 45, 1.0),
            ('below', 0, -45, 1.0),
            ('far', 0, 0, 2.0),
            ('close', 0, 0, 0.7),
        ]

        results = {}

        for name, azimuth, elevation, distance in positions:
            start_time = time.time()
            processed = spatial.process_stereo(stereo_signal, azimuth, elevation, distance)
            processing_time = (time.time() - start_time) * 1000

            # Analyze stereo field
            left_rms = np.sqrt(np.mean(processed[:, 0] ** 2))
            right_rms = np.sqrt(np.mean(processed[:, 1] ** 2))
            stereo_width = abs(left_rms - right_rms) / (left_rms + right_rms + 1e-8)

            results[name] = {
                'azimuth': azimuth,
                'elevation': elevation,
                'distance': distance,
                'processing_time_ms': processing_time,
                'stereo_width': stereo_width,
                'left_rms': left_rms,
                'right_rms': right_rms,
                'clipping': np.any(np.abs(processed) > 1.0)
            }

            print(f"\n  Position: {name:12s}  Az:{azimuth:4.0f}°  El:{elevation:4.0f}°  "
                  f"Dist:{distance:.1f}  Width:{stereo_width:.3f}  "
                  f"Time:{processing_time:.2f}ms")

        all_valid = all(
            not r['clipping'] and r['processing_time_ms'] < 1000
            for r in results.values()
        )

        print(f"\n  ✓ All positions tested: {all_valid}")

        return {
            'test_name': 'Spatial Audio',
            'passed': all_valid,
            'details': results
        }

    def test_multi_mode_anc(self) -> Dict[str, any]:
        """Test Multi-Mode ANC System"""
        print("\n" + "=" * 70)
        print("TEST 6: Multi-Mode ANC System")
        print("=" * 70)

        anc_system = MultiModeANCSystem(
            sample_rate=self.sample_rate,
            block_size=2048,
            mode=AudioMode.ANC,
            noise_reduction_level="normal"
        )

        signals = self.generate_test_signals()

        # Create noise profile from low frequency hum
        noise_audio = signals['low_freq_hum']
        import librosa
        stft = librosa.stft(noise_audio, n_fft=2048, hop_length=512)
        anc_system.noise_profile = np.median(np.abs(stft), axis=1)
        anc_system.is_calibrated = True

        # Test each mode
        test_signal = signals['speech_plus_noise'][:2048]  # One block
        modes = [AudioMode.OFF, AudioMode.TRANSPARENCY, AudioMode.ANC, AudioMode.ADAPTIVE]

        results = {}

        for mode in modes:
            anc_system.set_mode(mode)

            start_time = time.time()
            processed = anc_system.process_chunk(test_signal)
            processing_time = (time.time() - start_time) * 1000

            # Calculate noise reduction (if any)
            input_power = np.mean(test_signal ** 2)
            output_power = np.mean(processed ** 2)
            reduction_db = 10 * np.log10((input_power + 1e-10) / (output_power + 1e-10))

            results[mode.value] = {
                'processing_time_ms': processing_time,
                'reduction_db': reduction_db,
                'input_rms': np.sqrt(input_power),
                'output_rms': np.sqrt(output_power)
            }

            print(f"\n  Mode: {mode.value:12s}  Time: {processing_time:6.2f} ms  "
                  f"Reduction: {reduction_db:+6.2f} dB")

        # Get performance stats
        stats = anc_system.get_performance_stats()
        print(f"\n  System Stats:")
        print(f"    Avg latency: {stats.get('avg_latency_ms', 0):.2f} ms")
        print(f"    Max latency: {stats.get('max_latency_ms', 0):.2f} ms")

        all_valid = all(r['processing_time_ms'] < 100 for r in results.values())

        print(f"\n  ✓ All modes tested: {all_valid}")

        return {
            'test_name': 'Multi-Mode ANC',
            'passed': all_valid,
            'details': results
        }

    def test_real_time_performance(self) -> Dict[str, any]:
        """Test real-time performance under load"""
        print("\n" + "=" * 70)
        print("TEST 7: Real-Time Performance")
        print("=" * 70)

        anc_system = MultiModeANCSystem(
            sample_rate=self.sample_rate,
            block_size=2048,
            mode=AudioMode.ADAPTIVE
        )

        # Simulate noise profile
        signals = self.generate_test_signals()
        noise_audio = signals['pink_noise']
        import librosa
        stft = librosa.stft(noise_audio, n_fft=2048, hop_length=512)
        anc_system.noise_profile = np.median(np.abs(stft), axis=1)
        anc_system.is_calibrated = True

        # Process multiple chunks to simulate real-time
        num_chunks = 100
        chunk_size = 2048

        processing_times = []
        test_audio = signals['speech_plus_noise']

        print(f"\n  Processing {num_chunks} chunks...")

        for i in range(num_chunks):
            # Extract chunk
            start_idx = (i * chunk_size) % (len(test_audio) - chunk_size)
            chunk = test_audio[start_idx:start_idx + chunk_size]

            # Process
            start_time = time.time()
            processed = anc_system.process_chunk(chunk)
            processing_time = (time.time() - start_time) * 1000

            processing_times.append(processing_time)

        # Statistics
        avg_time = np.mean(processing_times)
        max_time = np.max(processing_times)
        min_time = np.min(processing_times)
        std_time = np.std(processing_times)

        # Real-time requirement: processing must be faster than audio duration
        audio_duration_ms = (chunk_size / self.sample_rate) * 1000  # ~46.4 ms for 2048 samples
        real_time_factor = audio_duration_ms / avg_time

        print(f"\n  Processing Statistics:")
        print(f"    Chunks processed: {num_chunks}")
        print(f"    Average time: {avg_time:.2f} ms")
        print(f"    Min time: {min_time:.2f} ms")
        print(f"    Max time: {max_time:.2f} ms")
        print(f"    Std dev: {std_time:.2f} ms")
        print(f"    Audio duration: {audio_duration_ms:.2f} ms")
        print(f"    Real-time factor: {real_time_factor:.2f}x")

        # Pass if we can process faster than real-time
        passed = real_time_factor > 1.0

        print(f"\n  ✓ Real-time capable: {passed}")

        return {
            'test_name': 'Real-Time Performance',
            'passed': passed,
            'avg_time_ms': avg_time,
            'max_time_ms': max_time,
            'real_time_factor': real_time_factor
        }

    def run_all_tests(self) -> List[Dict]:
        """Run all tests and generate report"""
        print("\n" + "=" * 70)
        print("  ADVANCED ANC SYSTEM - COMPREHENSIVE TEST SUITE")
        print("=" * 70)
        print(f"\nSample Rate: {self.sample_rate} Hz")
        print(f"Test Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        tests = [
            self.test_voice_activity_detection,
            self.test_transparency_mode,
            self.test_hearing_aid_processing,
            self.test_equalizer,
            self.test_spatial_audio,
            self.test_multi_mode_anc,
            self.test_real_time_performance
        ]

        results = []

        for test_func in tests:
            try:
                result = test_func()
                results.append(result)
            except Exception as e:
                print(f"\n  ✗ Test failed with exception: {e}")
                results.append({
                    'test_name': test_func.__name__,
                    'passed': False,
                    'error': str(e)
                })

        # Summary
        print("\n" + "=" * 70)
        print("  TEST SUMMARY")
        print("=" * 70)

        passed_tests = sum(1 for r in results if r.get('passed', False))
        total_tests = len(results)

        for result in results:
            status = "✓ PASS" if result.get('passed', False) else "✗ FAIL"
            print(f"  {status}  {result.get('test_name', 'Unknown')}")

        print(f"\n  Total: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
        print("=" * 70)

        return results


if __name__ == "__main__":
    tester = ANCSystemTester(sample_rate=44100)
    results = tester.run_all_tests()

    # Save results
    import json
    with open('test_results.json', 'w') as f:
        # Convert to JSON-serializable format
        json_results = []
        for result in results:
            json_result = {
                'test_name': result.get('test_name', 'Unknown'),
                'passed': result.get('passed', False)
            }
            if 'accuracy' in result:
                json_result['accuracy'] = float(result['accuracy'])
            if 'real_time_factor' in result:
                json_result['real_time_factor'] = float(result['real_time_factor'])

            json_results.append(json_result)

        json.dump(json_results, f, indent=2)

    print(f"\n✓ Test results saved to test_results.json")
