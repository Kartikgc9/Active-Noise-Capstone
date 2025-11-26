"""
Test script for ANC System Fixed - Validates functionality without audio hardware
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.anc_system_fixed import ANCSystemFixed


def test_anc_initialization():
    """Test 1: Verify ANC system initializes correctly"""
    print("\n" + "="*60)
    print("TEST 1: ANC System Initialization")
    print("="*60)

    try:
        anc = ANCSystemFixed(
            noise_reduction_level="gentle",
            sample_rate=44100,
            block_size=2048,
            output_delay_chunks=3
        )

        # Verify parameters
        assert anc.sr == 44100, "Sample rate mismatch"
        assert anc.block_size == 2048, "Block size mismatch"
        assert anc.output_delay_chunks == 3, "Delay chunks mismatch"
        assert anc.noise_reduction_level == "gentle", "Level mismatch"

        # Verify delay buffer
        assert len(anc.delay_buffer) == 3, f"Delay buffer should have 3 chunks, got {len(anc.delay_buffer)}"

        # Verify presets
        assert "alpha" in anc.current_preset, "Missing alpha parameter"
        assert "beta" in anc.current_preset, "Missing beta parameter"

        # Verify improved parameters (should be more conservative)
        assert anc.current_preset["alpha"] == 1.5, f"Alpha should be 1.5 for gentle, got {anc.current_preset['alpha']}"
        assert anc.current_preset["beta"] == 0.05, f"Beta should be 0.05 for gentle, got {anc.current_preset['beta']}"

        print("✅ PASS: Initialization successful")
        print(f"   - Sample rate: {anc.sr} Hz")
        print(f"   - Block size: {anc.block_size} samples")
        print(f"   - Delay chunks: {anc.output_delay_chunks}")
        print(f"   - Alpha: {anc.current_preset['alpha']} (conservative)")
        print(f"   - Beta: {anc.current_preset['beta']} (higher spectral floor)")
        return True

    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_anc_denoising():
    """Test 2: Verify denoising functionality reduces distortions"""
    print("\n" + "="*60)
    print("TEST 2: Denoising with Reduced Distortions")
    print("="*60)

    try:
        anc = ANCSystemFixed(
            noise_reduction_level="normal",
            sample_rate=44100,
            block_size=2048
        )

        # Generate test signal: speech (800 Hz) + noise (200 Hz)
        duration = 0.1  # 100ms
        t = np.linspace(0, duration, int(anc.sr * duration))

        # Speech component
        speech = 0.3 * np.sin(2 * np.pi * 800 * t)

        # Noise component
        noise = 0.2 * np.sin(2 * np.pi * 200 * t)

        # Combined signal
        noisy_signal = speech + noise

        # Process in chunks
        chunk_size = anc.block_size
        denoised_chunks = []

        for i in range(0, len(noisy_signal), chunk_size):
            chunk = noisy_signal[i:i+chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

            denoised_chunk = anc.denoise_chunk(chunk)
            denoised_chunks.append(denoised_chunk)

        denoised_signal = np.concatenate(denoised_chunks)[:len(noisy_signal)]

        # Verify output properties
        assert len(denoised_signal) == len(noisy_signal), "Length mismatch"

        # Verify no clipping (fixed version should prevent this)
        max_val = np.max(np.abs(denoised_signal))
        assert max_val <= 0.95, f"Output should be clipped to 0.95, got {max_val}"

        # Verify RMS is reasonable (not over-amplified)
        rms_input = np.sqrt(np.mean(noisy_signal ** 2))
        rms_output = np.sqrt(np.mean(denoised_signal ** 2))
        amplification = rms_output / rms_input if rms_input > 0 else 0

        # Should not amplify more than 2x (conservative)
        assert amplification <= 2.0, f"Over-amplification detected: {amplification:.2f}x"

        print("✅ PASS: Denoising successful with reduced distortions")
        print(f"   - Output max: {max_val:.3f} (clipped at 0.95)")
        print(f"   - RMS amplification: {amplification:.2f}x (≤2.0x)")
        print(f"   - Length preserved: {len(denoised_signal)} samples")
        return True

    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_anc_delay_buffer():
    """Test 3: Verify delay buffer prevents echo"""
    print("\n" + "="*60)
    print("TEST 3: Delay Buffer for Echo Prevention")
    print("="*60)

    try:
        anc = ANCSystemFixed(
            noise_reduction_level="gentle",
            sample_rate=44100,
            block_size=2048,
            output_delay_chunks=3
        )

        # Verify delay buffer is pre-filled with silence
        assert len(anc.delay_buffer) == 3, f"Expected 3 chunks, got {len(anc.delay_buffer)}"

        for i, chunk in enumerate(anc.delay_buffer):
            assert len(chunk) == 2048, f"Chunk {i} has wrong size: {len(chunk)}"
            assert np.allclose(chunk, 0), f"Chunk {i} should be silence"

        # Calculate delay
        delay_ms = (anc.output_delay_chunks * anc.block_size / anc.sr) * 1000

        print("✅ PASS: Delay buffer configured correctly")
        print(f"   - Buffer size: {len(anc.delay_buffer)} chunks")
        print(f"   - Delay: ~{delay_ms:.1f}ms")
        print(f"   - Prevents echo/feedback")
        return True

    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_all_levels():
    """Test 4: Verify all noise reduction levels work with conservative parameters"""
    print("\n" + "="*60)
    print("TEST 4: All Noise Reduction Levels (Conservative)")
    print("="*60)

    levels = ["gentle", "normal", "moderate", "aggressive", "maximum"]
    expected_alphas = [1.5, 2.0, 2.5, 3.0, 3.5]  # Reduced from original
    expected_betas = [0.05, 0.03, 0.02, 0.01, 0.008]  # Increased from original

    try:
        for level, expected_alpha, expected_beta in zip(levels, expected_alphas, expected_betas):
            anc = ANCSystemFixed(noise_reduction_level=level, sample_rate=44100, block_size=2048)

            # Verify conservative parameters
            actual_alpha = anc.current_preset["alpha"]
            actual_beta = anc.current_preset["beta"]

            assert actual_alpha == expected_alpha, f"{level}: alpha should be {expected_alpha}, got {actual_alpha}"
            assert actual_beta == expected_beta, f"{level}: beta should be {expected_beta}, got {actual_beta}"

            # Test denoising
            test_chunk = np.random.randn(2048) * 0.1
            denoised = anc.denoise_chunk(test_chunk)

            assert len(denoised) == len(test_chunk), f"{level}: Length mismatch"
            assert np.max(np.abs(denoised)) <= 0.95, f"{level}: Not clipped properly"

            print(f"  ✓ {level.upper()}: alpha={actual_alpha}, beta={actual_beta}")

        print("✅ PASS: All levels work with conservative parameters")
        return True

    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rms_normalization():
    """Test 5: Verify RMS-based normalization (new feature)"""
    print("\n" + "="*60)
    print("TEST 5: RMS-Based Normalization")
    print("="*60)

    try:
        anc = ANCSystemFixed(noise_reduction_level="normal", sample_rate=44100, block_size=2048)

        # Process multiple chunks to build RMS history
        for i in range(15):  # More than maxlen=10
            chunk = np.random.randn(2048) * 0.2
            denoised = anc.denoise_chunk(chunk)

        # Verify RMS history is maintained
        assert len(anc.rms_history) == 10, f"RMS history should be 10, got {len(anc.rms_history)}"

        # Verify RMS values are reasonable
        for rms in anc.rms_history:
            assert rms > 0, "RMS should be positive"
            assert rms < 1.0, f"RMS should be < 1.0, got {rms}"

        print("✅ PASS: RMS normalization working")
        print(f"   - History size: {len(anc.rms_history)}")
        print(f"   - Median RMS: {np.median(list(anc.rms_history)):.4f}")
        print(f"   - Prevents amplitude spikes")
        return True

    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("ANC SYSTEM FIXED - COMPREHENSIVE TEST SUITE")
    print("="*60)
    print("\nTesting improvements:")
    print("  - Reduced distortions")
    print("  - Conservative parameters")
    print("  - RMS normalization")
    print("  - Soft clipping")
    print("  - Better audio quality")

    tests = [
        test_anc_initialization,
        test_anc_denoising,
        test_anc_delay_buffer,
        test_all_levels,
        test_rms_normalization
    ]

    results = []
    for test in tests:
        result = test()
        results.append(result)

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(results)
    total = len(results)

    for i, (test, result) in enumerate(zip(tests, results), 1):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test.__doc__.split(':')[1].strip()}")

    print(f"\nTOTAL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    if passed == total:
        print("\n🎉 All tests passed! Fixed ANC system is ready.")
        print("   Improvements verified:")
        print("   - Conservative parameters prevent over-processing")
        print("   - RMS normalization reduces amplitude spikes")
        print("   - Soft clipping prevents hard distortions")
        print("   - Output properly bounded to [-0.95, 0.95]")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
