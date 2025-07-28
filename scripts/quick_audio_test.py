"""
Quick Audio Denoising Test
=========================

Simple test script to verify the audio denoising system works.
"""

import numpy as np
import soundfile as sf
from pathlib import Path

def create_test_audio():
    """Create a simple test audio file"""
    
    print("🎵 Creating test audio...")
    
    # Create test directory
    test_dir = Path("test_audio")
    test_dir.mkdir(exist_ok=True)
    
    # Parameters
    duration = 3.0  # seconds
    sr = 44100
    t = np.linspace(0, duration, int(duration * sr))
    
    # Create clean signal (sine wave)
    clean_signal = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    
    # Add noise
    noise = 0.2 * np.random.normal(0, 1, len(t))
    noisy_signal = clean_signal + noise
    
    # Normalize
    noisy_signal = noisy_signal / np.max(np.abs(noisy_signal)) * 0.8
    
    # Save test file
    test_file = test_dir / "test_noisy_audio.wav"
    sf.write(test_file, noisy_signal, sr)
    
    print(f"✅ Test audio created: {test_file}")
    return test_file

def test_denoising():
    """Test the denoising system"""
    
    print("🧪 Testing Audio Denoising System")
    print("=" * 40)
    
    # Create test audio
    test_file = create_test_audio()
    
    # Test the denoiser
    try:
        from audio_denoiser import AudioDenoiser
        
        # Initialize denoiser (without trained model)
        denoiser = AudioDenoiser()
        
        # Process test file
        output_file = test_file.parent / "denoised_test_audio.wav"
        
        print(f"\n🔄 Processing test file...")
        success = denoiser.denoise_audio_file(test_file, output_file)
        
        if success:
            print(f"🎉 Test completed successfully!")
            print(f"📁 Check these files:")
            print(f"   Original (noisy): {test_file}")
            print(f"   Processed: {output_file}")
            print(f"💡 Listen to both files to compare")
        else:
            print(f"❌ Test failed!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_denoising()
