"""
Visualization script for audio processing steps
Generates graphs and spectrograms to visualize the noise reduction process
"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
import soundfile as sf
from audio_denoiser import EffectiveAudioDenoiser

def plot_waveforms(original, denoised, sr, title, save_path):
    """Plot original and denoised waveforms"""
    plt.figure(figsize=(15, 6))
    
    # Original waveform
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(original, sr=sr)
    plt.title('Original Audio Waveform')
    plt.ylabel('Amplitude')
    
    # Denoised waveform
    plt.subplot(2, 1, 2)
    librosa.display.waveshow(denoised, sr=sr)
    plt.title('Denoised Audio Waveform')
    plt.ylabel('Amplitude')
    
    plt.tight_layout()
    plt.savefig(save_path / f'{title}_waveforms.png')
    plt.close()

def plot_spectrograms(original, denoised, sr, title, save_path):
    """Plot original and denoised spectrograms"""
    plt.figure(figsize=(15, 8))
    
    # Original spectrogram
    plt.subplot(2, 1, 1)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(original)), ref=np.max)
    librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Original Audio Spectrogram')
    
    # Denoised spectrogram
    plt.subplot(2, 1, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(denoised)), ref=np.max)
    librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Denoised Audio Spectrogram')
    
    plt.tight_layout()
    plt.savefig(save_path / f'{title}_spectrograms.png')
    plt.close()

def plot_frequency_response(original, denoised, sr, title, save_path):
    """Plot frequency response before and after denoising"""
    plt.figure(figsize=(15, 6))
    
    # Ensure same length for comparison
    min_len = min(len(original), len(denoised))
    original = original[:min_len]
    denoised = denoised[:min_len]
    
    # Calculate frequency responses
    freq_orig = np.abs(np.fft.rfft(original))
    freq_denoised = np.abs(np.fft.rfft(denoised))
    freqs = np.fft.rfftfreq(min_len, 1/sr)
    
    # Plot both on same axes
    plt.semilogy(freqs, freq_orig, label='Original', alpha=0.7)
    plt.semilogy(freqs, freq_denoised, label='Denoised', alpha=0.7)
    
    plt.title('Frequency Response Comparison')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path / f'{title}_frequency_response.png')
    plt.close()

def plot_noise_reduction_levels(audio_path, save_path):
    """Plot comparison of different noise reduction levels"""
    levels = ['gentle', 'normal', 'moderate', 'aggressive', 'maximum']
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=44100)
    
    plt.figure(figsize=(15, 10))
    
    # Original spectrogram
    plt.subplot(3, 2, 1)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr)
    plt.title('Original Audio')
    plt.colorbar(format='%+2.0f dB')
    
    # Process with each level
    for i, level in enumerate(levels, 2):
        denoiser = EffectiveAudioDenoiser(noise_reduction_level=level)
        denoised = denoiser.adaptive_noise_reduction(audio)
        
        plt.subplot(3, 2, i)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(denoised)), ref=np.max)
        librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr)
        plt.title(f'{level.capitalize()} Noise Reduction')
        plt.colorbar(format='%+2.0f dB')
    
    plt.tight_layout()
    plt.savefig(save_path / 'noise_reduction_levels.png')
    plt.close()

def plot_stages_visualization(audio_path, save_path):
    """Visualize the different stages of processing"""
    # Load audio
    audio, sr = librosa.load(audio_path, sr=44100)
    
    # Initialize denoiser
    denoiser = EffectiveAudioDenoiser(noise_reduction_level='moderate')
    
    # Get processed audio at different stages
    stage1 = denoiser.advanced_spectral_subtraction(audio)
    stage2 = denoiser.wiener_filter_denoising(stage1)
    final = denoiser._apply_light_filtering(stage2)
    
    plt.figure(figsize=(15, 12))
    
    # Original
    plt.subplot(4, 1, 1)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr)
    plt.title('Original Audio')
    plt.colorbar(format='%+2.0f dB')
    
    # After Spectral Subtraction
    plt.subplot(4, 1, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(stage1)), ref=np.max)
    librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr)
    plt.title('After Spectral Subtraction')
    plt.colorbar(format='%+2.0f dB')
    
    # After Wiener Filtering
    plt.subplot(4, 1, 3)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(stage2)), ref=np.max)
    librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr)
    plt.title('After Wiener Filtering')
    plt.colorbar(format='%+2.0f dB')
    
    # Final Result
    plt.subplot(4, 1, 4)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(final)), ref=np.max)
    librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr)
    plt.title('Final Result')
    plt.colorbar(format='%+2.0f dB')
    
    plt.tight_layout()
    plt.savefig(save_path / 'processing_stages.png')
    plt.close()

def main():
    """Generate all visualizations"""
    # Setup paths
    base_dir = Path(__file__).parent.parent
    input_dir = base_dir / "audio_files" / "input"
    output_dir = base_dir / "audio_files" / "output"
    vis_dir = base_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    
    print("🎨 Generating visualizations...")
    
    # Process each input file
    for audio_file in input_dir.glob('*.wav'):
        print(f"📊 Processing {audio_file.name}")
        
        # Load original audio
        audio, sr = librosa.load(audio_file, sr=44100)
        
        # Process with normal settings
        denoiser = EffectiveAudioDenoiser(noise_reduction_level='normal')
        denoised = denoiser.adaptive_noise_reduction(audio)
        
        # Generate basic visualizations
        plot_waveforms(audio, denoised, sr, audio_file.stem, vis_dir)
        plot_spectrograms(audio, denoised, sr, audio_file.stem, vis_dir)
        plot_frequency_response(audio, denoised, sr, audio_file.stem, vis_dir)
        
        # Generate comparison of noise reduction levels
        plot_noise_reduction_levels(audio_file, vis_dir)
        
        # Generate processing stages visualization
        plot_stages_visualization(audio_file, vis_dir)
        
        print(f"✅ Generated visualizations for {audio_file.name}")
    
    print(f"\n🎉 All visualizations saved to {vis_dir}")
    print("Generated visualizations include:")
    print("1. Waveform comparisons")
    print("2. Spectrogram comparisons")
    print("3. Frequency response analysis")
    print("4. Noise reduction level comparisons")
    print("5. Processing stages visualization")

if __name__ == "__main__":
    main()
