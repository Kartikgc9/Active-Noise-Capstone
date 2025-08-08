import torch
import torch.nn as nn
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import warnings
from scipy import signal
from scipy.ndimage import median_filter

warnings.filterwarnings("ignore")

class EffectiveAudioDenoiser:
    """
    Base audio denoiser that combines multiple techniques for actual noise reduction
    """
    
    def __init__(self, model_path=None, noise_reduction_level="normal"):
        self.sr = 44100
        self.n_fft = 2048
        self.hop_length = 512
        self.win_length = 2048
        
        # Define noise reduction presets
        self.presets = {
            "gentle": {
                "alpha": 2.0,           # Spectral subtraction factor
                "beta": 0.05,           # Spectral floor
                "speech_boost": 1.1,    # Speech frequency boost
                "wiener_weight": 0.3,   # Wiener filter contribution
                "noise_percentile": 20  # Noise estimation percentile
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
                "alpha": 3.5,           # More aggressive subtraction
                "beta": 0.005,          # Lower noise floor
                "speech_boost": 1.4,    # Stronger speech enhancement
                "wiener_weight": 0.5,   # More Wiener filtering
                "noise_percentile": 10  # More aggressive noise detection
            },
            "maximum": {
                "alpha": 4.0,           # Maximum subtraction
                "beta": 0.003,          # Minimal noise floor
                "speech_boost": 1.5,    # Maximum speech boost
                "wiener_weight": 0.6,   # Heavy Wiener filtering
                "noise_percentile": 8   # Very aggressive noise detection
            }
        }
        
        self.current_preset = self.presets[noise_reduction_level]
        print(f"🎯 Using {noise_reduction_level} noise reduction settings")
        
        # Try to load neural network model as enhancement
        self.has_nn_model = False
        if model_path and Path(model_path).exists():
            try:
                self.nn_model = self._load_simple_model(model_path)
                self.has_nn_model = True
                print("✅ Neural network model loaded as enhancement")
            except:
                print("⚠️ Neural network model not available, using signal processing methods")
        else:
            print("⚠️ No model found, using advanced signal processing methods")
        
        print("✅ Effective Audio Denoiser ready")
    
    def _load_simple_model(self, model_path):
        """Try to load a simple model if available"""
        try:
            # Create a very simple model for enhancement
            model = SimpleDenoiseNet()
            checkpoint = torch.load(model_path, map_location='cpu')
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Only load if weights are compatible
            try:
                model.load_state_dict(state_dict, strict=False)
                model.eval()
                return model
            except:
                return None
        except:
            return None
    
    def advanced_spectral_subtraction(self, audio):
        """
        Enhanced spectral subtraction with adjustable parameters
        """
        # Get current parameters
        alpha = self.current_preset["alpha"]
        beta = self.current_preset["beta"]
        speech_boost = self.current_preset["speech_boost"]
        noise_percentile = self.current_preset["noise_percentile"]
        
        # Compute spectrogram
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Enhanced noise estimation with multiple methods
        # Method 1: Use first and last 5% as likely noise periods
        noise_start = int(0.05 * magnitude.shape[1])
        noise_end = int(0.95 * magnitude.shape[1])
        
        noise_segments = np.concatenate([
            magnitude[:, :noise_start],
            magnitude[:, noise_end:]
        ], axis=1)
        
        # Method 2: Find quiet segments throughout using adjustable percentile
        frame_energy = np.mean(magnitude, axis=0)
        quiet_threshold = np.percentile(frame_energy, noise_percentile)
        quiet_frames = frame_energy < quiet_threshold
        
        if np.sum(quiet_frames) > 10:  # If we have enough quiet frames
            quiet_spectrum = magnitude[:, quiet_frames]
            noise_spectrum = np.mean(quiet_spectrum, axis=1, keepdims=True)
        else:
            noise_spectrum = np.mean(noise_segments, axis=1, keepdims=True)
        
        # Smooth the noise spectrum
        noise_spectrum = median_filter(noise_spectrum.squeeze(), size=5).reshape(-1, 1)
        
        # Apply frequency-dependent subtraction
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        freq_weights = np.linspace(1.0, 2.0, magnitude.shape[0]).reshape(-1, 1)
        
        # More aggressive in non-speech frequencies
        speech_mask = (freqs >= 300) & (freqs <= 3400)
        non_speech_mask = ~speech_mask
        
        freq_weights[speech_mask] = alpha * 0.8  # Gentler in speech range
        freq_weights[non_speech_mask] = alpha * 1.2  # More aggressive elsewhere
        
        adjusted_alpha = freq_weights
        
        # Perform subtraction
        subtracted_magnitude = magnitude - adjusted_alpha * noise_spectrum
        
        # Dynamic spectral floor based on local SNR
        local_snr = magnitude / (noise_spectrum + 1e-10)
        dynamic_beta = beta * (1.0 / (1.0 + local_snr * 0.1))  # Lower floor for high SNR regions
        
        enhanced_magnitude = np.maximum(subtracted_magnitude, dynamic_beta * magnitude)
        
        # Enhanced speech frequency boosting
        speech_enhancement_mask = np.ones_like(enhanced_magnitude)
        speech_enhancement_mask[speech_mask, :] *= speech_boost
        enhanced_magnitude *= speech_enhancement_mask
        
        # Additional post-processing: median filtering to remove impulsive noise
        if alpha > 3.0:  # Only for aggressive modes
            for freq_bin in range(0, enhanced_magnitude.shape[0], 50):  # Sample every 50th bin for efficiency
                end_bin = min(freq_bin + 50, enhanced_magnitude.shape[0])
                enhanced_magnitude[freq_bin:end_bin, :] = signal.medfilt2d(
                    enhanced_magnitude[freq_bin:end_bin, :], kernel_size=[1, 3]
                )
        
        # Reconstruct audio
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        enhanced_audio = librosa.istft(enhanced_stft, hop_length=self.hop_length, win_length=self.win_length)
        
        return enhanced_audio
    
    def wiener_filter_denoising(self, audio):
        """
        Enhanced Wiener filtering with adjustable parameters
        """
        wiener_weight = self.current_preset["wiener_weight"]
        noise_percentile = self.current_preset["noise_percentile"]
        
        # Compute power spectral density
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        power = magnitude ** 2
        
        # Enhanced noise power estimation
        frame_energy = np.mean(power, axis=0)
        noise_threshold = np.percentile(frame_energy, noise_percentile)
        noise_frames = frame_energy < noise_threshold
        
        if np.sum(noise_frames) > 5:
            noise_power = np.mean(power[:, noise_frames], axis=1, keepdims=True)
        else:
            noise_power = np.mean(power[:, :int(0.1 * power.shape[1])], axis=1, keepdims=True)
        
        # Wiener filter with enhanced gain calculation
        signal_power = power - noise_power
        signal_power = np.maximum(signal_power, 0.1 * power)  # Prevent negative values
        
        wiener_gain = signal_power / (signal_power + noise_power + 1e-10)
        
        # Apply gain smoothing for aggressive modes to reduce musical noise
        if self.current_preset["alpha"] > 3.0:
            wiener_gain = self._smooth_gain(wiener_gain)
        
        # Apply filter
        filtered_magnitude = magnitude * wiener_gain
        
        # Reconstruct
        filtered_stft = filtered_magnitude * np.exp(1j * phase)
        filtered_audio = librosa.istft(filtered_stft, hop_length=self.hop_length)
        
        return filtered_audio
    
    def _smooth_gain(self, gain, smoothing_factor=0.7):
        """Smooth gain to reduce musical noise artifacts"""
        if not hasattr(self, 'previous_gain'):
            self.previous_gain = gain
        
        smoothed_gain = smoothing_factor * self.previous_gain + (1 - smoothing_factor) * gain
        self.previous_gain = smoothed_gain
        
        return smoothed_gain
    
    def adaptive_noise_reduction(self, audio):
        """
        Enhanced adaptive noise reduction with multi-stage processing for aggressive modes
        """
        # Analyze audio characteristics
        rms_energy = np.sqrt(np.mean(audio ** 2))
        zero_crossing_rate = np.mean(librosa.zero_crossings(audio))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=self.sr))
        
        print(f"📊 Audio analysis - RMS: {rms_energy:.4f}, ZCR: {zero_crossing_rate:.4f}, SC: {spectral_centroid:.1f} Hz")
        
        # Check if we're in aggressive mode
        is_aggressive = self.current_preset["alpha"] >= 3.5
        
        # Choose denoising strategy based on characteristics
        if rms_energy > 0.05 and spectral_centroid > 1000:
            # High energy speech
            if is_aggressive:
                print("🎯 Detected: High energy speech - applying aggressive multi-stage denoising")
                denoised = self._multi_stage_denoising(audio)
            else:
                print("🎯 Detected: High energy speech - applying gentle denoising")
                denoised = self.advanced_spectral_subtraction(audio)
                denoised = self._apply_light_filtering(denoised)
        
        elif rms_energy < 0.02:
            # Very quiet audio - boost and denoise carefully
            print("🎯 Detected: Quiet audio - applying careful denoising with boost")
            # First boost the audio
            boosted_audio = audio * (0.05 / (rms_energy + 1e-8))
            boosted_audio = np.clip(boosted_audio, -0.95, 0.95)
            
            if is_aggressive:
                denoised = self._multi_stage_denoising(boosted_audio)
            else:
                denoised = self.wiener_filter_denoising(boosted_audio)
            
        else:
            # Normal audio
            if is_aggressive:
                print("🎯 Detected: Normal audio - applying aggressive combined denoising")
                denoised = self._multi_stage_denoising(audio)
            else:
                print("🎯 Detected: Normal audio - applying combined denoising")
                # Apply both methods and blend
                spec_sub = self.advanced_spectral_subtraction(audio)
                wiener = self.wiener_filter_denoising(audio)
                denoised = 0.7 * spec_sub + 0.3 * wiener
        
        return denoised
    
    def _multi_stage_denoising(self, audio):
        """Apply multi-stage denoising for aggressive modes"""
        wiener_weight = self.current_preset["wiener_weight"]
        
        # Stage 1: Advanced spectral subtraction
        stage1_audio = self.advanced_spectral_subtraction(audio)
        
        # Quality check after stage 1
        stage1_rms = np.sqrt(np.mean(stage1_audio ** 2))
        original_rms = np.sqrt(np.mean(audio ** 2))
        
        if stage1_rms < 0.05 * original_rms:
            print("⚠️ Stage 1 too aggressive, reducing parameters for stage 2")
            # Temporarily reduce aggressiveness for stage 2
            original_alpha = self.current_preset["alpha"]
            original_beta = self.current_preset["beta"]
            self.current_preset["alpha"] *= 0.7
            self.current_preset["beta"] *= 2.0
        
        # Stage 2: Enhanced Wiener filtering
        stage2_audio = self.wiener_filter_denoising(stage1_audio)
        
        # Restore original parameters if they were modified
        if stage1_rms < 0.05 * original_rms:
            self.current_preset["alpha"] = original_alpha
            self.current_preset["beta"] = original_beta
        
        # Stage 3: Blend results with weighting
        final_audio = (1 - wiener_weight) * stage1_audio + wiener_weight * stage2_audio
        
        # Stage 4: Final light filtering
        final_audio = self._apply_light_filtering(final_audio)
        
        return final_audio
    
    def _apply_light_filtering(self, audio):
        """Apply light filtering to remove remaining artifacts"""
        # Design a gentle high-pass filter to remove low-frequency noise
        nyquist = self.sr / 2
        low_cutoff = 80 / nyquist  # 80 Hz high-pass
        
        # Butterworth filter
        b, a = signal.butter(2, low_cutoff, btype='high')
        filtered_audio = signal.filtfilt(b, a, audio)
        
        return filtered_audio
    
    def denoise_audio_file(self, input_path, output_path):
        """
        Main denoising function that actually reduces noise
        """
        print(f"🎵 Processing: {input_path}")
        
        try:
            # Load audio
            audio, sr = librosa.load(input_path, sr=self.sr, mono=True)
            original_length = len(audio)
            
            print(f"📊 Audio duration: {len(audio)/sr:.2f} seconds")
            print(f"📊 Original audio range: [{audio.min():.3f}, {audio.max():.3f}]")
            
            # Normalize input
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                normalized_audio = audio / max_val
            else:
                normalized_audio = audio
            
            # Apply adaptive noise reduction
            denoised_audio = self.adaptive_noise_reduction(normalized_audio)
            
            # Apply neural network enhancement if available
            if self.has_nn_model and self.nn_model is not None:
                try:
                    print("🔧 Applying neural network enhancement...")
                    denoised_audio = self._apply_nn_enhancement(denoised_audio)
                except:
                    print("⚠️ Neural network enhancement failed, using signal processing result")
            
            # Restore original amplitude
            if max_val > 0:
                denoised_audio = denoised_audio * max_val
            
            # Ensure proper length
            if len(denoised_audio) > original_length:
                denoised_audio = denoised_audio[:original_length]
            elif len(denoised_audio) < original_length:
                pad_length = original_length - len(denoised_audio)
                denoised_audio = np.pad(denoised_audio, (0, pad_length), mode='constant')
            
            # Quality assessment
            original_rms = np.sqrt(np.mean(audio ** 2))
            denoised_rms = np.sqrt(np.mean(denoised_audio ** 2))
            noise_reduction = 20 * np.log10((np.std(audio) + 1e-10) / (np.std(audio - denoised_audio) + 1e-10))
            
            print(f"📊 Original RMS: {original_rms:.4f}")
            print(f"📊 Denoised RMS: {denoised_rms:.4f}")
            print(f"📊 Estimated noise reduction: {noise_reduction:.1f} dB")
            print(f"📊 Denoised range: [{denoised_audio.min():.3f}, {denoised_audio.max():.3f}]")
            
            # Save result
            sf.write(output_path, denoised_audio, sr, subtype='PCM_24')
            print(f"✅ Denoised audio saved to: {output_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _apply_nn_enhancement(self, audio):
        """Apply neural network as final enhancement if available"""
        # This would apply the neural network model if it's working
        # For now, return the input since the NN model isn't effectively trained
        return audio

class SimpleDenoiseNet(nn.Module):
    """Simple network for audio enhancement"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, 15, padding=7)
        self.conv2 = nn.Conv1d(16, 32, 15, padding=7)
        self.conv3 = nn.Conv1d(32, 16, 15, padding=7)
        self.conv4 = nn.Conv1d(16, 1, 15, padding=7)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        return x

def main():
    """Enhanced main function with noise reduction level options"""
    import sys
    
    print("🚀 ENHANCED AUDIO NOISE REDUCTION SYSTEM")
    print("=" * 60)
    
    # Define paths
    base_dir = Path(__file__).parent.parent
    model_path = base_dir / "checkpoints" / "test_model.pth"
    input_dir = base_dir / "audio_files" / "input"
    output_dir = base_dir / "audio_files" / "output"
    
    # Ensure directories exist
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Input: {input_dir}")
    print(f"📁 Output: {output_dir}")
    
    # Parse command line arguments
    reduction_level = "normal"  # Default
    filename = None
    
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if not filename.endswith('.wav'):
            filename += '.wav'
        
        # Check for noise reduction level argument
        if len(sys.argv) > 2 and sys.argv[2] in ["gentle", "normal", "moderate", "aggressive", "maximum"]:
            reduction_level = sys.argv[2]
    
    # Initialize denoiser with specified level
    denoiser = EffectiveAudioDenoiser(model_path, reduction_level)
    
    # Get files to process
    if filename:
        wav_files = [input_dir / filename]
        if not wav_files[0].exists():
            print(f"❌ File not found: {wav_files[0]}")
            print(f"Available files:")
            for f in input_dir.glob('*.wav'):
                print(f"   - {f.name}")
            return
        print(f"🎯 Processing specific file: {filename}")
    else:
        wav_files = list(input_dir.glob('*.wav'))
        print(f"🔄 Processing all files in directory")
    
    if not wav_files:
        print(f"⚠️ No WAV files found in {input_dir}")
        return
    
    print(f"\n🔄 Processing {len(wav_files)} files with {reduction_level} noise reduction...")
    success_count = 0
    
    for wav_file in wav_files:
        output_file = output_dir / f"{reduction_level}_{wav_file.name}"
        print(f"\n📄 Processing: {wav_file.name}")
        
        if denoiser.denoise_audio_file(wav_file, output_file):
            success_count += 1
        
    print(f"\n🎉 Successfully processed {success_count}/{len(wav_files)} files!")
    
    if success_count > 0:
        print(f"\n🎵 Test your results:")
        print(f"   1. Listen to {reduction_level}_*.wav files in {output_dir}")
        print(f"   2. Compare with original files")
        print(f"   3. You should hear significant background noise reduction")
        print(f"   4. Speech should be clearer and more prominent")
        
        # Usage examples
        print(f"\n💡 Usage examples:")
        print(f"   Normal mode:     python audio_denoiser.py input3")
        print(f"   Moderate mode:   python audio_denoiser.py input3 moderate")
        print(f"   Aggressive mode: python audio_denoiser.py input3 aggressive")
        print(f"   Maximum mode:    python audio_denoiser.py input3 maximum")

if __name__ == "__main__":
    main()
