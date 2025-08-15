import torch
import numpy as np
from scipy.signal import stft
from pesq import pesq
from pystoi import stoi
import librosa
from pathlib import Path
from typing import Dict, Tuple

class AudioMetrics:
    """Class for computing audio quality metrics"""
    
    def __init__(self, sr: int = 44100):
        """
        Initialize metrics calculator
        
        Args:
            sr: Sample rate of audio
        """
        self.sr = sr
        self.resample_16k = lambda x: librosa.resample(x, orig_sr=sr, target_sr=16000)
    
    def snr(self, 
            clean: np.ndarray, 
            denoised: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio"""
        noise = clean - denoised
        return 10 * np.log10(
            np.sum(clean ** 2) / (np.sum(noise ** 2) + 1e-10)
        )
    
    def pesq_score(self, 
                  clean: np.ndarray, 
                  denoised: np.ndarray) -> float:
        """Calculate PESQ score"""
        # PESQ requires 16kHz sample rate
        clean_16k = self.resample_16k(clean)
        denoised_16k = self.resample_16k(denoised)
        
        try:
            return pesq(16000, clean_16k, denoised_16k, 'wb')
        except:
            return 0.0
    
    def stoi_score(self, 
                   clean: np.ndarray, 
                   denoised: np.ndarray) -> float:
        """Calculate STOI score"""
        clean_16k = self.resample_16k(clean)
        denoised_16k = self.resample_16k(denoised)
        
        try:
            return stoi(clean_16k, denoised_16k, 16000, extended=False)
        except:
            return 0.0
    
    def segmental_snr(self,
                     clean: np.ndarray,
                     denoised: np.ndarray,
                     window_size: int = 2048) -> float:
        """Calculate Segmental SNR"""
        n_segments = len(clean) // window_size
        snr_values = []
        
        for i in range(n_segments):
            start = i * window_size
            end = start + window_size
            
            clean_seg = clean[start:end]
            denoised_seg = denoised[start:end]
            
            snr_values.append(self.snr(clean_seg, denoised_seg))
        
        return np.mean(snr_values)
    
    def frequency_rmse(self,
                      clean: np.ndarray,
                      denoised: np.ndarray,
                      n_fft: int = 2048) -> float:
        """Calculate RMSE in frequency domain"""
        clean_stft = np.abs(librosa.stft(clean, n_fft=n_fft))
        denoised_stft = np.abs(librosa.stft(denoised, n_fft=n_fft))
        
        return np.sqrt(np.mean((clean_stft - denoised_stft) ** 2))
    
    def compute_all_metrics(self,
                          clean: np.ndarray,
                          denoised: np.ndarray) -> Dict[str, float]:
        """Compute all available metrics"""
        return {
            'snr': self.snr(clean, denoised),
            'segmental_snr': self.segmental_snr(clean, denoised),
            'pesq': self.pesq_score(clean, denoised),
            'stoi': self.stoi_score(clean, denoised),
            'freq_rmse': self.frequency_rmse(clean, denoised)
        }

def validate_model_performance(model_path: Path,
                             test_files: Dict[str, Tuple[Path, Path]],
                             device: torch.device) -> Dict[str, Dict[str, float]]:
    """
    Validate model performance on test files
    
    Args:
        model_path: Path to model checkpoint
        test_files: Dictionary mapping test names to (clean_path, noisy_path) tuples
        device: Torch device to use
        
    Returns:
        Dictionary of metrics for each test file
    """
    # Load model
    model = torch.load(model_path, map_location=device)
    model.eval()
    
    # Initialize metrics calculator
    metrics_calc = AudioMetrics()
    results = {}
    
    for test_name, (clean_path, noisy_path) in test_files.items():
        # Load audio files
        clean, _ = librosa.load(clean_path, sr=44100)
        noisy, _ = librosa.load(noisy_path, sr=44100)
        
        # Process through model
        with torch.no_grad():
            noisy_tensor = torch.FloatTensor(noisy).unsqueeze(0).unsqueeze(0).to(device)
            denoised_tensor = model(noisy_tensor)
            denoised = denoised_tensor.squeeze().cpu().numpy()
        
        # Calculate metrics
        results[test_name] = metrics_calc.compute_all_metrics(clean, denoised)
    
    return results

def print_validation_results(results: Dict[str, Dict[str, float]]):
    """Pretty print validation results"""
    print("\n📊 Validation Results:")
    print("=" * 60)
    
    # Calculate averages
    metric_sums = {}
    for test_results in results.values():
        for metric, value in test_results.items():
            if metric not in metric_sums:
                metric_sums[metric] = []
            metric_sums[metric].append(value)
    
    # Print individual results
    for test_name, metrics in results.items():
        print(f"\n🔍 Test file: {test_name}")
        for metric, value in metrics.items():
            print(f"  {metric:15s}: {value:.4f}")
    
    # Print averages
    print("\n📈 Average Results:")
    print("-" * 60)
    for metric, values in metric_sums.items():
        avg = np.mean(values)
        std = np.std(values)
        print(f"{metric:15s}: {avg:.4f} ± {std:.4f}")
    
    print("=" * 60)
