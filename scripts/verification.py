# scripts/verification.py
import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model

def test_demucs_installation():
    """Test if Demucs is properly installed"""
    print("Testing Demucs installation...")
    
    try:
        # Test basic imports
        print("✅ torch imported successfully")
        print("✅ torchaudio imported successfully")
        print("✅ demucs.pretrained imported successfully")
        print("✅ demucs.apply imported successfully")
        
        # Test model loading
        print("\nTesting model loading...")
        model = get_model('htdemucs')  # Use 'htdemucs' instead of 'mdx_extra_q'
        print(f"✅ Successfully loaded model: {model.__class__.__name__}")
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Available sources: {model.sources}")
        
        # Test with dummy audio
        print("\nTesting with dummy audio...")
        dummy_audio = torch.randn(1, 2, 44100)  # 1 second of stereo audio
        
        with torch.no_grad():
            sources = apply_model(model, dummy_audio)
        
        print(f"✅ Model processing successful")
        print(f"Output shape: {sources.shape}")
        print(f"Number of sources: {len(model.sources)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_demucs_installation()
    if success:
        print("\n🎉 Demucs installation verified successfully!")
    else:
        print("\n❌ Demucs installation verification failed.")
