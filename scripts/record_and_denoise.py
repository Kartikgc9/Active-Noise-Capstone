import pyaudio
import wave
import time
import subprocess
from pathlib import Path
import numpy as np

def record_audio(duration=30, sample_rate=44100):
    """Record audio from the microphone for a specified duration"""
    print(" Initializing audio recording...")
    
    # Audio recording parameters
    chunk = 1024
    audio_format = pyaudio.paFloat32
    channels = 1
    
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    
    print("🔴 Recording will start in 3 seconds...")
    time.sleep(3)
    
    # Open audio stream
    stream = p.open(format=audio_format,
                   channels=channels,
                   rate=sample_rate,
                   input=True,
                   frames_per_buffer=chunk)
    
    print("🎵 Recording...")
    
    frames = []
    
    # Record audio in chunks
    for i in range(0, int(sample_rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(np.frombuffer(data, dtype=np.float32))
        
        # Print progress every 5 seconds
        if i % (5 * sample_rate // chunk) == 0:
            seconds = i * chunk / sample_rate
            print(f" Recorded {seconds:.1f} seconds...")
    
    print(" Recording finished!")
    
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Combine all frames
    return np.concatenate(frames)

def save_wav(audio_data, filename, sample_rate=44100):
    """Save audio data as WAV file"""
    # Convert to 16-bit PCM
    audio_data = (audio_data * 32767).astype(np.int16)
    
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 2 bytes for 16-bit audio
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())

def main():
    print(" AUTOMATED RECORDING AND DENOISING SYSTEM")
    print("=" * 60)
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    input_dir = base_dir / "audio_files" / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename based on timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    input_file = input_dir / f"recording_{timestamp}.wav"
    
    try:
        # Record audio
        print(f"\n Recording will be saved to: {input_file}")
        audio_data = record_audio(duration=30)  # 30 seconds recording
        
        # Save the recorded audio
        print(f"\n Saving recorded audio...")
        save_wav(audio_data, str(input_file))
        print(f" Audio saved successfully!")
        
        # Run the denoiser script
        print(f"\n Starting denoising process...")
        denoiser_script = base_dir / "scripts" / "audio_denoiser.py"
        subprocess.run(["python", str(denoiser_script), f"recording_{timestamp}.wav", "normal"])
        
        print(f"\n Process completed!")
        print(f" Check the output folder for your denoised audio.")
        
    except Exception as e:
        print(f"\n Error: {str(e)}")
        return

if __name__ == "__main__":
    main()
