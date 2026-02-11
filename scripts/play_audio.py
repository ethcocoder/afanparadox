"""
Playback Tool for AfanParadox
Plays back the most recent recording to verify audio quality
"""

import sounddevice as sd
import soundfile as sf
import os
import glob

def play_last_recording():
    output_dir = "data/recordings"
    
    # Get all .wav files in the output directory
    files = glob.glob(os.path.join(output_dir, "*.wav"))
    
    if not files:
        print("❌ No recordings found in data/recordings")
        return

    # Find the most recently created file
    latest_file = max(files, key=os.path.getctime)
    
    print(f"🔊 Playing back: {latest_file}")
    
    try:
        # Read the audio file
        data, fs = sf.read(latest_file)
        
        # Play it
        sd.play(data, fs)
        sd.wait()  # Wait until playback is finished
        print("✅ Playback finished!")
        
    except Exception as e:
        print(f"❌ Error during playback: {e}")

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🇪🇹 AfanParadox - Audio Playback Test")
    print("=" * 60)
    play_last_recording()
