"""
Audio Recording Tool for Data Collection
Helps community members record Ethiopian speech data
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
from datetime import datetime
import os
import json


class AudioRecorder:
    """Simple audio recorder for collecting Ethiopian speech data"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        output_dir: str = "data/recordings"
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        print("🎙️ Audio Recorder Initialized")
        print(f"Sample Rate: {sample_rate} Hz")
        print(f"Channels: {channels}")
        print(f"Output Directory: {output_dir}")
    
    def record(self, max_duration: int = 30, prompt: str = None) -> str:
        """
        Record audio with manual stop
        
        Args:
            max_duration: Maximum recording duration in seconds
            prompt: Optional text prompt for speaker to read
            
        Returns:
            Path to saved audio file
        """
        if prompt:
            print(f"\n📝 Please read: {prompt}")
        
        print(f"🔴 Recording... Press ENTER to stop.")
        
        try:
            # Use a list to store audio chunks
            audio_data = []
            
            def callback(indata, frames, time, status):
                if status:
                    print(status)
                audio_data.append(indata.copy())

            # Start recording in a stream
            with sd.InputStream(samplerate=self.sample_rate, channels=self.channels, callback=callback):
                input() # Wait for the user to press Enter

            print("✅ Recording complete!")
            
            # Combine chunks
            audio = np.concatenate(audio_data, axis=0)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.wav"
            filepath = os.path.join(self.output_dir, filename)
            
            # Save audio
            sf.write(filepath, audio, self.sample_rate)
            print(f"💾 Saved to: {filepath}")
            
            # Save metadata
            metadata = {
                "filename": filename,
                "duration": len(audio) / self.sample_rate,
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "prompt": prompt,
                "timestamp": timestamp
            }
            
            metadata_path = filepath.replace(".wav", ".json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            return filepath
            
        except Exception as e:
            print(f"❌ Error during recording: {e}")
            return None
    
    def interactive_session(self, prompts: list = None):
        """
        Interactive recording session
        
        Args:
            prompts: List of text prompts for speaker to read
        """
        print("\n" + "=" * 60)
        print("🎤 Interactive Recording Session")
        print("=" * 60)
        
        if prompts is None:
            # Default Amharic prompts
            prompts = [
                "ሰላም። እንዴት ነህ?",
                "ዛሬ የአየር ሁኔታው ጥሩ ነው።",
                "አዲስ አበባ የኢትዮጵያ ዋና ከተማ ነች።",
                "የትምህርት ቤቶች በሴፕቴምበር ይከፈታሉ።",
                "ቡና የኢትዮጵያ የተለመደ መጠጥ ነው።"
            ]
        
        print(f"\n📋 You will record {len(prompts)} prompts")
        print("Press Enter to start each recording...\n")
        
        for i, prompt in enumerate(prompts, 1):
            input(f"[{i}/{len(prompts)}] Press Enter to START recording...")
            self.record(prompt=prompt)
            print()
        
        print("🎉 Recording session complete!")
        print(f"✅ {len(prompts)} recordings saved to {self.output_dir}")


def main():
    """Main entry point for recording tool"""
    print("\n" + "=" * 60)
    print("🇪🇹 AfanParadox - Audio Recording Tool")
    print("=" * 60)
    
    recorder = AudioRecorder()
    
    # Interactive session
    print("\nChoose mode:")
    print("1. Interactive session (multiple prompts)")
    print("2. Single recording")
    print("3. Exit")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        recorder.interactive_session()
    elif choice == "2":
        prompt = input("Prompt (optional): ").strip() or None
        recorder.record(prompt=prompt)
    elif choice == "3":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Recording cancelled. Goodbye!")
