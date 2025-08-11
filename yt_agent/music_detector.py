import os
import sys
import tempfile
import shutil
import numpy as np
import urllib.request
import librosa
from transformers import pipeline
import yt_dlp


class YouTubeMusicDetector:
    def __init__(self):
        print("Loading audio classification model...")
        try:
            # Use Hugging Face transformers instead of TensorFlow Hub
            self.classifier = pipeline(
                "audio-classification",
                model="MIT/ast-finetuned-audioset-10-10-0.4593",
                device=-1  # Use CPU
            )
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Model loading error: {e}")
            print("Trying alternative approach...")
            
            # Fallback to simple frequency analysis
            self.classifier = None
            print("✅ Using frequency analysis fallback")
    
    def download_audio(self, url):
        """Download audio from YouTube."""
        temp_dir = tempfile.mkdtemp()
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(temp_dir, 'audio.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }]
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                title = info.get('title', 'Unknown')
                print(f"📹 {title}")
                
                ydl.download([url])
            
            # Find the downloaded file
            for file in os.listdir(temp_dir):
                if file.endswith(('.wav', '.mp3', '.m4a', '.webm')):
                    return os.path.join(temp_dir, file), temp_dir
            
            raise Exception("No audio file found")
            
        except Exception as e:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise Exception(f"Download failed: {e}")
    
    def analyze_with_transformers(self, audio_file):
        """Analyze using Hugging Face transformers."""
        try:
            # Load audio with librosa
            audio, sr = librosa.load(audio_file, sr=16000, duration=30)  # Limit to 30 seconds
            
            # Get predictions
            results = self.classifier(audio)
            
            # Look for music-related classes
            music_keywords = ['music', 'song', 'singing', 'piano', 'guitar', 'drum', 'instrument']
            music_score = 0.0
            
            for result in results:
                label = result['label'].lower()
                score = result['score']
                
                if any(keyword in label for keyword in music_keywords):
                    music_score = max(music_score, score)
            
            return music_score > 0.3, music_score
            
        except Exception as e:
            print(f"Transformers analysis error: {e}")
            return self.analyze_with_frequency(audio_file)
    
    def analyze_with_frequency(self, audio_file):
        """Fallback: Simple frequency-based music detection."""
        try:
            # Load audio
            audio, sr = librosa.load(audio_file, sr=22050, duration=30)
            
            # Extract features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            
            # Simple heuristics for music detection
            # Music typically has:
            # - Consistent tempo
            # - Rich harmonic content
            # - Structured patterns
            
            spectral_mean = np.mean(spectral_centroids)
            tempo_score = 1.0 if 60 <= tempo <= 200 else 0.5
            harmonic_score = min(1.0, spectral_mean / 3000)
            
            # Combine scores
            music_score = (tempo_score + harmonic_score) / 2
            
            return music_score > 0.4, music_score
            
        except Exception as e:
            print(f"Frequency analysis error: {e}")
            return False, 0.0
    
    def detect_music(self, url):
        """Main detection function."""
        temp_dir = None
        
        try:
            print("📥 Downloading audio...")
            audio_file, temp_dir = self.download_audio(url)
            
            print("🔍 Analyzing for music...")
            
            if self.classifier:
                has_music, confidence = self.analyze_with_transformers(audio_file)
            else:
                has_music, confidence = self.analyze_with_frequency(audio_file)
            
            return has_music, confidence
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return False, 0.0
            
        finally:
            # Always clean up
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)

# def main():
#     print("🎵 Simple YouTube Music Detector")
#     print("=" * 40)
#     print("(No TensorFlow required!)")
    
#     # Initialize detector
#     try:
#         detector = YouTubeMusicDetector()
#     except Exception as e:
#         print(f"❌ Setup failed: {e}")
#         return
    
#     while True:
#         print()
#         url = input("Enter YouTube URL (or 'quit'): ").strip()
        
#         if url.lower() in ['quit', 'exit', 'q']:
#             print("👋 Goodbye!")
#             break
        
#         if not url or ('youtube.com' not in url and 'youtu.be' not in url):
#             print("❌ Please enter a valid YouTube URL")
#             continue
        
#         print(f"🔗 Processing: {url}")
#         has_music, score = detector.detect_music(url)
        
#         print()
#         if has_music:
#             print(f"✅ MUSIC DETECTED! 🎵")
#             print(f"   Confidence: {score:.2f}")
#         else:
#             print(f"❌ No music detected")
#             print(f"   Score: {score:.2f}")
        
#         print("-" * 40)

# if __name__ == "__main__":
#     main()
    