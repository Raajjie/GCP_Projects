import cv2
import numpy as np
import speech_recognition as sr
import moviepy.editor as mp
from moviepy.video.io.VideoFileClip import VideoFileClip
from collections import Counter
import json
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import yt_dlp
import tempfile
import shutil
import re
from pydub import AudioSegment



class YouTubeShortAnalyzer:
    def __init__(self, youtube_url: str):
        """
        Initialize the analyzer with a YouTube URL
        
        Args:
            youtube_url (str): YouTube video URL (including Shorts)
        """
        self.youtube_url = youtube_url
        self.temp_dir = tempfile.mkdtemp()
        self.video_path = None
        self.video = None
        self.fps = None
        self.frame_count = None
        self.duration = None
        
        # Analysis results
        self.results = {
            'video_info': {},
            'visual_analysis': {},
            'audio_analysis': {},
            'final_classification': '',
            'confidence_score': 0.0,
            'reasoning': []
        }
        
        # Download and initialize video
        self._download_video()
    
    def _is_youtube_url(self, url: str) -> bool:
        """Check if the URL is a valid YouTube URL"""
        youtube_patterns = [
            r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=[\w-]+',
            r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/shorts\/[\w-]+',
            r'(?:https?:\/\/)?youtu\.be\/[\w-]+',
            r'(?:https?:\/\/)?(?:m\.)?youtube\.com\/watch\?v=[\w-]+'
        ]
        return any(re.match(pattern, url) for pattern in youtube_patterns)
    
    def _download_video(self):
        """Download video from YouTube URL"""
        if not self._is_youtube_url(self.youtube_url):
            raise ValueError("Invalid YouTube URL provided")
        
        print(f"Downloading video from: {self.youtube_url}")
        
        # Configure yt-dlp options
        ydl_opts = {
            'format': 'mp4[height<=720]/best[height<=720]/best',  # Prefer smaller files for analysis
            'outtmpl': os.path.join(self.temp_dir, 'video_%(id)s.%(ext)s'),  # Use video ID instead of title
            'quiet': True,
            'no_warnings': True,
            'encoding': 'utf-8',
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract video info first
                info = ydl.extract_info(self.youtube_url, download=False)
                
                # Store video metadata (clean Unicode characters)
                def clean_text(text):
                    if text is None:
                        return 'Unknown'
                    # Remove or replace problematic Unicode characters
                    return text.encode('ascii', 'ignore').decode('ascii') or 'Unknown'
                
                self.results['video_info'] = {
                    'title': clean_text(info.get('title')),
                    'duration': info.get('duration', 0),
                    'view_count': info.get('view_count', 0),
                    'uploader': clean_text(info.get('uploader')),
                    'upload_date': info.get('upload_date', 'Unknown'),
                    'description': clean_text(info.get('description', ''))[:200] + '...' if info.get('description') else '',
                    'video_id': info.get('id', 'unknown')
                }
                
                # Download the video
                ydl.download([self.youtube_url])
                
                # Find the downloaded file
                for file in os.listdir(self.temp_dir):
                    if file.endswith(('.mp4', '.webm', '.mkv')):
                        self.video_path = os.path.join(self.temp_dir, file)
                        break
                
                if not self.video_path:
                    raise Exception("Downloaded video file not found")
                
                # Initialize video capture
                self.video = cv2.VideoCapture(self.video_path)
                self.fps = self.video.get(cv2.CAP_PROP_FPS)
                self.frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
                self.duration = self.frame_count / self.fps if self.fps > 0 else info.get('duration', 0)
                
                print(f"Video downloaded successfully: {self.results['video_info']['title']}")
                print(f"Duration: {self.duration:.2f} seconds")
                
        except Exception as e:
            self._cleanup()
            raise Exception(f"Failed to download video: {str(e)}")
    
    def _cleanup(self):
        """Clean up temporary files"""
        if self.video:
            self.video.release()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def analyze_visual_content(self) -> Dict:
        """
        Analyze visual content to determine if it's footage-dependent
        """
        print("Analyzing visual content...")
        
        if not self.video or not self.video.isOpened():
            return {'error': 'Video not available for analysis'}
        
        # Sample frames at regular intervals
        sample_interval = max(1, self.frame_count // 30)  # Sample ~30 frames
        frames = []
        scene_changes = 0
        motion_scores = []
        
        prev_frame = None
        for i in range(0, self.frame_count, sample_interval):
            self.video.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = self.video.read()
            if not ret:
                break
            
            frames.append(frame)
            
            # Calculate motion between frames
            if prev_frame is not None:
                # Convert to grayscale for motion detection
                gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Calculate frame difference
                diff = cv2.absdiff(gray1, gray2)
                motion_score = np.mean(diff)
                motion_scores.append(motion_score)
                
                # Detect scene changes using histogram comparison
                hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
                hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
                correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                
                if correlation < 0.7:  # Threshold for scene change
                    scene_changes += 1
            
            prev_frame = frame
        
        # Analyze visual complexity and variety
        avg_motion = np.mean(motion_scores) if motion_scores else 0
        scene_change_rate = scene_changes / len(frames) if frames else 0
        
        # Detect if video is primarily static (like talking head or simple graphics)
        is_static = bool(avg_motion < 20 and scene_change_rate < 0.1)
        
        # Detect if video has dynamic footage
        is_dynamic = bool(avg_motion > 40 or scene_change_rate > 0.3)
        
        visual_analysis = {
            'average_motion': float(avg_motion),
            'scene_change_rate': float(scene_change_rate),
            'total_scene_changes': int(scene_changes),
            'is_static': is_static,
            'is_dynamic': is_dynamic,
            'visual_complexity': 'high' if is_dynamic else 'medium' if not is_static else 'low',
            'frames_analyzed': int(len(frames))
        }
        
        return visual_analysis
    
    def extract_and_analyze_audio(self) -> Dict:
        """
        Extract audio and analyze speech content with multiple diverse fallback strategies.
        """
        print("Analyzing audio content...")
        
        video_clip = None
        audio_clip = None
        temp_files = []
        
        try:
            from pydub.effects import normalize, strip_silence
            
            # Load video and audio
            video_clip = VideoFileClip(self.video_path)
            audio_clip = video_clip.audio

            if audio_clip is None:
                return {
                    'transcript': "",
                    'word_count': 0,
                    'chars_per_second': 0,
                    'words_per_minute': 0,
                    'has_narration': False,
                    'is_speech_heavy': False,
                    'duration': self.duration,
                    'successful_method': 'none',
                    'confidence_score': 0,
                    'error': 'No audio track found'
                }

            # Export to WAV with proper settings
            temp_audio_path = os.path.join(self.temp_dir, "temp_audio.wav")
            temp_files.append(temp_audio_path)
            
            # Export with consistent settings for better speech recognition
            audio_clip.write_audiofile(
                temp_audio_path, 
                verbose=False, 
                logger=None,
                codec='pcm_s16le',  # Ensure consistent format
                ffmpeg_params=['-ar', '16000']  # 16kHz sample rate for speech recognition
            )

            # Initialize recognizer with better settings
            recognizer = sr.Recognizer()
            recognizer.energy_threshold = 300
            recognizer.dynamic_energy_threshold = True
            recognizer.pause_threshold = 0.8

            def transcribe(path: str) -> str:
                """Transcribe audio file with proper error handling."""
                try:
                    with sr.AudioFile(path) as source:
                        # Adjust for ambient noise
                        recognizer.adjust_for_ambient_noise(source, duration=1.0)
                        audio_data = recognizer.record(source)
                        
                        # Try Google first, then fallback to other engines
                        try:
                            return recognizer.recognize_google(audio_data, language='en-US')
                        except sr.RequestError:
                            # Fallback to offline recognition if available
                            try:
                                return recognizer.recognize_sphinx(audio_data)
                            except (sr.RequestError, sr.UnknownValueError):
                                return ""
                except Exception as e:
                    print(f"Transcription error: {e}")
                    return ""

            def transform_audio(original_path: str, transforms: list) -> str:
                """Apply a sequence of audio transformations using pydub."""
                try:
                    from pydub import AudioSegment
                    sound = AudioSegment.from_wav(original_path)
                    
                    for transform in transforms:
                        if transform == "mono":
                            sound = sound.set_channels(1)
                        elif transform == "boost":
                            sound = sound + 10  # Increase volume by 10dB
                        elif transform == "trim":
                            sound = strip_silence(
                                sound, 
                                silence_len=300, 
                                silence_thresh=sound.dBFS - 14,
                                padding=100
                            )
                        elif transform == "normalize":
                            sound = normalize(sound)
                        elif transform == "slow":
                            # More reliable way to slow down audio
                            sound = sound._spawn(
                                sound.raw_data, 
                                overrides={"frame_rate": int(sound.frame_rate * 0.85)}
                            ).set_frame_rate(sound.frame_rate)
                        elif transform == "high_pass":
                            # Simple high-pass filter effect
                            sound = sound.high_pass_filter(80)
                        elif transform == "compress":
                            # Simple compression by reducing dynamic range
                            sound = sound.compress_dynamic_range()
                    
                    output_path = os.path.join(
                        self.temp_dir, 
                        f"temp_audio_{'_'.join(transforms)}.wav"
                    )
                    temp_files.append(output_path)
                    sound.export(output_path, format="wav")
                    return output_path
                    
                except Exception as e:
                    print(f"Audio transformation failed: {e}")
                    return original_path

            # Enhanced fallback strategies
            attempts = [
                {"label": "original", "transforms": []},
                {"label": "mono", "transforms": ["mono"]},
                {"label": "normalize", "transforms": ["normalize"]},
                {"label": "boost", "transforms": ["boost"]},
                {"label": "mono_normalize", "transforms": ["mono", "normalize"]},
                {"label": "mono_boost", "transforms": ["mono", "boost"]},
                {"label": "trim_silence", "transforms": ["trim"]},
                {"label": "mono_trim_boost", "transforms": ["mono", "trim", "boost"]},
                {"label": "high_pass_filter", "transforms": ["high_pass"]},
                {"label": "slow_down", "transforms": ["mono", "slow"]},
                {"label": "comprehensive", "transforms": ["mono", "normalize", "trim", "boost"]},
            ]

            transcript = ""
            successful_method = "none"
            
            for attempt in attempts:
                try:
                    print(f"Trying transcription method: {attempt['label']}")
                    
                    if attempt["transforms"]:
                        processed_path = transform_audio(temp_audio_path, attempt["transforms"])
                    else:
                        processed_path = temp_audio_path
                    
                    transcript = transcribe(processed_path)
                    
                    if transcript and len(transcript.strip()) > 0:
                        successful_method = attempt['label']
                        print(f"Successfully transcribed using: {successful_method}")
                        break
                        
                except Exception as e:
                    print(f"Method '{attempt['label']}' failed: {e}")
                    continue

            # Enhanced analysis
            words = transcript.split() if transcript else []
            word_count = len(words)
            
            # Better metrics calculation
            chars_per_second = len(transcript) / self.duration if self.duration > 0 else 0
            words_per_minute = (word_count / self.duration) * 60 if self.duration > 0 else 0
            
            # More nuanced classification
            has_narration = (
                word_count >= 15 and 
                chars_per_second >= 8 and 
                words_per_minute >= 30
            )
            
            is_speech_heavy = (
                word_count >= 40 and 
                chars_per_second >= 15 and
                words_per_minute >= 60
            )

            return {
                'transcript': transcript.strip(),
                'word_count': word_count,
                'chars_per_second': round(chars_per_second, 2),
                'words_per_minute': round(words_per_minute, 2),
                'has_narration': has_narration,
                'is_speech_heavy': is_speech_heavy,
                'duration': self.duration,
                'successful_method': successful_method,
                'confidence_score': min(1.0, word_count / 50) if word_count > 0 else 0
            }

        except Exception as e:
            print(f"Audio analysis failed: {str(e)}")
            return {
                'transcript': "",
                'word_count': 0,
                'chars_per_second': 0,
                'words_per_minute': 0,
                'has_narration': False,
                'is_speech_heavy': False,
                'duration': self.duration,
                'successful_method': 'none',
                'confidence_score': 0,
                'error': str(e)
            }
            
        finally:
            # Proper cleanup
            if audio_clip:
                audio_clip.close()
            if video_clip:
                video_clip.close()
                
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except Exception as e:
                    print(f"Failed to remove temp file {temp_file}: {e}")
    
    def analyze_text_content(self) -> Dict:
        """
        Analyze on-screen text using OCR
        """
        print("Analyzing on-screen text...")
        
        if not self.video or not self.video.isOpened():
            return {'error': 'Video not available for text analysis'}
        
        # Sample frames for text detection
        sample_interval = max(1, self.frame_count // 15)  # Sample ~15 frames
        detected_texts = []
        
        for i in range(0, self.frame_count, sample_interval):
            self.video.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = self.video.read()
            if not ret:
                break
            
            try:
                # Use OCR to detect text
                text = pytesseract.image_to_string(frame, config='--psm 6')
                cleaned_text = ' '.join(text.split())  # Clean up whitespace
                if cleaned_text and len(cleaned_text) > 3:  # Minimum text length
                    detected_texts.append(cleaned_text)
            except Exception as e:
                continue
        
        # Analyze text content
        all_text = " ".join(detected_texts)
        text_density = len(all_text) / self.duration if self.duration > 0 else 0
        has_significant_text = bool(text_density > 20 or len(all_text) > 100)
        
        text_analysis = {
            'detected_texts': detected_texts[:10],  # Limit to first 10 for storage
            'total_text_length': int(len(all_text)),
            'text_density': float(text_density),
            'has_significant_text': has_significant_text,
            'unique_text_segments': int(len(set(detected_texts)))
        }
        
        return text_analysis
    
        """
        Classify the content as footage-dependent or script-dependent
        
        Returns:
            Tuple of (classification, confidence_score, reasoning)
        """
        footage_score = 0
        script_score = 0
        reasoning = []
        
        # Visual indicators
        if visual_analysis.get('is_dynamic', False):
            footage_score += 3
            reasoning.append("High visual motion and scene changes indicate footage-dependent content")
        elif visual_analysis.get('is_static', False):
            script_score += 2
            reasoning.append("Static visuals suggest script-dependent content (talking head, simple graphics)")
        
        if visual_analysis.get('scene_change_rate', 0) > 0.2:
            footage_score += 2
            reasoning.append("Frequent scene changes indicate edited footage")
        
    def classify_content(self, visual_analysis: Dict, audio_analysis: Dict) -> Tuple[str, float, List[str]]:
        """
        Classify the content as footage-dependent, script-dependent, or footage-&-script-dependent
        
        Returns:
            Tuple of (classification, confidence_score, reasoning)
        """
        footage_score = 0
        script_score = 0
        reasoning = []
        
        # Visual indicators
        if visual_analysis.get('is_dynamic', False):
            footage_score += 4
            reasoning.append("High visual motion and scene changes indicate footage-dependent content")
        elif visual_analysis.get('is_static', False):
            script_score += 3
            reasoning.append("Static visuals suggest script-dependent content (talking head, simple graphics)")
        
        if visual_analysis.get('scene_change_rate', 0) > 0.2:
            footage_score += 3
            reasoning.append("Frequent scene changes indicate edited footage")
        elif visual_analysis.get('scene_change_rate', 0) < 0.05:
            script_score += 2
            reasoning.append("Very few scene changes suggest script-focused content")
        
        # Enhanced visual complexity scoring
        visual_complexity = visual_analysis.get('visual_complexity', 'low')
        if visual_complexity == 'high':
            footage_score += 2
            reasoning.append("High visual complexity indicates footage-dependent content")
        elif visual_complexity == 'low':
            script_score += 2
            reasoning.append("Low visual complexity suggests script-dependent content")
        
        # Audio/Speech indicators
        if audio_analysis.get('is_speech_heavy', False):
            script_score += 4
            reasoning.append("Heavy speech content suggests script-dependent format")
        
        if audio_analysis.get('has_narration', False) and not visual_analysis.get('is_dynamic', False):
            script_score += 3
            reasoning.append("Narration with static visuals indicates script-dependent content")
        
        # Enhanced audio analysis
        speech_ratio = audio_analysis.get('speech_ratio', 0)
        if speech_ratio > 20:  # Very speech-heavy
            script_score += 2
            reasoning.append("Very high speech density indicates script-dependent content")
        elif speech_ratio < 5:  # Very little speech
            footage_score += 2
            reasoning.append("Low speech content suggests footage-focused video")
        
        # Duration and format considerations
        if self.duration < 30:  # Very short videos
            if audio_analysis.get('speech_ratio', 0) > 15:
                script_score += 2
                reasoning.append("Short duration with high speech density suggests script format")
        
        # YouTube Shorts specific indicators
        if self.duration <= 60:  # Shorts are typically under 60 seconds
            if visual_analysis.get('visual_complexity') == 'high':
                footage_score += 2
                reasoning.append("High visual complexity in short format suggests footage-based content")
            
            # Additional Shorts-specific logic
            avg_motion = visual_analysis.get('average_motion', 0)
            if avg_motion > 60:  # High motion threshold for shorts
                footage_score += 1
                reasoning.append("High motion typical of engaging short-form footage content")
        
        # Word count considerations
        word_count = audio_analysis.get('word_count', 0)
        if word_count > 100:  # Lots of spoken content
            script_score += 2
            reasoning.append("High word count indicates script-heavy content")
        elif word_count < 10:  # Very little speech
            footage_score += 1
            reasoning.append("Low word count suggests visual-focused content")
        
        # Special case: Check for FOOTAGE-&-SCRIPT-DEPENDENT classification
        total_score = footage_score + script_score
        
        # Conditions for mixed classification
        has_significant_footage = (
            visual_analysis.get('is_dynamic', False) or 
            visual_analysis.get('scene_change_rate', 0) > 0.15 or
            visual_analysis.get('average_motion', 0) > 30
        )
        
        has_significant_script = (
            audio_analysis.get('has_narration', False) or
            audio_analysis.get('word_count', 0) > 30 or
            audio_analysis.get('speech_ratio', 0) > 10
        )
        
        # Mixed classification thresholds
        footage_ratio = footage_score / total_score if total_score > 0 else 0
        script_ratio = script_score / total_score if total_score > 0 else 0
        
        # Determine final classification
        if total_score == 0:
            classification = "uncertain"
            confidence = 0.0
            reasoning.append("Insufficient indicators to determine content type")
        
        # Check for mixed classification (both significant footage and script elements)
        elif (has_significant_footage and has_significant_script and 
              0.3 <= footage_ratio <= 0.7 and 0.3 <= script_ratio <= 0.7):
            classification = "footage-&-script-dependent"
            # Confidence based on how balanced the content is
            balance_score = 1.0 - abs(footage_ratio - script_ratio)  # Higher when more balanced
            confidence = float(min(0.95, 0.6 + (balance_score * 0.35)))  # Range: 0.6-0.95
            reasoning.append("Video contains both significant visual elements and substantial script content")
            reasoning.append(f"Visual elements: {footage_score} points, Script elements: {script_score} points")
        
        # Pure classifications
        elif footage_score > script_score:
            score_difference = footage_score - script_score
            if score_difference >= 3 and not has_significant_script:
                classification = "footage-dependent"
                confidence = float(footage_score / total_score)
            elif has_significant_script:
                # Has some script elements, check if should be mixed
                classification = "footage-&-script-dependent"
                confidence = float(0.65 + (score_difference / total_score * 0.25))
                reasoning.append("Primarily footage-based but with meaningful script content")
            else:
                classification = "footage-dependent"
                confidence = float(footage_score / total_score)
        
        else:  # script_score >= footage_score
            score_difference = script_score - footage_score
            if score_difference >= 3 and not has_significant_footage:
                classification = "script-dependent"
                confidence = float(script_score / total_score)
            elif has_significant_footage:
                # Has some footage elements, check if should be mixed
                classification = "footage-&-script-dependent"
                confidence = float(0.65 + (score_difference / total_score * 0.25))
                reasoning.append("Primarily script-based but with meaningful visual elements")
            else:
                classification = "script-dependent"
                confidence = float(script_score / total_score)
        
        return classification, confidence, reasoning
        
        return classification, confidence, reasoning
    
    def analyze(self) -> Dict:
        """
        Run complete analysis and return results
        """
        print(f"Analyzing YouTube video: {self.results['video_info'].get('title', 'Unknown')}")
        
        try:
            # Run analyses (removed text analysis)
            self.results['visual_analysis'] = self.analyze_visual_content()
            self.results['audio_analysis'] = self.extract_and_analyze_audio()
            
            # Classify content
            classification, confidence, reasoning = self.classify_content(
                self.results['visual_analysis'],
                self.results['audio_analysis']
            )
            
            self.results['final_classification'] = classification
            self.results['confidence_score'] = confidence
            self.results['reasoning'] = reasoning
            
        except Exception as e:
            self.results['error'] = str(e)
            print(f"Analysis error: {str(e)}")
        
        finally:
            # Clean up resources
            if self.video:
                self.video.release()
        print(self.results['final_classification'])
        return self.results
    
    def print_summary(self):
        """
        Print a summary of the analysis
        """
        print("\n" + "="*60)
        print("YOUTUBE SHORT ANALYSIS SUMMARY")
        print("="*60)
        
        # Video info
        video_info = self.results.get('video_info', {})
        print(f"Title: {video_info.get('title', 'Unknown')}")
        print(f"Duration: {self.duration:.2f} seconds")
        print(f"Uploader: {video_info.get('uploader', 'Unknown')}")
        
        if 'error' in self.results:
            print(f"Error: {self.results['error']}")
            return
        
        print(f"\nClassification: {self.results['final_classification'].upper()}")
        print(f"Confidence: {self.results['confidence_score']:.2f}")
        
        print("\nReasoning:")
        for reason in self.results['reasoning']:
            print(f"• {reason}")
        
        print(f"\nVisual Analysis:")
        va = self.results.get('visual_analysis', {})
        print(f"• Motion Level: {va.get('average_motion', 0):.2f}")
        print(f"• Scene Changes: {va.get('total_scene_changes', 0)}")
        print(f"• Visual Complexity: {va.get('visual_complexity', 'unknown')}")
        
        print(f"\nAudio Analysis:")
        aa = self.results.get('audio_analysis', {})
        print(f"• Word Count: {aa.get('word_count', 0)}")
        print(f"• Has Narration: {aa.get('has_narration', False)}")
        print(f"• Speech Heavy: {aa.get('is_speech_heavy', False)}")
        print(f"• Speech Ratio: {aa.get('speech_ratio', 0):.2f} chars/sec")
    
    def save_results(self, output_path: str):
        """
        Save analysis results to JSON file
        """
        # Remove video path from results for privacy
        results_copy = self.results.copy()
        if 'video_path' in results_copy:
            del results_copy['video_path']
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        results_serializable = convert_types(results_copy)
        
        # Ensure UTF-8 encoding for JSON output
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {output_path}")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self._cleanup()
