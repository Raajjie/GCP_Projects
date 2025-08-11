import os
import json
import tempfile
import re
from datetime import datetime
from typing import Dict, List, Any, TypedDict, Annotated
import google.generativeai as genai
from langgraph.graph import StateGraph, END
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

# Import your custom modules
from fetcher import YouTubeShortsFetcher
from content_classifier import YouTubeShortAnalyzer
from music_detector import YouTubeMusicDetector
from sentiment_analyzer import YouTubeCommentSentimentAnalyzer

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class WorkflowState(TypedDict):
    """State object that will be passed between nodes"""
    user_request: str
    region_code: str
    search_term: str
    hours_threshold: int
    max_videos: int  # Added for specific video count
    specific_url: str  # Added for specific URL analysis
    analysis_mode: str  # Added: 'trending', 'specific_url', 'search'
    fetched_data: List[Dict]
    collection_name: str
    from_cache: bool
    video_analysis_results: List[Dict]
    video_analysis_plans: List[Dict]
    final_summary: Dict
    messages: List[str]
    next_action: str
    error: str


class YouTubeTrendsWorkflow:
    """Main workflow class for YouTube Shorts trend analysis"""
    
    def __init__(self):
        # Initialize tools
        self.fetcher = YouTubeShortsFetcher()
        self.music_detector = YouTubeMusicDetector()
        self.sentiment_analyzer = YouTubeCommentSentimentAnalyzer()
        
        # Initialize Gemini model
        self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Create the workflow graph
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """Create and configure the LangGraph workflow"""
        
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("reasoner", self.reasoner_node)
        workflow.add_node("tools", self.tools_node)
        workflow.add_node("finalize", self.finalize_node)
        
        # Set entry point
        workflow.set_entry_point("reasoner")
        
        # Add edges
        workflow.add_edge("reasoner", "tools")
        workflow.add_edge("tools", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    def _parse_user_request(self, user_request: str) -> Dict:
        """
        Enhanced parsing to handle:
        1. Specific video counts: "Get 5 trends", "analyze 10 videos"
        2. Specific URLs: "analyze this video youtube.com/shorts/..."
        3. Region codes: "from Japan", "in US"
        4. Time constraints: "last 2 hours", "recent"
        """
        
        # Clean and normalize the request
        request_lower = user_request.lower().strip()
        
        # Default values
        parsed = {
            "region_code": "US",
            "search_term": "#shorts",
            "hours_threshold": 1,
            "max_videos": 15,
            "specific_url": None,
            "analysis_mode": "trending"
        }
        
        # 1. Check for specific YouTube URLs
        youtube_patterns = [
            r'(?:https?://)?(?:www\.)?youtube\.com/shorts/([a-zA-Z0-9_-]+)',
            r'(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]+)',
            r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]+)'
        ]
        
        for pattern in youtube_patterns:
            match = re.search(pattern, user_request)
            if match:
                video_id = match.group(1)
                parsed["specific_url"] = f"https://www.youtube.com/shorts/{video_id}"
                parsed["analysis_mode"] = "specific_url"
                parsed["max_videos"] = 1
                break
        
        # 2. Extract video count if not analyzing specific URL
        if parsed["analysis_mode"] != "specific_url":
            # Look for numbers followed by video-related words
            count_patterns = [
                r'(\d+)\s*(?:videos?|trends?|shorts?)',
                r'(?:get|analyze|show|find)\s*(\d+)',
                r'(?:top|first|only)\s*(\d+)'
            ]
            
            for pattern in count_patterns:
                match = re.search(pattern, request_lower)
                if match:
                    count = int(match.group(1))
                    if 1 <= count <= 50:  # Reasonable limits
                        parsed["max_videos"] = count
                    break
        
        # 3. Extract region code
        region_patterns = {
            r'\b(?:from|in)\s+(us|usa|america|united states)\b': 'US',
            r'\b(?:from|in)\s+(uk|britain|england|united kingdom)\b': 'GB',
            r'\b(?:from|in)\s+(japan|jp)\b': 'JP',
            r'\b(?:from|in)\s+(germany|de|deutschland)\b': 'DE',
            r'\b(?:from|in)\s+(france|fr)\b': 'FR',
            r'\b(?:from|in)\s+(canada|ca)\b': 'CA',
            r'\b(?:from|in)\s+(australia|au)\b': 'AU',
            r'\b(?:from|in)\s+(india|in)\b': 'IN',
            r'\b(?:from|in)\s+(brazil|br)\b': 'BR',
            r'\b(?:from|in)\s+(korea|kr|south korea)\b': 'KR',
        }
        
        for pattern, code in region_patterns.items():
            if re.search(pattern, request_lower):
                parsed["region_code"] = code
                break
        
        # 4. Extract time threshold
        time_patterns = {
            r'(?:last|past|recent)\s*(\d+)\s*hours?': lambda m: int(m.group(1)),
            r'(?:last|past|recent)\s*(\d+)\s*mins?': lambda m: max(1, int(m.group(1)) // 60),
            r'(?:last|past|recent)\s*hour': lambda m: 1,
            r'(?:fresh|new|latest)': lambda m: 1,
            r'today': lambda m: 24,
        }
        
        for pattern, extractor in time_patterns.items():
            match = re.search(pattern, request_lower)
            if match:
                parsed["hours_threshold"] = extractor(match)
                break
        
        # 5. Extract search terms or hashtags
        if parsed["analysis_mode"] != "specific_url":
            # Look for hashtags
            hashtag_match = re.search(r'#(\w+)', user_request)
            if hashtag_match:
                parsed["search_term"] = f"#{hashtag_match.group(1)}"
            
            # Look for quoted search terms
            quoted_match = re.search(r'"([^"]+)"', user_request)
            if quoted_match:
                parsed["search_term"] = quoted_match.group(1)
                parsed["analysis_mode"] = "search"
            
            # Look for specific topics
            topic_keywords = {
                'music': '#music',
                'dance': '#dance', 
                'comedy': '#comedy',
                'cooking': '#cooking',
                'fashion': '#fashion',
                'gaming': '#gaming',
                'fitness': '#fitness',
                'travel': '#travel'
            }
            
            for keyword, hashtag in topic_keywords.items():
                if keyword in request_lower:
                    parsed["search_term"] = hashtag
                    parsed["analysis_mode"] = "search"
                    break
        
        return parsed
    
    def reasoner_node(self, state: WorkflowState) -> Dict:
        """
        Enhanced reasoning node with better request parsing
        """
        try:
            user_request = state.get("user_request", "")
            messages = state.get("messages", [])
            
            messages.append(f"🧠 Reasoner: Parsing request: '{user_request}'")
            
            # Parse the user request with enhanced logic
            parsed = self._parse_user_request(user_request)
            
            # Log what was parsed
            messages.append(f"🧠 Reasoner: Detected analysis mode: {parsed['analysis_mode']}")
            
            if parsed["analysis_mode"] == "specific_url":
                messages.append(f"🧠 Reasoner: Analyzing specific video: {parsed['specific_url']}")
            else:
                messages.append(f"🧠 Reasoner: Region: {parsed['region_code']}, Videos: {parsed['max_videos']}")
                messages.append(f"🧠 Reasoner: Search term: {parsed['search_term']}, Time: {parsed['hours_threshold']}h")
            
            # Use Gemini for additional context if needed
            reasoning_prompt = f"""
            Analyze this YouTube request and confirm the parsing:
            
            Original request: "{user_request}"
            
            Parsed parameters:
            - Analysis mode: {parsed['analysis_mode']}
            - Region: {parsed['region_code']}
            - Max videos: {parsed['max_videos']}
            - Search term: {parsed['search_term']}
            - Time threshold: {parsed['hours_threshold']} hours
            - Specific URL: {parsed['specific_url']}
            
            Create an analysis strategy based on the parsed parameters.
            
            Respond in JSON:
            {{
                "strategy": "Brief description of analysis approach",
                "confirmed_params": {{
                    "analysis_mode": "{parsed['analysis_mode']}",
                    "max_videos": {parsed['max_videos']},
                    "region_code": "{parsed['region_code']}",
                    "search_term": "{parsed['search_term']}",
                    "hours_threshold": {parsed['hours_threshold']},
                    "specific_url": "{parsed['specific_url']}"
                }},
                "next_steps": ["step1", "step2", "step3"]
            }}
            """
            
            try:
                response = self.gemini_model.generate_content(reasoning_prompt)
                strategy = json.loads(response.text)
                confirmed_params = strategy.get("confirmed_params", parsed)
                messages.append(f"🧠 Reasoner: Strategy confirmed: {strategy.get('strategy', 'Standard analysis')}")
            except:
                confirmed_params = parsed
                messages.append("🧠 Reasoner: Using parsed parameters (Gemini confirmation failed)")
            
            # Generate ReAct plans if we have data
            fetched_data = state.get("fetched_data", [])
            video_analysis_plans = []
            
            if fetched_data:
                messages.append("🧠 Reasoner: Starting ReAct THOUGHT phase...")
                
                # Limit to the requested number of videos
                videos_to_analyze = fetched_data[:confirmed_params["max_videos"]]
                
                for i, video in enumerate(videos_to_analyze):
                    video_title = video.get('title', 'Unknown')[:50]
                    video_url = video.get("video_url")
                    
                    if not video_url:
                        continue
                    
                    messages.append(f"🧠 Reasoner: Planning analysis for video {i+1}/{len(videos_to_analyze)}: {video_title}...")
                    
                    # Generate ReAct THOUGHT for this video
                    react_prompt = f"""
                    Analyze video {i+1} of {len(videos_to_analyze)} using ReAct pattern.
                    
                    Video Info:
                    - Title: {video.get('title', '')}
                    - Description: {video.get('description', '')[:200]}
                    - Duration: {video.get('duration', 'Unknown')}
                    - View count: {video.get('view_count', 'Unknown')}
                    
                    Analysis context: {confirmed_params['analysis_mode']} mode, focusing on {confirmed_params['search_term']}
                    
                    THOUGHT: What type of content is this and what analysis is needed?
                    ACTION PLAN: Which analysis tools should be applied?
                    REASONING: Why this approach makes sense?
                    
                    JSON response:
                    {{
                        "video_id": "{video.get('video_id', '')}",
                        "thought": "Content type analysis and needs assessment",
                        "predicted_classification": "script-dependent|footage-dependent|both|uncertain",
                        "action_plan": {{
                            "content_classification": true,
                            "music_detection": true/false,
                            "transcription_analysis": true/false,
                            "sentiment_analysis": true
                        }},
                        "reasoning": "Why this plan fits the video and request context"
                    }}
                    """
                    
                    try:
                        react_response = self.gemini_model.generate_content(react_prompt)
                        plan = json.loads(react_response.text)
                        video_analysis_plans.append(plan)
                        
                        predicted = plan.get("predicted_classification", "uncertain")
                        messages.append(f"   💭 THOUGHT: Predicted '{predicted}'")
                        
                    except Exception as e:
                        messages.append(f"   ❌ Planning failed for video {i+1}: {str(e)}")
                        # Fallback plan
                        fallback_plan = {
                            "video_id": video.get('video_id', ''),
                            "thought": "Comprehensive analysis due to planning failure",
                            "predicted_classification": "uncertain",
                            "action_plan": {
                                "content_classification": True,
                                "music_detection": True,
                                "transcription_analysis": True,
                                "sentiment_analysis": True
                            },
                            "reasoning": "Fallback comprehensive analysis"
                        }
                        video_analysis_plans.append(fallback_plan)
                
                messages.append(f"🧠 Reasoner: Generated {len(video_analysis_plans)} analysis plans")
            
            return {
                **state,
                **confirmed_params,  # Unpack all confirmed parameters
                "video_analysis_plans": video_analysis_plans,
                "messages": messages
            }
                
        except Exception as e:
            return {**state, "error": f"Reasoning failed: {str(e)}"}
    
    def tools_node(self, state: WorkflowState) -> Dict:
        """
        Enhanced tools node that handles different analysis modes
        """
        try:
            messages = state.get("messages", [])
            analysis_mode = state.get("analysis_mode", "trending")
            specific_url = state.get("specific_url")
            max_videos = state.get("max_videos", 15)
            region_code = state.get("region_code", "US")
            search_term = state.get("search_term", None)
            hours_threshold = state.get("hours_threshold", 1)
            video_analysis_plans = state.get("video_analysis_plans", [])
            
            # Step 1: Fetch data based on analysis mode
            fetched_data = state.get("fetched_data", [])
            collection_name = state.get("collection_name", "")
            from_cache = state.get("from_cache", False)
            
            if not fetched_data:
                if analysis_mode == "specific_url" and specific_url:
                    messages.append(f"🔧 Tools: Analyzing specific URL: {specific_url}")
                    
                    # For specific URL, create a mock video object
                    try:
                        # Extract video ID from URL
                        video_id_match = re.search(r'/shorts/([a-zA-Z0-9_-]+)', specific_url)
                        if not video_id_match:
                            video_id_match = re.search(r'v=([a-zA-Z0-9_-]+)', specific_url)
                        
                        if video_id_match:
                            video_id = video_id_match.group(1)
                            
                            # Create a basic video object for the specific URL
                            fetched_data = [{
                                'video_id': video_id,
                                'video_url': specific_url,
                                'title': 'User-specified video',
                                'description': 'Analyzing specific video provided by user',
                                'view_count': 'Unknown',
                                'duration': 'Unknown',
                                'published_at': datetime.now().isoformat()
                            }]
                            
                            collection_name = f"specific_video_{video_id}"
                            from_cache = False
                            
                            messages.append("🔧 Tools: Created analysis object for specific video")
                        else:
                            return {**state, "error": "Could not extract video ID from URL", "messages": messages}
                    
                    except Exception as e:
                        return {**state, "error": f"Failed to process specific URL: {str(e)}", "messages": messages}
                
                else:
                    # Standard trending/search analysis
                    messages.append("🔧 Tools: Checking cache for trending videos...")
                    
                    search_type = f"search_{search_term.replace('#', '').replace(' ', '_')}"
                    is_recent, recent_collection, last_search_time = self.fetcher.check_recent_search(
                        region_code, search_type, hours_threshold
                    )
                    
                    if is_recent and recent_collection:
                        messages.append(f"🔧 Tools: Found recent data in {recent_collection}")
                        all_fetched_data = self.fetcher.get_from_firestore(recent_collection)
                        # Limit to requested number
                        fetched_data = all_fetched_data[:max_videos]
                        from_cache = True
                        collection_name = recent_collection
                    else:
                        messages.append("🔧 Tools: Fetching fresh data from YouTube...")
                        shorts, collection_name, from_cache = self.fetcher.get_trending_shorts(
                            region_code=region_code,
                            max_results=max_videos,  # Use the parsed max_videos
                            search_term=search_term,
                            save_to_firestore=True,
                            force_new_search=True
                        )
                        fetched_data = shorts[:max_videos]  # Ensure we don't exceed limit
                
                if not fetched_data:
                    return {**state, "error": "No videos found to analyze", "messages": messages}
                
                messages.append(f"🔧 Tools: Processing {len(fetched_data)} videos (requested: {max_videos})")
                
                # Generate plans if we don't have them
                if not video_analysis_plans and len(fetched_data) > 0:
                    messages.append("🔧 Tools: No pre-made plans found, generating them now...")
                    reasoner_result = self.reasoner_node({
                        **state, 
                        "fetched_data": fetched_data,
                        "messages": messages
                    })
                    video_analysis_plans = reasoner_result.get("video_analysis_plans", [])
                    messages = reasoner_result.get("messages", messages)
            
            # Step 2: Execute analysis plans (limit to requested number of videos)
            messages.append(f"🔧 Tools: Executing ReAct ACTION plans for {len(fetched_data)} videos...")
            video_analysis_results = []
            
            for i, video in enumerate(fetched_data):
                try:
                    video_url = video.get("video_url")
                    if not video_url:
                        continue
                    
                    # Find corresponding plan
                    video_id = video.get('video_id', '')
                    plan = None
                    for p in video_analysis_plans:
                        if p.get('video_id') == video_id:
                            plan = p
                            break
                    
                    if not plan:
                        messages.append(f"🔧 Tools: No plan found for video {i+1}, using comprehensive analysis...")
                        plan = {
                            "action_plan": {
                                "content_classification": True,
                                "music_detection": True,
                                "transcription_analysis": True,
                                "sentiment_analysis": True
                            },
                            "thought": "Comprehensive analysis (no specific plan)",
                            "reasoning": "Default analysis for unplanned video"
                        }
                    
                    video_title = video.get('title', 'Unknown')[:50]
                    messages.append(f"🔧 Tools: [{i+1}/{len(fetched_data)}] Analyzing: {video_title}...")
                    
                    action_plan = plan.get("action_plan", {})
                    video_result = {**video}
                    
                    # Execute planned actions
                    if action_plan.get("content_classification", True):
                        messages.append(f"   🎬 ACTION: Content classification...")
                        analyzer = YouTubeShortAnalyzer(video_url)
                        analysis_results = analyzer.analyze()
                        
                        classification = analysis_results.get("final_classification", "uncertain")
                        confidence = analysis_results.get("confidence_score", 0)
                        
                        video_result.update({
                            "classification": classification,
                            "classification_confidence": confidence,
                            "classification_reasoning": analysis_results.get("reasoning", [])
                        })
                        
                        messages.append(f"   📊 RESULT: '{classification}' (conf: {confidence:.2f})")
                    
                    if action_plan.get("music_detection", False):
                        messages.append(f"   🎵 ACTION: Music detection...")
                        has_music, music_confidence = self.music_detector.detect_music(video_url)
                        video_result.update({
                            "has_music": has_music,
                            "music_confidence": music_confidence
                        })
                        messages.append(f"   🎼 RESULT: Music: {has_music} (conf: {music_confidence:.2f})")
                    
                    if action_plan.get("transcription_analysis", False):
                        messages.append(f"   🎤 ACTION: Audio transcription...")
                        audio_analysis = analyzer.extract_and_analyze_audio()
                        transcript = audio_analysis.get("transcript", "")
                        
                        if transcript:
                            topic_data = self._extract_topic_with_gemini(transcript)
                            messages.append(f"   📝 RESULT: Topic: '{topic_data.get('topic', 'Unknown')[:30]}...'")
                        else:
                            topic_data = {"topic": "No transcript", "category": "Other", "keywords": []}
                            messages.append(f"   ⚠️ RESULT: No transcript extracted")
                        
                        video_result.update({
                            "transcript": transcript,
                            "topic": topic_data.get("topic", "Unknown"),
                            "category": topic_data.get("category", "Other"),
                            "keywords": topic_data.get("keywords", [])
                        })
                    
                    if action_plan.get("sentiment_analysis", True):
                        messages.append(f"   😊 ACTION: Sentiment analysis...")
                        sentiment_results = self.sentiment_analyzer.analyze_video_comments(video_url)
                        video_result["sentiment_analysis"] = sentiment_results
                        
                        overall_sentiment = sentiment_results.get("overall_sentiment", "neutral")
                        messages.append(f"   💭 RESULT: Sentiment: {overall_sentiment}")
                    
                    video_result["executed_plan"] = plan
                    video_analysis_results.append(video_result)
                    messages.append(f"   ✅ Video {i+1} completed")
                    
                except Exception as e:
                    messages.append(f"   ❌ ERROR: Video {i+1} failed: {str(e)}")
                    continue
            
            return {
                **state,
                "fetched_data": fetched_data,
                "collection_name": collection_name,
                "from_cache": from_cache,
                "video_analysis_results": video_analysis_results,
                "messages": messages + [f"🔧 Tools: Completed analysis of {len(video_analysis_results)} videos"]
            }
        
        except Exception as e:
            return {**state, "error": f"Tools execution failed: {str(e)}", "messages": messages}
    
    def _extract_topic_with_gemini(self, transcript: str) -> Dict:
        """Helper function to extract topic using Gemini"""
        try:
            topic_prompt = f"""
            Analyze this video transcript and extract the main topic/theme:
            
            Transcript: "{transcript}"
            
            Provide a concise topic summary (1-2 sentences) and categorize it.
            
            Categories: Entertainment, Education, Commentary, Tutorial, Lifestyle, News, Gaming, Music, Comedy, Other
            
            JSON format:
            {{
                "topic": "Main topic description",
                "category": "Category name",
                "keywords": ["key1", "key2", "key3"]
            }}
            """
            
            response = self.gemini_model.generate_content(topic_prompt)
            return json.loads(response.text)
        except:
            return {"topic": "Analysis failed", "category": "Other", "keywords": []}
    
    def finalize_node(self, state: WorkflowState) -> Dict:
        """
        Enhanced finalization with professional PDF formatting
        """
        try:
            messages = state.get("messages", [])
            video_results = state.get("video_analysis_results", [])
            analysis_mode = state.get("analysis_mode", "trending")
            region_code = state.get("region_code", "US")
            search_term = state.get("search_term", None)
            max_videos = state.get("max_videos", 15)
            specific_url = state.get("specific_url")
            
            if not video_results:
                return {**state, "error": "No video results to summarize"}
            
            messages.append("📝 Finalize: Generating comprehensive summary...")
            
            # Mode-specific summary prompt
            if analysis_mode == "specific_url":
                summary_context = f"Single video analysis for: {specific_url}"
            else:
                # Clean search term for display (remove hashtags)
                clean_search_term = search_term.replace('#', '') if search_term else "shorts"
                summary_context = f"""
                YouTube Shorts trend analysis:
                - Region: {region_code}
                - Search: {clean_search_term}
                - Requested videos: {max_videos}
                - Actually analyzed: {len(video_results)}
                - Analysis mode: {analysis_mode}
                """
            
            summary_prompt = f"""
            Create a comprehensive analysis report for:
            {summary_context}
            
            Video Analysis Data:
            {json.dumps([{
                'title': v.get('title', '')[:100],
                'classification': v.get('classification', ''),
                'has_music': v.get('has_music', False),
                'topic': v.get('topic', '')[:100],
                'category': v.get('category', ''),
                'sentiment': v.get('sentiment_analysis', {}).get('overall_sentiment', ''),
                'executed_plan_reasoning': v.get('executed_plan', {}).get('reasoning', '')
            } for v in video_results], indent=2)}
            
            Structure the report based on analysis type:
            
            {'For Single Video Analysis:' if analysis_mode == 'specific_url' else 'For Trending Analysis:'}
            1. Executive Summary
            2. {'Individual Video Deep Dive' if analysis_mode == 'specific_url' else 'Content Distribution Overview'}
            3. {'Content Analysis' if analysis_mode == 'specific_url' else 'Popular Topics and Themes'}
            4. Sentiment Analysis {'Results' if analysis_mode == 'specific_url' else 'Overview'}
            5. {'Technical Findings' if analysis_mode == 'specific_url' else 'Music Usage Patterns'}
            6. Key Insights and {'Recommendations' if analysis_mode == 'specific_url' else 'Trends'}
            7. ReAct Analysis Effectiveness
            8. {'Next Steps' if analysis_mode == 'specific_url' else 'Recommendations for Content Creators'}
            
            Make it comprehensive and actionable. Use clear section headers without hashtags or special characters.
            """
            
            response = self.gemini_model.generate_content(summary_prompt)
            
            # Clean search term for final summary
            clean_search_term = search_term.replace('#', '') if search_term else None
            
            final_summary = {
                "analysis_date": datetime.now().isoformat(),
                "analysis_mode": analysis_mode,
                "region": region_code,
                "search_term": clean_search_term,
                "requested_videos": max_videos,
                "analyzed_videos": len(video_results),
                "specific_url": specific_url,
                "detailed_analysis": response.text,
                "video_results": video_results
            }
            
            messages.append("📄 Finalize: Creating enhanced PDF report...")
            
            # Create PDF with enhanced formatting
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if analysis_mode == "specific_url":
                filename = f"youtube_video_analysis_{timestamp}.pdf"
            else:
                clean_search_for_filename = clean_search_term.replace(' ', '_') if clean_search_term else 'shorts'
                filename = f"youtube_trends_{region_code}_{clean_search_for_filename}_{max_videos}videos_{timestamp}.pdf"
            
            doc = SimpleDocTemplate(filename, pagesize=letter, topMargin=1*inch, bottomMargin=1*inch)
            styles = getSampleStyleSheet()
            story = []
            
            # Enhanced styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=20,
                textColor=colors.darkblue,
                spaceAfter=30,
                alignment=1  # Center alignment
            )
            
            subtitle_style = ParagraphStyle(
                'CustomSubtitle',
                parent=styles['Heading2'],
                fontSize=14,
                textColor=colors.darkred,
                spaceAfter=20
            )
            
            header_style = ParagraphStyle(
                'CustomHeader',
                parent=styles['Heading3'],
                fontSize=12,
                textColor=colors.darkblue,
                spaceAfter=12
            )
            
            # Title
            if analysis_mode == "specific_url":
                title_text = "YouTube Video Analysis Report"
            else:
                title_text = "YouTube Shorts Trend Analysis Report"
            
            story.append(Paragraph(title_text, title_style))
            story.append(Spacer(1, 20))
            
            # Analysis Information Table
            story.append(Paragraph("Analysis Overview", subtitle_style))
            
            # Create metadata table
            metadata_data = [
                ['Analysis Date', datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                ['Analysis Mode', analysis_mode.replace('_', ' ').title()],
            ]
            
            if analysis_mode == "specific_url":
                metadata_data.append(['Video URL', specific_url or 'N/A'])
            else:
                metadata_data.extend([
                    ['Region', region_code],
                    ['Search Term', clean_search_term or 'General Shorts'],
                    ['Requested Videos', str(max_videos)],
                ])
            
            metadata_data.append(['Videos Analyzed', str(len(video_results))])
            
            metadata_table = Table(metadata_data, colWidths=[2*inch, 4*inch])
            metadata_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            
            story.append(metadata_table)
            story.append(Spacer(1, 30))
            
            # Video Results Summary Table (if multiple videos)
            if len(video_results) > 1:
                story.append(Paragraph("Video Analysis Summary", subtitle_style))
                
                # Prepare video data for table
                video_table_data = [['#', 'Title', 'Classification', 'Sentiment', 'Music', 'Category']]
                
                for i, video in enumerate(video_results[:10], 1):  # Limit to 10 for readability
                    title = video.get('title', 'Unknown')[:40] + '...' if len(video.get('title', '')) > 40 else video.get('title', 'Unknown')
                    classification = video.get('classification', 'N/A')
                    sentiment = video.get('sentiment_analysis', {}).get('overall_sentiment', 'N/A')
                    has_music = 'Yes' if video.get('has_music') else 'No'
                    category = video.get('category', 'N/A')
                    
                    video_table_data.append([
                        str(i), title, classification, sentiment, has_music, category
                    ])
                
                if len(video_results) > 10:
                    video_table_data.append(['...', f'And {len(video_results) - 10} more videos', '', '', '', ''])
                
                video_table = Table(video_table_data, colWidths=[0.5*inch, 2.5*inch, 1.2*inch, 1*inch, 0.6*inch, 1*inch])
                video_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
                ]))
                
                story.append(video_table)
                story.append(Spacer(1, 30))
            
            # Statistics Table
            if len(video_results) > 1:
                story.append(Paragraph("Content Distribution Statistics", subtitle_style))
                
                # Calculate statistics
                from collections import Counter
                
                classifications = [v.get('classification', 'Unknown') for v in video_results]
                categories = [v.get('category', 'Unknown') for v in video_results]
                sentiments = [v.get('sentiment_analysis', {}).get('overall_sentiment', 'Unknown') for v in video_results]
                music_count = sum(1 for v in video_results if v.get('has_music'))
                
                class_counter = Counter(classifications)
                category_counter = Counter(categories)
                sentiment_counter = Counter(sentiments)
                
                # Create statistics tables
                stats_data = []
                
                # Classifications
                stats_data.append(['Content Classification', 'Count', 'Percentage'])
                for classification, count in class_counter.most_common():
                    percentage = f"{(count/len(video_results)*100):.1f}%"
                    stats_data.append([classification, str(count), percentage])
                
                stats_data.append(['', '', ''])  # Empty row
                
                # Categories
                stats_data.append(['Content Category', 'Count', 'Percentage'])
                for category, count in category_counter.most_common()[:5]:  # Top 5
                    percentage = f"{(count/len(video_results)*100):.1f}%"
                    stats_data.append([category, str(count), percentage])
                
                stats_data.append(['', '', ''])  # Empty row
                
                # Additional stats
                stats_data.append(['Music Usage', str(music_count), f"{(music_count/len(video_results)*100):.1f}%"])
                stats_data.append(['Total Videos Analyzed', str(len(video_results)), '100%'])
                
                stats_table = Table(stats_data, colWidths=[3*inch, 1*inch, 1.5*inch])
                stats_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                    ('BACKGROUND', (0, len(class_counter)+1), (-1, len(class_counter)+1), colors.darkred),
                    ('BACKGROUND', (0, len(class_counter)+len(category_counter)+3), (-1, -2), colors.darkgreen),
                    ('BACKGROUND', (0, -1), (-1, -1), colors.darkgrey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('TEXTCOLOR', (0, len(class_counter)+1), (-1, len(class_counter)+1), colors.whitesmoke),
                    ('TEXTCOLOR', (0, len(class_counter)+len(category_counter)+3), (-1, -1), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTNAME', (0, len(class_counter)+1), (-1, len(class_counter)+1), 'Helvetica-Bold'),
                    ('FONTNAME', (0, len(class_counter)+len(category_counter)+3), (-1, -1), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ]))
                
                story.append(stats_table)
                story.append(Spacer(1, 30))
            
            # Detailed Analysis Report
            story.append(Paragraph("Detailed Analysis Report", subtitle_style))
            
            # Process and format the AI-generated analysis
            analysis_text = final_summary['detailed_analysis']
            
            # Clean the analysis text
            analysis_text = re.sub(r'#+\s*', '', analysis_text)  # Remove markdown headers
            analysis_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', analysis_text)  # Bold formatting
            analysis_text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', analysis_text)  # Italic formatting
            
            # Split into sections and format properly
            sections = re.split(r'\n\s*\n', analysis_text)
            
            for section in sections:
                section = section.strip()
                if not section:
                    continue
                
                # Check if it's a section header (starts with number or is short and contains key words)
                if (re.match(r'^\d+\.', section) or 
                    (len(section) < 80 and any(word in section.lower() for word in 
                    ['summary', 'overview', 'analysis', 'findings', 'insights', 'recommendations']))):
                    story.append(Paragraph(section, header_style))
                else:
                    # Format as regular paragraph
                    # Handle bullet points
                    if '•' in section or section.strip().startswith('-'):
                        # It's a list
                        lines = section.split('\n')
                        for line in lines:
                            line = line.strip()
                            if line:
                                if line.startswith('•') or line.startswith('-'):
                                    story.append(Paragraph(line, styles['Normal']))
                                else:
                                    story.append(Paragraph(line, styles['Normal']))
                    else:
                        # Regular paragraph
                        # Split long paragraphs
                        sentences = re.split(r'(?<=[.!?])\s+', section)
                        current_para = ""
                        
                        for sentence in sentences:
                            if len(current_para + sentence) > 400:  # Max paragraph length
                                if current_para:
                                    story.append(Paragraph(current_para.strip(), styles['Normal']))
                                    story.append(Spacer(1, 6))
                                current_para = sentence + " "
                            else:
                                current_para += sentence + " "
                        
                        if current_para.strip():
                            story.append(Paragraph(current_para.strip(), styles['Normal']))
                    
                    story.append(Spacer(1, 12))
            
            # Add footer with generation info
            story.append(Spacer(1, 30))
            footer_style = ParagraphStyle(
                'Footer',
                parent=styles['Normal'],
                fontSize=8,
                textColor=colors.grey,
                alignment=1  # Center
            )
            
            footer_text = f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | YouTube Shorts Analysis Workflow"
            story.append(Paragraph(footer_text, footer_style))
            
            # Build PDF
            doc.build(story)
            
            # ✅ Upload to Google Drive
            try:
                gauth = GoogleAuth()
                gauth.LocalWebserverAuth()

                drive = GoogleDrive(gauth)
                gfile = drive.CreateFile({'title': filename})
                gfile.SetContentFile(filename)
                gfile.Upload()
                messages.append(f"📤 Uploaded to Google Drive: {gfile['alternateLink']}")
            except Exception as upload_err:
                messages.append(f"⚠️ Failed to upload to Google Drive: {upload_err}")

            messages.append(f"✅ Enhanced analysis complete! Professional report saved as {filename}")
            
            return {
                **state,
                "final_summary": final_summary,
                "pdf_filename": filename,
                "messages": messages
            }
            
        except Exception as e:
            return {**state, "error": f"Finalization failed: {str(e)}", "messages": messages}
    
    def run_workflow(self, user_request: str) -> Dict:
        """Run the complete workflow"""
        try:
            initial_state = {
                "user_request": user_request,
                "messages": []
            }
            
            print(f"\n🚀 Starting analysis for: '{user_request}'")
            print("="*60)
            
            result = self.workflow.invoke(initial_state)
            
            # Print all messages for transparency
            for message in result.get("messages", []):
                print(message)
            
            return result
            
        except Exception as e:
            return {"error": f"Workflow execution failed: {str(e)}"}
    

        
        print("\n" + "="*80)
        print("🔧 ENHANCED LANGGRAPH WORKFLOW - INPUT PARSING IMPROVEMENTS")
        print("="*80)
        print("New Features Added:")
        print("• ✅ Parse specific video counts: 'Get 5 trends', 'analyze 10 videos'")
        print("• ✅ Handle specific URLs: 'analyze this video youtube.com/shorts/ABC'")
        print("• ✅ Extract regions: 'from Japan', 'in Germany', 'UK trends'")
        print("• ✅ Parse time constraints: 'last 2 hours', 'recent', 'today'")
        print("• ✅ Detect search terms: hashtags, quoted terms, topics")
        print("• ✅ Three analysis modes: trending, specific_url, search")
        print("="*80)
        print(mermaid_code)
        print("="*80)
        print("Supported Input Examples:")
        print('• "Get 5 trends from US"')
        print('• "Analyze this video https://youtube.com/shorts/ABC123"')
        print('• "Show 10 comedy shorts from Japan"')
        print('• "Find trending music videos last 2 hours"')
        print('• "Get recent UK trends" (defaults to 15 videos)')
        print('• "Only analyze 3 videos with #dance from Germany"')
        print("="*80)


# def main():
#     """Enhanced interactive terminal interface"""
#     print("🎬 Enhanced YouTube Shorts Trend Analysis Workflow")
#     print("="*60)
    
#     # Print the enhanced diagram
    
#     print("\n🔧 NEW FEATURES - Smart Input Parsing:")
#     print("="*60)
#     print("✅ Specific Video Counts:")
#     print("   • 'Get 5 trends from US'")
#     print("   • 'Analyze only 3 videos'")
#     print("   • 'Show me top 10 shorts'")
#     print()
#     print("✅ Specific Video URLs:")
#     print("   • 'Analyze this video: youtube.com/shorts/ABC123'")
#     print("   • 'Only analyze https://youtu.be/XYZ789'")
#     print()
#     print("✅ Region Detection:")
#     print("   • 'Get trends from Japan'")
#     print("   • 'UK comedy shorts'")
#     print("   • 'Germany music videos'")
#     print()
#     print("✅ Time Constraints:")
#     print("   • 'Last 2 hours trends'")
#     print("   • 'Recent viral shorts'")
#     print("   • 'Today's popular videos'")
#     print()
#     print("✅ Topic/Hashtag Search:")
#     print("   • 'Find #dance videos'")
#     print("   • 'Comedy shorts from US'")
#     print("   • 'Music trending videos'")
#     print("\nType 'quit' to exit")
#     print("="*60)
    
#     # Initialize workflow
#     try:
#         workflow = YouTubeTrendsWorkflow()
#         print("✅ Enhanced workflow initialized successfully!")
#     except Exception as e:
#         print(f"❌ Failed to initialize workflow: {e}")
#         return
    
#     # Example requests for testing
#     example_requests = [
#         "Get 5 trends from US",
#         "Analyze this video: https://youtube.com/shorts/dQw4w9WgXcQ",
#         "Show me 3 comedy shorts from Japan",
#         "Find trending music videos last 2 hours",
#         "Get 7 #dance videos from Germany"
#     ]
    
#     print(f"\n🧪 Quick Test Examples (type number 1-{len(example_requests)} or custom request):")
#     for i, example in enumerate(example_requests, 1):
#         print(f"   {i}. {example}")
    
#     while True:
#         try:
#             print(f"\n{'='*60}")
#             user_input = input("📝 Enter your request (or 1-5 for examples): ").strip()
            
#             if user_input.lower() in ['quit', 'exit', 'q']:
#                 print("👋 Goodbye!")
#                 break
            
#             if not user_input:
#                 print("❌ Please enter a request")
#                 continue
            
#             # Handle example selection
#             if user_input.isdigit() and 1 <= int(user_input) <= len(example_requests):
#                 user_input = example_requests[int(user_input) - 1]
#                 print(f"🧪 Testing example: '{user_input}'")
            
#             # Run the workflow
#             result = workflow.run_workflow(user_input)
            
#             print("\n" + "="*60)
#             print("📊 ENHANCED RESULTS SUMMARY")
#             print("="*60)
            
#             if "error" in result:
#                 print(f"❌ Error: {result['error']}")
#             else:
#                 print(f"✅ Analysis completed successfully!")
                
#                 # Enhanced result display
#                 analysis_mode = result.get('analysis_mode', 'unknown')
#                 max_videos = result.get('max_videos', 0)
#                 analyzed_count = len(result.get('video_analysis_results', []))
                
#                 print(f"📈 Analysis Mode: {analysis_mode}")
#                 print(f"📱 Requested Videos: {max_videos}")
#                 print(f"📊 Actually Analyzed: {analyzed_count}")
#                 print(f"📁 PDF Report: {result.get('pdf_filename', 'N/A')}")
                
#                 if analysis_mode == "specific_url":
#                     print(f"🎯 Target URL: {result.get('specific_url', 'N/A')}")
#                 else:
#                     print(f"🌍 Region: {result.get('region_code', 'N/A')}")
#                     print(f"🔍 Search Term: {result.get('search_term', 'N/A')}")
#                     print(f"⏰ Time Window: {result.get('hours_threshold', 0)} hours")
                
#                 print(f"♻️  From Cache: {'Yes' if result.get('from_cache') else 'No'}")
#                 print(f"🧠 Analysis Plans: {len(result.get('video_analysis_plans', []))}")
                
#                 # Content analysis summary
#                 videos = result.get('video_analysis_results', [])
#                 if videos:
#                     classifications = [v.get('classification', 'unknown') for v in videos]
#                     from collections import Counter
#                     class_counts = Counter(classifications)
                    
#                     print(f"\n📈 Content Distribution:")
#                     for classification, count in class_counts.most_common():
#                         print(f"   • {classification}: {count} videos")
                    
#                     # ReAct effectiveness
#                     plans = result.get('video_analysis_plans', [])
#                     if plans:
#                         predicted_types = [p.get('predicted_classification', 'unknown') for p in plans]
#                         actual_types = [v.get('classification', 'unknown') for v in videos]
                        
#                         matches = sum(1 for pred, actual in zip(predicted_types, actual_types) 
#                                     if pred.lower() in actual.lower() or actual.lower() in pred.lower())
#                         accuracy = (matches / len(plans)) * 100 if plans else 0
                        
#                         print(f"\n🎯 ReAct Planning Accuracy: {accuracy:.1f}%")
#                         print(f"   • Correct predictions: {matches}/{len(plans)}")
                
#                 # Parsing effectiveness display
#                 if analysis_mode == "specific_url":
#                     print(f"\n🎯 URL Parsing: Successfully detected and analyzed specific video")
#                 elif max_videos != 15:  # 15 is default
#                     print(f"\n🔢 Count Parsing: Successfully limited to {max_videos} videos")
            
#             print("\n" + "="*60)
            
#         except KeyboardInterrupt:
#             print("\n👋 Goodbye!")
#             break
#         except Exception as e:
#             print(f"❌ Unexpected error: {e}")
#             continue


# if __name__ == "__main__":
#     main()