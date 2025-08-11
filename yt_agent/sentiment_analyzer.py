import random
from typing import List, Dict, Optional, Tuple
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv
import os

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

class Comment:
    """Class to represent a YouTube comment"""
    
    def __init__(self, text: str, like_count: int, reply_count: int, author: str, published_at: str):
        self.text = text
        self.like_count = like_count
        self.reply_count = reply_count
        self.author = author
        self.published_at = published_at
    
    def get_interaction_score(self) -> int:
        """Calculate interaction score based on likes and replies"""
        return self.like_count + self.reply_count


class SentimentResult:
    """Class to represent sentiment analysis results"""
    
    def __init__(self, positive: float, negative: float, neutral: float, compound: float):
        self.positive = positive
        self.negative = negative
        self.neutral = neutral
        self.compound = compound
    
    def get_label(self) -> str:
        """Get sentiment label based on compound score"""
        if self.compound >= 0.05:
            return "Positive"
        elif self.compound <= -0.05:
            return "Negative"
        else:
            return "Neutral"


class YouTubeCommentSentimentAnalyzer:
    """
    A class to analyze sentiment of YouTube video comments using VADER sentiment analysis.
    
    Features:
    - Processes all comments if <= 50 comments
    - For >50 comments, analyzes 20% of comments with highest interactions
    - Uses free VADER sentiment analysis
    - Provides detailed sentiment breakdown
    """
    
    def __init__(self, api_key=YOUTUBE_API_KEY):
        """
        Initialize the analyzer with YouTube API key.
        
        Args:
            api_key (str): YouTube Data API v3 key
        """
        self.api_key = api_key
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
    
    def extract_video_id(self, url: str) -> Optional[str]:
        """
        Extract video ID from YouTube URL.
        
        Args:
            url (str): YouTube video URL
            
        Returns:
            Optional[str]: Video ID if valid URL, None otherwise
        """
        if "youtube.com/watch?v=" in url:
            return url.split("watch?v=")[1].split("&")[0]
        elif "youtu.be/" in url:
            return url.split("youtu.be/")[1].split("?")[0]
        elif len(url) == 11:  # Direct video ID
            return url
        return None
    
    def fetch_comments(self, video_id: str, max_results: int = 100) -> List[Comment]:
        """
        Fetch comments from a YouTube video.
        
        Args:
            video_id (str): YouTube video ID
            max_results (int): Maximum number of comments to fetch
            
        Returns:
            List[Comment]: List of Comment objects
        """
        comments = []
        
        try:
            # Get video comments
            request = self.youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=min(max_results, 100),  # API limit is 100 per request
                order='relevance'  # Get most relevant comments first
            )
            
            while request and len(comments) < max_results:
                response = request.execute()
                
                for item in response['items']:
                    snippet = item['snippet']['topLevelComment']['snippet']
                    comment = Comment(
                        text=snippet['textDisplay'],
                        like_count=snippet.get('likeCount', 0),
                        reply_count=item['snippet'].get('totalReplyCount', 0),
                        author=snippet['authorDisplayName'],
                        published_at=snippet['publishedAt']
                    )
                    comments.append(comment)
                
                # Check if there are more pages
                request = self.youtube.commentThreads().list_next(request, response)
                if not request:
                    break
                    
        except HttpError as e:
            print(f"An error occurred while fetching comments: {e}")
            return []
        
        return comments[:max_results]
    
    def filter_comments(self, comments: List[Comment]) -> List[Comment]:
        """
        Filter comments based on the rules:
        - If <= 50 comments: return all
        - If > 50 comments: return top 20% by interaction score
        
        Args:
            comments (List[Comment]): List of all comments
            
        Returns:
            List[Comment]: Filtered list of comments
        """
        if len(comments) <= 50:
            return comments
        
        # Sort by interaction score (likes + replies) in descending order
        sorted_comments = sorted(comments, key=lambda x: x.get_interaction_score(), reverse=True)
        
        # Take top 20%
        top_20_percent = max(1, int(len(comments) * 0.2))
        return sorted_comments[:top_20_percent]
    
    def analyze_sentiment(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of a single text using VADER.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            SentimentResult: Sentiment analysis results
        """
        scores = self.sentiment_analyzer.polarity_scores(text)
        return SentimentResult(
            positive=scores['pos'],
            negative=scores['neg'],
            neutral=scores['neu'],
            compound=scores['compound']
        )
    
    def analyze_video_comments(self, video_url_or_id: str) -> Dict:
        """
        Analyze sentiment of YouTube video comments.
        
        Args:
            video_url_or_id (str): YouTube video URL or video ID
            
        Returns:
            Dict: Analysis results including overall sentiment and individual comment analysis
        """
        # Extract video ID
        video_id = self.extract_video_id(video_url_or_id)
        if not video_id:
            return {"error": "Invalid YouTube URL or video ID"}
        
        # Fetch comments
        print("Fetching comments...")
        all_comments = self.fetch_comments(video_id, max_results=200)
        
        if not all_comments:
            return {"error": "No comments found or unable to fetch comments"}
        
        # Filter comments based on rules
        comments_to_analyze = self.filter_comments(all_comments)
        
        print(f"Total comments fetched: {len(all_comments)}")
        print(f"Comments to analyze: {len(comments_to_analyze)}")
        
        # Analyze sentiment for each comment
        comment_sentiments = []
        sentiment_scores = []
        
        for comment in comments_to_analyze:
            sentiment = self.analyze_sentiment(comment.text)
            comment_sentiments.append({
                'text': comment.text[:100] + "..." if len(comment.text) > 100 else comment.text,
                'author': comment.author,
                'likes': comment.like_count,
                'replies': comment.reply_count,
                'sentiment': sentiment.get_label(),
                'compound_score': round(sentiment.compound, 3)
            })
            sentiment_scores.append(sentiment.compound)
        
        # Calculate overall statistics
        positive_count = sum(1 for score in sentiment_scores if score >= 0.05)
        negative_count = sum(1 for score in sentiment_scores if score <= -0.05)
        neutral_count = len(sentiment_scores) - positive_count - negative_count
        
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        
        # Determine overall sentiment
        if avg_sentiment >= 0.05:
            overall_sentiment = "Positive"
        elif avg_sentiment <= -0.05:
            overall_sentiment = "Negative"
        else:
            overall_sentiment = "Neutral"
        
        return {
            "video_id": video_id,
            "total_comments_fetched": len(all_comments),
            "comments_analyzed": len(comments_to_analyze),
            "analysis_method": "All comments" if len(all_comments) <= 50 else "Top 20% by interactions",
            "overall_sentiment": overall_sentiment,
            "average_compound_score": round(avg_sentiment, 3),
            "sentiment_distribution": {
                "positive": positive_count,
                "negative": negative_count,
                "neutral": neutral_count,
                "positive_percentage": round((positive_count / len(sentiment_scores)) * 100, 1),
                "negative_percentage": round((negative_count / len(sentiment_scores)) * 100, 1),
                "neutral_percentage": round((neutral_count / len(sentiment_scores)) * 100, 1)
            },
            "individual_comments": comment_sentiments
        }


