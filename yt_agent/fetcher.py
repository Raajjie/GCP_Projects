import os
import csv
from datetime import datetime, timezone
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import firestore
import re

load_dotenv()  # Load environment variables if using .env file
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

class YouTubeShortsFetcher:
    def __init__(self, api_key=YOUTUBE_API_KEY):
        """Initialize the YouTube API client and Firestore using Google Cloud default credentials."""
        self.api_key = api_key
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        
        # Initialize Firebase Admin SDK with Google Cloud default credentials
        try:
            if not firebase_admin._apps:
                # Use Google Cloud default credentials (ADC - Application Default Credentials)
                firebase_admin.initialize_app()
            
            self.db = firestore.client()
            print("✅ Firestore initialized successfully with Google Cloud default credentials")
        except Exception as e:
            print(f"❌ Error initializing Firestore: {e}")
            print("💡 Make sure you're running in a Google Cloud environment or have gcloud auth application-default login configured")
            self.db = None
    
    def generate_collection_name(self, region_code, search_type, timestamp=None):
        """Generate a meaningful collection name for Firestore."""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Clean region code and search type
        region_clean = region_code.lower()
        search_clean = search_type.replace(' ', '_').replace('#', '').lower()
        
        return f"youtube_shorts_{region_clean}_{search_clean}_{timestamp}"
    
    def check_recent_search(self, region_code, search_type, hours_threshold=1):
        """
        Check if a similar search was performed recently (within threshold hours).
        
        Args:
            region_code (str): Region code to check
            search_type (str): Search type to check
            hours_threshold (int): Minimum hours between searches (default: 1)
        
        Returns:
            tuple: (bool: is_recent, str: collection_name if recent, datetime: last_search_time)
        """
        if not self.db:
            print("⚠️  Firestore not initialized - will proceed with API call")
            return False, None, None
        
        try:
            # Clean search parameters for matching
            region_clean = region_code.lower()
            search_clean = search_type.replace(' ', '_').replace('#', '').lower()
            
            # Search pattern for matching collections
            search_pattern = f"youtube_shorts_{region_clean}_{search_clean}_"
            
            print(f"🔍 Checking for recent collections matching pattern: {search_pattern}*")
            
            # Get all collections and filter client-side (Firestore doesn't support prefix queries on collection names)
            collections = self.db.collections()
            matching_collections = []
            
            for collection in collections:
                if collection.id.startswith(search_pattern):
                    try:
                        # Get metadata document
                        metadata_doc = collection.document('_metadata').get()
                        if metadata_doc.exists:
                            metadata = metadata_doc.to_dict()
                            created_at = metadata.get('created_at')
                            
                            if created_at:
                                # Convert Firestore timestamp to datetime with timezone awareness
                                if hasattr(created_at, 'timestamp'):
                                    created_datetime = datetime.fromtimestamp(created_at.timestamp(), tz=timezone.utc)
                                else:
                                    # Fallback: parse from collection name timestamp (assume UTC)
                                    timestamp_str = collection.id.split('_')[-2] + '_' + collection.id.split('_')[-1]
                                    created_datetime = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
                                
                                matching_collections.append({
                                    'name': collection.id,
                                    'created_at': created_datetime,
                                    'metadata': metadata
                                })
                                print(f"📁 Found matching collection: {collection.id} (created: {created_datetime})")
                    except Exception as e:
                        print(f"⚠️  Could not process collection {collection.id}: {e}")
                        continue
            
            if not matching_collections:
                print("📭 No matching collections found - new search will be performed")
                return False, None, None
            
            # Sort by creation time (most recent first)
            matching_collections.sort(key=lambda x: x['created_at'], reverse=True)
            most_recent = matching_collections[0]
            
            # Calculate time difference using timezone-aware datetime
            now = datetime.now(timezone.utc)
            time_diff = now - most_recent['created_at']
            hours_diff = time_diff.total_seconds() / 3600
            
            print(f"⏰ Most recent collection: {most_recent['name']}")
            print(f"   Created: {most_recent['created_at']}")
            print(f"   Time ago: {hours_diff:.2f} hours")
            print(f"   Threshold: {hours_threshold} hours")
            
            if hours_diff < hours_threshold:
                print(f"🔄 Recent search found within {hours_threshold}h threshold - using cached data")
                return True, most_recent['name'], most_recent['created_at']
            else:
                print(f"🆕 Last search was {hours_diff:.2f} hours ago (≥ {hours_threshold}h threshold) - will perform new search")
                return False, most_recent['name'], most_recent['created_at']
            
        except Exception as e:
            print(f"❌ Error checking recent searches: {e}")
            print("⚠️  Proceeding with new search due to error")
            return False, None, None
    
    def generate_document_id(self, video_id, rank):
        """Generate a meaningful document ID for Firestore."""
        return f"rank_{rank:03d}_{video_id}"
    
    def save_to_firestore(self, shorts, region_code='US', search_type='trending', collection_name=None):
        """Save trending shorts to Firestore."""
        if not self.db:
            print("❌ Firestore not initialized. Cannot save data.")
            return None
        
        if not shorts:
            print("❌ No shorts data to save.")
            return None
        
        try:
            # Generate collection name if not provided
            if collection_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                collection_name = self.generate_collection_name(region_code, search_type, timestamp)
            
            print(f"💾 Saving {len(shorts)} shorts to Firestore collection: {collection_name}")
            
            collection_ref = self.db.collection(collection_name)
            
            # Save metadata about the search as a special document with timezone-aware timestamp
            search_metadata = {
                'collection_type': 'youtube_shorts_trending',
                'region_code': region_code,
                'search_type': search_type,
                'total_videos': len(shorts),
                'created_at': firestore.SERVER_TIMESTAMP,  # This will be timezone-aware
                'api_used': 'youtube_data_api_v3',
                'max_duration_seconds': 60,
                'search_timestamp': datetime.now(timezone.utc).isoformat(),
                'hours_threshold_enforced': True
            }
            
            collection_ref.document('_metadata').set(search_metadata)
            
            # Save each video as a separate document
            batch = self.db.batch()
            saved_count = 0
            
            for rank, short in enumerate(shorts, 1):
                doc_id = self.generate_document_id(short['video_id'], rank)
                doc_ref = collection_ref.document(doc_id)
                
                # Prepare video data for Firestore
                video_data = {
                    'rank': rank,
                    'video_id': short['video_id'],
                    'title': short['title'],
                    'channel_title': short['channel_title'],
                    'published_at': short['published_at'],
                    'description': short['description'],
                    'thumbnail_url': short['thumbnail_url'],
                    'video_url': short['video_url'],
                    'shorts_url': short['shorts_url'],
                    'duration_seconds': short['duration_seconds'],
                    'duration_iso': short['duration'],
                    'tags': short['tags'],
                    'statistics': {
                        'view_count': int(short['view_count']) if short['view_count'] != 'N/A' and str(short['view_count']).isdigit() else None,
                        'like_count': int(short['like_count']) if short['like_count'] != 'N/A' and str(short['like_count']).isdigit() else None,
                        'comment_count': int(short['comment_count']) if short['comment_count'] != 'N/A' and str(short['comment_count']).isdigit() else None,
                        'view_count_raw': str(short['view_count']),
                        'like_count_raw': str(short['like_count']),
                        'comment_count_raw': str(short['comment_count'])
                    },
                    'metadata': {
                        'saved_at': firestore.SERVER_TIMESTAMP,
                        'region_code': region_code,
                        'search_type': search_type
                    }
                }
                
                batch.set(doc_ref, video_data)
                saved_count += 1
                
                # Commit batch every 500 documents (Firestore limit)
                if saved_count % 500 == 0:
                    batch.commit()
                    batch = self.db.batch()
                    print(f"   Committed batch of {saved_count} documents...")
            
            # Commit remaining documents
            if saved_count % 500 != 0:
                batch.commit()
            
            print(f"✅ Successfully saved {saved_count} YouTube Shorts to Firestore collection: {collection_name}")
            print(f"📁 Collection path: {collection_name}")
            print(f"📊 Documents: _metadata + {saved_count} video documents")
            
            return collection_name
            
        except Exception as e:
            print(f"❌ Error saving to Firestore: {e}")
            return None
    
    def get_from_firestore(self, collection_name, limit=None):
        """Retrieve YouTube Shorts data from Firestore."""
        if not self.db:
            print("❌ Firestore not initialized. Cannot retrieve data.")
            return []
        
        try:
            collection_ref = self.db.collection(collection_name)
            
            # Get all documents except metadata
            query = collection_ref.where('rank', '>=', 1)
            if limit:
                query = query.limit(limit)
            
            docs = query.order_by('rank').stream()
            
            shorts = []
            for doc in docs:
                doc_data = doc.to_dict()
                # Convert back to original format for compatibility
                short = {
                    'video_id': doc_data.get('video_id'),
                    'title': doc_data.get('title'),
                    'channel_title': doc_data.get('channel_title'),
                    'published_at': doc_data.get('published_at'),
                    'description': doc_data.get('description'),
                    'thumbnail_url': doc_data.get('thumbnail_url'),
                    'video_url': doc_data.get('video_url'),
                    'shorts_url': doc_data.get('shorts_url'),
                    'view_count': doc_data.get('statistics', {}).get('view_count_raw', 'N/A'),
                    'like_count': doc_data.get('statistics', {}).get('like_count_raw', 'N/A'),
                    'comment_count': doc_data.get('statistics', {}).get('comment_count_raw', 'N/A'),
                    'duration': doc_data.get('duration_iso'),
                    'duration_seconds': doc_data.get('duration_seconds'),
                    'tags': doc_data.get('tags', [])
                }
                shorts.append(short)
            
            print(f"✅ Retrieved {len(shorts)} YouTube Shorts from collection: {collection_name}")
            return shorts
            
        except Exception as e:
            print(f"❌ Error retrieving from Firestore: {e}")
            return []
    
    def list_collections(self, prefix='youtube_shorts_'):
        """List all YouTube Shorts collections in Firestore."""
        if not self.db:
            print("❌ Firestore not initialized.")
            return []
        
        try:
            collections = self.db.collections()
            youtube_collections = []
            
            for collection in collections:
                if collection.id.startswith(prefix):
                    # Get metadata document for additional info
                    try:
                        metadata_doc = collection.document('_metadata').get()
                        if metadata_doc.exists:
                            metadata = metadata_doc.to_dict()
                            created_at = metadata.get('created_at')
                            
                            # Format creation time
                            created_at_str = "Unknown"
                            if created_at and hasattr(created_at, 'timestamp'):
                                created_datetime = datetime.fromtimestamp(created_at.timestamp(), tz=timezone.utc)
                                created_at_str = created_datetime.strftime("%Y-%m-%d %H:%M:%S UTC")
                            
                            collection_info = {
                                'name': collection.id,
                                'total_videos': metadata.get('total_videos', 0),
                                'region_code': metadata.get('region_code', 'Unknown'),
                                'search_type': metadata.get('search_type', 'Unknown'),
                                'created_at': created_at,
                                'created_at_str': created_at_str
                            }
                            youtube_collections.append(collection_info)
                        else:
                            # Collection exists but no metadata
                            youtube_collections.append({
                                'name': collection.id,
                                'total_videos': 'Unknown',
                                'region_code': 'Unknown',
                                'search_type': 'Unknown',
                                'created_at': None,
                                'created_at_str': 'Unknown'
                            })
                    except Exception as e:
                        print(f"⚠️  Could not read metadata for collection {collection.id}: {e}")
                        youtube_collections.append({
                            'name': collection.id,
                            'total_videos': 'Error',
                            'region_code': 'Error',
                            'search_type': 'Error',
                            'created_at': None,
                            'created_at_str': 'Error'
                        })
            
            # Sort by creation time (most recent first)
            youtube_collections.sort(key=lambda x: x['created_at'] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
            
            return youtube_collections
            
        except Exception as e:
            print(f"❌ Error listing collections: {e}")
            return []
    
    def parse_duration_seconds(self, duration):
        """Convert ISO 8601 duration to seconds."""
        # Parse ISO 8601 duration (e.g., PT4M13S, PT30S)
        pattern = re.compile(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?')
        match = pattern.match(duration)
        
        if not match:
            return 0
        
        hours, minutes, seconds = match.groups()
        
        total_seconds = 0
        if hours:
            total_seconds += int(hours) * 3600
        if minutes:
            total_seconds += int(minutes) * 60
        if seconds:
            total_seconds += int(seconds)
        
        return total_seconds
    
    def get_trending_shorts(self, region_code='US', max_results=50, search_term=None, save_to_firestore=True, force_new_search=False, hours_threshold=1):
        """
        Fetch trending YouTube Shorts (videos ≤ 60 seconds) and optionally save to Firestore.
        ENFORCES 1-HOUR RULE: Only creates new Firestore collection if last one was created > 1 hour ago.
        
        Args:
            region_code (str): ISO 3166-1 alpha-2 country code (e.g., 'US', 'GB', 'IN')
            max_results (int): Maximum number of results to return (1-50)
            search_term (str): Optional search term. If None, uses trending videos directly
            save_to_firestore (bool): Whether to automatically save to Firestore
            force_new_search (bool): Force new search even if recent data exists
            hours_threshold (int): Minimum hours between similar searches (default: 1)
        
        Returns:
            tuple: (list of shorts, collection_name, bool: is_from_cache)
        """
        # Determine search type based on whether search_term is provided
        if search_term:
            search_type = f"search_{search_term.replace('#', '').replace(' ', '_')}"
            search_method = "search"
        else:
            search_type = 'trending_popular'
            search_method = "trending"
        
        print(f"🎯 Starting YouTube Shorts {search_method}:")
        print(f"   Region: {region_code}")
        if search_term:
            print(f"   Search term: {search_term}")
        else:
            print(f"   Method: Direct trending videos")
        print(f"   Max results: {max_results}")
        print(f"   Hours threshold: {hours_threshold}")
        print(f"   Force new search: {force_new_search}")
        print(f"   Save to Firestore: {save_to_firestore}")
        
        # ENFORCED 1-HOUR RULE: Check for recent searches unless forced
        if save_to_firestore and not force_new_search:
            print(f"\n🔍 Checking for recent searches (within {hours_threshold} hour(s))...")
            is_recent, recent_collection, last_search_time = self.check_recent_search(
                region_code, search_type, hours_threshold
            )
            
            if is_recent and recent_collection:
                print(f"♻️  USING CACHED DATA from recent collection: {recent_collection}")
                print(f"💡 To force a new search, use force_new_search=True")
                cached_shorts = self.get_from_firestore(recent_collection, limit=max_results)
                return cached_shorts, recent_collection, True
        elif force_new_search:
            print(f"🔄 FORCED NEW SEARCH - bypassing 1-hour rule")
        
        print(f"🆕 Performing new YouTube API {search_method}...")
        
        try:
            video_ids = []
            
            if search_term:
                # Method 1: Use search with custom search term
                print(f"🔍 Searching for: {search_term}")
                search_request = self.youtube.search().list(
                    part='snippet',
                    q=search_term,
                    type='video',
                    order='viewCount',
                    regionCode=region_code,
                    maxResults=max_results * 2,  # Get more to filter for duration
                    videoDuration='short',  # This helps filter for shorter videos
                    publishedAfter=(datetime.now().replace(day=1).isoformat() + 'Z')  # This month
                )
                
                search_response = search_request.execute()
                video_ids = [item['id']['videoId'] for item in search_response['items']]
                
            else:
                # Method 2: Get trending videos directly (most popular)
                print(f"📈 Getting trending videos...")
                trending_request = self.youtube.videos().list(
                    part='snippet,statistics,contentDetails',
                    chart='mostPopular',
                    regionCode=region_code,
                    maxResults=50,  # Get max to have more shorts to choose from
                    videoCategoryId='24'  # Entertainment category - more likely to have shorts
                )
                
                trending_response = trending_request.execute()
                
                # For trending method, we already have the full video data
                shorts = []
                for item in trending_response['items']:
                    duration_seconds = self.parse_duration_seconds(item['contentDetails']['duration'])
                    
                    # Only include videos that are 60 seconds or less
                    if duration_seconds <= 60 and duration_seconds > 0:
                        video_info = {
                            'video_id': item['id'],
                            'title': item['snippet']['title'],
                            'channel_title': item['snippet']['channelTitle'],
                            'published_at': item['snippet']['publishedAt'],
                            'description': item['snippet']['description'][:200] + '...' if len(item['snippet']['description']) > 200 else item['snippet']['description'],
                            'thumbnail_url': item['snippet']['thumbnails']['medium']['url'],
                            'video_url': f"https://www.youtube.com/watch?v={item['id']}",
                            'shorts_url': f"https://www.youtube.com/shorts/{item['id']}",
                            'view_count': item['statistics'].get('viewCount', 'N/A'),
                            'like_count': item['statistics'].get('likeCount', 'N/A'),
                            'comment_count': item['statistics'].get('commentCount', 'N/A'),
                            'duration': item['contentDetails']['duration'],
                            'duration_seconds': duration_seconds,
                            'tags': item['snippet'].get('tags', [])
                        }
                        shorts.append(video_info)
                    
                    if len(shorts) >= max_results:
                        break
                
                print(f"⏱️  Found {len(shorts)} trending shorts (≤60s) from popular videos")
                
                # Save to Firestore if requested - ONLY if 1-hour rule allows it
                collection_name = None
                if save_to_firestore and shorts:
                    print(f"💾 Saving fresh trending data to new Firestore collection...")
                    collection_name = self.save_to_firestore(shorts, region_code, search_type)
                    if collection_name:
                        print(f"✅ Fresh trending data saved to new collection: {collection_name}")
                
                return shorts, collection_name, False
            
            # If we used search method, continue with getting detailed video information
            if not video_ids:
                print("❌ No videos found in search results")
                return [], None, False
            
            print(f"📺 Found {len(video_ids)} videos, getting detailed info...")
            
            # Get detailed video information
            videos_request = self.youtube.videos().list(
                part='snippet,statistics,contentDetails',
                id=','.join(video_ids)
            )
            
            videos_response = videos_request.execute()
            
            # Filter for actual shorts (≤ 60 seconds) and process
            shorts = []
            for item in videos_response['items']:
                duration_seconds = self.parse_duration_seconds(item['contentDetails']['duration'])
                
                # Only include videos that are 60 seconds or less
                if duration_seconds <= 60 and duration_seconds > 0:
                    video_info = {
                        'video_id': item['id'],
                        'title': item['snippet']['title'],
                        'channel_title': item['snippet']['channelTitle'],
                        'published_at': item['snippet']['publishedAt'],
                        'description': item['snippet']['description'][:200] + '...' if len(item['snippet']['description']) > 200 else item['snippet']['description'],
                        'thumbnail_url': item['snippet']['thumbnails']['medium']['url'],
                        'video_url': f"https://www.youtube.com/watch?v={item['id']}",
                        'shorts_url': f"https://www.youtube.com/shorts/{item['id']}",
                        'view_count': item['statistics'].get('viewCount', 'N/A'),
                        'like_count': item['statistics'].get('likeCount', 'N/A'),
                        'comment_count': item['statistics'].get('commentCount', 'N/A'),
                        'duration': item['contentDetails']['duration'],
                        'duration_seconds': duration_seconds,
                        'tags': item['snippet'].get('tags', [])
                    }
                    shorts.append(video_info)
                
                # Stop when we have enough shorts
                if len(shorts) >= max_results:
                    break
            
            if not shorts:
                print("❌ No shorts (≤60s videos) found after filtering")
                return [], None, False
            
            print(f"⏱️  Found {len(shorts)} shorts (≤60s) after filtering")
            
            # Sort by view count (descending)
            shorts.sort(key=lambda x: int(x['view_count']) if x['view_count'] != 'N/A' and str(x['view_count']).isdigit() else 0, reverse=True)
            
            shorts = shorts[:max_results]
            
            # Save to Firestore if requested - ONLY if 1-hour rule allows it
            collection_name = None
            if save_to_firestore and shorts:
                print(f"💾 Saving fresh data to new Firestore collection (1-hour rule enforced)...")
                collection_name = self.save_to_firestore(shorts, region_code, search_type)
                if collection_name:
                    print(f"✅ Fresh data saved to new collection: {collection_name}")
                else:
                    print(f"❌ Failed to save to Firestore")
            
            return shorts, collection_name, False
            
        except HttpError as e:
            print(f"❌ YouTube API HTTP error: {e}")
            return [], None, False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return [], None, False
        
    def get_trending_shorts_alternative(self, region_code='US', max_results=50, save_to_firestore=True, force_new_search=False, hours_threshold=1):
        """
        Alternative method: Get trending videos and filter for shorts, with Firestore integration and 1-hour rule enforcement.
        This might be more reliable for getting actual trending content.
        """
        search_type = 'trending_popular'
        
        print(f"🎯 Starting alternative trending search:")
        print(f"   Region: {region_code}")
        print(f"   Max results: {max_results}")
        print(f"   Hours threshold: {hours_threshold}")
        print(f"   Force new search: {force_new_search}")
        
        # ENFORCED 1-HOUR RULE: Check for recent searches unless forced
        if save_to_firestore and not force_new_search:
            print(f"\n🔍 Checking for recent trending searches (within {hours_threshold} hour(s))...")
            is_recent, recent_collection, last_search_time = self.check_recent_search(
                region_code, search_type, hours_threshold
            )
            
            if is_recent and recent_collection:
                print(f"♻️  USING CACHED TRENDING DATA from: {recent_collection}")
                print("💡 Use force_new_search=True to fetch fresh data")
                cached_shorts = self.get_from_firestore(recent_collection, limit=max_results)
                return cached_shorts, recent_collection, True
        
        print(f"🆕 Performing new YouTube API trending search...")
        
        try:
            # Get trending videos
            request = self.youtube.videos().list(
                part='snippet,statistics,contentDetails',
                chart='mostPopular',
                regionCode=region_code,
                maxResults=50  # Get max to have more shorts to choose from
            )
            
            response = request.execute()
            
            # Filter for shorts (≤ 60 seconds)
            shorts = []
            for item in response['items']:
                duration_seconds = self.parse_duration_seconds(item['contentDetails']['duration'])
                
                # Only include videos that are 60 seconds or less
                if duration_seconds <= 60 and duration_seconds > 0:
                    video_info = {
                        'video_id': item['id'],
                        'title': item['snippet']['title'],
                        'channel_title': item['snippet']['channelTitle'],
                        'published_at': item['snippet']['publishedAt'],
                        'description': item['snippet']['description'][:200] + '...' if len(item['snippet']['description']) > 200 else item['snippet']['description'],
                        'thumbnail_url': item['snippet']['thumbnails']['medium']['url'],
                        'video_url': f"https://www.youtube.com/watch?v={item['id']}",
                        'shorts_url': f"https://www.youtube.com/shorts/{item['id']}",
                        'view_count': item['statistics'].get('viewCount', 'N/A'),
                        'like_count': item['statistics'].get('likeCount', 'N/A'),
                        'comment_count': item['statistics'].get('commentCount', 'N/A'),
                        'duration': item['contentDetails']['duration'],
                        'duration_seconds': duration_seconds,
                        'tags': item['snippet'].get('tags', [])
                    }
                    shorts.append(video_info)
                
                if len(shorts) >= max_results:
                    break
            
            print(f"⏱️  Found {len(shorts)} trending shorts (≤60s)")
            
            # Save to Firestore if requested - ONLY if 1-hour rule allows it
            collection_name = None
            if save_to_firestore and shorts:
                print(f"💾 Saving fresh trending data to new Firestore collection...")
                collection_name = self.save_to_firestore(shorts, region_code, search_type)
                if collection_name:
                    print(f"✅ Fresh trending data saved to new collection: {collection_name}")
            
            return shorts, collection_name, False
            
        except HttpError as e:
            print(f"❌ YouTube API HTTP error: {e}")
            return [], None, False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return [], None, False
    
    def get_video_categories(self, region_code='US'):
        """
        Get available video categories for a specific region.
        
        Args:
            region_code (str): ISO 3166-1 alpha-2 country code
        
        Returns:
            dict: Dictionary of category IDs and names
        """
        try:
            request = self.youtube.videoCategories().list(
                part='snippet',
                regionCode=region_code
            )
            response = request.execute()
            
            categories = {}
            for item in response['items']:
                categories[item['id']] = item['snippet']['title']
            
            return categories
            
        except HttpError as e:
            print(f"❌ YouTube API HTTP error: {e}")
            return {}
    
    def format_duration(self, duration):
        """Convert ISO 8601 duration to readable format."""
        # Parse ISO 8601 duration (e.g., PT4M13S)
        pattern = re.compile(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?')
        match = pattern.match(duration)
        
        if not match:
            return duration
        
        hours, minutes, seconds = match.groups()
        
        parts = []
        if hours:
            parts.append(f"{hours}h")
        if minutes:
            parts.append(f"{minutes}m")
        if seconds:
            parts.append(f"{seconds}s")
        
        return " ".join(parts) if parts else "0s"
    
    def display_trending_shorts(self, shorts, show_details=True, auto_save_csv=True):
        """Display trending YouTube Shorts in a formatted way and optionally auto-save to CSV."""
        if not shorts:
            print("No trending YouTube Shorts found.")
            return
        
        # Auto-save to CSV if requested
        if auto_save_csv:
            self.save_to_csv(shorts, auto_save=True)
        
        print(f"\n{'='*80}")
        print(f"TRENDING YOUTUBE SHORTS ({len(shorts)} shorts)")
        print(f"{'='*80}")
        
        for i, short in enumerate(shorts, 1):
            print(f"\n{i}. {short['title']}")
            print(f"   Channel: {short['channel_title']}")
            
            # Format view count safely
            if short['view_count'] != 'N/A' and str(short['view_count']).isdigit():
                print(f"   Views: {int(short['view_count']):,}")
            else:
                print(f"   Views: {short['view_count']}")
            
            print(f"   Duration: {short['duration_seconds']}s")
            print(f"   Shorts URL: {short['shorts_url']}")
            print(f"   Regular URL: {short['video_url']}")
            
            if show_details:
                # Format like count safely
                if short['like_count'] != 'N/A' and str(short['like_count']).isdigit():
                    print(f"   Likes: {int(short['like_count']):,}")
                else:
                    print(f"   Likes: {short['like_count']}")
                
                # Format comment count safely
                if short['comment_count'] != 'N/A' and str(short['comment_count']).isdigit():
                    print(f"   Comments: {int(short['comment_count']):,}")
                else:
                    print(f"   Comments: {short['comment_count']}")
                
                print(f"   Published: {short['published_at']}")
                if short['description']:
                    print(f"   Description: {short['description']}")
            
            print("-" * 80)
    
    def save_to_csv(self, shorts, filename=None, auto_save=True):
        """Save trending shorts to a CSV file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trending_shorts_{timestamp}.csv"
        
        try:
            # Define CSV headers
            headers = [
                'rank',
                'video_id',
                'title',
                'channel_title',
                'view_count',
                'like_count',
                'comment_count',
                'duration_seconds',
                'published_at',
                'shorts_url',
                'video_url',
                'description',
                'tags'
            ]
            
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()
                
                for rank, short in enumerate(shorts, 1):
                    # Clean and prepare data for CSV
                    row = {
                        'rank': rank,
                        'video_id': short['video_id'],
                        'title': short['title'].replace('\n', ' ').replace('\r', ''),
                        'channel_title': short['channel_title'],
                        'view_count': short['view_count'],
                        'like_count': short['like_count'],
                        'comment_count': short['comment_count'],
                        'duration_seconds': short['duration_seconds'],
                        'published_at': short['published_at'],
                        'shorts_url': short['shorts_url'],
                        'video_url': short['video_url'],
                        'description': short['description'].replace('\n', ' ').replace('\r', '') if short['description'] else '',
                        'tags': '|'.join(short['tags']) if short['tags'] else ''
                    }
                    writer.writerow(row)
            
            if auto_save:
                print(f"✅ CSV data automatically saved to {filename}")
            else:
                print(f"📁 CSV data saved to {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ Error saving CSV file: {e}")
            return None


# # Example usage and setup instructions
# if __name__ == "__main__":
#     """
#     Setup Instructions for Google Cloud:
    
#     1. Enable APIs in Google Cloud Console:
#        - YouTube Data API v3
#        - Firestore API
    
#     2. Set up authentication (choose one):
#        Option A - For Google Cloud environments (recommended):
#          - Cloud Shell, Cloud Functions, Compute Engine, etc. automatically have credentials
       
#        Option B - For local development:
#          - Install gcloud CLI: https://cloud.google.com/sdk/docs/install
#          - Run: gcloud auth application-default login
#          - Run: gcloud config set project YOUR_PROJECT_ID
    
#     3. Environment Variables:
#        Only YOUTUBE_API_KEY is required in .env file:
#        YOUTUBE_API_KEY=your_youtube_api_key_here
    
#     4. Install dependencies:
#        pip install google-api-python-client python-dotenv firebase-admin
#     """
    
#     # Initialize the fetcher
#     fetcher = YouTubeShortsFirestoreFetcher()
    
#     # Example 1: Test the 1-HOUR RULE enforcement
#     print("="*80)
#     print("TESTING 1-HOUR RULE ENFORCEMENT")
#     print("="*80)
    
#     # First search - should create new collection
#     print("\n1️⃣ FIRST SEARCH (should create new collection):")
#     shorts1, collection1, from_cache1 = fetcher.get_trending_shorts(
#         region_code='US', 
#         max_results=10, 
#         search_term='#shorts',
#         save_to_firestore=True,
#         hours_threshold=1  # 1 hour rule
#     )
    
#     if shorts1:
#         print(f"✅ First search completed")
#         print(f"   From cache: {from_cache1}")
#         print(f"   Collection: {collection1}")
#         print(f"   Shorts found: {len(shorts1)}")
#     else:
#         print("❌ First search failed")
    
#     # Second search immediately after - should use cached data (1-hour rule)
#     print(f"\n2️⃣ IMMEDIATE SECOND SEARCH (should use cached data due to 1-hour rule):")
#     shorts2, collection2, from_cache2 = fetcher.get_trending_shorts(
#         region_code='US', 
#         max_results=10, 
#         search_term='#shorts',
#         save_to_firestore=True,
#         hours_threshold=1  # Same 1 hour rule
#     )
    
#     if shorts2:
#         print(f"✅ Second search completed")
#         print(f"   From cache: {from_cache2}")
#         print(f"   Collection: {collection2}")
#         print(f"   Shorts found: {len(shorts2)}")
        
#         if from_cache2:
#             print("✅ 1-HOUR RULE WORKING: Used cached data as expected!")
#         else:
#             print("⚠️  WARNING: 1-hour rule may not be working properly")
#     else:
#         print("❌ Second search failed")
    
#     # Third search with force flag - should create new collection despite 1-hour rule
#     print(f"\n3️⃣ FORCED SEARCH (should bypass 1-hour rule and create new collection):")
#     shorts3, collection3, from_cache3 = fetcher.get_trending_shorts(
#         region_code='US',
#         max_results=10,
#         search_term='#shorts',
#         save_to_firestore=True,
#         force_new_search=True  # This bypasses the 1-hour rule
#     )
    
#     if shorts3:
#         print(f"✅ Forced search completed")
#         print(f"   From cache: {from_cache3}")
#         print(f"   Collection: {collection3}")
#         print(f"   Shorts found: {len(shorts3)}")
        
#         if not from_cache3 and collection3 != collection1:
#             print("✅ FORCE FLAG WORKING: Created new collection as expected!")
#         else:
#             print("⚠️  WARNING: Force flag may not be working properly")
#     else:
#         print("❌ Forced search failed")
    
#     # Display results from the most recent successful search
#     if shorts3:
#         print(f"\n" + "="*80)
#         print("SAMPLE RESULTS FROM FORCED SEARCH:")
#         print("="*80)
#         fetcher.display_trending_shorts(shorts3[:5], show_details=True, auto_save_csv=False)
    
#     # Example 4: List all collections to verify 1-hour rule created appropriate collections
#     print("\n" + "="*80)
#     print("ALL YOUTUBE SHORTS COLLECTIONS (verifying 1-hour rule):")
#     print("="*80)
#     collections = fetcher.list_collections()
    
#     if collections:
#         for i, col in enumerate(collections, 1):
#             print(f"{i}. {col['name']}")
#             print(f"   Videos: {col['total_videos']}")
#             print(f"   Region: {col['region_code']}")
#             print(f"   Type: {col['search_type']}")
#             print(f"   Created: {col['created_at_str']}")
#             print("-" * 60)
        
#         print(f"\n📊 SUMMARY:")
#         print(f"   Total collections: {len(collections)}")
#         print(f"   Expected: 2 collections (first search + forced search)")
#         print(f"   1-hour rule prevented: 1 duplicate collection")
        
#         if len(collections) >= 2:
#             print("✅ 1-HOUR RULE VERIFICATION: Working correctly!")
#         else:
#             print("⚠️  1-hour rule verification inconclusive")
#     else:
#         print("❌ No collections found")
    
#     print(f"\n" + "="*80)
#     print("1-HOUR RULE TEST COMPLETED")
#     print("="*80)
#     print("Key improvements made:")
#     print("• Enhanced error handling and logging")
#     print("• Proper timezone-aware datetime handling")
#     print("• More robust data type checking")
#     print("• Better collection pattern matching")
#     print("• Clearer status messages and verification")
#     print("• Enforced 1-hour rule with detailed feedback")