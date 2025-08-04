from langchain.tools import BaseTool
from typing import Optional, Type, Dict, List, Optional, Type, Any, Union
from pydantic import BaseModel, Field
from google.cloud import firestore
from dotenv import load_dotenv
import time
import os
import asyncio
import requests
from duckduckgo_search import DDGS
from datetime import datetime
import json


from docx import Document
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import re
from pathlib import Path
from langchain_core.callbacks.manager import CallbackManager




#=================================================================
# Initialize Firestore Client and APIs
#=================================================================

load_dotenv()
db = firestore.Client() 
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")



#=================================================================
# Initialize Agent Tools
#=================================================================


# class AddtoFirestoreInput(BaseModel):
#     name: str = Field(description="The name of the document to add")
#     link: str = Field(description="The link of the document to add")
#     tags: list[str] = Field(description="The tags of the document to add")
#     description: str = Field(description="The description of the document to add")

# class AddtoFirestore(BaseTool):
#     name: str = "add_to_firestore"
#     description: str = "Add a new document to the firestore database"

#     args_schema: Optional[Type[BaseModel]] = AddtoFirestoreInput

#     def _run(self, name: str, link: str, tags: list[str], description: str) -> str:
#         try:

#             # Prepare the document data
#             doc_data = {
#                 "name": name,
#                 "link": link,
#                 "tags": tags,
#                 "description": description,
#             }

#             # Add to Firestore under a collection, e.g., "resources"
#             doc_ref = db.collection("resources").document()
#             doc_ref.set(doc_data)

#             return f"✅ Document '{name}' added successfully with ID: {doc_ref.id}"
#         except Exception as e:
#             return f"❌ Failed to add document: {str(e)}"

#     async def _arun(self, name: str, link: str, tags: list[str], description: str) -> str:
#         return self._run(name, link, tags, description)


class AddtoFirestoreInput(BaseModel):
    name: str = Field(description="The name of the document to add")
    link: str = Field(default="", description="The link of the document to add (optional for lecture notes)")
    tags: List[str] = Field(description="The tags of the document to add")
    description: str = Field(description="The description of the document to add")
    content_type: str = Field(default="document", description="Type of content: 'document' or 'lecture_notes'")
    lecture_content: Optional[str] = Field(default=None, description="Full lecture notes content (markdown format)")
    course_title: Optional[str] = Field(default=None, description="Course title for lecture notes")
    topic: Optional[str] = Field(default=None, description="Topic for lecture notes")
    key_concepts: Optional[List[str]] = Field(default=None, description="Key concepts extracted from lecture notes")
    search_query: Optional[str] = Field(default=None, description="Original search query used for lecture notes")

class AddtoFirestore(BaseTool):
    name: str = "add_to_firestore"
    description: str = """
    Add a new document or lecture notes to the firestore database.
    
    For regular documents:
    - Provide name, link, tags, and description
    - Set content_type to 'document' (default)
    
    For lecture notes:
    - Provide name, tags, description, and lecture_content
    - Set content_type to 'lecture_notes'
    - Optionally include course_title, topic, key_concepts, and search_query
    - Link is optional for lecture notes
    """

    args_schema: Optional[Type[BaseModel]] = AddtoFirestoreInput

    def _extract_metadata_from_lecture_notes(self, lecture_content: str) -> dict:
        """Extract metadata from lecture notes content"""
        metadata = {}
        lines = lecture_content.split('\n')
        
        for line in lines[:20]:  # Check first 20 lines for metadata
            line = line.strip()
            if line.startswith('**Generated:**'):
                metadata['generated_date'] = line.replace('**Generated:**', '').strip()
            elif line.startswith('**Based on Search:**'):
                metadata['search_query'] = line.replace('**Based on Search:**', '').strip()
            elif line.startswith('**Total Sections:**'):
                try:
                    metadata['total_sections'] = int(line.replace('**Total Sections:**', '').strip())
                except ValueError:
                    pass
            elif line.startswith('# '):
                metadata['course_title'] = line.replace('# ', '').strip()
            elif line.startswith('## Topic:'):
                metadata['topic'] = line.replace('## Topic:', '').strip()
        
        return metadata

    def _run(self, name: str, link: str = "", tags: List[str] = [], description: str = "", 
             content_type: str = "document", lecture_content: Optional[str] = None,
             course_title: Optional[str] = None, topic: Optional[str] = None,
             key_concepts: Optional[List[str]] = None, search_query: Optional[str] = None) -> str:
        try:
            # Base document data
            doc_data = {
                "name": name,
                "tags": tags,
                "description": description,
                "content_type": content_type,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }

            if content_type == "lecture_notes":
                # Handle lecture notes
                if not lecture_content:
                    return "❌ lecture_content is required when content_type is 'lecture_notes'"
                
                # Extract metadata from content if not provided
                extracted_metadata = self._extract_metadata_from_lecture_notes(lecture_content)
                
                doc_data.update({
                    "lecture_content": lecture_content,
                    "course_title": course_title or extracted_metadata.get('course_title', 'Unknown Course'),
                    "topic": topic or extracted_metadata.get('topic', 'General Topic'),
                    "key_concepts": key_concepts or [],
                    "search_query": search_query or extracted_metadata.get('search_query', ''),
                    "content_length": len(lecture_content),
                    "total_sections": extracted_metadata.get('total_sections', 0),
                    "generated_date": extracted_metadata.get('generated_date', ''),
                    "link": link  # Optional for lecture notes
                })
                
                # Add to lecture_notes collection
                collection_name = "lecture_notes"
                
            else:
                # Handle regular documents
                if not link:
                    return "❌ link is required when content_type is 'document'"
                
                doc_data["link"] = link
                
                # Add to resources collection
                collection_name = "resources"

            # Add to Firestore
            doc_ref = db.collection(collection_name).document()
            doc_ref.set(doc_data)

            # Create success message based on content type
            if content_type == "lecture_notes":
                return (f"✅ Lecture notes '{name}' added successfully!\n"
                       f"📄 Document ID: {doc_ref.id}\n"
                       f"📚 Course: {doc_data['course_title']}\n"
                       f"📖 Topic: {doc_data['topic']}\n"
                       f"🔑 Key Concepts: {len(doc_data['key_concepts'])}\n"
                       f"📊 Content Length: {doc_data['content_length']} characters")
            else:
                return f"✅ Document '{name}' added successfully with ID: {doc_ref.id}"
                
        except Exception as e:
            return f"❌ Failed to add {content_type}: {str(e)}"

    async def _arun(self, name: str, link: str = "", tags: List[str] = [], description: str = "",
                   content_type: str = "document", lecture_content: Optional[str] = None,
                   course_title: Optional[str] = None, topic: Optional[str] = None,
                   key_concepts: Optional[List[str]] = None, search_query: Optional[str] = None) -> str:
        return self._run(name, link, tags, description, content_type, lecture_content,
                        course_title, topic, key_concepts, search_query)



    

class RetrievefromFirestoreInput(BaseModel):
    description: str = Field(description="The description of the document to retrieve")
    tags: list[str] = Field(description="The tags of the document to retrieve")

class RetrievefromFirestore(BaseTool):
    name: str = "retrieve_from_firestore"
    description: str = "Retrieve a document from the firestore database"

    args_schema: Optional[Type[BaseModel]] = RetrievefromFirestoreInput

    def _run(self, description: str, tags: list[str]) -> str:
        try:

            # Start with documents matching the description
            query = db.collection("resources").where("description", "==", description)

            # Execute query
            docs = query.stream()

            # Filter by tags (manual filtering since Firestore can't do `array-contains-all`)
            matching_docs = []
            for doc in docs:
                doc_data = doc.to_dict()
                if all(tag in doc_data.get("tags", []) for tag in tags):
                    matching_docs.append(doc_data)

            if not matching_docs:
                return "❌ No matching documents found."

            # Format the matching docs for output
            response = "✅ Matching documents:\n"
            for doc in matching_docs:
                response += f"- Name: {doc.get('name')}\n  Link: {doc.get('link')}\n  Tags: {doc.get('tags')}\n  Description: {doc.get('description')}\n\n"

            return response.strip()

        except Exception as e:
            return f"❌ Failed to retrieve document: {str(e)}"

# RetrievefromFirestore = RetrievefromFirestore()
# res = RetrievefromFirestore.run({"description": "test", "tags": ["test"]})
# print(res)

class SerperSearchInput(BaseModel):
    """Input schema for SerperSearchTool"""
    query: str = Field(description="Search query to execute")
    api_key: str = Field(description="Serper API key")
    num_results: int = Field(default=10, description="Number of search results to retrieve (1-100)")
    search_type: str = Field(default="search", description="Type of search: 'search', 'news', 'images', 'videos', 'places', 'maps'")
    country: str = Field(default="us", description="Country code for localized results (e.g., 'us', 'uk', 'ca')")
    location: Optional[str] = Field(default=None, description="Location for localized results (e.g., 'New York, NY')")

class SerperSearchTool(BaseTool):
    """
    LangChain tool for searching using Serper API (Google Search API)
    """
    name: str = "serper_search"
    description: str = """
    Search for information using Serper API (Google Search). Returns structured search results
    including organic results, knowledge graph, answer box, and people also ask sections.
    
    Use this when you need to:
    - Search for current information on the web
    - Get Google search results in structured format
    - Find recent news, articles, or web content
    - Gather information for research or analysis
    
    Returns JSON with search results that can be processed by other tools.
    """
    
    args_schema: Type[BaseModel] = SerperSearchInput
    return_direct: bool = False
    
    def _run(self, query: str, api_key: str, num_results: int = 10, search_type: str = "search", country: str = "us", location: Optional[str] = None,run_manager: Optional[CallbackManager] = None,) -> str:
        """Execute the search"""
        try:
            if run_manager:
                run_manager.on_text(f"Searching for: {query}", color="blue")
            
            # Determine the correct endpoint
            endpoint_map = {
                "search": "search",
                "news": "news", 
                "images": "images",
                "videos": "videos",
                "places": "places",
                "maps": "maps"
            }
            
            endpoint = endpoint_map.get(search_type.lower(), "search")
            url = f"https://google.serper.dev/{endpoint}"
            
            # Build payload
            payload = {
                "q": query,
                "num": min(max(num_results, 1), 100),  # Clamp between 1-100
                "gl": country.lower()
            }
            
            if location:
                payload["location"] = location
            
            headers = {
                'X-API-KEY': api_key,
                'Content-Type': 'application/json'
            }
            
            if run_manager:
                run_manager.on_text(f"Calling Serper API endpoint: {endpoint}", color="yellow")
            
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
            response.raise_for_status()
            
            results = response.json()
            
            # Add metadata
            results['_metadata'] = {
                'search_query': query,
                'search_type': search_type,
                'timestamp': datetime.now().isoformat(),
                'num_results_requested': num_results,
                'country': country,
                'location': location
            }
            
            if run_manager:
                organic_count = len(results.get('organic', []))
                run_manager.on_text(f"Found {organic_count} organic results", color="green")
            
            return json.dumps(results, indent=2, ensure_ascii=False)
            
        except requests.exceptions.RequestException as e:
            error_msg = f"❌ Serper API error: {str(e)}"
            if run_manager:
                run_manager.on_text(error_msg, color="red")
            return json.dumps({"error": error_msg, "query": query})
        except Exception as e:
            error_msg = f"❌ Unexpected error: {str(e)}"
            if run_manager:
                run_manager.on_text(error_msg, color="red")
            return json.dumps({"error": error_msg, "query": query})


ResearchTool = SerperSearchTool()
res = ResearchTool.run({"query":"quantum computing", "api_key": SERPER_API_KEY, "num_results": 5})
print(res)



class LectureNotesInput(BaseModel):
    """Input schema for LectureNotesTool"""
    search_results: str = Field(description="JSON string containing search results (from Serper or other sources)")
    course_title: str = Field(default="Research Notes", description="Title for the course/subject")
    topic: str = Field(default="General Research", description="Specific topic being studied")
    output_format: str = Field(default="markdown", description="Output format: 'markdown' or 'html'")
    output_file: Optional[str] = Field(default=None, description="Output file path (optional)")
    include_sources: bool = Field(default=True, description="Include source citations in notes")
    max_sections: int = Field(default=10, description="Maximum number of sections to process")

class LectureNotesTool(BaseTool):
    """
    LangChain tool that converts search results into structured lecture notes
    """
    name: str = "generate_lecture_notes"
    description: str = """
    Convert search results (JSON format) into structured lecture notes with summaries,
    key points, and study materials. Works with Serper API results or similar JSON structures.
    
    Use this when you need to:
    - Convert search results into study materials
    - Generate structured lecture notes from web data  
    - Create academic-style notes with summaries and key points
    - Organize research data into educational format
    
    Input should be JSON search results, typically from the serper_search tool.
    """
    
    args_schema: Type[BaseModel] = LectureNotesInput
    return_direct: bool = False
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text"""
        common_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'cannot',
            'this', 'that', 'these', 'those', 'what', 'when', 'where', 'why', 'how', 'about'
        }
        
        # Find various types of important terms
        words = re.findall(r'\b[A-Z][a-zA-Z]+\b', text)
        quoted_terms = re.findall(r'"([^"]+)"', text)
        technical_terms = re.findall(r'\b[a-zA-Z]+(?:-[a-zA-Z]+)+\b', text)
        
        all_terms = words + quoted_terms + technical_terms
        key_terms = [
            term for term in set(all_terms) 
            if term.lower() not in common_words and len(term) > 2 and len(term) < 30
        ]
        
        return list(set(key_terms))[:10]
    
    def _generate_summary(self, content: str) -> str:
        """Generate a brief summary from content"""
        # Clean markdown formatting
        clean_content = re.sub(r'\*\*([^*]+)\*\*', r'\1', content)
        clean_content = re.sub(r'\*([^*]+)\*', r'\1', clean_content)
        clean_content = re.sub(r'\n+', ' ', clean_content).strip()
        
        sentences = re.split(r'[.!?]+', clean_content)
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if len(valid_sentences) >= 2:
            return f"{valid_sentences[0]}. {valid_sentences[1]}."
        elif len(valid_sentences) == 1:
            return f"{valid_sentences[0]}."
        else:
            return clean_content[:200] + "..." if len(clean_content) > 200 else clean_content
    
    def _extract_main_points(self, content: str) -> List[str]:
        """Extract main points from content"""
        points = []
        lines = content.split('\n')
        
        # Look for existing structure
        for line in lines:
            line = line.strip()
            if line.startswith('**') and line.endswith('**') and len(line) > 6:
                points.append(f"• {line.replace('**', '')}")
            elif re.match(r'^[-*•]\s*(.+)$', line):
                points.append(line if line.startswith('•') else f"• {line[2:]}")
        
        # If no structured points, extract key sentences
        if not points:
            sentences = re.split(r'[.!?]+', content)
            for sentence in sentences[:4]:
                sentence = sentence.strip()
                if len(sentence) > 30 and len(sentence) < 200:
                    points.append(f"• {sentence}")
        
        return points[:6]  # Limit to 6 points
    
    def _process_search_results(self, search_data: Dict) -> Dict:
        """Process search results into structured sections"""
        sections = []
        all_key_terms = set()
        
        # Get search metadata
        metadata = search_data.get('_metadata', {})
        search_query = metadata.get('search_query', search_data.get('searchParameters', {}).get('q', 'Unknown Query'))
        
        # Process organic results
        organic_results = search_data.get('organic', [])
        if organic_results:
            content_parts = []
            sources = []
            
            for i, result in enumerate(organic_results[:5], 1):
                title = result.get('title', f'Result {i}')
                snippet = result.get('snippet', 'No description available')
                link = result.get('link', '#')
                
                content_parts.append(f"**{title}**\n{snippet}")
                sources.append(f"[{title}]({link})")
            
            if content_parts:
                combined_content = '\n\n'.join(content_parts)
                all_text = ' '.join([r.get('title', '') + ' ' + r.get('snippet', '') for r in organic_results[:5]])
                key_terms = self._extract_key_terms(all_text)
                all_key_terms.update(key_terms)
                
                sections.append({
                    'title': f"Web Search Results: {search_query}",
                    'content': combined_content,
                    'source': f"Search Results ({len(organic_results)} total)",
                    'key_terms': key_terms,
                    'summary': self._generate_summary(combined_content),
                    'main_points': self._extract_main_points(combined_content),
                    'sources': sources[:3]  # Limit sources
                })
        
        # Process knowledge graph
        knowledge_graph = search_data.get('knowledgeGraph')
        if knowledge_graph:
            title = knowledge_graph.get('title', 'Knowledge Graph')
            description = knowledge_graph.get('description', '')
            
            # Include attributes
            attributes = knowledge_graph.get('attributes', {})
            attr_parts = []
            for key, value in list(attributes.items())[:5]:  # Limit attributes
                if isinstance(value, str) and len(value) < 200:
                    attr_parts.append(f"**{key}:** {value}")
            
            content_parts = [description] if description else []
            if attr_parts:
                content_parts.append('\n'.join(attr_parts))
            
            content = '\n\n'.join(content_parts)
            
            if content.strip():
                key_terms = self._extract_key_terms(title + ' ' + content)
                all_key_terms.update(key_terms)
                
                sections.append({
                    'title': f"Knowledge Summary: {title}",
                    'content': content,
                    'source': "Knowledge Graph",
                    'key_terms': key_terms,
                    'summary': self._generate_summary(content),
                    'main_points': self._extract_main_points(content),
                    'sources': []
                })
        
        # Process answer box
        answer_box = search_data.get('answerBox')
        if answer_box:
            answer = answer_box.get('answer', '')
            title = answer_box.get('title', 'Direct Answer')
            
            if answer and len(answer) > 20:
                key_terms = self._extract_key_terms(answer)
                all_key_terms.update(key_terms)
                
                sections.append({
                    'title': f"Quick Answer: {search_query}",
                    'content': answer,
                    'source': "Featured Snippet",
                    'key_terms': key_terms,
                    'summary': self._generate_summary(answer),
                    'main_points': self._extract_main_points(answer),
                    'sources': []
                })
        
        # Process people also ask
        people_also_ask = search_data.get('peopleAlsoAsk', [])
        if people_also_ask:
            qa_pairs = []
            
            for item in people_also_ask[:4]:  # Limit to 4 questions
                question = item.get('question', '')
                snippet = item.get('snippet', '')
                if question and snippet:
                    qa_pairs.append(f"**Q: {question}**\nA: {snippet}")
            
            if qa_pairs:
                content = '\n\n'.join(qa_pairs)
                all_text = ' '.join([item.get('question', '') + ' ' + item.get('snippet', '') for item in people_also_ask[:4]])
                key_terms = self._extract_key_terms(all_text)
                all_key_terms.update(key_terms)
                
                sections.append({
                    'title': f"Common Questions: {search_query}",
                    'content': content,
                    'source': "People Also Ask",
                    'key_terms': key_terms,
                    'summary': self._generate_summary(content),
                    'main_points': self._extract_main_points(content),
                    'sources': []
                })
        
        return {
            'sections': sections,
            'key_concepts': list(all_key_terms),
            'search_query': search_query,
            'metadata': metadata
        }
    
    def _generate_markdown_notes(self, processed_data: Dict, course_title: str, topic: str, include_sources: bool) -> str:
        """Generate markdown lecture notes"""
        sections = processed_data['sections']
        key_concepts = processed_data['key_concepts']
        search_query = processed_data['search_query']
        metadata = processed_data.get('metadata', {})
        
        md_lines = []
        
        # Header
        md_lines.extend([
            f"# {course_title}",
            f"## Topic: {topic}",
            "",
            f"**Generated:** {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
            f"**Based on Search:** {search_query}",
            f"**Total Sections:** {len(sections)}",
            ""
        ])
        
        # Add search metadata if available
        if metadata:
            search_type = metadata.get('search_type', 'search')
            timestamp = metadata.get('timestamp', '')
            if timestamp:
                search_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M UTC')
                md_lines.append(f"**Search Date:** {search_time}")
            if search_type != 'search':
                md_lines.append(f"**Search Type:** {search_type.title()}")
            md_lines.append("")
        
        md_lines.extend(["---", ""])
        
        # Table of Contents
        md_lines.extend(["## 📋 Table of Contents", ""])
        for i, section in enumerate(sections, 1):
            safe_title = re.sub(r'[^\w\s-]', '', section['title'].lower().replace(' ', '-').replace(':', ''))
            md_lines.append(f"{i}. [{section['title']}](#{safe_title})")
        md_lines.extend(["", "---", ""])
        
        # Key Concepts
        if key_concepts:
            md_lines.extend(["## 🔑 Key Concepts", ""])
            # Display in columns
            sorted_concepts = sorted(key_concepts)
            for i in range(0, len(sorted_concepts), 3):
                row_concepts = sorted_concepts[i:i+3]
                md_lines.append("| " + " | ".join([f"**{concept}**" for concept in row_concepts]) + " |")
                if i == 0:
                    md_lines.append("| " + " | ".join(["---" for _ in row_concepts]) + " |")
            md_lines.extend(["", "---", ""])
        
        # Main Sections
        for i, section in enumerate(sections, 1):
            safe_title = re.sub(r'[^\w\s-]', '', section['title'].lower().replace(' ', '-').replace(':', ''))
            
            md_lines.extend([
                f"## {i}. {section['title']} {{#{safe_title}}}",
                "",
                "### 📝 Summary",
                f"> {section['summary']}",
                ""
            ])
            
            # Key Points
            if section['main_points']:
                md_lines.extend(["### 🎯 Key Points", ""])
                for point in section['main_points']:
                    md_lines.append(point)
                md_lines.append("")
            
            # Main Content
            md_lines.extend([
                "### 📖 Details",
                section['content'],
                ""
            ])
            
            # Sources
            if include_sources:
                md_lines.extend(["### 📚 Sources", f"**Primary:** {section['source']}"])
                if section.get('sources'):
                    md_lines.extend(["**Links:**"] + [f"- {source}" for source in section['sources']])
                md_lines.append("")
            
            md_lines.extend(["---", ""])
        
        # Study Section
        md_lines.extend([
            "## 📚 Study Guide",
            "",
            "### Quick Review",
            "Use this section to test your understanding:",
            ""
        ])
        
        for i, section in enumerate(sections, 1):
            md_lines.append(f"**{i}. {section['title']}**")
            md_lines.extend([
                "- What are the main concepts?",
                "- How does this relate to the overall topic?",
                "- What are the practical implications?",
                ""
            ])
        
        return '\n'.join(md_lines)
    
    def _run(
        self,
        search_results: str,
        course_title: str = "Research Notes",
        topic: str = "General Research",
        output_format: str = "markdown",
        output_file: Optional[str] = None,
        include_sources: bool = True,
        max_sections: int = 10,
        run_manager: Optional[CallbackManager] = None,
    ) -> str:
        """Execute the lecture notes generation"""
        try:
            if run_manager:
                run_manager.on_text("Processing search results...", color="blue")
            
            # Parse search results
            try:
                search_data = json.loads(search_results)
            except json.JSONDecodeError:
                return "❌ Error: Invalid JSON format in search_results"
            
            # Check for errors in search results
            if 'error' in search_data:
                return f"❌ Cannot process search results due to error: {search_data['error']}"
            
            # Process the search data
            processed_data = self._process_search_results(search_data)
            
            if not processed_data['sections']:
                return "❌ No processable content found in search results"
            
            # Limit sections if needed
            if len(processed_data['sections']) > max_sections:
                processed_data['sections'] = processed_data['sections'][:max_sections]
                if run_manager:
                    run_manager.on_text(f"Limited to {max_sections} sections", color="yellow")
            
            if run_manager:
                run_manager.on_text(f"Generating {output_format} lecture notes...", color="blue")
            
            # Generate notes (currently only markdown supported)
            lecture_notes = self._generate_markdown_notes(
                processed_data, course_title, topic, include_sources
            )
            
            # Save to file if requested
            if output_file:
                try:
                    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(lecture_notes)
                    
                    if run_manager:
                        run_manager.on_text(f"Saved to {output_file}", color="green")
                    
                    return f"✅ Lecture notes generated successfully!\n📄 Saved to: {output_file}\n📊 Processed {len(processed_data['sections'])} sections\n🔑 Extracted {len(processed_data['key_concepts'])} key concepts\n\n**Preview:**\n{lecture_notes[:500]}..."
                    
                except Exception as e:
                    return f"✅ Lecture notes generated but failed to save to file: {str(e)}\n\n{lecture_notes}"
            else:
                return f"✅ Lecture notes generated successfully!\n📊 Processed {len(processed_data['sections'])} sections\n🔑 Extracted {len(processed_data['key_concepts'])} key concepts\n\n{lecture_notes}"
                
        except Exception as e:
            error_msg = f"❌ Error generating lecture notes: {str(e)}"
            if run_manager:
                run_manager.on_text(error_msg, color="red")
            return error_msg

notes_tool = LectureNotesTool()
lecture_notes = notes_tool._run(
        search_results=res,
        course_title="AI in Healthcare",
        topic="Machine Learning Applications",
        output_file="ml_healthcare_notes.md"
    )



tool = AddtoFirestore()
result = tool._run(
    name="AI Healthcare Notes",
    tags=["research", "AI"],
    description="Comprehensive notes on Python fundamentals",
    content_type="lecture_notes",
    lecture_content=lecture_notes,
    course_title="AI in Healthcare",
    topic="Machine Learning Application"
)


#=================================================================
# Initialize Agent with LLM
#=================================================================




#=================================================================
# Example Usage
#=================================================================
