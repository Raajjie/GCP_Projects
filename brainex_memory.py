import os
import json
from typing import Dict, List, Optional, TypedDict, Annotated, Literal, Tuple
from datetime import datetime
import re

from langchain.tools import BaseTool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from pydantic import BaseModel, Field
from google.cloud import firestore
from dotenv import load_dotenv
import requests

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables
load_dotenv()
db = firestore.Client()
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

#=================================================================
# State Definition
#=================================================================

class RAGState(TypedDict):
    """State for the RAG agent"""
    messages: Annotated[List, "The messages in the conversation"]
    query: str
    firestore_results: Optional[Dict]
    web_results: Optional[Dict]
    knowledge_saved: bool
    processing_step: str

#=================================================================
# Tools 
#=================================================================

class SearchFirestoreInput(BaseModel):
    query: str = Field(description="Search query to find relevant documents in the knowledge base")

class SearchFirestore(BaseTool):
    name: str = "search_firestore"
    description: str = """Search for existing documents and lecture notes in Firestore database."""
    args_schema = SearchFirestoreInput

    def _run(self, query: str) -> str:
        print(f"Running Firestore search with query!!!!")
        try:
            results = []
            query_lower = query.lower()
            query_words = [word for word in query_lower.split() if len(word) > 2]  # Filter out very short words
            
            # Remove common stop words that might cause false matches
            stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'what', 'when', 'where', 'why', 'best', 'good', 'about', 'from', 'they', 'this', 'that', 'with', 'have', 'will', 'your', 'there', 'been', 'were', 'said', 'each', 'which', 'their', 'time', 'would', 'like', 'into', 'than', 'find', 'over', 'also', 'back', 'after', 'first', 'well', 'down', 'work', 'life', 'only', 'then', 'come', 'some', 'could', 'should'}
            
            # Filter out stop words from query
            meaningful_words = [word for word in query_words if word not in stop_words]
            
            if not meaningful_words:  # If no meaningful words left, use original
                meaningful_words = query_words
            
            # Search in resources collection
            resources = db.collection("resources").stream()
            for doc in resources:
                data = doc.to_dict()
                search_text = f"{data.get('name', '')} {data.get('description', '')} {' '.join(data.get('tags', []))}".lower()
                
                # Calculate relevance score
                relevance_score = 0
                matched_words = 0
                
                for word in meaningful_words:
                    if word in search_text:
                        matched_words += 1
                        # Give higher score for exact matches in important fields
                        if word in data.get('name', '').lower():
                            relevance_score += 3
                        elif word in data.get('description', '').lower():
                            relevance_score += 2
                        elif word in ' '.join(data.get('tags', [])).lower():
                            relevance_score += 1
                
                # Calculate match percentage
                match_percentage = matched_words / len(meaningful_words) if meaningful_words else 0
                
                # Only include if it meets minimum criteria
                if match_percentage >= 0.3 and relevance_score >= 2:  # At least 30% of words match and score >= 2
                    results.append({
                        'id': doc.id,
                        'type': 'resource',
                        'name': data.get('name'),
                        'description': data.get('description'),
                        'link': data.get('link'),
                        'tags': data.get('tags', []),
                        'relevance_score': relevance_score,
                        'match_percentage': match_percentage,
                        'matched_words': matched_words
                    })
            
            # Search in lecture_notes collection
            lectures = db.collection("lecture_notes").stream()
            for doc in lectures:
                data = doc.to_dict()
                name = data.get('name', '')
                description = data.get('description', '')
                course_title = data.get('course_title', '')
                topic = data.get('topic', '')
                tags = ' '.join(data.get('tags', []))
                key_concepts = ' '.join(data.get('key_concepts', []))
                content = data.get('lecture_content', '')
                
                # Create weighted search text (more important fields get more weight)
                search_fields = {
                    'name': name.lower(),
                    'topic': topic.lower(),
                    'description': description.lower(),
                    'course_title': course_title.lower(),
                    'tags': tags.lower(),
                    'key_concepts': key_concepts.lower(),
                    'content': content.lower()
                }
                
                relevance_score = 0
                matched_words = 0
                field_matches = []
                
                for word in meaningful_words:
                    word_found = False
                    # Check each field with different weights
                    if word in search_fields['name']:
                        relevance_score += 5  # Highest weight for name matches
                        word_found = True
                        field_matches.append(f"name:{word}")
                    elif word in search_fields['topic']:
                        relevance_score += 4  # High weight for topic matches
                        word_found = True
                        field_matches.append(f"topic:{word}")
                    elif word in search_fields['key_concepts']:
                        relevance_score += 3  # Medium-high weight for key concepts
                        word_found = True
                        field_matches.append(f"concepts:{word}")
                    elif word in search_fields['description']:
                        relevance_score += 2  # Medium weight for description
                        word_found = True
                        field_matches.append(f"desc:{word}")
                    elif word in search_fields['tags']:
                        relevance_score += 2  # Medium weight for tags
                        word_found = True
                        field_matches.append(f"tags:{word}")
                    elif word in search_fields['course_title']:
                        relevance_score += 1  # Lower weight for course title
                        word_found = True
                        field_matches.append(f"course:{word}")
                    elif word in search_fields['content']:
                        relevance_score += 1  # Lowest weight for content matches
                        word_found = True
                        field_matches.append(f"content:{word}")
                    
                    if word_found:
                        matched_words += 1
                
                # Calculate match percentage
                match_percentage = matched_words / len(meaningful_words) if meaningful_words else 0
                
                # More strict criteria for lecture notes since they have more text
                # Require higher match percentage or very high relevance score
                if (match_percentage >= 0.4 and relevance_score >= 3) or relevance_score >= 8:
                    content_excerpt = content[:800] + "..." if len(content) > 800 else content
                    
                    results.append({
                        'id': doc.id,
                        'type': 'lecture_notes',
                        'name': name,
                        'description': description,
                        'course_title': course_title,
                        'topic': topic,
                        'content_excerpt': content_excerpt,
                        'full_content': content,
                        'tags': data.get('tags', []),
                        'key_concepts': data.get('key_concepts', []),
                        'relevance_score': relevance_score,
                        'match_percentage': match_percentage,
                        'matched_words': matched_words,
                        'field_matches': field_matches
                    })
            
            # Sort by relevance score (higher is better)
            results.sort(key=lambda x: (x['relevance_score'], x['match_percentage']), reverse=True)
            
            # Additional filtering: only return results above a minimum threshold
            min_score_threshold = 3
            filtered_results = [r for r in results if r['relevance_score'] >= min_score_threshold]
            
            if not filtered_results:
                return json.dumps({
                    "found": False, 
                    "message": "No sufficiently relevant documents found in knowledge base",
                    "documents": [],
                    "debug_info": {
                        "total_results_found": len(results),
                        "meaningful_query_words": meaningful_words,
                        "min_threshold": min_score_threshold
                    }
                })
            
            return json.dumps({
                "found": True, 
                "count": len(filtered_results),
                "documents": filtered_results[:5],  # Return top 5 most relevant
                "debug_info": {
                    "total_results_found": len(results),
                    "filtered_results": len(filtered_results),
                    "meaningful_query_words": meaningful_words,
                    "min_threshold": min_score_threshold
                }
            })
            
        except Exception as e:
            return json.dumps({"error": f"Firestore search failed: {str(e)}"})
        
class WebSearchInput(BaseModel):
    query: str = Field(description="Search query for web research")
    num_results: int = Field(default=5, description="Number of results to retrieve (1-10)")

class WebSearch(BaseTool):
    name: str = "web_search"
    description: str = """Search the web for current information using Serper API."""
    args_schema = WebSearchInput

    def _run(self, query: str, num_results: int = 5) -> str:
        print(f"Running web search with query!!!!!")
        try:
            url = "https://google.serper.dev/search"
            payload = {
                "q": query,
                "num": min(max(num_results, 1), 10),
                "gl": "us"
            }
            headers = {
                'X-API-KEY': SERPER_API_KEY,
                'Content-Type': 'application/json'
            }
            
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
            response.raise_for_status()
            
            results = response.json()
            
            structured_results = {
                "search_query": query,
                "timestamp": datetime.now().isoformat(),
                "organic_results": results.get('organic', []),
                "answer_box": results.get('answerBox'),
                "knowledge_graph": results.get('knowledgeGraph'),
                "people_also_ask": results.get('peopleAlsoAsk', [])
            }
            
            return json.dumps(structured_results, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Web search failed: {str(e)}", "query": query})

class SaveKnowledgeInput(BaseModel):
    name: str = Field(description="Name/title for the knowledge document")
    content: str = Field(description="The knowledge content to save")
    topic: str = Field(description="Main topic of the content")
    tags: List[str] = Field(description="Tags to categorize the content")
    source_query: str = Field(description="Original query that generated this content")

class SaveKnowledge(BaseTool):
    name: str = "save_knowledge"
    description: str = """Save new knowledge to the Firestore database for future retrieval."""
    args_schema = SaveKnowledgeInput

    def _run(self, name: str, content: str, topic: str, tags: List[str], source_query: str) -> str:
        print(f"Saving knowledge with name!!!")
        try:
            key_concepts = self._extract_key_concepts(content)
            
            doc_data = {
                "name": name,
                "tags": tags,
                "description": f"Knowledge base entry on {topic}",
                "content_type": "lecture_notes",
                "lecture_content": content,
                "course_title": "Knowledge Base",
                "topic": topic,
                "key_concepts": key_concepts,
                "search_query": source_query,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "content_length": len(content)
            }
            
            doc_ref = db.collection("lecture_notes").document()
            doc_ref.set(doc_data)
            
            return json.dumps({
                "success": True,
                "message": f"Knowledge saved successfully with ID: {doc_ref.id}",
                "document_id": doc_ref.id,
                "key_concepts_extracted": len(key_concepts)
            })
            
        except Exception as e:
            return json.dumps({"success": False, "error": f"Failed to save knowledge: {str(e)}"})
    
    def _extract_key_concepts(self, content: str) -> List[str]:
        """Simple key concept extraction"""
        concepts = set()
        
        # Extract from markdown headers
        headers = re.findall(r'#+\s*([^#\n]+)', content)
        for header in headers:
            words = re.findall(r'\b[A-Za-z]{3,}\b', header)
            concepts.update(word.lower() for word in words if len(word) > 3)
        
        # Extract capitalized terms
        capitalized = re.findall(r'\b[A-Z][a-z]{2,}\b', content)
        concepts.update(word.lower() for word in capitalized if len(word) > 3)
        
        # Remove common words
        common_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'}
        concepts -= common_words
        
        return list(concepts)[:10]

#=================================================================
# Rule-Based Reasoning Engine
#=================================================================

class RuleBasedReasoner:
    """Rule-based reasoning engine for RAG operations"""
    
    def __init__(self):
        self.search_firestore = SearchFirestore()
        self.web_search = WebSearch()
        self.save_knowledge = SaveKnowledge()
    
    def extract_query_intent(self, query: str) -> Dict[str, any]:
        """Extract intent and parameters from user query"""
        query_lower = query.lower()
        
        # Classify query type
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which']
        is_question = any(word in query_lower for word in question_words) or query.endswith('?')
        
        # Extract key topics
        topics = self._extract_topics(query)
        
        # Determine urgency/recency need
        time_indicators = ['latest', 'recent', 'current', 'new', 'today', 'now', '2024', '2025']
        needs_recent_info = any(indicator in query_lower for indicator in time_indicators)
        
        return {
            'is_question': is_question,
            'topics': topics,
            'needs_recent_info': needs_recent_info,
            'query_length': len(query.split()),
            'complexity': self._assess_complexity(query)
        }
    
    def _extract_topics(self, query: str) -> List[str]:
        """Extract main topics from query"""
        # Simple topic extraction
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'what', 'how', 'why', 'when', 'where', 'who', 'which'}
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
        topics = [word for word in words if word not in stop_words]
        
        # Extract potential compound terms (capitalized words in original)
        compound_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        topics.extend([term.lower() for term in compound_terms])
        
        return list(set(topics))[:10]  # Top 10 unique topics
    
    def _assess_complexity(self, query: str) -> str:
        """Assess query complexity"""
        word_count = len(query.split())
        if word_count <= 5:
            return 'simple'
        elif word_count <= 15:
            return 'moderate'
        else:
            return 'complex'
    
    def decide_next_action(self, state: RAGState) -> str:
        """Rule-based decision making for next action"""
        step = state.get('processing_step', 'start')
        
        if step == 'start':
            return 'search_knowledge_base'
        
        elif step == 'knowledge_searched':
            firestore_results = state.get('firestore_results')
            if firestore_results and firestore_results.get('found', False):
                return 'generate_response_from_kb'
            else:
                return 'search_web'
        
        elif step == 'web_searched':
            web_results = state.get('web_results')
            if web_results and not web_results.get('error'):
                return 'save_and_respond'
            else:
                return 'generate_fallback_response'
        
        elif step == 'knowledge_saved':
            return 'generate_final_response'
        
        else:
            return 'end'
    
    def generate_response_from_knowledge_base(self, query: str, firestore_results: Dict) -> str:
        """Generate response using knowledge base results"""
        if not firestore_results.get('found', False):
            return "I couldn't find relevant information in the knowledge base."
        
        documents = firestore_results.get('documents', [])
        if not documents:
            return "No relevant documents found in the knowledge base."
        
        # Build response from retrieved documents
        response_parts = [f"Based on the information in our knowledge base:\n"]
        
        for i, doc in enumerate(documents[:3], 1):  # Use top 3 documents
            if doc['type'] == 'lecture_notes':
                response_parts.append(f"\n**{doc['name']}** (Topic: {doc.get('topic', 'General')})")
                content = doc.get('full_content', doc.get('content_excerpt', ''))
                
                # Extract relevant excerpts
                relevant_excerpt = self._extract_relevant_excerpt(content, query)
                response_parts.append(relevant_excerpt)
                
                if doc.get('key_concepts'):
                    response_parts.append(f"Key concepts: {', '.join(doc['key_concepts'][:5])}")
            
            elif doc['type'] == 'resource':
                response_parts.append(f"\n**{doc['name']}**")
                if doc.get('description'):
                    response_parts.append(doc['description'])
                if doc.get('link'):
                    response_parts.append(f"Resource link: {doc['link']}")
        
        return '\n'.join(response_parts)
    
    def _extract_relevant_excerpt(self, content: str, query: str, max_length: int = 500) -> str:
        """Extract most relevant excerpt from content"""
        query_words = set(query.lower().split())
        sentences = re.split(r'[.!?]+', content)
        
        # Score sentences based on query word overlap
        scored_sentences = []
        for sentence in sentences:
            if len(sentence.strip()) < 20:  # Skip very short sentences
                continue
            
            sentence_words = set(sentence.lower().split())
            overlap = len(query_words.intersection(sentence_words))
            scored_sentences.append((overlap, sentence.strip()))
        
        # Sort by relevance and get top sentences
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        
        # Build excerpt
        excerpt = ""
        for score, sentence in scored_sentences:
            if len(excerpt + sentence) > max_length:
                break
            if score > 0:  # Only include sentences with some relevance
                excerpt += sentence + ". "
        
        return excerpt[:max_length] + "..." if len(excerpt) > max_length else excerpt
    
    def generate_web_search_query(self, original_query: str, intent: Dict) -> str:
        """Generate optimized web search query"""
        # For recent info needs, add time indicators
        if intent['needs_recent_info']:
            return f"{original_query} 2024 2025 latest"
        
        # For complex queries, extract key terms
        if intent['complexity'] == 'complex':
            key_topics = intent['topics'][:3]  # Top 3 topics
            return ' '.join(key_topics)
        
        return original_query
    
    def synthesize_web_results(self, query: str, web_results: Dict) -> str:
        """Synthesize web search results into structured content"""
        if web_results.get('error'):
            return f"Error during web search: {web_results['error']}"
        
        content_parts = [f"# {query.title()}\n"]
        
        # Add answer box if available
        answer_box = web_results.get('answer_box')
        if answer_box:
            content_parts.append(f"## Quick Answer\n{answer_box.get('answer', '')}\n")
        
        # Add knowledge graph info
        kg = web_results.get('knowledge_graph')
        if kg:
            content_parts.append(f"## Overview\n{kg.get('description', '')}\n")
            if kg.get('attributes'):
                content_parts.append("### Key Information:")
                for attr in kg['attributes'][:5]:
                    content_parts.append(f"- **{attr.get('name', '')}**: {attr.get('value', '')}")
                content_parts.append("")
        
        # Add organic results
        organic = web_results.get('organic_results', [])
        if organic:
            content_parts.append("## Detailed Information\n")
            for i, result in enumerate(organic[:5], 1):
                title = result.get('title', 'Untitled')
                snippet = result.get('snippet', '')
                link = result.get('link', '')
                
                content_parts.append(f"### {i}. {title}")
                content_parts.append(snippet)
                content_parts.append(f"Source: {link}\n")
        
        # Add related questions
        paa = web_results.get('people_also_ask', [])
        if paa:
            content_parts.append("## Related Questions")
            for question in paa[:3]:
                content_parts.append(f"- {question.get('question', '')}")
            content_parts.append("")
        
        return '\n'.join(content_parts)
    
    def prepare_knowledge_for_saving(self, query: str, synthesized_content: str) -> Dict:
        """Prepare synthesized content for saving to knowledge base"""
        # Extract title
        title_match = re.search(r'^#\s+(.+)$', synthesized_content, re.MULTILINE)
        title = title_match.group(1) if title_match else f"Knowledge about {query}"
        
        # Extract main topic
        topics = self._extract_topics(query)
        main_topic = topics[0] if topics else "General Knowledge"
        
        # Generate tags
        tags = topics[:5] + ['web_research', 'knowledge_base']
        
        return {
            'name': title,
            'content': synthesized_content,
            'topic': main_topic,
            'tags': list(set(tags)),  # Remove duplicates
            'source_query': query
        }

#=================================================================
# Rule-Based Nodes
#=================================================================

def initialize_state(state: RAGState) -> RAGState:
    """Initialize the state with query analysis"""
    messages = state["messages"]
    if not messages:
        return state
    
    # Get the latest human message
    human_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]
    if not human_messages:
        return state
    
    query = human_messages[-1].content
    
    return {
        **state,
        "query": query,
        "firestore_results": None,
        "web_results": None,
        "knowledge_saved": False,
        "processing_step": "start"
    }

def search_knowledge_base_node(state: RAGState) -> RAGState:
    """Search the knowledge base"""
    reasoner = RuleBasedReasoner()
    query = state["query"]
    
    try:
        # Search Firestore
        firestore_result = reasoner.search_firestore._run(query)
        firestore_data = json.loads(firestore_result)
        
        return {
            **state,
            "firestore_results": firestore_data,
            "processing_step": "knowledge_searched"
        }
    except Exception as e:
        return {
            **state,
            "firestore_results": {"found": False, "error": str(e)},
            "processing_step": "knowledge_searched"
        }

def search_web_node(state: RAGState) -> RAGState:
    """Search the web for information"""
    reasoner = RuleBasedReasoner()
    query = state["query"]
    
    try:
        # Analyze query intent
        intent = reasoner.extract_query_intent(query)
        
        # Generate optimized search query
        search_query = reasoner.generate_web_search_query(query, intent)
        
        # Perform web search
        web_result = reasoner.web_search._run(search_query)
        web_data = json.loads(web_result)
        
        return {
            **state,
            "web_results": web_data,
            "processing_step": "web_searched"
        }
    except Exception as e:
        return {
            **state,
            "web_results": {"error": str(e)},
            "processing_step": "web_searched"
        }

def save_knowledge_node(state: RAGState) -> RAGState:
    """Save new knowledge to the database"""
    reasoner = RuleBasedReasoner()
    query = state["query"]
    web_results = state["web_results"]
    
    try:
        # Synthesize web results
        synthesized_content = reasoner.synthesize_web_results(query, web_results)
        
        # Prepare for saving
        save_data = reasoner.prepare_knowledge_for_saving(query, synthesized_content)
        
        # Save to knowledge base
        save_result = reasoner.save_knowledge._run(**save_data)
        save_data_parsed = json.loads(save_result)
        
        return {
            **state,
            "knowledge_saved": save_data_parsed.get("success", False),
            "processing_step": "knowledge_saved"
        }
    except Exception as e:
        return {
            **state,
            "knowledge_saved": False,
            "processing_step": "knowledge_saved"
        }

def generate_response_node(state: RAGState) -> RAGState:
    """Generate final response based on available information"""
    reasoner = RuleBasedReasoner()
    messages = state["messages"]
    query = state["query"]
    firestore_results = state.get("firestore_results")
    web_results = state.get("web_results")
    knowledge_saved = state.get("knowledge_saved", False)
    
    # Generate response based on available data
    if firestore_results and firestore_results.get("found", False):
        # Use knowledge base
        response_content = reasoner.generate_response_from_knowledge_base(query, firestore_results)
        source_info = "\n\n*Response generated from existing knowledge base.*"
    
    elif web_results and not web_results.get("error"):
        # Use web results
        response_content = reasoner.synthesize_web_results(query, web_results)
        source_info = f"\n\n*Response generated from web research and {'saved to knowledge base' if knowledge_saved else 'could not be saved to knowledge base'}.*"
    
    else:
        # Fallback response
        response_content = f"I apologize, but I couldn't find sufficient information to answer your question about '{query}'. This might be due to connectivity issues or the topic being very specialized. Please try rephrasing your question or asking about a different topic."
        source_info = ""
    
    final_response = response_content + source_info
    ai_message = AIMessage(content=final_response)
    
    return {
        **state,
        "messages": messages + [ai_message],
        "processing_step": "completed"
    }

#=================================================================
# Routing Function
#=================================================================

def route_next_step(state: RAGState) -> str:
    """Route to next step based on current state"""
    reasoner = RuleBasedReasoner()
    next_action = reasoner.decide_next_action(state)
    
    routing_map = {
        'search_knowledge_base': 'search_kb',
        'generate_response_from_kb': 'generate_response',
        'search_web': 'search_web',
        'save_and_respond': 'save_knowledge',
        'generate_fallback_response': 'generate_response',
        'generate_final_response': 'generate_response',
        'end': 'end'
    }
    
    return routing_map.get(next_action, 'end')

#=================================================================
# Build the Graph
#=================================================================

def build_rule_based_rag_graph():
    """Build the rule-based RAG workflow graph"""
    workflow = StateGraph(RAGState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_state)
    workflow.add_node("search_kb", search_knowledge_base_node)
    workflow.add_node("search_web", search_web_node)
    workflow.add_node("save_knowledge", save_knowledge_node)
    workflow.add_node("generate_response", generate_response_node)
    
    # Set entry point
    workflow.set_entry_point("initialize")
    
    # Add edges
    workflow.add_edge("initialize", "search_kb")
    
    # Conditional routing from search_kb
    workflow.add_conditional_edges(
        "search_kb",
        route_next_step,
        {
            "search_web": "search_web",
            "generate_response": "generate_response",
            "end": END
        }
    )
    
    # From search_web, go to save_knowledge
    workflow.add_edge("search_web", "save_knowledge")
    
    # From save_knowledge, go to generate_response
    workflow.add_edge("save_knowledge", "generate_response")
    
    # End at generate_response
    workflow.add_edge("generate_response", END)
    
    # Compile with memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app

#=================================================================
# Rule-Based RAG Agent Class
#=================================================================

class RuleBasedRAGAgent:
    """Rule-based RAG Agent - no LLM reasoning required"""
    
    def __init__(self):
        self.app = build_rule_based_rag_graph()
        self.reasoner = RuleBasedReasoner()
    
    def query(self, question: str, thread_id: str = "default") -> str:
        """Query the rule-based RAG agent"""
        config = {"configurable": {"thread_id": thread_id}}
        
        # Create the initial state
        initial_state = {
            "messages": [HumanMessage(content=question)],
            "query": "",
            "firestore_results": None,
            "web_results": None,
            "knowledge_saved": False,
            "processing_step": "start"
        }
        
        try:
            # Run the workflow
            final_state = self.app.invoke(initial_state, config)
            
            # Get the final response
            messages = final_state["messages"]
            last_ai_message = None
            
            # Find the last AI message
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    last_ai_message = msg
                    break
            
            return last_ai_message.content if last_ai_message else "I apologize, but I couldn't generate a response."
        
        except Exception as e:
            return f"Error processing your query: {str(e)}. Please try again with a different question."
    
    def stream_query(self, question: str, thread_id: str = "default"):
        """Stream the RAG agent response"""
        config = {"configurable": {"thread_id": thread_id}}
        
        initial_state = {
            "messages": [HumanMessage(content=question)],
            "query": "",
            "firestore_results": None,
            "web_results": None,
            "knowledge_saved": False,
            "processing_step": "start"
        }
        
        # Stream the workflow
        for chunk in self.app.stream(initial_state, config):
            for step_name, step_data in chunk.items():
                if step_name == "generate_response" and "messages" in step_data:
                    messages = step_data["messages"]
                    if messages:
                        last_msg = messages[-1]
                        if isinstance(last_msg, AIMessage):
                            yield last_msg.content
    
    def get_conversation_history(self, thread_id: str = "default") -> List:
        """Get conversation history for a thread"""
        config = {"configurable": {"thread_id": thread_id}}
        state = self.app.get_state(config)
        return state.values.get("messages", []) if state.values else []
    
    def analyze_query(self, query: str) -> Dict:
        """Analyze a query and return intent information"""
        return self.reasoner.extract_query_intent(query)

#=================================================================
# Example Usage
#=================================================================

def main():
    """Example usage of the rule-based RAG agent"""
    agent = RuleBasedRAGAgent()
    
    # agent.analyze_query()
    
    # Test queries
    queries = ["Was there a volcanic eruption in the Russia today?",
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*80}")
        print(f"Query {i}: {query}")
        print(f"{'='*80}")
        
        # Analyze query first
        intent = agent.analyze_query(query)
        print(f"Query Analysis: {intent}")
        print(f"{'-'*80}")
        
        try:
            answer = agent.query(query, thread_id=f"test_{i}")
            print(f"\nAnswer:\n{answer}")
        except Exception as e:
            print(f"Error: {str(e)}")
        
        print(f"\n{'='*80}")

if __name__ == "__main__":
    main()