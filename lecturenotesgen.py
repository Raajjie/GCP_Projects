import json
import requests
from datetime import datetime
from pathlib import Path
import re
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from typing import Optional, Type
import json


class ResearchInput(BaseModel):
    topic: str = Field(description="The topic of the research")
    description: str = Field(description="The description of the research")


class SerperLectureNotesGenerator:
    name: str = "research_tool"
    description: str = "Search for information about a topic using Serper.dev (Google Search API)"

    args_schema: Optional[Type[BaseModel]] = ResearchInput


    def __init__(self, course_title="Research Notes", topic="General Research"):
        self.course_title = course_title
        self.topic = topic
        self.research_sections = []
        self.key_concepts = set()
        self.search_queries = []
        
    def search_with_serper(self, query: str, api_key: str, num_results: int = 10) -> Dict:
        """
        Search using Serper API and return results
        """
        url = "https://google.serper.dev/search"
        
        payload = json.dumps({
            "q": query,
            "num": num_results
        })
        
        headers = {
            'X-API-KEY': api_key,
            'Content-Type': 'application/json'
        }
        
        try:
            response = requests.request("POST", url, headers=headers, data=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error with Serper API: {e}")
            return {}
    
    def load_serper_json(self, json_file_path: str):
        """
        Load Serper API results from a saved JSON file
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.process_serper_results(data, f"Search Results from {json_file_path}")
        except FileNotFoundError:
            print(f"File {json_file_path} not found")
        except json.JSONDecodeError:
            print(f"Invalid JSON in {json_file_path}")
    
    def process_serper_results(self, serper_data: Dict, source_name: str = "Serper Search"):
        """
        Process Serper API results and extract relevant information
        """
        if not serper_data:
            return
            
        # Extract search query if available
        search_query = serper_data.get('searchParameters', {}).get('q', 'Unknown Query')
        self.search_queries.append(search_query)
        
        # Process organic results
        organic_results = serper_data.get('organic', [])
        if organic_results:
            self._process_organic_results(organic_results, search_query, source_name)
        
        # Process knowledge graph if available
        knowledge_graph = serper_data.get('knowledgeGraph')
        if knowledge_graph:
            self._process_knowledge_graph(knowledge_graph, search_query)
        
        # Process answer box if available
        answer_box = serper_data.get('answerBox')
        if answer_box:
            self._process_answer_box(answer_box, search_query)
        
        # Process people also ask
        people_also_ask = serper_data.get('peopleAlsoAsk', [])
        if people_also_ask:
            self._process_people_also_ask(people_also_ask, search_query)
    
    def _process_organic_results(self, organic_results: List[Dict], query: str, source_name: str):
        """Process organic search results"""
        content_parts = []
        sources = []
        
        for i, result in enumerate(organic_results[:5]):  # Limit to top 5 results
            title = result.get('title', 'No Title')
            snippet = result.get('snippet', 'No snippet available')
            link = result.get('link', '#')
            
            content_parts.append(f"**{title}**\n{snippet}")
            sources.append(f"{title} - {link}")
        
        if content_parts:
            combined_content = '\n\n'.join(content_parts)
            
            # Extract key terms from titles and snippets
            all_text = ' '.join([result.get('title', '') + ' ' + result.get('snippet', '') for result in organic_results[:5]])
            key_terms = self._extract_key_terms(all_text)
            
            self.add_research_section(
                title=f"Search Results: {query}",
                content=combined_content,
                source=f"{source_name} - {len(organic_results)} results",
                key_terms=key_terms,
                additional_sources=sources
            )
    
    def _process_knowledge_graph(self, kg_data: Dict, query: str):
        """Process knowledge graph data"""
        title = kg_data.get('title', 'Knowledge Graph')
        description = kg_data.get('description', '')
        
        # Combine attributes if available
        attributes = kg_data.get('attributes', {})
        attr_text = '\n'.join([f"**{k}:** {v}" for k, v in attributes.items()])
        
        content = f"{description}\n\n{attr_text}" if attr_text else description
        
        if content.strip():
            key_terms = self._extract_key_terms(title + ' ' + description)
            self.add_research_section(
                title=f"Knowledge Graph: {title}",
                content=content,
                source=f"Google Knowledge Graph for '{query}'",
                key_terms=key_terms
            )
    
    def _process_answer_box(self, answer_data: Dict, query: str):
        """Process answer box data"""
        answer = answer_data.get('answer', '')
        title = answer_data.get('title', 'Answer')
        
        if answer:
            key_terms = self._extract_key_terms(answer)
            self.add_research_section(
                title=f"Direct Answer: {query}",
                content=answer,
                source=f"Google Answer Box for '{query}'",
                key_terms=key_terms
            )
    
    def _process_people_also_ask(self, paa_data: List[Dict], query: str):
        """Process People Also Ask section"""
        questions_and_answers = []
        
        for item in paa_data[:5]:  # Limit to 5 questions
            question = item.get('question', '')
            snippet = item.get('snippet', '')
            if question and snippet:
                questions_and_answers.append(f"**Q: {question}**\nA: {snippet}")
        
        if questions_and_answers:
            content = '\n\n'.join(questions_and_answers)
            all_text = ' '.join([item.get('question', '') + ' ' + item.get('snippet', '') for item in paa_data])
            key_terms = self._extract_key_terms(all_text)
            
            self.add_research_section(
                title=f"Related Questions: {query}",
                content=content,
                source=f"Google People Also Ask for '{query}'",
                key_terms=key_terms
            )
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text"""
        # Remove common words
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'cannot', 'this', 'that', 'these', 'those', 'what', 'when', 'where', 'why', 'how'}
        
        # Find capitalized words and important terms
        words = re.findall(r'\b[A-Z][a-zA-Z]+\b', text)
        
        # Also find quoted terms and technical terms
        quoted_terms = re.findall(r'"([^"]+)"', text)
        technical_terms = re.findall(r'\b[a-zA-Z]+(?:-[a-zA-Z]+)*\b', text)
        
        all_terms = words + quoted_terms + [term for term in technical_terms if len(term) > 4]
        key_terms = [term for term in set(all_terms) if term.lower() not in common_words and len(term) > 2]
        
        return list(set(key_terms))[:10]  # Limit to 10 key terms
    
    def add_research_section(self, title: str, content: str, source: str = None, key_terms: List[str] = None, additional_sources: List[str] = None):
        """Add a research section with automatic processing"""
        section = {
            'title': title,
            'content': content,
            'source': source or 'Unknown Source',
            'key_terms': key_terms or [],
            'summary': self._generate_summary(content),
            'main_points': self._extract_main_points(content),
            'additional_sources': additional_sources or []
        }
        
        self.research_sections.append(section)
        
        # Add key terms to global set
        if key_terms:
            self.key_concepts.update(key_terms)
    
    def _generate_summary(self, content: str) -> str:
        """Generate a brief summary from content"""
        # Clean content of markdown formatting for summary
        clean_content = re.sub(r'\*\*([^*]+)\*\*', r'\1', content)
        clean_content = re.sub(r'\*([^*]+)\*', r'\1', clean_content)
        
        sentences = re.split(r'[.!?]+', clean_content)
        if len(sentences) >= 2:
            return f"{sentences[0].strip()}.{sentences[1].strip()}."
        return clean_content[:200] + "..." if len(clean_content) > 200 else clean_content
    
    def _extract_main_points(self, content: str) -> List[str]:
        """Extract main points from content"""
        points = []
        
        # Look for existing bullet points or bold text
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('**') and line.endswith('**'):
                points.append(f"• {line.replace('**', '')}")
            elif re.match(r'^[-*•]\s*(.+)$', line):
                points.append(line if line.startswith('•') else f"• {line[2:]}")
        
        # If no structured points found, extract key sentences
        if not points:
            sentences = re.split(r'[.!?]+', content)
            for sentence in sentences[:3]:
                if len(sentence.strip()) > 20:
                    points.append(f"• {sentence.strip()}")
        
        return points[:5]
    
    def export_to_lecture_notes_md(self, output_file: str):
        """Export as structured Markdown lecture notes"""
        with open(output_file, 'w', encoding='utf-8') as f:
            # Header
            f.write(f"# {self.course_title}\n")
            f.write(f"## Topic: {self.topic}\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%B %d, %Y')}\n")
            f.write(f"**Search Queries:** {', '.join(self.search_queries)}\n")
            f.write(f"**Total Sections:** {len(self.research_sections)}\n\n")
            f.write("---\n\n")
            
            # Table of Contents
            f.write("## 📋 Table of Contents\n\n")
            for i, section in enumerate(self.research_sections, 1):
                safe_title = re.sub(r'[^\w\s-]', '', section['title'].lower().replace(' ', '-'))
                f.write(f"{i}. [{section['title']}](#{safe_title})\n")
            f.write("\n---\n\n")
            
            # Key Concepts Overview
            if self.key_concepts:
                f.write("## 🔑 Key Concepts\n\n")
                concepts_per_row = 3
                concepts_list = list(sorted(self.key_concepts))
                for i in range(0, len(concepts_list), concepts_per_row):
                    row_concepts = concepts_list[i:i+concepts_per_row]
                    f.write("| " + " | ".join([f"**{concept}**" for concept in row_concepts]) + " |\n")
                    if i == 0:  # Add header separator only once
                        f.write("| " + " | ".join(["---" for _ in row_concepts]) + " |\n")
                f.write("\n---\n\n")
            
            # Main Content
            for i, section in enumerate(self.research_sections, 1):
                safe_title = re.sub(r'[^\w\s-]', '', section['title'].lower().replace(' ', '-'))
                f.write(f"## {i}. {section['title']} {{#{safe_title}}}\n\n")
                
                # Quick Summary
                f.write("### 📝 Summary\n")
                f.write(f"> {section['summary']}\n\n")
                
                # Key Points
                if section['main_points']:
                    f.write("### 🎯 Key Points\n")
                    for point in section['main_points']:
                        f.write(f"{point}\n")
                    f.write("\n")
                
                # Full Content
                f.write("### 📖 Detailed Information\n")
                f.write(f"{section['content']}\n\n")
                
                # Sources
                f.write("### 📚 Sources\n")
                f.write(f"**Primary:** {section['source']}\n")
                if section.get('additional_sources'):
                    f.write("**Additional Sources:**\n")
                    for source in section['additional_sources'][:3]:  # Limit to 3 additional sources
                        f.write(f"- {source}\n")
                f.write("\n")
                
                f.write("---\n\n")
            
            # Study Guide Section
            f.write("## 📚 Study Guide\n\n")
            f.write("### Review Checklist\n")
            for i, section in enumerate(self.research_sections, 1):
                f.write(f"- [ ] **{section['title']}** - Can you explain the key concepts?\n")
            
            f.write("\n### Practice Questions\n")
            for i, section in enumerate(self.research_sections, 1):
                f.write(f"**{i}. Based on '{section['title']}':**\n")
                f.write(f"- What are the main takeaways?\n")
                f.write(f"- How does this connect to other topics?\n")
                f.write(f"- What real-world applications exist?\n\n")

# Example usage for Serper API
def main():
    # Initialize the notes generator
    notes = SerperLectureNotesGenerator(
        course_title="AI Research Compilation",
        topic="Artificial Intelligence and Machine Learning"
    )
    
    # Method 1: Load from saved Serper JSON results
    # notes.load_serper_json('serper_results.json')
    
    # Method 2: Search directly with Serper API (requires API key)
    # api_key = "your_serper_api_key_here"
    # results = notes.search_with_serper("machine learning applications", api_key)
    # notes.process_serper_results(results, "Live Serper Search")
    
    # Method 3: Process sample Serper data (for demonstration)
    sample_serper_data = {
        "searchParameters": {"q": "artificial intelligence trends 2024"},
        "organic": [
            {
                "title": "AI Trends in 2024: What to Expect",
                "snippet": "Artificial intelligence continues to evolve rapidly with advances in large language models, computer vision, and autonomous systems. Key trends include improved reasoning capabilities and ethical AI development.",
                "link": "https://example.com/ai-trends-2024"
            },
            {
                "title": "Machine Learning Applications in Industry",
                "snippet": "Industries are adopting ML for predictive analytics, automation, and decision support. Healthcare, finance, and manufacturing lead in AI implementation.",
                "link": "https://example.com/ml-applications"
            }
        ],
        "answerBox": {
            "answer": "Artificial Intelligence (AI) refers to computer systems that can perform tasks typically requiring human intelligence, including learning, reasoning, and perception.",
            "title": "What is Artificial Intelligence?"
        }
    }
    
    notes.process_serper_results(sample_serper_data, "Sample Serper Search")
    
    # Generate lecture notes
    notes.export_to_lecture_notes_md('serper_lecture_notes.md')
    
    print("✅ Lecture notes from Serper API data generated successfully!")
    print("📄 Output file: serper_lecture_notes.md")
    print(f"📊 Processed {len(notes.research_sections)} sections")
    print(f"🔑 Extracted {len(notes.key_concepts)} key concepts")

if __name__ == "__main__":
    main()