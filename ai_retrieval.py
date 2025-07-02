import re
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime
from config import config
from vector_db import VectorDatabase
from knowledge_graph import KnowledgeGraphBuilder
import openai

logger = logging.getLogger(__name__)

class QueryProcessor:
    """Process and analyze user queries"""
    
    def __init__(self):
        self.intent_patterns = {
            'definition': [r'what is', r'define', r'meaning of', r'definition of'],
            'howto': [r'how to', r'how do i', r'how can i', r'steps to'],
            'comparison': [r'difference between', r'compare', r'vs', r'versus'],
            'list': [r'list', r'show me', r'give me all', r'what are the'],
            'troubleshooting': [r'problem', r'issue', r'error', r'not working', r'fix'],
            'location': [r'where', r'location', r'find'],
            'general': []  # Fallback
        }
    
    def analyze_query(self, query: str) -> Dict:
        """Analyze query to extract intent, entities, and keywords"""
        query_lower = query.lower()
        
        # Detect intent
        intent = self._detect_intent(query_lower)
        
        # Extract keywords (simple approach - remove stop words)
        keywords = self._extract_keywords(query)
        
        # Extract entities (basic named entity recognition)
        entities = self._extract_entities(query)
        
        return {
            'original_query': query,
            'intent': intent,
            'keywords': keywords,
            'entities': entities,
            'processed_at': datetime.now().isoformat()
        }
    
    def _detect_intent(self, query: str) -> str:
        """Detect the intent of the query"""
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query):
                    return intent
        return 'general'
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from the query"""
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can'
        }
        
        # Simple tokenization and filtering
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities (basic implementation)"""
        # This is a simple approach - in production, you'd use spaCy or similar
        entities = []
        
        # Look for capitalized words (potential proper nouns)
        capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', query)
        entities.extend(capitalized_words)
        
        # Look for quoted phrases
        quoted_phrases = re.findall(r'"([^"]*)"', query)
        entities.extend(quoted_phrases)
        
        return list(set(entities))

class AIRetrievalEngine:
    """Main AI retrieval engine that combines knowledge graph and vector search"""
    
    def __init__(self):
        self.vector_db = VectorDatabase()
        self.knowledge_graph = KnowledgeGraphBuilder()
        self.query_processor = QueryProcessor()
        
        # Initialize OpenAI if API key is available
        if config.OPENAI_API_KEY:
            openai.api_key = config.OPENAI_API_KEY
            self.use_openai = True
            logger.info("OpenAI integration enabled")
        else:
            self.use_openai = False
            logger.info("OpenAI integration disabled (no API key)")
    
    def answer_query(self, query: str, max_results: int = 5) -> Dict:
        """Process a query and return a comprehensive answer"""
        logger.info(f"Processing query: '{query}'")
        
        # Analyze the query
        query_analysis = self.query_processor.analyze_query(query)
        
        # Retrieve relevant information
        retrieval_results = self._retrieve_information(query, query_analysis, max_results)
        
        # Generate response
        response = self._generate_response(query, query_analysis, retrieval_results)
        
        return {
            'query': query,
            'query_analysis': query_analysis,
            'retrieval_results': retrieval_results,
            'response': response,
            'timestamp': datetime.now().isoformat()
        }
    
    def _retrieve_information(self, query: str, query_analysis: Dict, max_results: int) -> Dict:
        """Retrieve relevant information using multiple strategies"""
        results = {
            'vector_search': [],
            'graph_search': [],
            'combined_score': []
        }
        
        # Vector-based semantic search
        try:
            vector_results = self.vector_db.hybrid_search(
                query=query,
                keywords=query_analysis['keywords'],
                top_k=max_results * 2
            )
            results['vector_search'] = vector_results
        except Exception as e:
            logger.error(f"Vector search error: {e}")
        
        # Knowledge graph search
        try:
            graph_results = self._search_knowledge_graph(query_analysis)
            results['graph_search'] = graph_results
        except Exception as e:
            logger.error(f"Knowledge graph search error: {e}")
        
        # Combine and rank results
        results['combined_score'] = self._combine_results(
            vector_results=results['vector_search'],
            graph_results=results['graph_search'],
            max_results=max_results
        )
        
        return results
    
    def _search_knowledge_graph(self, query_analysis: Dict) -> List[Dict]:
        """Search the knowledge graph for relevant nodes"""
        graph_results = []
        
        # Search for nodes containing keywords
        for keyword in query_analysis['keywords']:
            for node_id, node_data in self.knowledge_graph.graph.nodes(data=True):
                if node_data.get('type') in ['heading', 'paragraph']:
                    text = node_data.get('text', '').lower()
                    if keyword.lower() in text:
                        # Get related content
                        related_content = self.knowledge_graph.get_related_content(node_id)
                        
                        graph_results.append({
                            'id': node_id,
                            'text': node_data.get('text', ''),
                            'type': node_data.get('type'),
                            'keyword_match': keyword,
                            'related_content': related_content[:3]  # Limit related content
                        })
        
        return graph_results[:10]  # Limit results
    
    def _combine_results(self, vector_results: List[Dict], graph_results: List[Dict], max_results: int) -> List[Dict]:
        """Combine and rank results from different sources"""
        combined = []
        seen_texts = set()
        
        # Add vector search results with scores
        for result in vector_results:
            text = result['text']
            if text not in seen_texts:
                combined.append({
                    **result,
                    'source': 'vector_search',
                    'relevance_score': 1.0 - (result.get('distance', 0.5))  # Convert distance to relevance
                })
                seen_texts.add(text)
        
        # Add graph search results
        for result in graph_results:
            text = result['text']
            if text not in seen_texts:
                combined.append({
                    **result,
                    'source': 'knowledge_graph',
                    'relevance_score': 0.8  # Base score for graph matches
                })
                seen_texts.add(text)
        
        # Sort by relevance score
        combined.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return combined[:max_results]
    
    def _generate_response(self, query: str, query_analysis: Dict, retrieval_results: Dict) -> Dict:
        """Generate a comprehensive response to the query"""
        combined_results = retrieval_results['combined_score']
        
        if not combined_results:
            return {
                'answer': "I couldn't find specific information related to your query in the knowledge base.",
                'sources': [],
                'confidence': 0.0,
                'method': 'fallback'
            }
        
        # Try to generate AI-powered response first
        if self.use_openai:
            ai_response = self._generate_ai_response(query, query_analysis, combined_results)
            if ai_response:
                return ai_response
        
        # Fallback to template-based response
        return self._generate_template_response(query, query_analysis, combined_results)
    
    def _generate_ai_response(self, query: str, query_analysis: Dict, results: List[Dict]) -> Optional[Dict]:
        """Generate response using OpenAI"""
        try:
            # Prepare context from retrieved results
            context_texts = []
            sources = []
            
            for result in results[:3]:  # Use top 3 results
                context_texts.append(result['text'])
                sources.append({
                    'id': result['id'],
                    'type': result.get('type', 'unknown'),
                    'relevance_score': result.get('relevance_score', 0)
                })
            
            context = "\n\n".join(context_texts)
            
            # Create prompt
            prompt = f"""Based on the following context information, please answer the user's question.

Context:
{context}

User Question: {query}

Please provide a helpful, accurate answer based on the context. If the context doesn't contain enough information to fully answer the question, acknowledge this and provide what information is available.

Answer:"""
            
            # Call OpenAI API
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )
            
            answer = response.choices[0].message.content.strip()
            
            return {
                'answer': answer,
                'sources': sources,
                'confidence': 0.9,
                'method': 'ai_generated'
            }
            
        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            return None
    
    def _generate_template_response(self, query: str, query_analysis: Dict, results: List[Dict]) -> Dict:
        """Generate template-based response"""
        intent = query_analysis['intent']
        
        if intent == 'definition':
            return self._generate_definition_response(results)
        elif intent == 'howto':
            return self._generate_howto_response(results)
        elif intent == 'list':
            return self._generate_list_response(results)
        else:
            return self._generate_general_response(results)
    
    def _generate_definition_response(self, results: List[Dict]) -> Dict:
        """Generate definition-style response"""
        if not results:
            return {
                'answer': "I couldn't find a definition for the requested term.",
                'sources': [],
                'confidence': 0.0,
                'method': 'template'
            }
        
        # Find the most relevant result
        best_result = results[0]
        
        answer = f"Based on the available information: {best_result['text']}"
        
        return {
            'answer': answer,
            'sources': [{'id': best_result['id'], 'type': best_result.get('type')}],
            'confidence': best_result.get('relevance_score', 0.5),
            'method': 'template'
        }
    
    def _generate_howto_response(self, results: List[Dict]) -> Dict:
        """Generate how-to style response"""
        if not results:
            return {
                'answer': "I couldn't find specific instructions for your request.",
                'sources': [],
                'confidence': 0.0,
                'method': 'template'
            }
        
        # Combine multiple results for comprehensive instructions
        steps = []
        sources = []
        
        for result in results[:3]:
            steps.append(result['text'])
            sources.append({'id': result['id'], 'type': result.get('type')})
        
        answer = "Here's what I found:\n\n" + "\n\n".join(f"• {step}" for step in steps)
        
        return {
            'answer': answer,
            'sources': sources,
            'confidence': 0.7,
            'method': 'template'
        }
    
    def _generate_list_response(self, results: List[Dict]) -> Dict:
        """Generate list-style response"""
        if not results:
            return {
                'answer': "I couldn't find the requested list information.",
                'sources': [],
                'confidence': 0.0,
                'method': 'template'
            }
        
        items = []
        sources = []
        
        for result in results:
            items.append(result['text'])
            sources.append({'id': result['id'], 'type': result.get('type')})
        
        answer = "Here are the relevant items I found:\n\n" + "\n".join(f"• {item}" for item in items)
        
        return {
            'answer': answer,
            'sources': sources,
            'confidence': 0.6,
            'method': 'template'
        }
    
    def _generate_general_response(self, results: List[Dict]) -> Dict:
        """Generate general response"""
        if not results:
            return {
                'answer': "I couldn't find specific information related to your query.",
                'sources': [],
                'confidence': 0.0,
                'method': 'template'
            }
        
        # Use the most relevant result
        best_result = results[0]
        
        answer = f"Based on the available information: {best_result['text']}"
        
        if len(results) > 1:
            answer += f"\n\nAdditional relevant information: {results[1]['text']}"
        
        sources = [{'id': r['id'], 'type': r.get('type')} for r in results[:2]]
        
        return {
            'answer': answer,
            'sources': sources,
            'confidence': best_result.get('relevance_score', 0.5),
            'method': 'template'
        }
    
    def get_stats(self) -> Dict:
        """Get statistics about the retrieval system"""
        vector_stats = self.vector_db.get_collection_stats()
        graph_stats = {
            'nodes': self.knowledge_graph.graph.number_of_nodes(),
            'edges': self.knowledge_graph.graph.number_of_edges()
        }
        
        return {
            'vector_database': vector_stats,
            'knowledge_graph': graph_stats,
            'ai_integration': self.use_openai
        }