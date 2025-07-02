#!/usr/bin/env python3
"""
Demo script for the AI Help Bot system
Demonstrates key features with sample data
"""

import time
import requests
from typing import List
from knowledge_graph import KnowledgeGraphBuilder
from vector_db import EmbeddingManager
from ai_retrieval import AIRetrievalEngine

def demo_knowledge_graph_creation():
    """Demonstrate knowledge graph creation"""
    print("🏗️ DEMO: Knowledge Graph Creation")
    print("=" * 50)
    
    # Sample URLs for demonstration (using public documentation)
    sample_urls = [
        "https://docs.python.org/3/tutorial/introduction.html",
        "https://docs.python.org/3/tutorial/controlflow.html"
    ]
    
    print(f"Building knowledge graph from {len(sample_urls)} sample URLs...")
    
    builder = KnowledgeGraphBuilder()
    
    try:
        # Build with limited scope for demo
        graph = builder.build_from_urls(sample_urls, max_depth=1, max_pages=10)
        
        print(f"✅ Knowledge graph created successfully!")
        print(f"   - Nodes: {graph.number_of_nodes()}")
        print(f"   - Edges: {graph.number_of_edges()}")
        
        # Show some sample nodes
        print("\n📋 Sample content nodes:")
        count = 0
        for node_id, node_data in graph.nodes(data=True):
            if node_data.get('type') == 'paragraph' and count < 3:
                text = node_data.get('text', '')[:100] + "..."
                print(f"   • {text}")
                count += 1
        
        return builder
        
    except Exception as e:
        print(f"❌ Error creating knowledge graph: {e}")
        return None

def demo_vector_embeddings(knowledge_graph_builder):
    """Demonstrate vector embedding creation"""
    print("\n🔍 DEMO: Vector Embeddings")
    print("=" * 50)
    
    if not knowledge_graph_builder:
        print("❌ Cannot create embeddings without knowledge graph")
        return None
    
    try:
        embedding_manager = EmbeddingManager()
        
        print("Creating vector embeddings from knowledge graph...")
        embedding_manager.build_embeddings_from_knowledge_graph(knowledge_graph_builder)
        
        # Test search
        print("\n🔎 Testing semantic search:")
        test_queries = [
            "Python introduction",
            "control flow",
            "functions"
        ]
        
        for query in test_queries:
            results = embedding_manager.vector_db.search(query, top_k=3)
            print(f"\nQuery: '{query}'")
            print(f"Found {len(results)} results")
            if results:
                for i, result in enumerate(results[:2]):
                    text = result['text'][:80] + "..."
                    distance = result.get('distance', 0)
                    print(f"   {i+1}. {text} (distance: {distance:.3f})")
        
        return embedding_manager
        
    except Exception as e:
        print(f"❌ Error creating embeddings: {e}")
        return None

def demo_ai_retrieval():
    """Demonstrate AI-powered retrieval"""
    print("\n🤖 DEMO: AI-Powered Retrieval")
    print("=" * 50)
    
    try:
        engine = AIRetrievalEngine()
        
        # Sample queries
        test_queries = [
            "What is Python?",
            "How to use if statements?",
            "Define a function in Python",
            "What are the Python data types?"
        ]
        
        for query in test_queries:
            print(f"\n❓ Query: {query}")
            print("-" * 40)
            
            result = engine.answer_query(query, max_results=3)
            
            response = result['response']
            print(f"🤖 Answer: {response['answer']}")
            print(f"📊 Confidence: {response['confidence']:.2%}")
            print(f"⚙️ Method: {response['method']}")
            print(f"📚 Sources: {len(response['sources'])}")
            
            time.sleep(1)  # Brief pause between queries
        
        return True
        
    except Exception as e:
        print(f"❌ Error with AI retrieval: {e}")
        return False

def demo_api_integration():
    """Demonstrate API integration"""
    print("\n🌐 DEMO: API Integration")
    print("=" * 50)
    
    api_url = "http://localhost:8000"
    
    try:
        # Check if API is running
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API is running and healthy")
            
            # Test query endpoint
            query_data = {
                "query": "What is Python?",
                "max_results": 3
            }
            
            response = requests.post(f"{api_url}/query", json=query_data, timeout=10)
            if response.status_code == 200:
                result = response.json()
                print(f"🤖 API Response: {result['answer'][:100]}...")
                print(f"📊 Confidence: {result['confidence']:.2%}")
            else:
                print(f"❌ Query failed: {response.status_code}")
            
            # Get stats
            response = requests.get(f"{api_url}/stats", timeout=5)
            if response.status_code == 200:
                stats = response.json()
                print(f"📈 System Stats:")
                print(f"   - Documents: {stats['vector_database'].get('document_count', 0)}")
                print(f"   - Graph Nodes: {stats['knowledge_graph'].get('nodes', 0)}")
                print(f"   - AI Integration: {stats['ai_integration']}")
                
        else:
            print(f"❌ API health check failed: {response.status_code}")
            
    except requests.exceptions.RequestException:
        print("⚠️ API is not running. Start it with: python manage.py api")
    except Exception as e:
        print(f"❌ Error testing API: {e}")

def demo_sample_data():
    """Create sample data for demonstration"""
    print("\n📝 DEMO: Creating Sample Data")
    print("=" * 50)
    
    # Sample content that would typically come from web scraping
    sample_content = [
        {
            'id': 'sample_1',
            'text': 'Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming.',
            'type': 'paragraph',
            'metadata': {'source': 'demo', 'topic': 'python_intro'}
        },
        {
            'id': 'sample_2', 
            'text': 'Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.',
            'type': 'paragraph',
            'metadata': {'source': 'demo', 'topic': 'ml_intro'}
        },
        {
            'id': 'sample_3',
            'text': 'A REST API (Representational State Transfer) is an architectural style for building web services that use HTTP methods like GET, POST, PUT, and DELETE.',
            'type': 'paragraph', 
            'metadata': {'source': 'demo', 'topic': 'api_intro'}
        }
    ]
    
    try:
        from vector_db import VectorDatabase
        vector_db = VectorDatabase()
        
        # Add sample documents
        vector_db.add_documents(sample_content)
        
        print(f"✅ Added {len(sample_content)} sample documents")
        
        # Test search with sample data
        print("\n🔎 Testing search with sample data:")
        queries = ["What is Python?", "machine learning", "REST API"]
        
        for query in queries:
            results = vector_db.search(query, top_k=2)
            print(f"\nQuery: '{query}'")
            if results:
                for result in results:
                    print(f"   • {result['text'][:60]}...")
            else:
                print("   No results found")
                
        return True
        
    except Exception as e:
        print(f"❌ Error creating sample data: {e}")
        return False

def main():
    """Run the complete demo"""
    print("🤖 AI Help Bot System Demo")
    print("=" * 60)
    print("This demo showcases the key features of the AI Help Bot system.")
    print("For the full experience, make sure the API server is running.")
    print()
    
    # Demo 1: Sample data creation (always works)
    demo_sample_data()
    
    # Demo 2: Knowledge graph creation (requires internet)
    try:
        kg_builder = demo_knowledge_graph_creation()
        if kg_builder:
            # Demo 3: Vector embeddings
            embedding_manager = demo_vector_embeddings(kg_builder)
    except Exception as e:
        print(f"⚠️ Skipping knowledge graph demo (internet required): {e}")
    
    # Demo 4: AI retrieval
    try:
        demo_ai_retrieval()
    except Exception as e:
        print(f"⚠️ AI retrieval demo failed: {e}")
    
    # Demo 5: API integration (requires API server)
    demo_api_integration()
    
    print("\n🎉 Demo Complete!")
    print("\nNext steps:")
    print("1. Start the full system: python manage.py start")
    print("2. Visit the web interface: http://localhost:8501")
    print("3. Build your own knowledge graph with: python manage.py build <urls>")
    print("4. Ask questions through the web interface or API")

if __name__ == "__main__":
    main()