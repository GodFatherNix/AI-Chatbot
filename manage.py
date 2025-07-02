#!/usr/bin/env python3
"""
Management script for the AI Help Bot system
Provides command-line interface for common operations
"""

import argparse
import sys
import os
import json
import time
import subprocess
from typing import List
from config import config
from knowledge_graph import KnowledgeGraphBuilder
from vector_db import EmbeddingManager
from ai_retrieval import AIRetrievalEngine

def build_knowledge_graph(urls: List[str], max_depth: int = 2, max_pages: int = 100):
    """Build knowledge graph from URLs"""
    print(f"Building knowledge graph from {len(urls)} URLs...")
    print(f"Max depth: {max_depth}, Max pages: {max_pages}")
    
    builder = KnowledgeGraphBuilder()
    
    try:
        # Build the graph
        graph = builder.build_from_urls(urls, max_depth, max_pages)
        
        # Save the graph
        builder.save_to_file(config.KNOWLEDGE_GRAPH_PATH)
        print(f"Knowledge graph saved to {config.KNOWLEDGE_GRAPH_PATH}")
        
        # Build embeddings
        print("Building vector embeddings...")
        embedding_manager = EmbeddingManager()
        embedding_manager.build_embeddings_from_knowledge_graph(builder)
        
        print("✅ Knowledge graph and embeddings built successfully!")
        print(f"   - Nodes: {graph.number_of_nodes()}")
        print(f"   - Edges: {graph.number_of_edges()}")
        
    except Exception as e:
        print(f"❌ Error building knowledge graph: {e}")
        sys.exit(1)

def query_system(query: str, max_results: int = 5):
    """Query the AI system"""
    print(f"Querying: {query}")
    
    try:
        engine = AIRetrievalEngine()
        result = engine.answer_query(query, max_results)
        
        print("\n" + "="*60)
        print("QUERY RESULT")
        print("="*60)
        print(f"Query: {result['query']}")
        print(f"Answer: {result['response']['answer']}")
        print(f"Confidence: {result['response']['confidence']:.2%}")
        print(f"Method: {result['response']['method']}")
        print(f"Sources: {len(result['response']['sources'])}")
        
        if result['response']['sources']:
            print("\nSources:")
            for i, source in enumerate(result['response']['sources'][:3]):
                print(f"  {i+1}. {source['id']} ({source.get('type', 'unknown')})")
        
    except Exception as e:
        print(f"❌ Error querying system: {e}")
        sys.exit(1)

def show_stats():
    """Show system statistics"""
    try:
        engine = AIRetrievalEngine()
        stats = engine.get_stats()
        
        print("\n" + "="*60)
        print("SYSTEM STATISTICS")
        print("="*60)
        
        print("\nVector Database:")
        vector_stats = stats['vector_database']
        print(f"  Collection: {vector_stats.get('collection_name', 'N/A')}")
        print(f"  Documents: {vector_stats.get('document_count', 0)}")
        print(f"  Embedding Model: {vector_stats.get('embedding_model', 'N/A')}")
        
        print("\nKnowledge Graph:")
        graph_stats = stats['knowledge_graph']
        print(f"  Nodes: {graph_stats.get('nodes', 0)}")
        print(f"  Edges: {graph_stats.get('edges', 0)}")
        
        print("\nAI Integration:")
        print(f"  Status: {'✅ Active' if stats['ai_integration'] else '❌ Disabled'}")
        
    except Exception as e:
        print(f"❌ Error getting stats: {e}")
        sys.exit(1)

def clear_system():
    """Clear all system data"""
    print("⚠️  This will delete all knowledge graph and vector database data!")
    confirm = input("Type 'yes' to confirm: ")
    
    if confirm.lower() != 'yes':
        print("Operation cancelled.")
        return
    
    try:
        # Clear vector database
        from vector_db import VectorDatabase
        vector_db = VectorDatabase()
        vector_db.clear_collection()
        
        # Clear knowledge graph file
        if os.path.exists(config.KNOWLEDGE_GRAPH_PATH):
            os.remove(config.KNOWLEDGE_GRAPH_PATH)
        
        print("✅ System data cleared successfully!")
        
    except Exception as e:
        print(f"❌ Error clearing system: {e}")
        sys.exit(1)

def start_api():
    """Start the API server"""
    print("Starting API server...")
    try:
        subprocess.run([
            sys.executable, "api.py"
        ])
    except KeyboardInterrupt:
        print("\nAPI server stopped.")

def start_streamlit():
    """Start the Streamlit web interface"""
    print("Starting Streamlit web interface...")
    try:
        subprocess.run([
            "streamlit", "run", "streamlit_app.py",
            "--server.port", str(config.STREAMLIT_PORT),
            "--server.address", "0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\nStreamlit app stopped.")

def start_both():
    """Start both API and Streamlit"""
    import threading
    import time
    
    def run_api():
        subprocess.run([sys.executable, "api.py"])
    
    def run_streamlit():
        time.sleep(2)  # Wait for API to start
        subprocess.run([
            "streamlit", "run", "streamlit_app.py",
            "--server.port", str(config.STREAMLIT_PORT),
            "--server.address", "0.0.0.0"
        ])
    
    print("Starting both API server and Streamlit interface...")
    
    api_thread = threading.Thread(target=run_api)
    streamlit_thread = threading.Thread(target=run_streamlit)
    
    api_thread.start()
    streamlit_thread.start()
    
    try:
        api_thread.join()
        streamlit_thread.join()
    except KeyboardInterrupt:
        print("\nBoth services stopped.")

def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="AI Help Bot Management Script")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Build knowledge graph command
    build_parser = subparsers.add_parser('build', help='Build knowledge graph from URLs')
    build_parser.add_argument('urls', nargs='+', help='URLs to crawl')
    build_parser.add_argument('--max-depth', type=int, default=2, help='Maximum crawl depth')
    build_parser.add_argument('--max-pages', type=int, default=100, help='Maximum pages to crawl')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the knowledge base')
    query_parser.add_argument('query', help='Query string')
    query_parser.add_argument('--max-results', type=int, default=5, help='Maximum results to return')
    
    # Stats command
    subparsers.add_parser('stats', help='Show system statistics')
    
    # Clear command
    subparsers.add_parser('clear', help='Clear all system data')
    
    # Server commands
    subparsers.add_parser('api', help='Start API server')
    subparsers.add_parser('web', help='Start Streamlit web interface')
    subparsers.add_parser('start', help='Start both API and web interface')
    
    # Install command
    subparsers.add_parser('install', help='Install dependencies')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Ensure data directories exist
    config.ensure_data_dirs()
    
    if args.command == 'build':
        build_knowledge_graph(args.urls, args.max_depth, args.max_pages)
    
    elif args.command == 'query':
        query_system(args.query, args.max_results)
    
    elif args.command == 'stats':
        show_stats()
    
    elif args.command == 'clear':
        clear_system()
    
    elif args.command == 'api':
        start_api()
    
    elif args.command == 'web':
        start_streamlit()
    
    elif args.command == 'start':
        start_both()
    
    elif args.command == 'install':
        install_dependencies()

if __name__ == "__main__":
    main()