import streamlit as st
import requests
import json
from datetime import datetime
from typing import Dict, List
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from config import config

# Configure Streamlit page
st.set_page_config(
    page_title="AI Help Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API base URL
API_BASE_URL = f"http://localhost:{config.API_PORT}"

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1e88e5;
        margin-bottom: 2rem;
    }
    .query-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1e88e5;
        margin: 1rem 0;
    }
    .response-box {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
    .source-item {
        background-color: #fff3e0;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.3rem 0;
        border-left: 3px solid #ff9800;
    }
    .stats-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_system_stats():
    """Get system statistics from the API"""
    try:
        response = requests.get(f"{API_BASE_URL}/stats")
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def query_knowledge_base(query: str, max_results: int = 5):
    """Query the knowledge base through the API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={"query": query, "max_results": max_results}
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error querying knowledge base: {response.text}")
    except Exception as e:
        st.error(f"Failed to connect to API: {str(e)}")
    return None

def build_knowledge_graph(urls: List[str], max_depth: int = 2, max_pages: int = 100):
    """Build knowledge graph from URLs"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/build-knowledge-graph",
            json={
                "urls": urls,
                "max_depth": max_depth,
                "max_pages": max_pages
            }
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error building knowledge graph: {response.text}")
    except Exception as e:
        st.error(f"Failed to connect to API: {str(e)}")
    return None

def get_example_queries():
    """Get example queries from the API"""
    try:
        response = requests.get(f"{API_BASE_URL}/example-queries")
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 AI Help Bot</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">AI-powered information retrieval from knowledge graphs</p>', unsafe_allow_html=True)
    
    # Check API health
    if not check_api_health():
        st.error("⚠️ API is not running. Please start the API server first.")
        st.code("python api.py")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("🛠️ Configuration")
        
        # System stats
        stats = get_system_stats()
        if stats:
            st.subheader("📊 System Status")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", stats['vector_database'].get('document_count', 0))
            with col2:
                st.metric("Graph Nodes", stats['knowledge_graph'].get('nodes', 0))
            
            st.metric("Graph Edges", stats['knowledge_graph'].get('edges', 0))
            
            if stats['ai_integration']:
                st.success("✅ AI Integration Active")
            else:
                st.warning("⚠️ AI Integration Disabled")
        
        st.divider()
        
        # Knowledge Graph Management
        st.subheader("🕸️ Knowledge Graph")
        
        with st.expander("Build Knowledge Graph"):
            st.write("Add new content sources to the knowledge base:")
            
            urls_input = st.text_area(
                "URLs (one per line)",
                placeholder="https://example.com\nhttps://docs.example.com",
                height=100
            )
            
            col1, col2 = st.columns(2)
            with col1:
                max_depth = st.number_input("Max Depth", min_value=1, max_value=5, value=2)
            with col2:
                max_pages = st.number_input("Max Pages", min_value=1, max_value=1000, value=100)
            
            if st.button("🚀 Build Knowledge Graph"):
                if urls_input.strip():
                    urls = [url.strip() for url in urls_input.strip().split('\n') if url.strip()]
                    
                    with st.spinner("Building knowledge graph... This may take a while."):
                        result = build_knowledge_graph(urls, max_depth, max_pages)
                        if result:
                            st.success("Knowledge graph building started! Check the logs for progress.")
                else:
                    st.error("Please provide at least one URL.")
        
        with st.expander("Clear Knowledge Base"):
            st.warning("⚠️ This will delete all stored information.")
            if st.button("🗑️ Clear Knowledge Base"):
                try:
                    response = requests.delete(f"{API_BASE_URL}/clear-knowledge-base")
                    if response.status_code == 200:
                        st.success("Knowledge base cleared successfully!")
                        st.experimental_rerun()
                    else:
                        st.error("Failed to clear knowledge base.")
                except:
                    st.error("Failed to connect to API.")
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📈 Analytics", "❓ Help"])
    
    with tab1:
        chat_interface()
    
    with tab2:
        analytics_interface()
    
    with tab3:
        help_interface()

def chat_interface():
    """Chat interface for querying the knowledge base"""
    st.header("💬 Ask the AI Help Bot")
    
    # Query input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_input(
            "Enter your question:",
            placeholder="e.g., What is machine learning?",
            key="query_input"
        )
    
    with col2:
        max_results = st.selectbox("Max Results", [3, 5, 10], index=1)
    
    # Example queries
    example_queries = get_example_queries()
    if example_queries:
        st.subheader("💡 Example Queries")
        
        cols = st.columns(3)
        for i, example in enumerate(example_queries['examples'][:6]):
            with cols[i % 3]:
                if st.button(f"📝 {example['query']}", key=f"example_{i}"):
                    st.session_state.query_input = example['query']
                    st.experimental_rerun()
    
    # Process query
    if query:
        st.markdown(f'<div class="query-box">🔍 <strong>Your Question:</strong> {query}</div>', unsafe_allow_html=True)
        
        with st.spinner("🤖 AI is thinking..."):
            result = query_knowledge_base(query, max_results)
        
        if result:
            # Display response
            st.markdown(f'<div class="response-box">🤖 <strong>AI Response:</strong><br>{result["answer"]}</div>', unsafe_allow_html=True)
            
            # Response metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Confidence", f"{result['confidence']:.2%}")
            with col2:
                st.metric("Method", result['method'].replace('_', ' ').title())
            with col3:
                st.metric("Sources", len(result['sources']))
            
            # Display sources
            if result['sources']:
                st.subheader("📚 Sources")
                
                for i, source in enumerate(result['sources']):
                    with st.expander(f"Source {i+1}: {source.get('type', 'Unknown').title()}"):
                        st.markdown(f"**ID:** {source['id']}")
                        if 'relevance_score' in source:
                            st.markdown(f"**Relevance:** {source['relevance_score']:.2%}")
                        st.markdown("---")
            
            # Feedback
            st.subheader("📝 Feedback")
            feedback_col1, feedback_col2 = st.columns(2)
            
            with feedback_col1:
                if st.button("👍 Helpful"):
                    submit_feedback(query, result, "positive")
                    st.success("Thank you for your feedback!")
            
            with feedback_col2:
                if st.button("👎 Not Helpful"):
                    submit_feedback(query, result, "negative")
                    st.success("Thank you for your feedback!")

def analytics_interface():
    """Analytics dashboard"""
    st.header("📈 System Analytics")
    
    stats = get_system_stats()
    if not stats:
        st.error("Unable to load system statistics")
        return
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📄 Total Documents",
            stats['vector_database'].get('document_count', 0)
        )
    
    with col2:
        st.metric(
            "🔗 Graph Nodes",
            stats['knowledge_graph'].get('nodes', 0)
        )
    
    with col3:
        st.metric(
            "🌐 Graph Edges",
            stats['knowledge_graph'].get('edges', 0)
        )
    
    with col4:
        ai_status = "Active" if stats['ai_integration'] else "Disabled"
        st.metric("🤖 AI Integration", ai_status)
    
    # Knowledge graph visualization
    st.subheader("🕸️ Knowledge Graph Overview")
    
    if stats['knowledge_graph'].get('nodes', 0) > 0:
        # Create a simple visualization
        nodes = stats['knowledge_graph'].get('nodes', 0)
        edges = stats['knowledge_graph'].get('edges', 0)
        
        # Simple network density calculation
        if nodes > 1:
            density = (2 * edges) / (nodes * (nodes - 1))
        else:
            density = 0
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Network metrics
            metrics_data = {
                'Metric': ['Nodes', 'Edges', 'Density', 'Avg. Connections'],
                'Value': [nodes, edges, f"{density:.3f}", f"{edges/nodes if nodes > 0 else 0:.1f}"]
            }
            
            df_metrics = pd.DataFrame(metrics_data)
            st.table(df_metrics)
        
        with col2:
            # Simple pie chart for visualization
            fig = go.Figure(data=[go.Pie(
                labels=['Connected', 'Potential Connections'],
                values=[edges, max(0, nodes * (nodes - 1) / 2 - edges)],
                hole=.3
            )])
            fig.update_layout(title="Graph Connectivity")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No knowledge graph data available. Build a knowledge graph first!")
    
    # Vector database info
    st.subheader("🔍 Vector Database Status")
    
    vector_info = stats['vector_database']
    
    st.write(f"**Collection:** {vector_info.get('collection_name', 'N/A')}")
    st.write(f"**Embedding Model:** {vector_info.get('embedding_model', 'N/A')}")
    st.write(f"**Document Count:** {vector_info.get('document_count', 0)}")

def help_interface():
    """Help and documentation interface"""
    st.header("❓ Help & Documentation")
    
    st.markdown("""
    ### 🚀 Getting Started
    
    1. **Build Knowledge Graph**: Add URLs in the sidebar to create a knowledge base
    2. **Ask Questions**: Use the Chat tab to query your knowledge base
    3. **View Analytics**: Monitor system performance in the Analytics tab
    
    ### 💬 Query Types
    
    The AI Help Bot can handle various types of questions:
    
    - **Definitions**: "What is machine learning?"
    - **How-to**: "How do I implement authentication?"
    - **Lists**: "List all available APIs"
    - **Comparisons**: "Compare React vs Vue"
    - **Troubleshooting**: "Fix login error"
    
    ### 🔧 Features
    
    - **Semantic Search**: Uses AI embeddings for intelligent content matching
    - **Knowledge Graphs**: Maintains relationships between content pieces
    - **Multi-source**: Can integrate content from multiple web portals
    - **Real-time**: Instant responses to user queries
    """)
    
    st.subheader("🛠️ API Endpoints")
    
    api_endpoints = [
        {"Endpoint": "/query", "Method": "POST", "Description": "Query the knowledge base"},
        {"Endpoint": "/build-knowledge-graph", "Method": "POST", "Description": "Build knowledge graph from URLs"},
        {"Endpoint": "/stats", "Method": "GET", "Description": "Get system statistics"},
        {"Endpoint": "/health", "Method": "GET", "Description": "Health check"},
        {"Endpoint": "/example-queries", "Method": "GET", "Description": "Get example queries"},
    ]
    
    df_endpoints = pd.DataFrame(api_endpoints)
    st.table(df_endpoints)
    
    st.subheader("⚙️ Configuration")
    
    st.code(f"""
    API URL: {API_BASE_URL}
    Vector DB Path: {config.VECTOR_DB_PATH}
    Knowledge Graph Path: {config.KNOWLEDGE_GRAPH_PATH}
    Embedding Model: {config.EMBEDDING_MODEL}
    """)
    
    st.subheader("🐛 Troubleshooting")
    
    with st.expander("Common Issues"):
        st.markdown("""
        **API not responding**
        - Make sure the API server is running: `python api.py`
        - Check if the port 8000 is available
        
        **No search results**
        - Build a knowledge graph first using the sidebar
        - Ensure URLs are accessible and contain content
        
        **Poor answer quality**
        - Add more diverse content sources
        - Use specific keywords in your queries
        - Provide feedback to improve the system
        """)

def submit_feedback(query: str, result: Dict, feedback_type: str):
    """Submit feedback to the API"""
    try:
        feedback_data = {
            "query": query,
            "result_id": result.get("timestamp"),
            "feedback_type": feedback_type,
            "confidence": result.get("confidence"),
            "method": result.get("method"),
            "timestamp": datetime.now().isoformat()
        }
        
        requests.post(f"{API_BASE_URL}/feedback", json=feedback_data)
    except:
        pass  # Feedback is optional

if __name__ == "__main__":
    main()