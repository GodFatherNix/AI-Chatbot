# 🤖 AI Help Bot

An AI-powered help bot for information retrieval from knowledge graphs created from static/dynamic content at web portals.

## 🌟 Features

- **Knowledge Graph Creation**: Automatically extracts and structures content from web portals
- **Semantic Search**: Uses AI embeddings for intelligent content matching
- **Multi-Modal Retrieval**: Combines vector search with graph-based relationships
- **AI-Powered Responses**: Generates contextual answers using OpenAI (optional)
- **Web Interface**: Beautiful Streamlit-based UI for easy interaction
- **REST API**: FastAPI backend for integration with other systems
- **Real-time Processing**: Instant responses to user queries

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Portal    │───▶│  Content        │───▶│  Knowledge      │
│   Content       │    │  Extraction     │    │  Graph          │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐           │
│   Streamlit     │◀───│   FastAPI       │◀──────────┤
│   Frontend      │    │   Backend       │           │
└─────────────────┘    └─────────────────┘           │
                                │                     │
                                ▼                     ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   AI Retrieval  │    │   Vector        │
                       │   Engine        │───▶│   Database      │
                       └─────────────────┘    │   (ChromaDB)    │
                                │             └─────────────────┘
                                ▼
                       ┌─────────────────┐
                       │   OpenAI API    │
                       │   (Optional)    │
                       └─────────────────┘
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd ai-chatbot

# Install dependencies
python manage.py install
# or manually:
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

Edit `.env` to configure your settings:
```env
# Optional: Add OpenAI API key for enhanced AI responses
OPENAI_API_KEY=your_openai_api_key_here

# Paths and settings (defaults are usually fine)
KNOWLEDGE_GRAPH_PATH=./data/knowledge_graph.json
VECTOR_DB_PATH=./data/chroma_db
```

### 3. Start the System

```bash
# Start both API and web interface
python manage.py start

# Or start them separately:
python manage.py api      # Start API server (port 8000)
python manage.py web      # Start web interface (port 8501)
```

### 4. Build Knowledge Graph

#### Via Web Interface:
1. Open http://localhost:8501
2. Use the sidebar to add URLs
3. Click "Build Knowledge Graph"

#### Via Command Line:
```bash
python manage.py build https://example.com https://docs.example.com
```

### 5. Start Querying!

Visit http://localhost:8501 and start asking questions!

## 💻 Usage Examples

### Command Line Interface

```bash
# Build knowledge graph from URLs
python manage.py build https://docs.python.org/3/ --max-depth 2 --max-pages 50

# Query the system
python manage.py query "What is machine learning?"

# View system statistics
python manage.py stats

# Clear all data
python manage.py clear
```

### REST API

```bash
# Query the knowledge base
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is machine learning?", "max_results": 5}'

# Build knowledge graph
curl -X POST "http://localhost:8000/build-knowledge-graph" \
     -H "Content-Type: application/json" \
     -d '{"urls": ["https://example.com"], "max_depth": 2, "max_pages": 100}'

# Get system stats
curl http://localhost:8000/stats
```

### Python API

```python
from ai_retrieval import AIRetrievalEngine

# Initialize the engine
engine = AIRetrievalEngine()

# Query the knowledge base
result = engine.answer_query("What is machine learning?")
print(f"Answer: {result['response']['answer']}")
print(f"Confidence: {result['response']['confidence']:.2%}")
```

## 📚 API Documentation

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/query` | POST | Query the knowledge base |
| `/build-knowledge-graph` | POST | Build knowledge graph from URLs |
| `/stats` | GET | Get system statistics |
| `/health` | GET | Health check |
| `/example-queries` | GET | Get example queries |
| `/clear-knowledge-base` | DELETE | Clear all data |

### Query Request Format

```json
{
  "query": "What is machine learning?",
  "max_results": 5
}
```

### Query Response Format

```json
{
  "query": "What is machine learning?",
  "answer": "Machine learning is a subset of artificial intelligence...",
  "confidence": 0.85,
  "sources": [
    {
      "id": "source_id",
      "type": "paragraph",
      "relevance_score": 0.92
    }
  ],
  "method": "ai_generated",
  "timestamp": "2024-01-01T12:00:00"
}
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (optional) | None |
| `KNOWLEDGE_GRAPH_PATH` | Path to save knowledge graph | `./data/knowledge_graph.json` |
| `VECTOR_DB_PATH` | Vector database path | `./data/chroma_db` |
| `WEB_SCRAPING_DELAY` | Delay between requests (seconds) | `1.0` |
| `MAX_CONTENT_LENGTH` | Max content length per chunk | `10000` |
| `API_HOST` | API server host | `0.0.0.0` |
| `API_PORT` | API server port | `8000` |
| `STREAMLIT_PORT` | Streamlit port | `8501` |

### Model Settings

The system uses `all-MiniLM-L6-v2` for embeddings by default. You can modify this in `config.py`:

```python
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast and efficient
# EMBEDDING_MODEL = "all-mpnet-base-v2"  # Higher quality, slower
```

## 🎯 Query Types

The AI Help Bot can handle various types of questions:

### 1. **Definitions**
- "What is machine learning?"
- "Define neural networks"
- "Meaning of API"

### 2. **How-to Questions**
- "How to implement authentication?"
- "How do I deploy a model?"
- "Steps to create a database"

### 3. **Lists**
- "List all available endpoints"
- "Show me all features"
- "What are the requirements?"

### 4. **Comparisons**
- "Compare React vs Vue"
- "Difference between SQL and NoSQL"
- "Python vs JavaScript"

### 5. **Troubleshooting**
- "Fix authentication error"
- "Resolve connection issues"
- "Debug performance problems"

## 🔍 How It Works

### 1. **Content Extraction**
- Crawls web pages using BeautifulSoup
- Extracts structured content (headings, paragraphs, links)
- Respects rate limiting and robots.txt

### 2. **Knowledge Graph Creation**
- Builds NetworkX graph with content relationships
- Creates nodes for pages, headings, and paragraphs
- Establishes edges for content hierarchy and links

### 3. **Vector Embeddings**
- Generates embeddings using Sentence Transformers
- Stores in ChromaDB for fast semantic search
- Chunks large content for optimal embedding quality

### 4. **Query Processing**
- Analyzes query intent and extracts keywords
- Performs hybrid search (semantic + keyword)
- Combines vector search with graph traversal

### 5. **Response Generation**
- Uses OpenAI for enhanced responses (if configured)
- Falls back to template-based responses
- Provides source attribution and confidence scores

## 🛠️ Development

### Project Structure

```
ai-chatbot/
├── api.py                 # FastAPI backend
├── streamlit_app.py       # Streamlit frontend
├── manage.py              # Management CLI
├── config.py              # Configuration
├── knowledge_graph.py     # Knowledge graph builder
├── vector_db.py           # Vector database interface
├── ai_retrieval.py        # AI retrieval engine
├── requirements.txt       # Dependencies
├── .env.example          # Environment template
└── data/                 # Data storage
    ├── knowledge_graph.json
    └── chroma_db/
```

### Adding New Features

1. **Custom Content Extractors**: Extend `ContentExtractor` class
2. **New Query Types**: Add patterns to `QueryProcessor.intent_patterns`
3. **Response Templates**: Add methods to `AIRetrievalEngine`
4. **API Endpoints**: Add routes to `api.py`

### Testing

```bash
# Test the system with example data
python manage.py build https://docs.python.org/3/tutorial/
python manage.py query "What is a function in Python?"
```

## 🔒 Security Considerations

- **Rate Limiting**: Implements delays between web requests
- **Input Validation**: Validates URLs and query parameters
- **Content Filtering**: Removes scripts and potentially harmful content
- **API Security**: Add authentication for production use

## 📈 Performance Tips

1. **Optimize Embedding Model**: Choose between speed and quality
2. **Chunk Size**: Adjust `CHUNK_SIZE` for your content type
3. **Vector Database**: ChromaDB is optimized for small to medium datasets
4. **Caching**: Implement caching for frequently queried content
5. **Parallel Processing**: Use async operations for large crawls

## 🐛 Troubleshooting

### Common Issues

**API not responding**
```bash
# Check if port is available
netstat -an | grep 8000
# Restart the API
python manage.py api
```

**No search results**
```bash
# Check system status
python manage.py stats
# Rebuild knowledge graph
python manage.py clear
python manage.py build <your-urls>
```

**Poor answer quality**
- Add more diverse content sources
- Use specific keywords in queries
- Configure OpenAI API key for better responses
- Adjust embedding model in config

**Memory issues with large datasets**
- Reduce `max_pages` parameter
- Increase `CHUNK_SIZE` to reduce total chunks
- Use more efficient embedding model

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Sentence Transformers** for embeddings
- **ChromaDB** for vector storage
- **NetworkX** for graph operations
- **FastAPI** for the API framework
- **Streamlit** for the web interface
- **OpenAI** for enhanced AI responses

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the API documentation

---

**Built with ❤️ for intelligent information retrieval**