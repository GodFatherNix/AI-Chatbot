# MOSDAC AI Help Bot

An intelligent virtual assistant leveraging NLP/ML for query understanding and precise information retrieval from the MOSDAC (Meteorological and Oceanographic Satellite Data Archival Centre) portal. This system extracts and models structured/unstructured content into a dynamic knowledge graph, supporting geospatial data intelligence for spatially-aware question answering.

## 🚀 Features

- **Intelligent Web Crawling**: Extracts content from static and dynamic web pages
- **Multi-format Content Processing**: Handles PDF, DOCX, XLSX, HTML, and text files
- **Advanced Entity Extraction**: Domain-specific entity recognition for satellite data
- **Knowledge Graph Construction**: Creates structured relationships between entities
- **Geospatial Intelligence**: Specialized parsing for satellite and location data
- **Interactive Chat Interface**: Multiple frontend options (Streamlit, Gradio)
- **RESTful API**: Scalable backend services
- **Visualization Tools**: Interactive knowledge graph visualization
- **Modular Architecture**: Easily deployable across different portals

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data          │    │   Knowledge      │    │   AI Models     │
│   Ingestion     │───▶│   Graph          │───▶│   & Chatbot     │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Crawler   │    │   Graph Builder  │    │   Query Engine  │
│   Content       │    │   Visualizer     │    │   RAG System    │
│   Extractor     │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   Streamlit      │    │   Gradio        │
│   Backend       │───▶│   Frontend       │    │   Interface     │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- Chrome/Chromium browser (for Selenium)
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd mosdac-ai-helpbot
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download required NLP models**
   ```bash
   python -m spacy download en_core_web_sm
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

5. **Set up configuration** (optional)
   ```bash
   cp .env.example .env
   # Edit .env with your specific settings
   ```

## 📖 Usage

### Quick Start

```python
from src.data_ingestion import MOSDACCrawler, ContentExtractor, DataPreprocessor
from src.knowledge_graph import EntityExtractor, RelationExtractor, GraphBuilder
from src.knowledge_graph import GraphVisualizer

# 1. Crawl MOSDAC portal
crawler = MOSDACCrawler()
pages = list(crawler.crawl_portal())

# 2. Extract and process content
extractor = ContentExtractor()
preprocessor = DataPreprocessor()

# Process crawled pages
processed_docs = []
for page in pages:
    # Extract content if it's a file
    if page.url.endswith(('.pdf', '.docx', '.xlsx')):
        content = extractor.extract_from_url(page.url)
        if content:
            processed_doc = preprocessor.process_document(content.__dict__)
            processed_docs.append(processed_doc)
    else:
        # Process web page content
        processed_doc = preprocessor.process_document({
            'file_hash': page.hash,
            'title': page.title,
            'text_content': page.content,
            'content_type': page.content_type,
            'metadata': page.metadata
        })
        processed_docs.append(processed_doc)

# 3. Build knowledge graph
entity_extractor = EntityExtractor()
relation_extractor = RelationExtractor()
graph_builder = GraphBuilder()

# Extract entities and relations
entities_by_doc = {}
relations_by_doc = {}

for doc in processed_docs:
    doc_id = doc.document_id
    entities = entity_extractor.extract_entities(doc.content, doc_id)
    relations = relation_extractor.extract_relations(doc.content, entities)
    
    entities_by_doc[doc_id] = entities
    relations_by_doc[doc_id] = relations

# Build the graph
graph_builder.build_graph_from_extractions(entities_by_doc, relations_by_doc)

# 4. Visualize the knowledge graph
visualizer = GraphVisualizer(graph_builder)
fig = visualizer.plot_interactive_graph()
fig.show()

# 5. Query the knowledge graph
results = graph_builder.query_graph(
    'find_entity',
    entity_name='INSAT',
    entity_type='Satellite'
)
print(f"Found {len(results)} satellite entities")
```

### Advanced Usage

#### Custom Web Crawling
```python
from src.data_ingestion import MOSDACCrawler

crawler = MOSDACCrawler()

# Crawl specific URLs
custom_urls = [
    'https://www.mosdac.gov.in/data/product/insat-3d',
    'https://www.mosdac.gov.in/data/product/oceansat-2'
]

for url in custom_urls:
    page = crawler.crawl_url(url, use_selenium=True)
    if page:
        print(f"Crawled: {page.title}")
```

#### Entity Extraction with Custom Patterns
```python
from src.knowledge_graph import EntityExtractor

extractor = EntityExtractor()

# Add custom entity patterns
extractor.entity_patterns['CustomType'] = {
    'patterns': [r'\b(Custom-\w+)\b'],
    'keywords': ['custom', 'special']
}

entities = extractor.extract_entities(text, document_id)
```

#### Knowledge Graph Queries
```python
# Find entities
entities = graph_builder.query_graph(
    'find_entity',
    entity_name='INSAT-3D',
    entity_type='Satellite'
)

# Find relationships
relations = graph_builder.query_graph(
    'find_relations',
    subject_id='satellite_id',
    relation_type='carries'
)

# Path queries
paths = graph_builder.query_graph(
    'path_query',
    source_name='INSAT-3D',
    target_name='ISRO'
)

# Neighborhood queries
neighborhood = graph_builder.query_graph(
    'neighborhood_query',
    entity_name='INSAT-3D',
    max_hops=2
)
```

## 🚀 Running the Application

### Backend API
```bash
cd src/api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Streamlit Frontend
```bash
streamlit run src/frontend/streamlit_app.py --server.port 8501
```

### Gradio Interface
```bash
python src/frontend/gradio_app.py
```

## 📁 Project Structure

```
mosdac-ai-helpbot/
├── src/
│   ├── data_ingestion/          # Web crawling and content extraction
│   │   ├── web_crawler.py       # MOSDAC portal crawler
│   │   ├── content_extractor.py # Multi-format content extraction
│   │   ├── data_preprocessor.py # Content cleaning and structuring
│   │   └── geospatial_parser.py # Geospatial metadata extraction
│   │
│   ├── knowledge_graph/         # Knowledge graph construction
│   │   ├── entity_extractor.py  # NLP-based entity extraction
│   │   ├── relation_extractor.py# Relationship identification
│   │   ├── graph_builder.py     # Graph construction and queries
│   │   └── graph_visualizer.py  # Interactive visualizations
│   │
│   ├── models/                  # AI models and training
│   ├── chatbot/                 # Conversational interface
│   ├── api/                     # FastAPI backend
│   ├── frontend/                # User interfaces
│   └── utils/                   # Configuration and utilities
│
├── data/                        # Data storage
├── tests/                       # Test suites
├── docs/                        # Documentation
├── config/                      # Configuration files
└── requirements.txt             # Dependencies
```

## 🔧 Configuration

### Main Configuration (`config/settings.yaml`)

```yaml
# Portal settings
portal:
  base_url: "https://www.mosdac.gov.in"
  crawl_depth: 3
  crawl_delay: 1

# Model settings  
models:
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  ner_model: "en_core_web_sm"

# Knowledge graph settings
knowledge_graph:
  entity_types:
    - "Satellite"
    - "Mission"
    - "Product"
    - "Service"
    - "Location"
    - "Date"
    - "Organization"
    - "Technology"
```

### Environment Variables (`.env`)

```bash
MOSDAC_DEBUG=true
MOSDAC_LOG_LEVEL=INFO
MOSDAC_API_HOST=0.0.0.0
MOSDAC_API_PORT=8000
MOSDAC_DB_PATH=data/mosdac_bot.db
```

## 📊 Data Pipeline

### 1. Data Ingestion
- **Web Crawling**: Crawls MOSDAC portal using BeautifulSoup and Selenium
- **Content Extraction**: Processes PDF, DOCX, XLSX files
- **Data Preprocessing**: Cleans and structures text content
- **Geospatial Parsing**: Extracts satellite and location metadata

### 2. Knowledge Graph Construction
- **Entity Extraction**: Identifies satellites, organizations, technologies, etc.
- **Relation Extraction**: Finds relationships between entities
- **Graph Building**: Creates NetworkX-based knowledge graph
- **Visualization**: Interactive plots using Plotly and Matplotlib

### 3. AI Models
- **Intent Classification**: Understands user queries
- **Entity Recognition**: Identifies mentioned entities
- **Response Generation**: RAG-based answer generation
- **Context Management**: Maintains conversation context

## 🎯 Key Features

### Intelligent Content Processing
- Handles multiple file formats (PDF, DOCX, XLSX, HTML, TXT)
- Cleans and normalizes text content
- Extracts structured information from unstructured text

### Domain-Specific Entity Recognition
- Recognizes satellite names (INSAT, IRS, Oceansat, etc.)
- Identifies organizations (ISRO, NASA, ESA, etc.)
- Extracts technologies and sensors
- Parses geospatial and temporal information

### Advanced Relationship Extraction
- Pattern-based relation identification
- Dependency parsing for complex relationships
- Proximity-based relationship inference
- Confidence scoring for all extractions

### Interactive Knowledge Graph
- Multiple visualization options
- Query capabilities for exploration
- Subgraph extraction and analysis
- Export options for external tools

### Modular Architecture
- Easy configuration for different portals
- Pluggable components
- Scalable deployment options
- Comprehensive testing framework

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/unit/
python -m pytest tests/integration/
python -m pytest tests/evaluation/

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## 📈 Evaluation Metrics

The system tracks several evaluation metrics:

- **Intent Recognition Accuracy**: How accurately user queries are interpreted
- **Entity Recognition F1**: Precision and recall of entity extraction
- **Response Completeness**: Coverage of answer relative to query context
- **Response Consistency**: Logical consistency across conversations
- **Retrieval Precision/Recall**: Quality of information retrieval

## 🚀 Deployment

### Docker Deployment
```bash
# Build image
docker build -t mosdac-ai-bot .

# Run container
docker run -p 8000:8000 -p 8501:8501 mosdac-ai-bot
```

### Production Considerations
- Use environment-specific configuration
- Set up proper logging and monitoring
- Configure rate limiting for API endpoints
- Implement caching for frequently accessed data
- Set up backup and recovery procedures

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the documentation in the `docs/` folder
- Review configuration examples in `config/`

## 🔮 Future Enhancements

- **Multi-language Support**: Extend to support regional languages
- **Real-time Updates**: Live synchronization with MOSDAC portal
- **Advanced Analytics**: Usage analytics and query insights
- **Mobile App**: Native mobile application
- **Voice Interface**: Speech-to-text query capabilities
- **Federated Search**: Integration with other satellite data portals

## 📚 References

- [MOSDAC Portal](https://www.mosdac.gov.in)
- [spaCy NLP Library](https://spacy.io/)
- [NetworkX Documentation](https://networkx.org/)
- [FastAPI Framework](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

**Note**: This project is designed to be modular and can be adapted for other web portals with similar architectures. The MOSDAC-specific components can be easily replaced or extended for different domains.