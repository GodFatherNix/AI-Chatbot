# MOSDAC AI Chatbot - Complete Implementation Guide

A comprehensive AI-powered conversational help bot for the MOSDAC (Meteorological & Oceanographic Satellite Data Archival Centre) portal that leverages NLP/ML for intelligent information retrieval from satellite data and services.

## 🎯 Project Overview

This system provides:
- **Intelligent Query Understanding**: Intent classification and entity recognition for satellite data queries
- **Knowledge Graph**: Entity and relationship mapping across MOSDAC portal content
- **Vector Search**: Semantic search over crawled documents and FAQs
- **RAG Pipeline**: Retrieval-Augmented Generation for contextual responses
- **Geospatial Intelligence**: Location-aware querying for satellite coverage
- **Modern UI**: React-based chat interface with citation support

## 🏗️ Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Frontend  │────│     API      │────│   Vector    │
│  (React)    │    │  (FastAPI)   │    │ DB (Qdrant) │
└─────────────┘    └──────────────┘    └─────────────┘
                          │
                   ┌──────────────┐    ┌─────────────┐
                   │ Knowledge    │────│   Neo4j     │
                   │ Graph Engine │    │ Database    │
                   └──────────────┘    └─────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- Docker & Docker Compose
- OpenAI API Key (for LLM)

### 1. Clone and Setup

```bash
git clone <repository>
cd mosdac-chatbot

# Create environment file
cat > .env << EOF
OPENAI_API_KEY=your_openai_api_key_here
WANDB_API_KEY=your_wandb_key_optional
EOF
```

### 2. Start Infrastructure

```bash
# Start databases and core services
docker-compose up -d qdrant neo4j

# Wait for services to be ready
sleep 30
```

### 3. Data Ingestion Pipeline

```bash
# Install dependencies
pip install -r ingestion/requirements.txt
pip install -r processing/requirements.txt
pip install -r kg/requirements.txt

# Crawl MOSDAC portal
cd ingestion/mosdac_crawler
scrapy crawl mosdac
cd ../..

# Process documents and create embeddings
python processing/process_documents.py --input data/mosdac_crawl_*.jl

# Build knowledge graph
python kg/build_kg.py --input data/mosdac_crawl_*.jl

# Install spaCy model
python -m spacy download en_core_web_sm
```

### 4. Start Services

```bash
# Install service dependencies
pip install -r service/requirements.txt

# Start backend API
uvicorn service.app:app --reload --port 8000 &

# Install and start frontend
cd frontend
npm install
npm run dev &
cd ..
```

### 5. Access the Application

- **Frontend UI**: http://localhost:5173
- **API Documentation**: http://localhost:8000/docs
- **Neo4j Browser**: http://localhost:7474
- **Qdrant Dashboard**: http://localhost:6333/dashboard

## 📋 Implementation Steps

### Step 1: Data Ingestion
```bash
# Navigate to crawler
cd ingestion/mosdac_crawler

# Run web crawler
scrapy crawl mosdac

# Check output
ls -la data/
```

### Step 2: Document Processing
```bash
# Process crawled data
python processing/process_documents.py \
  --input data/mosdac_crawl_20240101T120000Z.jl \
  --max_tokens 500 \
  --overlap 50
```

### Step 3: Knowledge Graph Construction
```bash
# Build knowledge graph
python kg/build_kg.py --input data/mosdac_crawl_20240101T120000Z.jl

# Verify in Neo4j browser
# Navigate to http://localhost:7474
# Run: MATCH (n) RETURN n LIMIT 25
```

### Step 4: Backend Service
```bash
# Set environment variables
export OPENAI_API_KEY=your_key
export QDRANT_HOST=localhost
export NEO4J_URI=bolt://localhost:7687

# Start API server
uvicorn service.app:app --reload --port 8000
```

### Step 5: Frontend Interface
```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

## 🔧 Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...                    # OpenAI API key
QDRANT_HOST=localhost                    # Vector DB host
NEO4J_URI=bolt://localhost:7687          # Graph DB URI
NEO4J_USER=neo4j                         # Graph DB user
NEO4J_PASSWORD=password                  # Graph DB password

# Optional
OPENAI_MODEL=gpt-3.5-turbo-0125         # LLM model
QDRANT_PORT=6333                        # Vector DB port
QDRANT_COLLECTION=mosdac_embeddings     # Collection name
WANDB_API_KEY=...                       # Experiment tracking
```

### Model Configuration

```python
# service/app.py - Update these settings
EMBED_MODEL_NAME = "all-MiniLM-L12-v2"  # Embedding model
OPENAI_MODEL = "gpt-3.5-turbo-0125"     # Chat model
MEMORY_MAX_TURNS = 5                     # Conversation memory
```

## 🧪 Training & Evaluation

### Training Pipeline
```bash
# Install training dependencies
pip install -r training/requirements.txt

# Run training pipeline
python training/train_pipeline.py

# Check results
cat training_report.json
```

### Evaluation Metrics
```bash
# Install evaluation dependencies
pip install -r evaluation/requirements.txt

# Run comprehensive evaluation
python evaluation/evaluate_system.py

# View metrics
cat evaluation_results.json
```

### Geospatial Queries
```bash
# Install geospatial dependencies
pip install -r geospatial/requirements.txt

# Test geospatial service
python geospatial/geospatial_service.py
```

## 🌍 Geospatial Features

### Bounding Box Queries
```python
from geospatial.geospatial_service import GeospatialService

service = GeospatialService()

# Query Bay of Bengal region
bbox = [80, 10, 95, 22]  # [min_lon, min_lat, max_lon, max_lat]
results = service.query_by_bbox(bbox, mission="OCEANSAT-2")
```

### Location-Based Search
```python
# Query products covering Chennai
results = service.query_by_location(13.0827, 80.2707)
```

## 📊 Monitoring & Metrics

### Key Performance Indicators

| Metric | Target | Description |
|--------|--------|-------------|
| Intent Accuracy | >85% | Correct intent classification |
| Entity F1 Score | >80% | Named entity recognition |
| Response Completeness | >75% | Coverage of query aspects |
| Response Consistency | >90% | Multi-turn consistency |
| Retrieval Precision@5 | >70% | Relevant document retrieval |
| Generation ROUGE-L | >65% | Response quality |
| End-to-End Latency | <3s | Query to response time |

### Evaluation Commands
```bash
# Run full evaluation suite
python evaluation/evaluate_system.py

# Training metrics
python training/train_pipeline.py

# View evaluation plots
ls evaluation_report.png
```

## 🐳 Production Deployment

### Docker Compose (Recommended)
```bash
# Full stack deployment
docker-compose up -d

# Processing pipeline
docker-compose --profile processing up -d

# Training pipeline
docker-compose --profile training up -d
```

### Individual Services
```bash
# Vector database
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:v1.7.4

# Knowledge graph
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password neo4j:5.15-community

# API service
docker build -f Dockerfile.api -t mosdac-api .
docker run -d --name api -p 8000:8000 mosdac-api

# Frontend
cd frontend && docker build -t mosdac-ui .
docker run -d --name ui -p 3000:3000 mosdac-ui
```

### Kubernetes Deployment
```bash
# Create namespace
kubectl create namespace mosdac

# Deploy services
kubectl apply -f deployment/k8s/

# Check status
kubectl get pods -n mosdac
```

## 🔍 Usage Examples

### Basic Queries
```
User: "What satellite data is available for India?"
Bot: "MOSDAC provides data from multiple satellites covering India including INSAT-3D for meteorological data, OCEANSAT-2 for ocean observations, and SCATSAT-1 for wind measurements..."

User: "How do I download INSAT-3D temperature data?"
Bot: "To download INSAT-3D temperature data: 1) Register on MOSDAC portal 2) Navigate to INSAT-3D section 3) Select temperature products 4) Choose your desired format and time range..."
```

### Geospatial Queries
```
User: "Show me ocean color data for Bay of Bengal"
Bot: "OCEANSAT-2 provides ocean color data for Bay of Bengal region. Available products include chlorophyll concentration, suspended sediments, and sea surface temperature with 360m resolution..."

User: "What's the coverage area of MEGHA-TROPIQUES?"
Bot: "MEGHA-TROPIQUES covers tropical regions between 30°N and 30°S, providing precipitation, water vapor, and cloud property data with 10km resolution..."
```

## 🛠️ Development

### Adding New Data Sources
1. Update crawler in `ingestion/mosdac_crawler/spiders/`
2. Modify processing pipeline in `processing/process_documents.py`
3. Extend knowledge graph schema in `kg/build_kg.py`

### Customizing Models
1. Update embedding model in `processing/process_documents.py`
2. Fine-tune intent classifier in `training/train_pipeline.py`
3. Extend NER labels in `kg/build_kg.py`

### Frontend Customization
1. Modify components in `frontend/src/components/`
2. Update styling in `frontend/src/index.css`
3. Add new features in `frontend/src/App.tsx`

## 🔧 Troubleshooting

### Common Issues

**Docker Services Not Starting**
```bash
# Check logs
docker-compose logs qdrant
docker-compose logs neo4j

# Restart services
docker-compose restart
```

**Frontend TypeScript Errors**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run build
```

**API Connection Issues**
```bash
# Check service status
curl http://localhost:8000/docs

# Verify environment variables
env | grep -E "(OPENAI|QDRANT|NEO4J)"
```

**Empty Search Results**
```bash
# Check if data was processed
python -c "
from qdrant_client import QdrantClient
client = QdrantClient('localhost', 6333)
print(client.count('mosdac_embeddings'))
"
```

## 📈 Performance Optimization

### Vector Search Optimization
- Use approximate nearest neighbor search (HNSW)
- Optimize chunk size (400-600 tokens)
- Implement hybrid search (BM25 + semantic)

### Knowledge Graph Optimization
- Create indexes on frequently queried properties
- Use graph algorithms for relationship discovery
- Implement caching for common patterns

### API Performance
- Add Redis caching for frequent queries
- Implement request rate limiting
- Use async processing for long operations

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
- Create GitHub issues for bugs
- Check documentation for common solutions
- Contact the development team for enterprise support

---

**Note**: This system is designed to be modular and can be adapted for other scientific data portals by modifying the crawling targets, entity types, and domain-specific knowledge.