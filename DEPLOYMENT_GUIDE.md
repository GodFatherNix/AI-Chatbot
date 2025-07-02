# MOSDAC AI Chatbot - Production Deployment Guide

## 🎯 Complete Implementation Overview

This guide provides step-by-step instructions for deploying the complete MOSDAC AI chatbot system, including all core components that have been implemented:

### ✅ **Implemented Core Components**

1. **🕷️ Data Ingestion System** - Web crawler for MOSDAC portal content
2. **🔄 Document Processing Pipeline** - Text extraction and embedding generation
3. **🕸️ Knowledge Graph Builder** - Entity/relationship extraction and Neo4j population
4. **🤖 RAG-Powered Chatbot Service** - FastAPI backend with LLM integration
5. **💻 React Frontend Interface** - Modern chat UI with citations and geospatial features
6. **🌍 Geospatial Querying** - Location-aware satellite data queries
7. **🧠 Training Pipeline** - Intent classification and NER model training
8. **📊 Evaluation Metrics** - Comprehensive performance assessment
9. **🐳 Production Deployment** - Docker, Kubernetes, and monitoring setup

## 🚀 Quick Start (5 Minutes)

### **One-Command Deployment**
```bash
# Clone repository
git clone <your-repo>
cd mosdac-chatbot

# Set your OpenAI API key
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# Deploy everything
./run.sh deploy

# Access your app at http://localhost:5173 🎉
```

## 📋 Detailed Implementation Steps

### **Prerequisites**
- Docker & Docker Compose
- Python 3.9+
- Node.js 18+
- OpenAI API Key
- 8GB RAM, 10GB disk space

### **Step 1: Environment Setup**
```bash
# Create environment file
cat > .env << EOF
OPENAI_API_KEY=sk-your-actual-key-here
WANDB_API_KEY=your-wandb-key-optional
NEO4J_PASSWORD=password
EOF

# Verify prerequisites
docker --version
python3 --version
node --version
```

### **Step 2: Infrastructure Services**
```bash
# Start vector database and knowledge graph
docker-compose up -d qdrant neo4j

# Verify services are running
curl http://localhost:6333/dashboard  # Qdrant
curl http://localhost:7474            # Neo4j
```

### **Step 3: Data Ingestion Pipeline**
```bash
# Install Python dependencies
pip install -r ingestion/requirements.txt
pip install -r processing/requirements.txt
pip install -r kg/requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Crawl MOSDAC portal
cd ingestion/mosdac_crawler
scrapy crawl mosdac
cd ../..

# Process documents and create embeddings
python processing/process_documents.py --input data/mosdac_crawl_*.jl

# Build knowledge graph
python kg/build_kg.py --input data/mosdac_crawl_*.jl
```

### **Step 4: Backend Services**
```bash
# Install service dependencies
pip install -r service/requirements.txt
pip install -r geospatial/requirements.txt

# Start FastAPI backend
uvicorn service.app:app --host 0.0.0.0 --port 8000 &

# Verify API is running
curl http://localhost:8000/docs
```

### **Step 5: Frontend Interface**
```bash
# Install and start React frontend
cd frontend
npm install
npm run dev &
cd ..

# Verify frontend is running
curl http://localhost:5173
```

### **Step 6: Training & Evaluation**
```bash
# Install training dependencies
pip install -r training/requirements.txt
pip install -r evaluation/requirements.txt

# Run training pipeline
python training/train_pipeline.py

# Run evaluation
python evaluation/evaluate_system.py

# View results
cat training_report.json
cat evaluation_results.json
```

## 🌍 Geospatial Features

### **Query by Bounding Box**
```python
# Example: Bay of Bengal region
bbox = [80, 10, 95, 22]  # [min_lon, min_lat, max_lon, max_lat]
curl -X POST "http://localhost:8000/geospatial/query_bbox" \
  -H "Content-Type: application/json" \
  -d '{"bbox": [80, 10, 95, 22], "mission": "OCEANSAT-2"}'
```

### **Query by Location**
```python
# Example: Chennai coordinates
curl -X POST "http://localhost:8000/geospatial/query_location" \
  -H "Content-Type: application/json" \
  -d '{"lat": 13.0827, "lon": 80.2707}'
```

### **Mission Coverage Map**
```bash
curl http://localhost:8000/geospatial/coverage
```

## 📊 Performance Monitoring

### **Key Metrics Tracked**

| Component | Metric | Target |
|-----------|--------|--------|
| Intent Recognition | Accuracy | >85% |
| Entity Recognition | F1 Score | >80% |
| Response Quality | ROUGE-L | >65% |
| Response Time | End-to-End | <3s |
| Retrieval | Precision@5 | >70% |
| Consistency | Multi-turn | >90% |

### **Monitoring Commands**
```bash
# Check system health
./run.sh test

# View logs
tail -f api.log
tail -f frontend.log

# Check database status
docker-compose logs qdrant
docker-compose logs neo4j
```

## 🐳 Production Deployment Options

### **Option 1: Docker Compose (Recommended)**
```bash
# Full production stack
docker-compose up -d

# View all services
docker-compose ps

# Scale API service
docker-compose up -d --scale api=3
```

### **Option 2: Kubernetes**
```bash
# Create namespace and deploy
kubectl create namespace mosdac
kubectl apply -f deployment/k8s/

# Check deployment status
kubectl get pods -n mosdac
kubectl get services -n mosdac
```

### **Option 3: Cloud Deployment**

#### **AWS ECS**
```bash
# Build and push images
docker build -f Dockerfile.api -t your-repo/mosdac-api .
docker push your-repo/mosdac-api

# Deploy with ECS task definition
aws ecs register-task-definition --cli-input-json file://ecs-task.json
```

#### **Google Cloud Run**
```bash
# Deploy API service
gcloud run deploy mosdac-api \
  --image gcr.io/your-project/mosdac-api \
  --platform managed \
  --region us-central1
```

## 🔧 Configuration & Customization

### **Environment Variables**
```bash
# Core Configuration
OPENAI_API_KEY=sk-...              # Required: OpenAI API key
OPENAI_MODEL=gpt-3.5-turbo-0125   # LLM model choice
QDRANT_HOST=localhost              # Vector DB host
NEO4J_URI=bolt://localhost:7687    # Graph DB URI

# Optional Enhancements
WANDB_API_KEY=...                  # Experiment tracking
EMBEDDING_MODEL=all-MiniLM-L12-v2  # Embedding model
MEMORY_MAX_TURNS=5                 # Conversation memory
```

### **Model Customization**
```python
# Update embedding model
# File: processing/process_documents.py
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Change LLM model
# File: service/app.py
OPENAI_MODEL = "gpt-4-turbo-preview"

# Add new entity types
# File: kg/build_kg.py
ENTITY_LABELS = {"SATELLITE", "PARAMETER", "LOCATION", "MISSION", "SENSOR"}
```

### **Frontend Customization**
```typescript
// Add new message types
// File: frontend/src/types.ts
interface GeospatialQuery {
  bbox?: number[];
  location?: {lat: number, lon: number};
}

// Customize UI theme
// File: frontend/src/index.css
:root {
  --primary-color: #0078D4;
  --secondary-color: #106EBE;
}
```

## 🧪 Testing & Quality Assurance

### **Unit Tests**
```bash
# Run component tests
python -m pytest ingestion/tests/
python -m pytest processing/tests/
python -m pytest service/tests/

# Frontend tests
cd frontend && npm test
```

### **Integration Tests**
```bash
# End-to-end system test
./run.sh test

# Load testing
python tests/load_test.py --users 100 --duration 60s
```

### **Performance Benchmarks**
```bash
# Benchmark embedding generation
python benchmarks/embedding_speed.py

# Test retrieval performance
python benchmarks/retrieval_benchmark.py

# Evaluate response quality
python evaluation/evaluate_system.py
```

## 🔍 Troubleshooting Guide

### **Common Issues & Solutions**

#### **🚨 Services Not Starting**
```bash
# Check Docker status
docker ps
docker-compose logs

# Restart services
docker-compose restart
```

#### **🚨 Empty Search Results**
```bash
# Verify data was processed
python -c "
from qdrant_client import QdrantClient
client = QdrantClient('localhost', 6333)
print(f'Collection size: {client.count(\"mosdac_embeddings\")}')"

# Check Neo4j data
docker exec -it neo4j cypher-shell -u neo4j -p password
> MATCH (n) RETURN count(n);
```

#### **🚨 API Connection Errors**
```bash
# Check API health
curl http://localhost:8000/docs

# Verify environment variables
env | grep -E "(OPENAI|QDRANT|NEO4J)"

# Check logs
tail -f api.log
```

#### **🚨 Frontend Build Errors**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run build
```

### **Performance Optimization**

#### **Vector Search Optimization**
```python
# Use HNSW index for faster search
# File: processing/process_documents.py
from qdrant_client.http.models import VectorParams, Distance

vectors_config = VectorParams(
    size=384,
    distance=Distance.COSINE,
    hnsw_config=HnswConfig(m=16, ef_construct=100)
)
```

#### **Knowledge Graph Optimization**
```cypher
// Create indexes for faster queries
CREATE INDEX satellite_name FOR (s:Satellite) ON (s.name);
CREATE INDEX parameter_type FOR (p:Parameter) ON (p.type);
```

#### **API Performance**
```python
# Add Redis caching
# File: service/app.py
import redis
from functools import wraps

redis_client = redis.Redis(host='localhost', port=6379)

def cache_response(expire=300):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args)+str(kwargs))}"
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
            result = await func(*args, **kwargs)
            redis_client.setex(cache_key, expire, json.dumps(result))
            return result
        return wrapper
    return decorator
```

## 📈 Scaling Considerations

### **Horizontal Scaling**
```yaml
# Kubernetes scaling
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: mosdac-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: mosdac-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### **Database Scaling**
```bash
# Qdrant cluster
docker run -d --name qdrant-1 -p 6333:6333 qdrant/qdrant
docker run -d --name qdrant-2 -p 6334:6333 qdrant/qdrant

# Neo4j cluster (Enterprise)
docker run -d --name neo4j-core-1 -p 7474:7474 -p 7687:7687 neo4j:enterprise
```

## 🎯 Success Metrics

### **Technical KPIs**
- ✅ **Response Time**: 95th percentile < 3 seconds
- ✅ **Availability**: 99.9% uptime
- ✅ **Accuracy**: Intent recognition > 85%
- ✅ **Throughput**: 1000+ queries/hour
- ✅ **Memory Usage**: < 4GB per service

### **Business KPIs**
- ✅ **User Satisfaction**: CSAT > 4.0/5.0
- ✅ **Query Resolution**: 80% answered without escalation
- ✅ **Usage Growth**: 20% month-over-month increase
- ✅ **Support Ticket Reduction**: 40% decrease in manual queries

## 🤝 Maintenance & Support

### **Regular Maintenance Tasks**
```bash
# Weekly: Update embeddings with new content
python processing/process_documents.py --input data/latest_crawl.jl

# Monthly: Retrain models
python training/train_pipeline.py

# Quarterly: Full system evaluation
python evaluation/evaluate_system.py
```

### **Backup & Recovery**
```bash
# Backup vector database
docker exec qdrant tar -czf /tmp/qdrant_backup.tar.gz /qdrant/storage

# Backup knowledge graph
docker exec neo4j neo4j-admin dump --database=neo4j --to=/tmp/neo4j_backup.dump

# Restore from backup
docker exec neo4j neo4j-admin load --from=/tmp/neo4j_backup.dump --database=neo4j
```

## 🎉 Congratulations!

You now have a **complete, production-ready MOSDAC AI chatbot** with:

- ✅ **Intelligent Query Understanding** via NLP/ML
- ✅ **Knowledge Graph** for relationship-aware responses  
- ✅ **Vector Search** for semantic document retrieval
- ✅ **Geospatial Intelligence** for location-aware queries
- ✅ **Modern UI** with React and TypeScript
- ✅ **Training Pipeline** for continuous improvement
- ✅ **Comprehensive Evaluation** metrics
- ✅ **Production Deployment** with Docker/Kubernetes

The system is **modular and extensible** - you can adapt it for other scientific data portals by modifying the crawling targets, entity types, and domain knowledge.

---

**🆘 Need Help?**
- 📧 Create GitHub issues for bugs
- 📚 Check logs in `api.log` and `frontend.log`
- 🔧 Run `./run.sh test` for basic diagnostics
- 💬 Contact the development team for enterprise support