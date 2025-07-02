#!/bin/bash

# MOSDAC AI Chatbot - Master Deployment Script
# This script sets up and runs the complete MOSDAC chatbot system

set -e

echo "🚀 MOSDAC AI Chatbot Deployment Script"
echo "======================================="

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ .env file not found. Creating template..."
    cat > .env << EOF
OPENAI_API_KEY=your_openai_api_key_here
WANDB_API_KEY=your_wandb_key_optional
NEO4J_PASSWORD=password
EOF
    echo "📝 Please edit .env file with your API keys and run again."
    exit 1
fi

# Load environment variables
source .env

if [ "$OPENAI_API_KEY" = "your_openai_api_key_here" ]; then
    echo "❌ Please set your OPENAI_API_KEY in .env file"
    exit 1
fi

echo "✅ Environment variables loaded"

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        echo "❌ Docker is not running. Please start Docker and try again."
        exit 1
    fi
    echo "✅ Docker is running"
}

# Function to wait for service
wait_for_service() {
    local service=$1
    local port=$2
    local max_attempts=30
    local attempt=1

    echo "⏳ Waiting for $service to be ready..."
    while [ $attempt -le $max_attempts ]; do
        if curl -f "http://localhost:$port" > /dev/null 2>&1; then
            echo "✅ $service is ready"
            return 0
        fi
        echo "   Attempt $attempt/$max_attempts - $service not ready yet..."
        sleep 5
        attempt=$((attempt + 1))
    done
    
    echo "❌ $service failed to start after $max_attempts attempts"
    return 1
}

# Main deployment function
deploy() {
    echo "🔧 Starting infrastructure services..."
    
    # Start databases
    docker-compose up -d qdrant neo4j
    
    # Wait for databases
    wait_for_service "Qdrant" 6333
    wait_for_service "Neo4j" 7474
    
    echo "📦 Installing Python dependencies..."
    pip install -q -r ingestion/requirements.txt
    pip install -q -r processing/requirements.txt
    pip install -q -r kg/requirements.txt
    pip install -q -r service/requirements.txt
    
    # Download spaCy model
    python -m spacy download en_core_web_sm > /dev/null 2>&1 || echo "spaCy model already installed"
    
    echo "🕷️ Running web crawler..."
    cd ingestion/mosdac_crawler
    if [ ! -d "data" ] || [ -z "$(ls -A data 2>/dev/null)" ]; then
        scrapy crawl mosdac
    else
        echo "   Crawl data already exists, skipping..."
    fi
    cd ../..
    
    echo "🔄 Processing documents and building embeddings..."
    if [ -f data/mosdac_crawl_*.jl ]; then
        python processing/process_documents.py --input data/mosdac_crawl_*.jl
    else
        echo "   No crawl data found, skipping processing..."
    fi
    
    echo "🕸️ Building knowledge graph..."
    if [ -f data/mosdac_crawl_*.jl ]; then
        python kg/build_kg.py --input data/mosdac_crawl_*.jl
    else
        echo "   No crawl data found, skipping KG build..."
    fi
    
    echo "🚀 Starting backend API..."
    uvicorn service.app:app --host 0.0.0.0 --port 8000 > api.log 2>&1 &
    API_PID=$!
    
    # Wait for API
    wait_for_service "API" 8000
    
    echo "💻 Starting frontend..."
    cd frontend
    if [ ! -d "node_modules" ]; then
        npm install
    fi
    npm run dev > ../frontend.log 2>&1 &
    FRONTEND_PID=$!
    cd ..
    
    # Wait for frontend
    sleep 10
    wait_for_service "Frontend" 5173
    
    echo ""
    echo "🎉 MOSDAC AI Chatbot is now running!"
    echo "=================================="
    echo "🌐 Frontend UI: http://localhost:5173"
    echo "📚 API Docs: http://localhost:8000/docs"
    echo "🗄️ Neo4j Browser: http://localhost:7474 (neo4j/password)"
    echo "🔍 Qdrant Dashboard: http://localhost:6333/dashboard"
    echo ""
    echo "📊 System Status:"
    echo "   API PID: $API_PID"
    echo "   Frontend PID: $FRONTEND_PID"
    echo ""
    echo "🛑 To stop the system, run: ./run.sh stop"
    echo ""
    
    # Save PIDs for cleanup
    echo "$API_PID" > api.pid
    echo "$FRONTEND_PID" > frontend.pid
}

# Stop function
stop() {
    echo "🛑 Stopping MOSDAC AI Chatbot..."
    
    # Kill processes
    if [ -f api.pid ]; then
        kill $(cat api.pid) 2>/dev/null || true
        rm api.pid
    fi
    
    if [ -f frontend.pid ]; then
        kill $(cat frontend.pid) 2>/dev/null || true
        rm frontend.pid
    fi
    
    # Stop Docker services
    docker-compose down
    
    echo "✅ System stopped"
}

# Training function
train() {
    echo "🧠 Running training pipeline..."
    pip install -q -r training/requirements.txt
    pip install -q -r evaluation/requirements.txt
    
    python training/train_pipeline.py
    python evaluation/evaluate_system.py
    
    echo "📊 Training complete! Check training_report.json and evaluation_results.json"
}

# Test function
test() {
    echo "🧪 Running system tests..."
    
    # Test API endpoints
    curl -f http://localhost:8000/docs > /dev/null && echo "✅ API is responding"
    curl -f http://localhost:5173 > /dev/null && echo "✅ Frontend is responding"
    
    # Test chat endpoint
    response=$(curl -s -X POST "http://localhost:8000/chat" \
        -H "Content-Type: application/json" \
        -d '{"session_id": "test", "message": "What is INSAT-3D?"}')
    
    if echo "$response" | grep -q "answer"; then
        echo "✅ Chat endpoint is working"
    else
        echo "❌ Chat endpoint failed"
    fi
    
    echo "🧪 Basic tests complete"
}

# Help function
help() {
    echo "MOSDAC AI Chatbot Deployment Script"
    echo ""
    echo "Usage: ./run.sh [command]"
    echo ""
    echo "Commands:"
    echo "  deploy  - Deploy the complete system (default)"
    echo "  stop    - Stop all services"
    echo "  train   - Run training pipeline"
    echo "  test    - Run basic system tests"
    echo "  help    - Show this help message"
    echo ""
    echo "Prerequisites:"
    echo "  - Docker and Docker Compose"
    echo "  - Python 3.9+"
    echo "  - Node.js 18+"
    echo "  - OpenAI API key in .env file"
}

# Main script logic
case "${1:-deploy}" in
    "deploy")
        check_docker
        deploy
        ;;
    "stop")
        stop
        ;;
    "train")
        train
        ;;
    "test")
        test
        ;;
    "help"|"-h"|"--help")
        help
        ;;
    *)
        echo "❌ Unknown command: $1"
        help
        exit 1
        ;;
esac