#!/bin/bash

# AI Help Bot Management Script
# This script provides easy commands to manage the AI Help Bot system

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Python is installed
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3.8 or higher."
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    print_status "Python version: $PYTHON_VERSION"
}

# Function to check if virtual environment exists
check_venv() {
    if [ ! -d "venv" ]; then
        print_warning "Virtual environment not found. Creating one..."
        python3 -m venv venv
        print_success "Virtual environment created."
    fi
}

# Function to activate virtual environment
activate_venv() {
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        print_status "Virtual environment activated."
    else
        print_warning "Virtual environment not found. Running with system Python."
    fi
}

# Function to install dependencies
install_deps() {
    print_status "Installing dependencies..."
    pip install -r requirements.txt
    print_success "Dependencies installed successfully!"
}

# Function to start the system
start_system() {
    print_status "Starting AI Help Bot system..."
    python manage.py start
}

# Function to start only API
start_api() {
    print_status "Starting API server..."
    python manage.py api
}

# Function to start only web interface
start_web() {
    print_status "Starting web interface..."
    python manage.py web
}

# Function to build knowledge graph
build_kg() {
    if [ $# -eq 0 ]; then
        print_error "Please provide URLs to build knowledge graph."
        echo "Usage: $0 build <url1> <url2> ..."
        exit 1
    fi
    
    print_status "Building knowledge graph from provided URLs..."
    python manage.py build "$@"
}

# Function to query the system
query_system() {
    if [ $# -eq 0 ]; then
        print_error "Please provide a query."
        echo "Usage: $0 query \"Your question here\""
        exit 1
    fi
    
    print_status "Querying: $1"
    python manage.py query "$1"
}

# Function to show stats
show_stats() {
    print_status "Getting system statistics..."
    python manage.py stats
}

# Function to clear system
clear_system() {
    print_warning "This will delete all stored data!"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python manage.py clear
        print_success "System cleared."
    else
        print_status "Operation cancelled."
    fi
}

# Function to setup the system
setup() {
    print_status "Setting up AI Help Bot system..."
    
    check_python
    check_venv
    activate_venv
    install_deps
    
    # Create .env if it doesn't exist
    if [ ! -f ".env" ]; then
        print_status "Creating .env file from template..."
        cp .env.example .env
        print_warning "Please edit .env file to configure your settings."
    fi
    
    print_success "Setup complete! You can now start the system with: $0 start"
}

# Function to show help
show_help() {
    echo -e "${BLUE}AI Help Bot Management Script${NC}"
    echo ""
    echo "Usage: $0 <command> [arguments]"
    echo ""
    echo "Commands:"
    echo "  setup                 - Initial setup (install dependencies, create venv)"
    echo "  start                 - Start both API and web interface"
    echo "  api                   - Start only the API server"
    echo "  web                   - Start only the web interface"
    echo "  build <urls>          - Build knowledge graph from URLs"
    echo "  query \"<question>\"   - Query the knowledge base"
    echo "  stats                 - Show system statistics"
    echo "  clear                 - Clear all system data"
    echo "  help                  - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 setup"
    echo "  $0 start"
    echo "  $0 build https://docs.python.org/3/"
    echo "  $0 query \"What is machine learning?\""
    echo ""
}

# Main script logic
if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

# Check if we're in setup mode
if [ "$1" = "setup" ]; then
    setup
    exit 0
fi

# For all other commands, check if system is set up
if [ ! -f "requirements.txt" ]; then
    print_error "This doesn't appear to be the AI Help Bot directory."
    exit 1
fi

# Activate virtual environment if it exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Parse command
case "$1" in
    start)
        start_system
        ;;
    api)
        start_api
        ;;
    web)
        start_web
        ;;
    build)
        shift
        build_kg "$@"
        ;;
    query)
        shift
        query_system "$@"
        ;;
    stats)
        show_stats
        ;;
    clear)
        clear_system
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac