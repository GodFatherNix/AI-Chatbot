#!/usr/bin/env python3
"""
System validation script for AI Help Bot
Checks dependencies, configuration, and system readiness
"""

import sys
import os
import subprocess
import importlib
from typing import Tuple, List

def check_python_version() -> Tuple[bool, str]:
    """Check if Python version is compatible"""
    version = sys.version_info
    required_major, required_minor = 3, 8
    
    if version.major >= required_major and version.minor >= required_minor:
        return True, f"✅ Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"❌ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)"

def check_dependencies() -> Tuple[bool, List[str]]:
    """Check if required dependencies are installed"""
    dependencies = [
        'fastapi',
        'uvicorn', 
        'streamlit',
        'requests',
        'beautifulsoup4',
        'networkx',
        'sentence_transformers',
        'chromadb',
        'pandas',
        'numpy',
        'plotly'
    ]
    
    results = []
    all_good = True
    
    for dep in dependencies:
        try:
            importlib.import_module(dep.replace('-', '_'))
            results.append(f"✅ {dep}")
        except ImportError:
            results.append(f"❌ {dep} (missing)")
            all_good = False
    
    return all_good, results

def check_optional_dependencies() -> Tuple[bool, List[str]]:
    """Check optional dependencies"""
    optional_deps = [
        ('openai', 'Enhanced AI responses'),
        ('lxml', 'Better HTML parsing'),
        ('python-dotenv', 'Environment file support')
    ]
    
    results = []
    for dep, description in optional_deps:
        try:
            importlib.import_module(dep.replace('-', '_'))
            results.append(f"✅ {dep} - {description}")
        except ImportError:
            results.append(f"⚠️ {dep} - {description} (optional)")
    
    return True, results

def check_configuration() -> Tuple[bool, List[str]]:
    """Check configuration files and directories"""
    results = []
    all_good = True
    
    # Check config file
    if os.path.exists('config.py'):
        results.append("✅ config.py found")
    else:
        results.append("❌ config.py missing")
        all_good = False
    
    # Check environment file
    if os.path.exists('.env'):
        results.append("✅ .env file found")
    elif os.path.exists('.env.example'):
        results.append("⚠️ .env.example found but .env missing (will use defaults)")
    else:
        results.append("❌ No environment configuration found")
        all_good = False
    
    # Check data directory
    if os.path.exists('data'):
        results.append("✅ data directory exists")
    else:
        results.append("ℹ️ data directory will be created on first run")
    
    return all_good, results

def check_core_modules() -> Tuple[bool, List[str]]:
    """Check core application modules"""
    modules = [
        'knowledge_graph.py',
        'vector_db.py',
        'ai_retrieval.py',
        'api.py',
        'streamlit_app.py',
        'manage.py'
    ]
    
    results = []
    all_good = True
    
    for module in modules:
        if os.path.exists(module):
            results.append(f"✅ {module}")
        else:
            results.append(f"❌ {module} missing")
            all_good = False
    
    return all_good, results

def check_system_resources() -> Tuple[bool, List[str]]:
    """Check system resources"""
    results = []
    
    # Check available disk space
    try:
        import shutil
        total, used, free = shutil.disk_usage('.')
        free_gb = free // (1024**3)
        
        if free_gb >= 2:
            results.append(f"✅ Disk space: {free_gb} GB available")
        else:
            results.append(f"⚠️ Disk space: {free_gb} GB available (recommend 2+ GB)")
    except:
        results.append("⚠️ Could not check disk space")
    
    # Check memory (basic check)
    try:
        import psutil
        memory = psutil.virtual_memory()
        memory_gb = memory.total // (1024**3)
        
        if memory_gb >= 4:
            results.append(f"✅ RAM: {memory_gb} GB total")
        else:
            results.append(f"⚠️ RAM: {memory_gb} GB total (recommend 4+ GB)")
    except ImportError:
        results.append("ℹ️ Install 'psutil' to check memory usage")
    
    return True, results

def check_ports() -> Tuple[bool, List[str]]:
    """Check if required ports are available"""
    results = []
    all_good = True
    
    try:
        import socket
        
        ports_to_check = [8000, 8501]
        
        for port in ports_to_check:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            
            if result != 0:
                results.append(f"✅ Port {port} available")
            else:
                results.append(f"⚠️ Port {port} in use")
                # Not marking as failure since user might want to check running system
    
    except Exception as e:
        results.append(f"⚠️ Could not check ports: {e}")
    
    return all_good, results

def run_quick_import_test() -> Tuple[bool, str]:
    """Test importing main modules"""
    try:
        from config import config
        from knowledge_graph import KnowledgeGraphBuilder
        from vector_db import VectorDatabase
        from ai_retrieval import AIRetrievalEngine
        
        return True, "✅ All main modules import successfully"
    except Exception as e:
        return False, f"❌ Import error: {e}"

def main():
    """Run all validation checks"""
    print("🔍 AI Help Bot System Validation")
    print("=" * 50)
    
    all_checks_passed = True
    
    # Python version check
    python_ok, python_msg = check_python_version()
    print(f"\n📋 Python Version:")
    print(f"  {python_msg}")
    if not python_ok:
        all_checks_passed = False
    
    # Dependencies check
    deps_ok, deps_results = check_dependencies()
    print(f"\n📦 Required Dependencies:")
    for result in deps_results:
        print(f"  {result}")
    if not deps_ok:
        all_checks_passed = False
    
    # Optional dependencies
    _, opt_results = check_optional_dependencies()
    print(f"\n📦 Optional Dependencies:")
    for result in opt_results:
        print(f"  {result}")
    
    # Configuration check
    config_ok, config_results = check_configuration()
    print(f"\n⚙️ Configuration:")
    for result in config_results:
        print(f"  {result}")
    if not config_ok:
        all_checks_passed = False
    
    # Core modules check
    modules_ok, modules_results = check_core_modules()
    print(f"\n🏗️ Core Modules:")
    for result in modules_results:
        print(f"  {result}")
    if not modules_ok:
        all_checks_passed = False
    
    # System resources
    _, resource_results = check_system_resources()
    print(f"\n💻 System Resources:")
    for result in resource_results:
        print(f"  {result}")
    
    # Port availability
    _, port_results = check_ports()
    print(f"\n🌐 Port Availability:")
    for result in port_results:
        print(f"  {result}")
    
    # Import test
    import_ok, import_msg = run_quick_import_test()
    print(f"\n🧪 Import Test:")
    print(f"  {import_msg}")
    if not import_ok:
        all_checks_passed = False
    
    # Summary
    print(f"\n" + "=" * 50)
    if all_checks_passed:
        print("🎉 All checks passed! System is ready to run.")
        print("\nNext steps:")
        print("1. Start the system: python manage.py start")
        print("2. Or run the demo: python demo.py")
        print("3. Or use the shell script: ./run.sh start")
    else:
        print("❌ Some checks failed. Please resolve the issues above.")
        print("\nCommon solutions:")
        print("• Install missing dependencies: pip install -r requirements.txt")
        print("• Create .env file: cp .env.example .env")
        print("• Check Python version: python3 --version")
    
    print(f"\n💡 For help, see the README.md file or run: python manage.py help")

if __name__ == "__main__":
    main()