#!/bin/bash
# enhanced_setup.sh - Setup script cho Zalo OA AI Auto Learning System

set -e  # Exit on any error

echo "🤖 =================================="
echo "🤖  ZALO OA AI AUTO LEARNING SETUP"
echo "🤖 =================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Check system requirements
check_requirements() {
    log "Checking system requirements..."
    
    # Check Python
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        log "Python found: $PYTHON_VERSION"
    else
        error "Python 3.8+ is required but not found"
    fi
    
    # Check pip
    if ! command -v pip3 &> /dev/null; then
        error "pip3 not found. Please install pip3"
    fi
    
    # Check Git
    if ! command -v git &> /dev/null; then
        warn "Git not found. Some features may not work"
    fi
    
    # Check available disk space (minimum 10GB)
    AVAILABLE_SPACE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
    if [ "$AVAILABLE_SPACE" -lt 10 ]; then
        warn "Less than 10GB disk space available. Models may not download properly."
    fi
    
    # Check RAM (minimum 8GB recommended)
    if command -v free &> /dev/null; then
        TOTAL_RAM=$(free -g | awk 'NR==2{print $2}')
        if [ "$TOTAL_RAM" -lt 8 ]; then
            warn "Less than 8GB RAM detected. Performance may be limited."
        fi
        log "System RAM: ${TOTAL_RAM}GB"
    fi
    
    # Check CUDA if available
    if command -v nvidia-smi &> /dev/null; then
        log "NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        if [ "$GPU_MEMORY" -lt 6144 ]; then  # Less than 6GB
            warn "GPU has less than 6GB VRAM. Consider using smaller models."
        fi
    else
        warn "No NVIDIA GPU detected. Will use CPU mode (slower training)."
    fi
}

# Create project structure
create_structure() {
    log "Creating project structure..."
    
    mkdir -p {logs,models,data,cache,reports,backups}
    mkdir -p data/{crawled,processed,training}
    mkdir -p models/{base,fine_tuned,checkpoints}
    
    # Create .gitignore
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Data & Models
*.db
*.sqlite3
models/
cache/
logs/*.log
data/crawled/
data/processed/

# OS
.DS_Store
Thumbs.db

# Reports
reports/*.png
reports/*.html
EOF
    
    log "Project structure created"
}

# Setup Python virtual environment
setup_python_env() {
    log "Setting up Python virtual environment..."
    
    # Create virtual environment
    python3 -m venv venv
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    log "Virtual environment created and activated"
}

# Install dependencies
install_dependencies() {
    log "Installing Python dependencies..."
    
    # Ensure we're in virtual environment
    source venv/bin/activate
    
    # Install PyTorch first (with CUDA if available)
    if command -v nvidia-smi &> /dev/null; then
        info "Installing PyTorch with CUDA support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        info "Installing PyTorch CPU version..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    # Install other dependencies
    pip install -r enhanced_requirements.txt
    
    # Install additional Vietnamese NLP tools
    log "Setting up Vietnamese NLP tools..."
    python -c "
import underthesea
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('punkt')
nltk.download('stopwords')
print('Vietnamese NLP tools ready')
"
    
    # Install Rich for beautiful CLI
    pip install rich click typer
    
    log "Dependencies installed successfully"
}

# Download base models
download_models() {
    log "Downloading base AI models..."
    
    source venv/bin/activate
    
    python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

models = [
    'microsoft/DialoGPT-small',
    'microsoft/DialoGPT-medium'
]

for model_name in models:
    try:
        print(f'Downloading {model_name}...')
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='./cache')
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir='./cache')
        print(f'✅ {model_name} downloaded successfully')
    except Exception as e:
        print(f'❌ Error downloading {model_name}: {e}')
"
    
    log "Base models downloaded"
}

# Setup configuration files
setup_config() {
    log "Creating configuration files..."
    
    # Main config
    cat > config.json << 'EOF'
{
    "zalo": {
        "app_secret": "MJO6GxU8NydtN7hERS5n",
        "oa_secret_key": "xEpeaB5Gnb64mO5bbHsb",
        "access_token": "71c37aYKOnGI4uaQJPbY6HWNZ1fDXcLGG679FtQmMKfTDzSSUE1uJsOTz3Cw-dvQFGB4K1cAMp87PkLTFhCFF6fJxp1jX4DyU4ZE37Q-Nb5FUTa8Rv10KcfM-ZPGZd8vTtlXIdFpLXj79S1fJTDQ9qu_u2bOx7T80HVAB3oOAtGvGyi468fzL15DaoWodWL06tYdDm64INXOSFy1TOqI3sjqXMra-Zi2OYUp3KBwMN0SDgm19_j6ImWXyZychcur8tJnO1M2HHqVJQjg6SerEYKUWqrmaZORPbwN4L_H6N9nDeGuICGTT7H8bm1maorG94Ez03sI4qm3P-ObLvTSR7z9t3zLX798S7sR5sYZ75jDAeCCPVXi6cLrxKTikK5BOcNf8c7hKr1S6iqoOeWaC3jJZc8jimCURB_OEqI8QHK"
    },
    "ai": {
        "model_name": "microsoft/DialoGPT-small",
        "max_length": 512,
        "batch_size": 4,
        "learning_rate": 5e-5,
        "auto_training_threshold": 50,
        "incremental_learning": true,
        "cache_dir": "./cache"
    },
    "crawler": {
        "rate_limit": 1.0,
        "max_concurrent": 5,
        "timeout": 30,
        "user_agent": "ZaloOA-AI-Bot/1.0",
        "sources": {
            "stackoverflow": true,
            "reddit": true,
            "vietnamese_forums": true,
            "github": false
        }
    },
    "learning": {
        "auto_crawl_interval": 24,
        "check_interval": 6,
        "min_confidence": 0.3,
        "max_knowledge_items": 100000,
        "backup_interval": 168
    },
    "server": {
        "host": "0.0.0.0",
        "port": 8000,
        "workers": 1,
        "reload": false,
        "log_level": "info"
    }
}
EOF
    
    # Environment variables
    cat > .env << 'EOF'
# Zalo OA Configuration
ZALO_APP_SECRET=MJO6GxU8NydtN7hERS5n
ZALO_OA_SECRET_KEY=xEpeaB5Gnb64mO5bbHsb
ZALO_ACCESS_TOKEN=71c37aYKOnGI4uaQJPbY6HWNZ1fDXcLGG679FtQmMKfTDzSSUE1uJsOTz3Cw-dvQFGB4K1cAMp87PkLTFhCFF6fJxp1jX4DyU4ZE37Q-Nb5FUTa8Rv10KcfM-ZPGZd8vTtlXIdFpLXj79S1fJTDQ9qu_u2bOx7T80HVAB3oOAtGvGyi468fzL15DaoWodWL06tYdDm64INXOSFy1TOqI3sjqXMra-Zi2OYUp3KBwMN0SDgm19_j6ImWXyZychcur8tJnO1M2HHqVJQjg6SerEYKUWqrmaZORPbwN4L_H6N9nDeGuICGTT7H8bm1maorG94Ez03sI4qm3P-ObLvTSR7z9t3zLX798S7sR5sYZ75jDAeCCPVXi6cLrxKTikK5BOcNf8c7hKr1S6iqoOeWaC3jJZc8jimCURB_OEqI8QHK

# AI Configuration
AI_MODEL_NAME=microsoft/DialoGPT-small
AI_CACHE_DIR=./cache
AI_MAX_LENGTH=512
AI_BATCH_SIZE=4

# Database
DB_PATH=conversations.db
KNOWLEDGE_DB_PATH=computer_knowledge.db

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/ai_server.log
EOF
    
    log "Configuration files created"
}

# Create startup scripts
create_scripts() {
    log "Creating startup scripts..."
    
    # Main startup script
    cat > start_ai_server.sh << 'EOF'
#!/bin/bash
# start_ai_server.sh - Main AI server startup script

echo "🤖 Starting Zalo OA AI Auto Learning Server..."

# Activate virtual environment
source venv/bin/activate

# Check if server is already running
if pgrep -f "enhanced_ai_server.py" > /dev/null; then
    echo "❌ AI Server is already running"
    exit 1
fi

# Start server with logging
python enhanced_ai_server.py 2>&1 | tee logs/ai_server.log
EOF
    
    # Management script
    cat > manage.sh << 'EOF'
#!/bin/bash
# manage.sh - Management wrapper script

source venv/bin/activate
python advanced_management.py "$@"
EOF
    
    # Quick test script
    cat > quick_test.sh << 'EOF'
#!/bin/bash
# quick_test.sh - Quick system test

echo "🧪 Running quick system test..."

source venv/bin/activate

# Test 1: Check server health
echo "1. Checking server health..."
python advanced_management.py health

# Test 2: Test AI response
echo "2. Testing AI response..."
python advanced_management.py test --query "CPU là gì?"

# Test 3: Check learning status
echo "3. Checking learning status..."
python advanced_management.py dashboard

echo "✅ Quick test completed!"
EOF
    
    # Auto backup script
    cat > backup.sh << 'EOF'
#!/bin/bash
# backup.sh - Auto backup script

BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "📦 Creating backup in $BACKUP_DIR..."

# Backup databases
cp *.db "$BACKUP_DIR/" 2>/dev/null || true

# Backup models
cp -r models/fine_tuned* "$BACKUP_DIR/" 2>/dev/null || true

# Backup config
cp config.json .env "$BACKUP_DIR/" 2>/dev/null || true

# Create archive
tar -czf "$BACKUP_DIR.tar.gz" "$BACKUP_DIR"
rm -rf "$BACKUP_DIR"

echo "✅ Backup created: $BACKUP_DIR.tar.gz"
EOF
    
    # Make scripts executable
    chmod +x start_ai_server.sh manage.sh quick_test.sh backup.sh
    
    log "Startup scripts created"
}

# Setup systemd service (Linux only)
setup_service() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        log "Setting up systemd service..."
        
        SERVICE_FILE="/etc/systemd/system/zalo-ai-learning.service"
        
        sudo tee "$SERVICE_FILE" > /dev/null << EOF
[Unit]
Description=Zalo OA AI Auto Learning Server
After=network.target
StartLimitIntervalSec=0

[Service]
Type=simple
Restart=always
RestartSec=1
User=$USER
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/venv/bin
ExecStart=$(pwd)/venv/bin/python $(pwd)/enhanced_ai_server.py
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
        
        sudo systemctl daemon-reload
        sudo systemctl enable zalo-ai-learning.service
        
        log "Systemd service created and enabled"
        info "Use: sudo systemctl start zalo-ai-learning to start service"
        info "Use: sudo systemctl status zalo-ai-learning to check status"
    else
        info "Systemd service setup skipped (not Linux)"
    fi
}

# Run initial tests
run_tests() {
    log "Running initial tests..."
    
    source venv/bin/activate
    
    # Test imports
    python -c "
import torch
import transformers
import fastapi
import aiohttp
import underthesea
print('✅ All imports successful')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
"
    
    # Test Vietnamese NLP
    python -c "
from pyvi import ViTokenizer
import underthesea
test_text = 'Máy tính của tôi bị lỗi màn hình xanh'
tokens = ViTokenizer.tokenize(test_text)
print(f'✅ Vietnamese tokenization: {tokens}')
"
    
    log "Initial tests completed successfully"
}

# Main setup function
main() {
    log "Starting enhanced setup for Zalo OA AI Auto Learning System..."
    
    check_requirements
    create_structure
    setup_python_env
    install_dependencies
    download_models
    setup_config
    create_scripts
    setup_service
    run_tests
    
    echo ""
    echo "🎉 =================================="
    echo "🎉  SETUP COMPLETED SUCCESSFULLY!"
    echo "🎉 =================================="
    echo ""
    echo "📋 NEXT STEPS:"
    echo ""
    echo "1. 🚀 Start AI Server:"
    echo "   ./start_ai_server.sh"
    echo ""
    echo "2. 🌐 Deploy Cloudflare Worker:"
    echo "   - Copy code from cloudflare_worker.js"
    echo "   - Update AI_SERVER_URL to your server IP"
    echo "   - Deploy to Cloudflare Workers"
    echo ""
    echo "3. ⚙️  Configure Zalo OA:"
    echo "   - Set webhook URL: https://YOUR_WORKER_URL/webhook"
    echo "   - Enable events: user_send_text, user_send_image"
    echo ""
    echo "4. 🎮 Management Commands:"
    echo "   ./manage.sh dashboard    # Live dashboard"
    echo "   ./manage.sh learn        # Trigger learning"
    echo "   ./manage.sh test --query 'CPU là gì?'"
    echo "   ./manage.sh report       # Generate report"
    echo ""
    echo "5. 🧪 Quick Test:"
    echo "   ./quick_test.sh"
    echo ""
    echo "📊 URLs:"
    echo "   AI Server: http://localhost:8000"
    echo "   API Docs: http://localhost:8000/docs"
    echo "   Health: http://localhost:8000/health"
    echo ""
    echo "📚 Features:"
    echo "   ✅ Auto crawling from StackOverflow, Reddit, VN forums"
    echo "   ✅ Vietnamese NLP processing"
    echo "   ✅ Incremental learning every 24h"
    echo "   ✅ Smart response with knowledge base"
    echo "   ✅ Real-time monitoring dashboard"
    echo "   ✅ Advanced analytics & reporting"
    echo ""
    echo "🔧 Need help? Check logs in: logs/ai_server.log"
    echo ""
}

# Run main function
main "$@"