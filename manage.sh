#!/bin/bash
# manage.sh - Enhanced management script với Rich CLI
# Không đơn giản hóa - Phức tạp và chất lượng cao

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
else
    echo -e "${RED}[✗]${NC} Virtual environment not found!"
    exit 1
fi

print_header() {
    echo -e "${PURPLE}"
    echo "🎮 =================================="
    echo "🎮  ZALO AI BOT MANAGEMENT CONSOLE"
    echo "🎮 =================================="
    echo -e "${NC}"
}

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[ℹ]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

check_server() {
    if curl -s http://localhost:8000/ > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

case "$1" in
    "live")
        print_header
        print_info "🔴 Starting live dashboard..."
        if check_server; then
            python management.py live-dashboard
        else
            print_error "AI Server is not running! Start it first with: ./start_ai_server.sh"
        fi
        ;;
    
    "learn")
        print_header
        print_info "🧠 Triggering manual learning..."
        if check_server; then
            curl -X POST http://localhost:8000/trigger-training
            print_status "Learning triggered successfully!"
        else
            print_error "AI Server is not running!"
        fi
        ;;
    
    "test")
        print_header
        print_info "🧪 Testing AI response..."
        if [ -z "$2" ]; then
            print_warning "Usage: ./manage.sh test --query 'Your message'"
            exit 1
        fi
        
        if [[ "$2" == "--query" && -n "$3" ]]; then
            python management.py test --message "$3"
        else
            print_error "Invalid syntax. Use: ./manage.sh test --query 'Your message'"
        fi
        ;;
    
    "search")
        print_header
        print_info "🔍 Searching knowledge base..."
        if [ -z "$2" ]; then
            print_warning "Usage: ./manage.sh search --query 'search term'"
            exit 1
        fi
        
        if [[ "$2" == "--query" && -n "$3" ]]; then
            python management.py search-knowledge --query "$3"
        else
            print_error "Invalid syntax. Use: ./manage.sh search --query 'search term'"
        fi
        ;;
    
    "report")
        print_header
        print_info "📊 Generating advanced report..."
        python management.py advanced-report
        print_status "Report generated in reports/ directory"
        ;;
    
    "add-knowledge")
        print_header
        print_info "📚 Adding knowledge to database..."
        if [ -z "$2" ]; then
            print_warning "Usage: ./manage.sh add-knowledge 'Category' 'Question' 'Answer'"
            exit 1
        fi
        python management.py add-knowledge "$@"
        ;;
    
    "backup")
        print_header
        print_info "💾 Creating system backup..."
        timestamp=$(date +"%Y%m%d_%H%M%S")
        backup_dir="backups/backup_$timestamp"
        mkdir -p "$backup_dir"
        
        cp conversations.db "$backup_dir/" 2>/dev/null || true
        cp config.json "$backup_dir/" 2>/dev/null || true
        cp -r models/ "$backup_dir/" 2>/dev/null || true
        cp -r logs/ "$backup_dir/" 2>/dev/null || true
        
        print_status "Backup created: $backup_dir"
        ;;
    
    "status")
        print_header
        print_info "📈 System status check..."
        
        echo -e "${CYAN}🔧 Server Status:${NC}"
        if check_server; then
            print_status "AI Server is running (http://localhost:8000)"
            
            # Get server stats
            echo -e "${CYAN}📊 Server Stats:${NC}"
            curl -s http://localhost:8000/stats | python -m json.tool
        else
            print_error "AI Server is not running"
        fi
        
        echo -e "\n${CYAN}💾 Database Status:${NC}"
        if [ -f "conversations.db" ]; then
            python -c "
import sqlite3
conn = sqlite3.connect('conversations.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM conversations')
conv_count = cursor.fetchone()[0]
cursor.execute('SELECT COUNT(DISTINCT user_id) FROM conversations')
user_count = cursor.fetchone()[0]
print(f'Conversations: {conv_count}')
print(f'Unique users: {user_count}')
conn.close()
"
        else
            print_warning "Database not found"
        fi
        
        echo -e "\n${CYAN}🖥️ System Resources:${NC}"
        python -c "
import psutil
import GPUtil
print(f'CPU Usage: {psutil.cpu_percent()}%')
print(f'Memory Usage: {psutil.virtual_memory().percent}%')
print(f'Disk Usage: {psutil.disk_usage(\"/\").percent}%')
try:
    gpus = GPUtil.getGPUs()
    if gpus:
        for gpu in gpus:
            print(f'GPU {gpu.id}: {gpu.name} - {gpu.memoryUtil*100:.1f}% VRAM')
except:
    print('GPU info not available')
" 2>/dev/null || echo "Resource monitoring not available"
        ;;
    
    "logs")
        print_header
        print_info "📋 Showing recent logs..."
        if [ -f "logs/ai_server.log" ]; then
            tail -n 50 logs/ai_server.log
        else
            print_warning "No log file found"
        fi
        ;;
    
    "restart")
        print_header
        print_info "🔄 Restarting AI Server..."
        
        # Kill existing process
        pkill -f "python ai_server.py" 2>/dev/null || true
        sleep 2
        
        # Start new process
        print_info "Starting AI Server..."
        nohup ./start_ai_server.sh > logs/ai_server.log 2>&1 &
        sleep 3
        
        if check_server; then
            print_status "AI Server restarted successfully!"
        else
            print_error "Failed to restart AI Server"
        fi
        ;;
    
    "install")
        print_header
        print_info "📦 Installing additional packages..."
        pip install rich typer colorama psutil GPUtil schedule
        print_status "Additional packages installed"
        ;;
    
    "update")
        print_header
        print_info "🔄 Updating system..."
        git pull 2>/dev/null || print_warning "Git not available"
        pip install -r requirements.txt --upgrade
        print_status "System updated"
        ;;
    
    *)
        print_header
        echo -e "${CYAN}🎮 Available Commands:${NC}"
        echo ""
        echo -e "${GREEN}Core Operations:${NC}"
        echo "  ./manage.sh live          - 🔴 Live dashboard với Rich UI"
        echo "  ./manage.sh status        - 📈 System status & resources"
        echo "  ./manage.sh restart       - 🔄 Restart AI server"
        echo ""
        echo -e "${GREEN}AI Operations:${NC}"
        echo "  ./manage.sh learn         - 🧠 Trigger manual learning"
        echo "  ./manage.sh test --query 'message'  - 🧪 Test AI response"
        echo "  ./manage.sh search --query 'term'   - 🔍 Search knowledge base"
        echo ""
        echo -e "${GREEN}Data Management:${NC}"
        echo "  ./manage.sh report        - 📊 Generate advanced report"
        echo "  ./manage.sh add-knowledge - 📚 Add knowledge manually"
        echo "  ./manage.sh backup        - 💾 Create system backup"
        echo ""
        echo -e "${GREEN}System Management:${NC}"
        echo "  ./manage.sh logs          - 📋 Show recent logs"
        echo "  ./manage.sh install       - 📦 Install additional packages"
        echo "  ./manage.sh update        - 🔄 Update system"
        echo ""
        echo -e "${YELLOW}Examples:${NC}"
        echo "  ./manage.sh test --query 'CPU Intel i7 có tốt không?'"
        echo "  ./manage.sh search --query 'GPU'"
        echo "  ./manage.sh add-knowledge 'CPU' 'i9 vs i7?' 'i9 mạnh hơn i7'"
        echo ""
        ;;
esac
