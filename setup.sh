#!/bin/bash
# setup.sh - Script cài đặt hệ thống

echo "=== SETUP ZALO OA AI SYSTEM ==="

# Kiểm tra Python
if ! command -v python3 &> /dev/null; then
    echo "Python 3 không được tìm thấy. Vui lòng cài đặt Python 3.8+"
    exit 1
fi

# Kiểm tra CUDA
echo "Kiểm tra CUDA..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv
else
    echo "Không tìm thấy NVIDIA GPU. Sẽ sử dụng CPU."
fi

# Tạo virtual environment
echo "Tạo virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Cài đặt PyTorch với CUDA support
echo "Cài đặt PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Cài đặt các package khác
echo "Cài đặt dependencies..."
pip install -r requirements.txt

# Tạo directories
mkdir -p logs
mkdir -p models
mkdir -p data

# Tạo config file
cat > config.json << EOF
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
        "auto_training_threshold": 50
    },
    "server": {
        "host": "0.0.0.0",
        "port": 8000,
        "workers": 1
    }
}
EOF

# Tạo systemd service (Linux)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Tạo systemd service..."
    sudo tee /etc/systemd/system/zalo-ai.service > /dev/null << EOF
[Unit]
Description=Zalo OA AI Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/venv/bin
ExecStart=$(pwd)/venv/bin/python ai_server.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    echo "Service created. Sử dụng: sudo systemctl start zalo-ai"
fi

# Tạo start script
cat > start.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
python ai_server.py
EOF

chmod +x start.sh

echo "=== SETUP HOÀN THÀNH ==="
echo ""
echo "HƯỚNG DẪN CHẠY:"
echo "1. Chạy AI Server: ./start.sh"
echo "2. Deploy Cloudflare Worker (xem file worker.js)"
echo "3. Cấu hình webhook Zalo OA: https://YOUR_WORKER_URL/webhook"
echo ""
echo "URLS:"
echo "- AI Server: http://localhost:8000"
echo "- API Docs: http://localhost:8000/docs"
echo "- Stats: http://localhost:8000/stats"