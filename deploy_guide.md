# 🚀 ZALO OA AI BOT - DEPLOYMENT GUIDE

## 📋 Quick Deploy

### 1. Clone Repository
```bash
git clone https://github.com/namhbcf1/zalooa.git
cd zalooa
```

### 2. Setup AI Server
```bash
# Install dependencies
pip install -r requirements.txt

# Run enhanced AI server with auto learning
python enhanced_working_server.py
```

### 3. Deploy Cloudflare Worker
```bash
# Copy cloudflare_worker.js to Cloudflare Workers
# Update YOUR_LOCAL_IP in the worker code
# Deploy to: https://zaloapi.bangachieu2.workers.dev
```

### 4. Configure Zalo OA
```
Webhook URL: https://zaloapi.bangachieu2.workers.dev/webhook
```

## 🎮 Management Commands

```bash
# Health check
python working_management.py health

# Test AI response
python working_management.py test --message "CPU Intel i7 có tốt không?"

# Database statistics
python working_management.py stats

# Trigger training
python working_management.py train --force

# Trigger auto learning
curl -X POST http://localhost:8000/trigger-learning
```

## 📊 API Endpoints

- **API Docs**: http://localhost:8000/docs
- **Health**: http://localhost:8000/health
- **Stats**: http://localhost:8000/stats
- **Knowledge**: http://localhost:8000/knowledge
- **Process Message**: POST http://localhost:8000/process-message
- **Trigger Training**: POST http://localhost:8000/trigger-training
- **Trigger Learning**: POST http://localhost:8000/trigger-learning

## 🔧 Configuration

Update IP address in `cloudflare_worker.js`:
```javascript
const AI_SERVER_URL = 'http://YOUR_LOCAL_IP:8000';
```

## 🎯 Features

- ✅ Auto Learning from Vietnamese tech websites
- ✅ GPU Training (GTX 1070 optimized)
- ✅ Real-time Q&A generation
- ✅ Cloudflare Workers integration
- ✅ Rich CLI management tools
- ✅ SQLite database with auto-learned knowledge
- ✅ Scheduled learning sessions
- ✅ Enhanced AI responses

## 📈 System Stats

- **Knowledge Articles**: Auto-crawled from tech sites
- **Q&A Pairs**: Auto-generated from articles
- **Training Sessions**: GPU-accelerated fine-tuning
- **Learning Sessions**: Scheduled every 30 minutes
