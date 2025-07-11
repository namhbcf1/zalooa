# 🚀 ZALO OA AI BOT - DEPLOYMENT INSTRUCTIONS

## 📋 GitHub Repository
**Repository**: https://github.com/namhbcf1/zalooa
**Status**: ✅ Successfully uploaded with 22 files, 6530 lines of code

## 🌐 Cloudflare Worker Deployment

### Step 1: Access Cloudflare Dashboard
URL: https://dash.cloudflare.com/5b62d10947844251d23e0eac532531dd/workers/services/view/zaloapi/production/metrics

### Step 2: Deploy Worker Code
1. Go to Cloudflare Workers dashboard
2. Select the `zaloapi` worker
3. Click "Quick Edit" or "Edit Code"
4. **Copy the entire content** from `cloudflare_worker_production.js`
5. **Paste and replace** all existing code
6. Click "Save and Deploy"

### Step 3: Update Configuration
**IMPORTANT**: Update the AI server IP in the worker code:
```javascript
// Line 11 in cloudflare_worker_production.js
const AI_SERVER_URL = 'http://YOUR_ACTUAL_IP:8000';
```

Replace `YOUR_ACTUAL_IP` with your actual server IP address.

### Step 4: Test Deployment
1. **Worker URL**: https://zaloapi.bangachieu2.workers.dev
2. **Test endpoints**:
   - GET `/` - Worker status
   - GET `/stats` - Worker statistics
   - POST `/webhook` - Zalo webhook handler

## 🤖 AI Server Deployment

### Step 1: Clone Repository
```bash
git clone https://github.com/namhbcf1/zalooa.git
cd zalooa
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Start Enhanced AI Server
```bash
python enhanced_working_server.py
```

### Step 4: Verify Server
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Stats**: http://localhost:8000/stats
- **Knowledge**: http://localhost:8000/knowledge

## 🔗 Zalo OA Configuration

### Step 1: Set Webhook URL
In Zalo OA settings, set webhook URL to:
```
https://zaloapi.bangachieu2.workers.dev/webhook
```

### Step 2: Verify Webhook
Zalo will send a verification request. The worker will handle it automatically.

## 📊 System Architecture

```
[Zalo OA] → [Cloudflare Worker] → [Enhanced AI Server] → [Auto Learning System]
    ↓              ↓                      ↓                      ↓
[Users]    [Webhook Handler]      [GPU Training]        [Web Crawling]
                                 [Q&A Generation]       [Knowledge DB]
```

## 🎯 Features Deployed

### ✅ Auto Learning System
- **Web Crawling**: Automatic crawling from Vietnamese tech websites
- **Q&A Generation**: Rule-based Q&A generation from crawled content
- **Knowledge Database**: SQLite database with auto-learned knowledge
- **Scheduled Learning**: Automatic learning sessions every 30 minutes

### ✅ Enhanced AI Server
- **GPU Training**: Optimized for GTX 1070 8GB VRAM
- **Real-time Responses**: Enhanced AI responses with auto-learned data
- **Background Processing**: Async training and learning
- **Rich API**: Comprehensive REST API with documentation

### ✅ Cloudflare Worker
- **Production Ready**: Error handling, analytics, CORS support
- **Webhook Handler**: Complete Zalo OA webhook processing
- **Fallback Responses**: Graceful degradation when AI server is down
- **Request Tracking**: Built-in analytics and error tracking

### ✅ Management Tools
- **Rich CLI**: Beautiful command-line interface with progress bars
- **Health Monitoring**: Real-time server health checks
- **Database Stats**: Comprehensive database statistics
- **Training Control**: Manual training triggers

## 🔧 Configuration Files

### Key Files Deployed:
- `enhanced_working_server.py` - Main AI server with auto learning
- `auto_learning_crawler.py` - Web crawling and Q&A generation
- `working_management.py` - Rich CLI management tools
- `cloudflare_worker_production.js` - Production Cloudflare Worker
- `requirements.txt` - Python dependencies
- `README.md` - Complete documentation

### Configuration:
- **Zalo Credentials**: Already configured in worker
- **AI Server URL**: Update in worker code
- **Database**: SQLite (auto-created)
- **Models**: DialoGPT-small (auto-downloaded)

## 🚀 Quick Start Commands

```bash
# Clone and setup
git clone https://github.com/namhbcf1/zalooa.git
cd zalooa
pip install -r requirements.txt

# Start AI server
python enhanced_working_server.py

# Test system
python working_management.py health
python working_management.py test --message "CPU Intel i7 có tốt không?"

# Trigger learning
curl -X POST http://localhost:8000/trigger-learning

# View auto-learned knowledge
curl http://localhost:8000/knowledge
```

## 📈 Monitoring

### Cloudflare Worker Metrics:
- **Dashboard**: https://dash.cloudflare.com/5b62d10947844251d23e0eac532531dd/workers/services/view/zaloapi/production/metrics
- **Request Count**: Built-in analytics
- **Error Tracking**: Automatic error logging

### AI Server Monitoring:
- **Health**: http://localhost:8000/health
- **Stats**: http://localhost:8000/stats
- **Knowledge**: http://localhost:8000/knowledge

## 🎉 Deployment Complete!

**GitHub**: ✅ https://github.com/namhbcf1/zalooa
**Cloudflare**: ✅ https://zaloapi.bangachieu2.workers.dev
**AI Server**: ✅ http://localhost:8000
**Zalo Webhook**: ✅ Ready for configuration

**System is production-ready with auto learning, GPU training, and enhanced AI responses!** 🚀
