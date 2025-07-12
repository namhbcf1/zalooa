# 🚀 SIMPLE CLOUDFLARE DEPLOYMENT - FINAL VERSION

## ✅ WORKER FEATURES (ĐÚNG CÁCH):
- ✅ **Zalo API Integration** (webhook, send message)
- ✅ **PC Gaming Database** (CPU, GPU, builds, troubleshooting)
- ✅ **Smart Response Matching** từ database
- ✅ **Fallback to AI Server** nếu không tìm thấy
- ❌ **KHÔNG CÓ** AI training logic (đúng!)

## 📋 DEPLOYMENT STEPS:

### 🔧 STEP 1: Open Cloudflare Dashboard
**URL**: https://dash.cloudflare.com/5b62d10947844251d23e0eac532531dd/workers/services/view/zaloapi/production/metrics

### 🔧 STEP 2: Edit Worker
1. Click **"Quick Edit"**
2. **Select All** (Ctrl+A) 
3. **Delete** all existing code

### 🔧 STEP 3: Copy This Code
```javascript
// 🚀 ZALO OA CLOUDFLARE WORKER - PRODUCTION VERSION
// Chỉ handle Zalo API + Database, không có AI training logic

const ZALO_CONFIG = {
  APP_SECRET: 'MJO6GxU8NydtN7hERS5n',
  OA_SECRET_KEY: 'xEpeaB5Gnb64mO5bbHsb',
  ACCESS_TOKEN: '71c37aYKOnGI4uaQJPbY6HWNZ1fDXcLGG679FtQmMKfTDzSSUE1uJsOTz3Cw-dvQFGB4K1cAMp87PkLTFhCFF6fJxp1jX4DyU4ZE37Q-Nb5FUTa8Rv10KcfM-ZPGZd8vTtlXIdFpLXj79S1fJTDQ9qu_u2bOx7T80HVAB3oOAtGvGyi468fzL15DaoWodWL06tYdDm64INXOSFy1TOqI3sjqXMra-Zi2OYUp3KBwMN0SDgm19_j6ImWXyZychcur8tJnO1M2HHqVJQjg6SerEYKUWqrmaZORPbwN4L_H6N9nDeGuICGTT7H8bm1maorG94Ez03sI4qm3P-ObLvTSR7z9t3zLX798S7sR5sYZ75jDAeCCPVXi6cLrxKTikK5BOcNf1M2HHqVJQjg6SerEYKUWqrmaZORPbwN4L_H6N9nDeGuICGTT7H8bm1maorG94Ez03sI4qm3P-ObLvTSR7z9t3zLX798S7sR5sYZ75jDAeCCPVXi6cLrxKTikK5BOcNf8c7hKr1S6iqoOeWaC3jJZc8jimCURB_OEqI8QHK'
};

// AI Server URL - chỉ để gọi API trained model
const AI_SERVER_URL = 'http://192.168.1.8:8000';

// Analytics
let requestCount = 0;
let errorCount = 0;
let pcQueries = 0;
let successfulResponses = 0;

// PC Gaming Knowledge Database - từ trained model
const PC_KNOWLEDGE_DB = {
  // CPU Knowledge
  cpu: {
    'i7 13700f': {
      name: 'Intel Core i7-13700F',
      price: '9.5 triệu',
      specs: '16 cores (8P+8E), 24 threads, 3.4-5.2GHz',
      gaming: 'Excellent cho gaming 1440p/4K',
      pros: 'Hiệu năng cao, đa nhiệm tốt, tiết kiệm điện',
      cons: 'Giá cao, cần tản nhiệt tốt'
    },
    'ryzen 7 7700x': {
      name: 'AMD Ryzen 7 7700X',
      price: '8.8 triệu',
      specs: '8 cores, 16 threads, 4.5-5.4GHz',
      gaming: 'Tuyệt vời cho gaming, hiệu năng/giá tốt',
      pros: 'Hiệu năng/giá tốt, tiết kiệm điện, overclock tốt',
      cons: 'Cần RAM DDR5, giá mainboard cao'
    },
    'i5 13400f': {
      name: 'Intel Core i5-13400F',
      price: '4.5 triệu',
      specs: '10 cores (6P+4E), 16 threads, 2.5-4.6GHz',
      gaming: 'Rất tốt cho gaming 1080p/1440p',
      pros: 'Giá tốt, hiệu năng ổn, tương thích rộng',
      cons: 'Ít core hơn i7, không có iGPU'
    }
  },
  
  // GPU Knowledge
  gpu: {
    'rtx 4070': {
      name: 'RTX 4070',
      price: '14.5 triệu',
      specs: '12GB GDDR6X, 2610MHz boost',
      gaming: '1440p Ultra 60+ FPS, 4K Medium-High',
      vram: '12GB đủ cho gaming hiện tại và tương lai',
      pros: 'VRAM 12GB, DLSS 3, hiệu năng tốt',
      cons: 'Giá cao, cần PSU 650W+'
    },
    'rtx 4060 ti': {
      name: 'RTX 4060 Ti',
      price: '10.5 triệu',
      specs: '16GB GDDR6, 2540MHz boost',
      gaming: '1440p High 60+ FPS, 1080p Ultra',
      vram: '16GB version tốt cho tương lai',
      pros: 'VRAM 16GB, tiết kiệm điện, giá hợp lý',
      cons: 'Bus 128-bit, hiệu năng 1440p hạn chế'
    },
    'rtx 4060': {
      name: 'RTX 4060',
      price: '8.2 triệu',
      specs: '8GB GDDR6, 2460MHz boost',
      gaming: '1080p Ultra 60+ FPS, 1440p Medium-High',
      vram: '8GB đủ cho 1080p gaming',
      pros: 'Tiết kiệm điện, giá tốt, compact',
      cons: 'VRAM 8GB hạn chế, hiệu năng 1440p thấp'
    }
  },
  
  // Build Configs
  builds: {
    '15 triệu': {
      cpu: 'i5-13400F (4.5tr)',
      gpu: 'RTX 4060 (8tr)',
      ram: 'DDR4-3200 16GB (1.2tr)',
      storage: 'SSD 500GB (1tr)',
      performance: '1080p Ultra gaming',
      total: '~15 triệu'
    },
    '25 triệu': {
      cpu: 'i7-13700F (9.5tr)',
      gpu: 'RTX 4060 Ti (10.5tr)',
      ram: 'DDR4-3200 32GB (2.4tr)',
      storage: 'SSD NVMe 1TB (2.2tr)',
      performance: '1440p High gaming',
      total: '~25 triệu'
    },
    '35 triệu': {
      cpu: 'i7-13700F (9.5tr)',
      gpu: 'RTX 4070 (14.5tr)',
      ram: 'DDR5-5600 32GB (3.8tr)',
      storage: 'SSD NVMe PCIe 4.0 1TB (2.2tr)',
      performance: '1440p Ultra, 4K Medium gaming',
      total: '~35 triệu'
    }
  },
  
  // Troubleshooting
  troubleshooting: {
    'không khởi động': 'Kiểm tra: 1) Nguồn điện, 2) RAM lắp chặt, 3) Cáp 24pin + 8pin CPU, 4) GPU lắp chặt, 5) Monitor cắm đúng cổng GPU.',
    'game lag': 'Nguyên nhân: 1) GPU yếu - giảm setting, 2) RAM không đủ - đóng app khác, 3) CPU bottleneck - upgrade, 4) Nhiệt độ cao - vệ sinh, 5) Driver cũ - update.',
    'pc nóng': 'Giải pháp: 1) Vệ sinh bụi quạt/heatsink, 2) Thay keo tản nhiệt, 3) Thêm quạt case, 4) Kiểm tra airflow, 5) Undervolt CPU/GPU.',
    'không có tín hiệu': 'Kiểm tra: 1) Cáp monitor chắc chắn, 2) Cắm vào GPU không phải mainboard, 3) RAM lắp đúng slot, 4) GPU có nguồn PCIe, 5) Monitor chọn đúng input.'
  }
};

addEventListener('fetch', event => {
  event.respondWith(handleRequest(event.request));
});

async function handleRequest(request) {
  const url = new URL(request.url);
  requestCount++;
  
  const corsHeaders = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
  };
  
  if (request.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }
  
  try {
    if (request.method === 'GET' && url.pathname === '/') {
      return new Response(JSON.stringify({
        message: 'Zalo OA AI Bot - Production Ready',
        status: 'running',
        version: '2.0.0-production',
        features: [
          'Trained PC Gaming Knowledge Database',
          'Zalo OA Integration',
          'Smart Response Matching',
          'Vietnamese PC Terminology'
        ],
        database: {
          cpu_models: Object.keys(PC_KNOWLEDGE_DB.cpu).length,
          gpu_models: Object.keys(PC_KNOWLEDGE_DB.gpu).length,
          build_configs: Object.keys(PC_KNOWLEDGE_DB.builds).length,
          troubleshooting_solutions: Object.keys(PC_KNOWLEDGE_DB.troubleshooting).length
        },
        analytics: {
          total_requests: requestCount,
          pc_queries: pcQueries,
          successful_responses: successfulResponses,
          error_rate: ((errorCount / requestCount) * 100).toFixed(2) + '%'
        }
      }), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      });
    }
    
    if (request.method === 'GET' && url.pathname === '/webhook') {
      return handleWebhookVerification(request);
    }
    
    if (request.method === 'POST' && url.pathname === '/webhook') {
      return handleIncomingMessage(request);
    }
    
    return new Response('Zalo OA AI Bot - Endpoint not found', { 
      status: 404,
      headers: corsHeaders 
    });
    
  } catch (error) {
    errorCount++;
    console.error('Request error:', error);
    return new Response(JSON.stringify({ 
      error: 'Internal server error',
      message: error.message 
    }), {
      status: 500,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' }
    });
  }
}

async function handleWebhookVerification(request) {
  const url = new URL(request.url);
  const challenge = url.searchParams.get('challenge');
  
  if (challenge) {
    console.log('Webhook verification:', challenge);
    return new Response(challenge, { status: 200 });
  }
  
  return new Response('Invalid verification', { status: 400 });
}

async function handleIncomingMessage(request) {
  try {
    const body = await request.text();
    console.log('Incoming webhook:', body.substring(0, 200));
    
    const data = JSON.parse(body);
    
    if (data.event_name === 'user_send_text') {
      await handleTextMessage(data);
    } else if (data.event_name === 'user_send_image') {
      await handleImageMessage(data);
    } else if (data.event_name === 'user_send_sticker') {
      await handleStickerMessage(data);
    }
    
    return new Response('OK', { status: 200 });
    
  } catch (error) {
    errorCount++;
    console.error('Webhook error:', error);
    return new Response('Error', { status: 500 });
  }
}

async function handleTextMessage(data) {
  const message = data.message;
  const userId = data.sender.id;
  const userMessage = message.text.toLowerCase();
  
  console.log(`User ${userId}: ${userMessage}`);
  
  try {
    // Tìm response từ database trước
    let response = findResponseFromDatabase(userMessage);
    
    if (response) {
      // Có response từ database
      pcQueries++;
      successfulResponses++;
      await sendTextMessage(userId, response);
      console.log('Response from database:', response.substring(0, 100));
    } else {
      // Fallback to AI server nếu không tìm thấy trong database
      try {
        const aiResponse = await fetch(`${AI_SERVER_URL}/process-message`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            user_id: userId,
            message: userMessage,
            timestamp: Date.now()
          }),
          timeout: 15000
        });
        
        if (aiResponse.ok) {
          const aiResult = await aiResponse.json();
          successfulResponses++;
          await sendTextMessage(userId, aiResult.response);
          console.log('Response from AI server:', aiResult.response.substring(0, 100));
        } else {
          throw new Error('AI server error');
        }
      } catch (aiError) {
        // Fallback response
        const fallbackResponse = getFallbackResponse(userMessage);
        await sendTextMessage(userId, fallbackResponse);
        console.log('Fallback response used');
      }
    }
    
  } catch (error) {
    errorCount++;
    console.error('Error handling message:', error);
    await sendTextMessage(userId, 'Xin lỗi, tôi đang gặp sự cố. Vui lòng thử lại sau!');
  }
}

function findResponseFromDatabase(userMessage) {
  const msg = userMessage.toLowerCase();
  
  // CPU queries
  for (const [key, cpu] of Object.entries(PC_KNOWLEDGE_DB.cpu)) {
    if (msg.includes(key) || msg.includes(cpu.name.toLowerCase())) {
      if (msg.includes('giá') || msg.includes('bao nhiêu')) {
        return `${cpu.name} hiện giá khoảng ${cpu.price}. ${cpu.gaming}. Specs: ${cpu.specs}. Ưu điểm: ${cpu.pros}.`;
      } else if (msg.includes('tốt') || msg.includes('gaming')) {
        return `${cpu.name} ${cpu.gaming}. Specs: ${cpu.specs}. Giá: ${cpu.price}. Ưu điểm: ${cpu.pros}. Nhược điểm: ${cpu.cons}.`;
      } else {
        return `${cpu.name}: ${cpu.specs}, giá ${cpu.price}. ${cpu.gaming}. Ưu điểm: ${cpu.pros}.`;
      }
    }
  }
  
  // GPU queries
  for (const [key, gpu] of Object.entries(PC_KNOWLEDGE_DB.gpu)) {
    if (msg.includes(key) || msg.includes(gpu.name.toLowerCase())) {
      if (msg.includes('giá') || msg.includes('bao nhiêu')) {
        return `${gpu.name} giá khoảng ${gpu.price}. ${gpu.gaming}. VRAM: ${gpu.vram}. Ưu điểm: ${gpu.pros}.`;
      } else if (msg.includes('gaming') || msg.includes('chơi game')) {
        return `${gpu.name} ${gpu.gaming}. VRAM: ${gpu.vram}. Specs: ${gpu.specs}. Ưu điểm: ${gpu.pros}.`;
      } else {
        return `${gpu.name}: ${gpu.specs}, giá ${gpu.price}. ${gpu.gaming}. VRAM: ${gpu.vram}.`;
      }
    }
  }
  
  // Build queries
  for (const [budget, build] of Object.entries(PC_KNOWLEDGE_DB.builds)) {
    if (msg.includes(budget) || msg.includes('build') || msg.includes('cấu hình')) {
      return `PC ${budget}: CPU ${build.cpu}, GPU ${build.gpu}, RAM ${build.ram}, SSD ${build.storage}. Hiệu năng: ${build.performance}. Tổng: ${build.total}.`;
    }
  }
  
  // Troubleshooting queries
  for (const [problem, solution] of Object.entries(PC_KNOWLEDGE_DB.troubleshooting)) {
    if (msg.includes(problem)) {
      return solution;
    }
  }
  
  return null;
}

function getFallbackResponse(userMessage) {
  const msg = userMessage.toLowerCase();
  
  if (msg.includes('cpu') || msg.includes('processor')) {
    return 'Tôi có thể tư vấn về CPU Intel i7-13700F, i5-13400F, AMD Ryzen 7 7700X. Bạn muốn hỏi về CPU nào?';
  } else if (msg.includes('gpu') || msg.includes('vga') || msg.includes('rtx')) {
    return 'Tôi có thể tư vấn về GPU RTX 4070, RTX 4060 Ti, RTX 4060. Bạn quan tâm card nào?';
  } else if (msg.includes('build') || msg.includes('cấu hình')) {
    return 'Tôi có thể tư vấn build PC 15 triệu, 25 triệu, 35 triệu. Budget của bạn là bao nhiêu?';
  } else if (msg.includes('lỗi') || msg.includes('không') || msg.includes('bị')) {
    return 'Tôi có thể giúp troubleshoot: PC không khởi động, game lag, PC nóng, không có tín hiệu. Bạn gặp vấn đề gì?';
  } else {
    return 'Xin chào! Tôi có thể tư vấn về CPU, GPU, build PC gaming, và troubleshooting. Bạn cần hỏi gì?';
  }
}

async function handleImageMessage(data) {
  const userId = data.sender.id;
  await sendTextMessage(userId, 'Cảm ơn bạn đã gửi ảnh! Tôi chuyên tư vấn PC gaming qua text. Bạn có thể hỏi về CPU, GPU, build PC nhé!');
}

async function handleStickerMessage(data) {
  const userId = data.sender.id;
  const responses = [
    'Sticker cute! Bạn cần tư vấn PC gaming gì không?',
    'Haha! Tôi có thể giúp bạn về CPU, GPU, build PC đấy!',
    'Nice! Bạn quan tâm cấu hình PC nào?'
  ];
  const response = responses[Math.floor(Math.random() * responses.length)];
  await sendTextMessage(userId, response);
}

async function sendTextMessage(userId, text) {
  const payload = {
    recipient: { user_id: userId },
    message: { text: text }
  };
  
  return await callZaloAPI('/message', payload);
}

async function callZaloAPI(endpoint, data) {
  const url = `https://openapi.zalo.me/v2.0/oa${endpoint}`;
  
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'access_token': ZALO_CONFIG.ACCESS_TOKEN
      },
      body: JSON.stringify(data)
    });
    
    const result = await response.json();
    
    if (result.error) {
      console.error('Zalo API error:', result.error);
      errorCount++;
    } else {
      console.log('Zalo API success');
    }
    
    return result;
    
  } catch (error) {
    errorCount++;
    console.error('Zalo API call failed:', error);
    return null;
  }
}
```

### 🔧 STEP 4: Update IP Address
**Line 12**: Change `http://192.168.1.8:8000` to your actual server IP

### 🔧 STEP 5: Save and Deploy
1. Click **"Save and Deploy"**
2. Wait for deployment (10-30 seconds)

### 🔧 STEP 6: Test
**Expected Response**: `"Zalo OA AI Bot - Production Ready"`

## ✅ FEATURES AFTER DEPLOYMENT:
- ✅ **PC Gaming Database** với CPU, GPU, builds, troubleshooting
- ✅ **Smart Matching** từ database trước
- ✅ **Fallback to AI Server** nếu cần
- ✅ **Analytics** tracking
- ✅ **Zalo Integration** hoàn chỉnh

**DEPLOY NGAY!** 🚀
