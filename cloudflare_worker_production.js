// 🚀 ZALO OA CLOUDFLARE WORKER - PRODUCTION VERSION
// Deploy này lên Cloudflare Workers: https://zaloapi.bangachieu2.workers.dev

const ZALO_CONFIG = {
  APP_SECRET: 'MJO6GxU8NydtN7hERS5n',
  OA_SECRET_KEY: 'xEpeaB5Gnb64mO5bbHsb',
  ACCESS_TOKEN: '71c37aYKOnGI4uaQJPbY6HWNZ1fDXcLGG679FtQmMKfTDzSSUE1uJsOTz3Cw-dvQFGB4K1cAMp87PkLTFhCFF6fJxp1jX4DyU4ZE37Q-Nb5FUTa8Rv10KcfM-ZPGZd8vTtlXIdFpLXj79S1fJTDQ9qu_u2bOx7T80HVAB3oOAtGvGyi468fzL15DaoWodWL06tYdDm64INXOSFy1TOqI3sjqXMra-Zi2OYUp3KBwMN0SDgm19_j6ImWXyZychcur8tJnO1M2HHqVJQjg6SerEYKUWqrmaZORPbwN4L_H6N9nDeGuICGTT7H8bm1maorG94Ez03sI4qm3P-ObLvTSR7z9t3zLX798S7sR5sYZ75jDAeCCPVXi6cLrxKTikK5BOcNf8c7hKr1S6iqoOeWaC3jJZc8jimCURB_OEqI8QHK'
};

// 🔧 UPDATE THIS IP TO YOUR SERVER IP
const AI_SERVER_URL = 'http://192.168.1.8:8000';

// 📊 Analytics tracking
let requestCount = 0;
let errorCount = 0;

addEventListener('fetch', event => {
  event.respondWith(handleRequest(event.request));
});

async function handleRequest(request) {
  const url = new URL(request.url);
  requestCount++;
  
  // Add CORS headers
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
        message: '🤖 Zalo OA AI Bot - Production Ready',
        status: 'running',
        version: '2.0.0',
        features: ['auto_learning', 'gpu_training', 'enhanced_ai'],
        endpoints: ['/webhook', '/send-message', '/stats'],
        ai_server: AI_SERVER_URL,
        requests: requestCount,
        errors: errorCount
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
    
    if (request.method === 'POST' && url.pathname === '/send-message') {
      return handleSendMessage(request);
    }
    
    if (request.method === 'GET' && url.pathname === '/stats') {
      return new Response(JSON.stringify({
        requests: requestCount,
        errors: errorCount,
        uptime: Date.now(),
        ai_server: AI_SERVER_URL,
        zalo_config: {
          app_secret: ZALO_CONFIG.APP_SECRET.substring(0, 8) + '...',
          access_token: ZALO_CONFIG.ACCESS_TOKEN.substring(0, 20) + '...'
        }
      }), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      });
    }
    
    return new Response('🤖 Zalo OA AI Bot - Endpoint not found', { 
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
    console.log('✅ Webhook verification:', challenge);
    return new Response(challenge, { status: 200 });
  }
  
  return new Response('❌ Invalid verification', { status: 400 });
}

async function handleIncomingMessage(request) {
  try {
    const body = await request.text();
    const signature = request.headers.get('X-ZEvent-Signature');
    
    console.log('📨 Incoming webhook:', body.substring(0, 200));
    
    // Parse webhook data
    const data = JSON.parse(body);
    
    // Handle different event types
    if (data.event_name === 'user_send_text') {
      await handleTextMessage(data);
    } else if (data.event_name === 'user_send_image') {
      await handleImageMessage(data);
    } else if (data.event_name === 'user_send_sticker') {
      await handleStickerMessage(data);
    } else if (data.event_name === 'user_send_file') {
      await handleFileMessage(data);
    } else {
      console.log('🔄 Unhandled event:', data.event_name);
    }
    
    return new Response('OK', { status: 200 });
    
  } catch (error) {
    errorCount++;
    console.error('❌ Webhook error:', error);
    return new Response('Error', { status: 500 });
  }
}

async function handleTextMessage(data) {
  const message = data.message;
  const userId = data.sender.id;
  const userMessage = message.text;
  
  console.log(`👤 User ${userId}: ${userMessage}`);
  
  try {
    // 🧠 Send to Enhanced AI Server
    const aiResponse = await fetch(`${AI_SERVER_URL}/process-message`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        user_id: userId,
        message: userMessage,
        timestamp: Date.now()
      }),
      timeout: 30000 // 30 second timeout
    });
    
    if (aiResponse.ok) {
      const aiResult = await aiResponse.json();
      
      console.log('🤖 AI Response:', aiResult.response);
      
      // Send enhanced response
      await sendTextMessage(userId, aiResult.response);
      
      // Send additional info if confidence is low
      if (aiResult.confidence < 0.5) {
        await sendTextMessage(userId, 
          '💡 Tôi đang học thêm để trả lời tốt hơn. Bạn có thể hỏi về CPU, GPU, RAM, SSD, hoặc build PC gaming!'
        );
      }
      
      // Show learning status
      if (aiResult.auto_learning_status === 'active') {
        console.log(`📚 Auto learning: ${aiResult.knowledge_articles} articles, ${aiResult.qa_pairs} Q&A pairs`);
      }
      
    } else {
      throw new Error(`AI Server error: ${aiResponse.status}`);
    }
    
  } catch (error) {
    errorCount++;
    console.error('🚨 AI Server error:', error);
    
    // Fallback responses
    const fallbackResponses = [
      '🤖 Xin lỗi, tôi đang gặp sự cố kỹ thuật. Vui lòng thử lại sau!',
      '🔧 Hệ thống AI đang bảo trì. Tôi sẽ trả lời bạn sớm nhất có thể!',
      '⚡ Server đang quá tải. Hãy thử hỏi lại sau vài phút nhé!'
    ];
    
    const fallback = fallbackResponses[Math.floor(Math.random() * fallbackResponses.length)];
    await sendTextMessage(userId, fallback);
  }
}

async function handleImageMessage(data) {
  const userId = data.sender.id;
  const imageUrl = data.message.attachments[0].payload.url;
  
  console.log(`🖼️ User ${userId} sent image: ${imageUrl}`);
  
  await sendTextMessage(userId, 
    '📸 Cảm ơn bạn đã gửi ảnh! Tôi đang học cách phân tích hình ảnh PC gaming. Hiện tại bạn có thể hỏi tôi bằng text về CPU, GPU, RAM nhé!'
  );
}

async function handleStickerMessage(data) {
  const userId = data.sender.id;
  
  const stickerResponses = [
    '😊 Sticker rất cute! Bạn cần tư vấn gì về PC gaming không?',
    '🎮 Haha, bạn vui quá! Tôi có thể giúp bạn build PC gaming đấy!',
    '💻 Sticker đẹp! Bạn quan tâm CPU Intel hay AMD?'
  ];
  
  const response = stickerResponses[Math.floor(Math.random() * stickerResponses.length)];
  await sendTextMessage(userId, response);
}

async function handleFileMessage(data) {
  const userId = data.sender.id;
  
  await sendTextMessage(userId,
    '📁 Cảm ơn bạn đã gửi file! Tôi chưa thể xử lý file nhưng có thể tư vấn PC gaming qua text. Bạn cần hỏi gì?'
  );
}

async function sendTextMessage(userId, text) {
  const payload = {
    recipient: {
      user_id: userId
    },
    message: {
      text: text
    }
  };
  
  return await callZaloAPI('/message', payload);
}

async function sendImageMessage(userId, imageUrl) {
  const payload = {
    recipient: {
      user_id: userId
    },
    message: {
      attachment: {
        type: "image",
        payload: {
          url: imageUrl
        }
      }
    }
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
      console.error('❌ Zalo API error:', result.error);
      errorCount++;
    } else {
      console.log('✅ Zalo API success');
    }
    
    return result;
    
  } catch (error) {
    errorCount++;
    console.error('🚨 Zalo API call failed:', error);
    return null;
  }
}

async function handleSendMessage(request) {
  try {
    const { user_id, message, type = 'text' } = await request.json();
    
    if (type === 'text') {
      const result = await sendTextMessage(user_id, message);
      return new Response(JSON.stringify(result), {
        headers: { 'Content-Type': 'application/json' }
      });
    }
    
    return new Response(JSON.stringify({ error: 'Unsupported message type' }), {
      status: 400,
      headers: { 'Content-Type': 'application/json' }
    });
    
  } catch (error) {
    errorCount++;
    return new Response(JSON.stringify({ error: error.message }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
}

// 🔐 Utility function for signature verification
function verifySignature(body, signature) {
  // TODO: Implement HMAC-SHA256 verification with OA_SECRET_KEY
  // For now, return true for development
  return true;
}

// 👤 Get user info from Zalo
async function getUserInfo(userId) {
  const url = `https://openapi.zalo.me/v2.0/oa/getprofile?data={"user_id":"${userId}"}`;
  
  try {
    const response = await fetch(url, {
      headers: {
        'access_token': ZALO_CONFIG.ACCESS_TOKEN
      }
    });
    
    return await response.json();
    
  } catch (error) {
    console.error('❌ Error getting user info:', error);
    return null;
  }
}
