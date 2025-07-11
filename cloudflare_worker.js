// Cloudflare Worker - Zalo OA Webhook Handler
// Deploy này lên Cloudflare Workers của bạn

const ZALO_CONFIG = {
  APP_SECRET: 'MJO6GxU8NydtN7hERS5n',
  OA_SECRET_KEY: 'xEpeaB5Gnb64mO5bbHsb',
  ACCESS_TOKEN: '71c37aYKOnGI4uaQJPbY6HWNZ1fDXcLGG679FtQmMKfTDzSSUE1uJsOTz3Cw-dvQFGB4K1cAMp87PkLTFhCFF6fJxp1jX4DyU4ZE37Q-Nb5FUTa8Rv10KcfM-ZPGZd8vTtlXIdFpLXj79S1fJTDQ9qu_u2bOx7T80HVAB3oOAtGvGyi468fzL15DaoWodWL06tYdDm64INXOSFy1TOqI3sjqXMra-Zi2OYUp3KBwMN0SDgm19_j6ImWXyZychcur8tJnO1M2HHqVJQjg6SerEYKUWqrmaZORPbwN4L_H6N9nDeGuICGTT7H8bm1maorG94Ez03sI4qm3P-ObLvTSR7z9t3zLX798S7sR5sYZ75jDAeCCPVXi6cLrxKTikK5BOcNf8c7hKr1S6iqoOeWaC3jJZc8jimCURB_OEqI8QHK'
};

// URL của server AI local của bạn (IP thực tế)
const AI_SERVER_URL = 'http://192.168.1.8:8000';

addEventListener('fetch', event => {
  event.respondWith(handleRequest(event.request));
});

async function handleRequest(request) {
  const url = new URL(request.url);
  
  if (request.method === 'GET' && url.pathname === '/webhook') {
    // Verify webhook từ Zalo
    return handleWebhookVerification(request);
  }
  
  if (request.method === 'POST' && url.pathname === '/webhook') {
    // Xử lý tin nhắn đến
    return handleIncomingMessage(request);
  }
  
  if (request.method === 'POST' && url.pathname === '/send-message') {
    // API để gửi tin nhắn
    return handleSendMessage(request);
  }
  
  return new Response('Zalo OA API Handler', { status: 200 });
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
    const signature = request.headers.get('X-ZEvent-Signature');
    
    // Verify signature (optional nhưng nên có)
    if (!verifySignature(body, signature)) {
      return new Response('Invalid signature', { status: 401 });
    }
    
    const data = JSON.parse(body);
    console.log('Incoming webhook:', data);
    
    // Xử lý các loại event khác nhau
    if (data.event_name === 'user_send_text') {
      await handleTextMessage(data);
    } else if (data.event_name === 'user_send_image') {
      await handleImageMessage(data);
    } else if (data.event_name === 'user_send_sticker') {
      await handleStickerMessage(data);
    }
    
    return new Response('OK', { status: 200 });
  } catch (error) {
    console.error('Error handling webhook:', error);
    return new Response('Error', { status: 500 });
  }
}

async function handleTextMessage(data) {
  const message = data.message;
  const userId = data.sender.id;
  const userMessage = message.text;
  
  console.log(`User ${userId} said: ${userMessage}`);
  
  try {
    // Gửi tin nhắn đến AI server để training và generate response
    const aiResponse = await fetch(`${AI_SERVER_URL}/process-message`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        user_id: userId,
        message: userMessage,
        timestamp: Date.now()
      })
    });
    
    const aiResult = await aiResponse.json();
    
    // Gửi response lại cho user
    if (aiResult.response) {
      await sendTextMessage(userId, aiResult.response);
    }
    
  } catch (error) {
    console.error('Error calling AI server:', error);
    // Fallback response
    await sendTextMessage(userId, 'Xin lỗi, tôi đang gặp sự cố. Vui lòng thử lại sau.');
  }
}

async function handleImageMessage(data) {
  const userId = data.sender.id;
  const imageUrl = data.message.attachments[0].payload.url;
  
  // Có thể xử lý ảnh hoặc gửi response đơn giản
  await sendTextMessage(userId, 'Cảm ơn bạn đã gửi ảnh! Tôi đang học cách xử lý hình ảnh.');
}

async function handleStickerMessage(data) {
  const userId = data.sender.id;
  await sendTextMessage(userId, 'Sticker rất cute! 😊');
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
    console.log('Zalo API response:', result);
    
    if (result.error) {
      console.error('Zalo API error:', result.error);
    }
    
    return result;
  } catch (error) {
    console.error('Error calling Zalo API:', error);
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
    return new Response(JSON.stringify({ error: error.message }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
}

function verifySignature(body, signature) {
  // Implement signature verification nếu cần
  // Zalo sử dụng HMAC-SHA256 với OA_SECRET_KEY
  return true; // Simplified for now
}

// Utility function để get user info
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
    console.error('Error getting user info:', error);
    return null;
  }
}