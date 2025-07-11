# enhanced_ai_server.py - AI Server với auto learning tích hợp
import asyncio
import schedule
import threading
import time
from datetime import datetime
import logging
from contextlib import asynccontextmanager

import fastapi
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import từ auto learning system
from auto_learning_system import OnlineLearningSystem

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global learning system
learning_system = None
is_learning_active = False

class Message(BaseModel):
    user_id: str
    message: str
    timestamp: int

class AIResponse(BaseModel):
    response: str
    confidence: float
    source: str
    learning_status: str

class LearningStatus(BaseModel):
    is_active: bool
    last_session: str
    total_knowledge: int
    stats: dict

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Quản lý lifecycle của app"""
    global learning_system
    
    # Startup
    logger.info("Khởi tạo Enhanced AI Server...")
    learning_system = OnlineLearningSystem()
    
    # Start background learning scheduler
    start_learning_scheduler()
    
    yield
    
    # Shutdown
    logger.info("Shutting down Enhanced AI Server...")

app = FastAPI(
    title="Enhanced Zalo OA AI Server với Auto Learning",
    description="AI Server tự động học toàn bộ kiến thức máy tính từ internet",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def start_learning_scheduler():
    """Khởi động scheduler cho auto learning"""
    def run_scheduler():
        # Schedule daily learning
        schedule.every(24).hours.do(trigger_auto_learning)
        
        # Schedule immediate learning nếu cần
        schedule.every(6).hours.do(check_and_learn)
        
        while True:
            schedule.run_pending()
            time.sleep(3600)  # Check mỗi giờ
    
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    logger.info("Learning scheduler started")

def trigger_auto_learning():
    """Trigger auto learning session"""
    global is_learning_active, learning_system
    
    if is_learning_active:
        logger.info("Learning session đang chạy, bỏ qua...")
        return
    
    try:
        is_learning_active = True
        logger.info("Bắt đầu scheduled auto learning...")
        
        # Run async learning in thread
        def run_learning():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(learning_system.auto_crawl_and_learn())
            loop.close()
        
        learning_thread = threading.Thread(target=run_learning)
        learning_thread.start()
        learning_thread.join(timeout=7200)  # 2 hours timeout
        
    except Exception as e:
        logger.error(f"Error in scheduled learning: {e}")
    finally:
        is_learning_active = False

def check_and_learn():
    """Kiểm tra và học nếu cần thiết"""
    global learning_system
    
    try:
        stats = learning_system.get_learning_stats()
        total_items = stats.get('total_knowledge_items', 0)
        
        # Nếu ít hơn 1000 items, trigger learning
        if total_items < 1000:
            logger.info(f"Chỉ có {total_items} items, trigger learning...")
            trigger_auto_learning()
        
    except Exception as e:
        logger.error(f"Error checking learning status: {e}")

@app.post("/process-message", response_model=AIResponse)
async def process_message(message_data: Message):
    """Xử lý tin nhắn với AI thông minh"""
    global learning_system
    
    try:
        # Generate response sử dụng knowledge base
        ai_response = learning_system.generate_smart_response(message_data.message)
        
        # Tính confidence
        confidence = calculate_response_confidence(message_data.message, ai_response)
        
        # Determine source
        source = "knowledge_base" if len(ai_response) > 50 else "generative_model"
        
        # Learning status
        learning_status = "active" if is_learning_active else "idle"
        
        # Lưu conversation để training future
        save_conversation(message_data.user_id, message_data.message, ai_response)
        
        return AIResponse(
            response=ai_response,
            confidence=confidence,
            source=source,
            learning_status=learning_status
        )
        
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def calculate_response_confidence(question: str, response: str) -> float:
    """Tính confidence score cho response"""
    confidence = 0.5  # Base confidence
    
    # Length bonus
    if len(response) > 100:
        confidence += 0.2
    elif len(response) > 50:
        confidence += 0.1
    
    # Technical terms bonus
    tech_terms = ['CPU', 'GPU', 'RAM', 'SSD', 'motherboard', 'BIOS']
    term_count = sum(1 for term in tech_terms if term.lower() in response.lower())
    confidence += min(0.2, term_count * 0.05)
    
    # Vietnamese context bonus
    if any(word in response for word in ['máy tính', 'phần cứng', 'vi xử lý']):
        confidence += 0.1
    
    return min(1.0, confidence)

def save_conversation(user_id: str, question: str, response: str):
    """Lưu conversation vào database chính"""
    import sqlite3
    
    try:
        conn = sqlite3.connect("conversations.db")
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO conversations (user_id, user_message, ai_response)
            VALUES (?, ?, ?)
        ''', (user_id, question, response))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error saving conversation: {e}")

@app.post("/manual-learning")
async def manual_learning(background_tasks: BackgroundTasks):
    """Trigger manual learning"""
    global is_learning_active
    
    if is_learning_active:
        return {"status": "Learning session đang chạy"}
    
    background_tasks.add_task(run_background_learning)
    return {"status": "Manual learning started"}

async def run_background_learning():
    """Chạy learning trong background"""
    global is_learning_active, learning_system
    
    try:
        is_learning_active = True
        await learning_system.auto_crawl_and_learn()
    except Exception as e:
        logger.error(f"Error in background learning: {e}")
    finally:
        is_learning_active = False

@app.get("/learning-status", response_model=LearningStatus)
async def get_learning_status():
    """Lấy trạng thái learning"""
    global learning_system, is_learning_active
    
    try:
        stats = learning_system.get_learning_stats()
        
        return LearningStatus(
            is_active=is_learning_active,
            last_session=datetime.now().isoformat(),
            total_knowledge=stats.get('total_knowledge_items', 0),
            stats=stats
        )
        
    except Exception as e:
        logger.error(f"Error getting learning status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add-knowledge")
async def add_knowledge(question: str, answer: str, topic: str = "general"):
    """Thêm knowledge thủ công"""
    global learning_system
    
    try:
        import hashlib
        from auto_learning_system import KnowledgeItem
        from datetime import datetime
        
        knowledge_item = KnowledgeItem(
            question=question,
            answer=answer,
            source="manual",
            language="vi",
            topic=topic,
            difficulty="medium",
            timestamp=datetime.now(),
            hash_id=hashlib.md5(f"{question}{answer}".encode()).hexdigest()
        )
        
        processed = await learning_system.process_and_store_knowledge([knowledge_item])
        
        return {"status": "success", "items_added": processed}
        
    except Exception as e:
        logger.error(f"Error adding knowledge: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search-knowledge")
async def search_knowledge(query: str, limit: int = 10):
    """Tìm kiếm trong knowledge base"""
    global learning_system
    
    try:
        similar_qa = learning_system.find_similar_questions(query, top_k=limit)
        
        return {
            "query": query,
            "results": similar_qa,
            "total_found": len(similar_qa)
        }
        
    except Exception as e:
        logger.error(f"Error searching knowledge: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/knowledge-stats")
async def get_knowledge_stats():
    """Lấy thống kê chi tiết knowledge base"""
    global learning_system
    
    try:
        stats = learning_system.get_learning_stats()
        
        # Thêm thống kê chi tiết
        import sqlite3
        conn = sqlite3.connect(learning_system.knowledge_db)
        cursor = conn.cursor()
        
        # Top used knowledge
        cursor.execute('''
            SELECT question, usage_count FROM knowledge_base 
            ORDER BY usage_count DESC LIMIT 10
        ''')
        top_used = [{"question": q, "usage_count": c} for q, c in cursor.fetchall()]
        
        # Recent additions
        cursor.execute('''
            SELECT question, source, timestamp FROM knowledge_base 
            ORDER BY timestamp DESC LIMIT 10
        ''')
        recent_additions = [{"question": q, "source": s, "timestamp": t} for q, s, t in cursor.fetchall()]
        
        # Learning sessions history
        cursor.execute('''
            SELECT session_type, items_learned, timestamp FROM learning_sessions 
            ORDER BY timestamp DESC LIMIT 10
        ''')
        recent_sessions = [{"type": t, "items": i, "timestamp": ts} for t, i, ts in cursor.fetchall()]
        
        conn.close()
        
        return {
            "overview": stats,
            "top_used_knowledge": top_used,
            "recent_additions": recent_additions,
            "recent_learning_sessions": recent_sessions
        }
        
    except Exception as e:
        logger.error(f"Error getting knowledge stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize-knowledge")
async def optimize_knowledge():
    """Tối ưu hóa knowledge base"""
    global learning_system
    
    try:
        import sqlite3
        conn = sqlite3.connect(learning_system.knowledge_db)
        cursor = conn.cursor()
        
        # Remove duplicate knowledge
        cursor.execute('''
            DELETE FROM knowledge_base 
            WHERE id NOT IN (
                SELECT MIN(id) FROM knowledge_base 
                GROUP BY hash_id
            )
        ''')
        deleted_duplicates = cursor.rowcount
        
        # Remove low quality knowledge
        cursor.execute('''
            DELETE FROM knowledge_base 
            WHERE confidence_score < 0.2 AND usage_count = 0
        ''')
        deleted_low_quality = cursor.rowcount
        
        # Vacuum database
        cursor.execute('VACUUM')
        
        conn.commit()
        conn.close()
        
        return {
            "status": "success",
            "deleted_duplicates": deleted_duplicates,
            "deleted_low_quality": deleted_low_quality
        }
        
    except Exception as e:
        logger.error(f"Error optimizing knowledge: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global learning_system, is_learning_active
    
    return {
        "status": "healthy",
        "learning_active": is_learning_active,
        "total_knowledge": learning_system.get_learning_stats().get('total_knowledge_items', 0),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
async def root():
    return {
        "message": "Enhanced Zalo OA AI Server với Auto Learning",
        "version": "2.0.0",
        "features": [
            "Auto crawling từ StackOverflow, Reddit, Forums VN",
            "Vietnamese NLP processing",
            "Incremental learning",
            "Smart response generation",
            "Knowledge base management"
        ]
    }

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=False,  # Disable reload để scheduler hoạt động ổn định
        workers=1
    )