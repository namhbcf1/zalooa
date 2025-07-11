# AI Training Server - FastAPI với GPU Support
# File: ai_server.py
# Chạy: python ai_server.py

import os
import json
import sqlite3
import asyncio
from datetime import datetime
from typing import List, Dict, Optional
import logging

import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from datasets import Dataset
import fastapi
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Kiểm tra GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Configuration
MODEL_NAME = "microsoft/DialoGPT-small"  # Mô hình nhỏ cho GTX 1070
DB_PATH = "conversations.db"
MODEL_SAVE_PATH = "./fine_tuned_model"
MAX_LENGTH = 512
BATCH_SIZE = 4  # Giảm batch size cho GPU 8GB
LEARNING_RATE = 5e-5

# FastAPI app
app = FastAPI(title="Zalo OA AI Training Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class Message(BaseModel):
    user_id: str
    message: str
    timestamp: int

class TrainingData(BaseModel):
    conversations: List[Dict[str, str]]

class AIResponse(BaseModel):
    response: str
    confidence: float
    training_updated: bool

# Database setup
def init_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            user_message TEXT NOT NULL,
            ai_response TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            is_training_data BOOLEAN DEFAULT FALSE
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_version TEXT,
            training_loss REAL,
            eval_loss REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

# AI Model Manager
class AIModelManager:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.is_training = False
        self.load_model()
    
    def load_model(self):
        """Load pre-trained hoặc fine-tuned model"""
        try:
            if os.path.exists(MODEL_SAVE_PATH):
                logger.info("Loading fine-tuned model...")
                self.tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_PATH)
                self.model = AutoModelForCausalLM.from_pretrained(MODEL_SAVE_PATH)
            else:
                logger.info("Loading base model...")
                self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
                self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
            
            # Thêm pad token nếu chưa có
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model.to(device)
            self.model.eval()
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def generate_response(self, user_message: str, user_id: str) -> tuple[str, float]:
        """Generate AI response"""
        try:
            # Lấy context từ conversation history
            context = self.get_conversation_context(user_id, limit=5)
            
            # Tạo prompt
            if context:
                prompt = f"Lịch sử hội thoại:\n{context}\nNgười dùng: {user_message}\nAI:"
            else:
                prompt = f"Người dùng: {user_message}\nAI:"
            
            # Tokenize
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
            inputs = inputs.to(device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 100,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the AI response part
            if "AI:" in response:
                ai_response = response.split("AI:")[-1].strip()
            else:
                ai_response = "Xin chào! Tôi là AI assistant của bạn."
            
            # Calculate confidence (simplified)
            confidence = min(0.8, len(ai_response) / 50)
            
            return ai_response, confidence
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Xin lỗi, tôi đang gặp sự cố. Vui lòng thử lại.", 0.1
    
    def get_conversation_context(self, user_id: str, limit: int = 5) -> str:
        """Lấy context từ database"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT user_message, ai_response FROM conversations 
                WHERE user_id = ? AND ai_response IS NOT NULL
                ORDER BY timestamp DESC LIMIT ?
            ''', (user_id, limit))
            
            conversations = cursor.fetchall()
            conn.close()
            
            context_parts = []
            for user_msg, ai_resp in reversed(conversations):
                context_parts.append(f"Người dùng: {user_msg}")
                context_parts.append(f"AI: {ai_resp}")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error getting context: {e}")
            return ""
    
    async def train_model(self):
        """Training model với dữ liệu mới"""
        if self.is_training:
            logger.info("Training đang diễn ra...")
            return
        
        self.is_training = True
        try:
            logger.info("Bắt đầu training...")
            
            # Lấy training data từ database
            training_data = self.prepare_training_data()
            
            if len(training_data) < 10:
                logger.info("Chưa đủ dữ liệu để training")
                return
            
            # Tạo dataset
            dataset = Dataset.from_dict({"text": training_data})
            
            # Tokenize dataset
            def tokenize_function(examples):
                return self.tokenizer(
                    examples["text"],
                    truncation=True,
                    padding=True,
                    max_length=MAX_LENGTH
                )
            
            tokenized_dataset = dataset.map(tokenize_function, batched=True)
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=f"{MODEL_SAVE_PATH}_temp",
                overwrite_output_dir=True,
                num_train_epochs=3,
                per_device_train_batch_size=BATCH_SIZE,
                gradient_accumulation_steps=2,
                learning_rate=LEARNING_RATE,
                warmup_steps=100,
                logging_steps=10,
                save_steps=500,
                save_total_limit=1,
                prediction_loss_only=True,
                fp16=True,  # Mixed precision cho GPU tiết kiệm VRAM
                dataloader_pin_memory=False,
            )
            
            # Trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=tokenized_dataset,
                tokenizer=self.tokenizer,
            )
            
            # Train
            trainer.train()
            
            # Save model
            trainer.save_model(MODEL_SAVE_PATH)
            self.tokenizer.save_pretrained(MODEL_SAVE_PATH)
            
            # Log training session
            self.log_training_session("v1.0", 0.0, 0.0)
            
            logger.info("Training hoàn thành!")
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
        finally:
            self.is_training = False
    
    def prepare_training_data(self) -> List[str]:
        """Chuẩn bị dữ liệu training từ conversations"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT user_message, ai_response FROM conversations 
                WHERE ai_response IS NOT NULL
                ORDER BY timestamp
            ''')
            
            conversations = cursor.fetchall()
            conn.close()
            
            training_texts = []
            for user_msg, ai_resp in conversations:
                text = f"Người dùng: {user_msg}\nAI: {ai_resp}{self.tokenizer.eos_token}"
                training_texts.append(text)
            
            return training_texts
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return []
    
    def log_training_session(self, version: str, train_loss: float, eval_loss: float):
        """Log training session"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO training_sessions (model_version, training_loss, eval_loss)
                VALUES (?, ?, ?)
            ''', (version, train_loss, eval_loss))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error logging training session: {e}")

# Global model manager
model_manager = AIModelManager()

@app.on_event("startup")
async def startup_event():
    init_database()
    logger.info("AI Server started successfully!")

@app.post("/process-message", response_model=AIResponse)
async def process_message(message_data: Message):
    """Xử lý tin nhắn từ user và generate response"""
    try:
        # Generate AI response
        ai_response, confidence = model_manager.generate_response(
            message_data.message, 
            message_data.user_id
        )
        
        # Lưu vào database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO conversations (user_id, user_message, ai_response)
            VALUES (?, ?, ?)
        ''', (message_data.user_id, message_data.message, ai_response))
        conn.commit()
        conn.close()
        
        # Auto training nếu đủ dữ liệu mới
        training_updated = await auto_training_check()
        
        return AIResponse(
            response=ai_response,
            confidence=confidence,
            training_updated=training_updated
        )
        
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def auto_training_check() -> bool:
    """Kiểm tra và tự động training nếu cần"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Đếm số conversation mới từ lần training cuối
        cursor.execute('''
            SELECT COUNT(*) FROM conversations 
            WHERE timestamp > (
                SELECT COALESCE(MAX(timestamp), '1970-01-01') 
                FROM training_sessions
            )
        ''')
        
        new_conversations = cursor.fetchone()[0]
        conn.close()
        
        # Auto training nếu có >= 50 conversation mới
        if new_conversations >= 50 and not model_manager.is_training:
            asyncio.create_task(model_manager.train_model())
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error in auto training check: {e}")
        return False

@app.post("/manual-training")
async def manual_training():
    """Trigger manual training"""
    if model_manager.is_training:
        return {"status": "Training đang diễn ra"}
    
    asyncio.create_task(model_manager.train_model())
    return {"status": "Training bắt đầu"}

@app.get("/stats")
async def get_stats():
    """Lấy thống kê hệ thống"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM conversations")
        total_conversations = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT user_id) FROM conversations")
        unique_users = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM training_sessions")
        training_sessions = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_conversations": total_conversations,
            "unique_users": unique_users,
            "training_sessions": training_sessions,
            "model_status": "training" if model_manager.is_training else "ready",
            "device": str(device)
        }
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Zalo OA AI Training Server", "status": "running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)