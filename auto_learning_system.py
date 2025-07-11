# auto_learning_system.py - AI tự động học toàn bộ kiến thức máy tính
import asyncio
import aiohttp
import sqlite3
import json
import re
import time
import random
from datetime import datetime, timedelta
from typing import List, Dict, Set
from dataclasses import dataclass
import hashlib
import logging

import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    AutoModelForSequenceClassification,
    pipeline, Trainer, TrainingArguments
)
from datasets import Dataset
import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Vietnamese NLP
import underthesea
from pyvi import ViTokenizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class KnowledgeItem:
    question: str
    answer: str
    source: str
    language: str
    topic: str
    difficulty: str
    timestamp: datetime
    hash_id: str

class ComputerKnowledgeCrawler:
    """Crawler tự động thu thập kiến thức máy tính từ internet"""
    
    def __init__(self):
        self.session = None
        self.computer_keywords = {
            'vi': [
                'máy tính', 'computer', 'PC', 'CPU', 'GPU', 'RAM', 'SSD', 'HDD',
                'mainboard', 'bo mạch chủ', 'card đồ họa', 'vi xử lý', 'bộ nhớ',
                'ổ cứng', 'nguồn máy tính', 'case máy tính', 'tản nhiệt',
                'BIOS', 'UEFI', 'Windows', 'Linux', 'macOS', 'driver',
                'phần mềm', 'phần cứng', 'hardware', 'software',
                'lắp ráp máy tính', 'nâng cấp máy tính', 'sửa chữa máy tính',
                'gaming PC', 'workstation', 'server', 'build PC'
            ],
            'en': [
                'computer', 'PC', 'desktop', 'workstation', 'gaming PC',
                'CPU', 'processor', 'GPU', 'graphics card', 'RAM', 'memory',
                'SSD', 'NVMe', 'HDD', 'storage', 'motherboard', 'PSU',
                'cooling', 'thermal', 'overclocking', 'BIOS', 'UEFI',
                'Windows', 'Linux', 'drivers', 'hardware', 'software',
                'build guide', 'PC build', 'computer assembly', 'troubleshooting'
            ]
        }
        
        # Sources to crawl
        self.sources = {
            'stackoverflow': 'https://stackoverflow.com/questions/tagged/',
            'reddit_buildapc': 'https://www.reddit.com/r/buildapc/',
            'reddit_pcmr': 'https://www.reddit.com/r/pcmasterrace/',
            'voz_computer': 'https://voz.vn/f/may-tinh.17/',
            'tinhte': 'https://tinhte.vn/forums/may-tinh-pc.54/',
            'techpowerup': 'https://www.techpowerup.com/forums/',
            'anandtech': 'https://forums.anandtech.com/',
            'tomshardware': 'https://forums.tomshardware.com/'
        }
        
    async def init_session(self):
        """Khởi tạo aiohttp session"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.session = aiohttp.ClientSession(headers=headers)
    
    async def close_session(self):
        """Đóng session"""
        if self.session:
            await self.session.close()
    
    async def crawl_stackoverflow(self, limit=1000):
        """Crawl Stack Overflow cho câu hỏi về máy tính"""
        knowledge_items = []
        
        computer_tags = ['computer', 'pc', 'hardware', 'cpu', 'gpu', 'ram', 'motherboard']
        
        for tag in computer_tags:
            try:
                url = f"https://api.stackexchange.com/2.3/questions?tagged={tag}&site=stackoverflow&pagesize=100&sort=votes"
                
                async with self.session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for item in data.get('items', []):
                            if 'laptop' not in item.get('title', '').lower():
                                question = item.get('title', '')
                                body = item.get('body', '')
                                
                                # Get answers if available
                                if item.get('answer_count', 0) > 0:
                                    answer_url = f"https://api.stackexchange.com/2.3/questions/{item['question_id']}/answers?site=stackoverflow&filter=withbody"
                                    
                                    async with self.session.get(answer_url) as ans_response:
                                        if ans_response.status == 200:
                                            ans_data = await ans_response.json()
                                            if ans_data.get('items'):
                                                answer = ans_data['items'][0].get('body', '')
                                                
                                                knowledge_item = KnowledgeItem(
                                                    question=question,
                                                    answer=self.clean_html(answer),
                                                    source='stackoverflow',
                                                    language='en',
                                                    topic=tag,
                                                    difficulty='medium',
                                                    timestamp=datetime.now(),
                                                    hash_id=hashlib.md5(f"{question}{answer}".encode()).hexdigest()
                                                )
                                                knowledge_items.append(knowledge_item)
                
                await asyncio.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error crawling StackOverflow {tag}: {e}")
        
        return knowledge_items
    
    async def crawl_reddit(self, subreddit, limit=500):
        """Crawl Reddit cho discussions về máy tính"""
        knowledge_items = []
        
        try:
            url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for post in data.get('data', {}).get('children', []):
                        post_data = post.get('data', {})
                        
                        title = post_data.get('title', '')
                        selftext = post_data.get('selftext', '')
                        
                        # Filter computer-related posts (exclude laptop)
                        if self.is_computer_related(title + ' ' + selftext) and 'laptop' not in title.lower():
                            knowledge_item = KnowledgeItem(
                                question=title,
                                answer=selftext,
                                source=f'reddit_{subreddit}',
                                language='en',
                                topic='general',
                                difficulty='beginner',
                                timestamp=datetime.now(),
                                hash_id=hashlib.md5(f"{title}{selftext}".encode()).hexdigest()
                            )
                            knowledge_items.append(knowledge_item)
        
        except Exception as e:
            logger.error(f"Error crawling Reddit {subreddit}: {e}")
        
        return knowledge_items
    
    async def crawl_vietnamese_forums(self):
        """Crawl các forum Việt Nam về máy tính"""
        knowledge_items = []
        
        # Tinhte.vn crawling
        try:
            url = "https://tinhte.vn/forums/may-tinh-pc.54/"
            async with self.session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    threads = soup.find_all('div', class_='structItem-title')
                    
                    for thread in threads[:100]:  # Limit to 100 threads
                        title_elem = thread.find('a')
                        if title_elem and 'laptop' not in title_elem.text.lower():
                            title = title_elem.text.strip()
                            
                            knowledge_item = KnowledgeItem(
                                question=title,
                                answer="",  # Will be filled later
                                source='tinhte',
                                language='vi',
                                topic='general',
                                difficulty='beginner',
                                timestamp=datetime.now(),
                                hash_id=hashlib.md5(title.encode()).hexdigest()
                            )
                            knowledge_items.append(knowledge_item)
        
        except Exception as e:
            logger.error(f"Error crawling Tinhte: {e}")
        
        return knowledge_items
    
    def is_computer_related(self, text: str) -> bool:
        """Kiểm tra text có liên quan đến máy tính không"""
        text_lower = text.lower()
        
        # Check Vietnamese keywords
        for keyword in self.computer_keywords['vi']:
            if keyword.lower() in text_lower:
                return True
        
        # Check English keywords
        for keyword in self.computer_keywords['en']:
            if keyword.lower() in text_lower:
                return True
        
        return False
    
    def clean_html(self, text: str) -> str:
        """Làm sạch HTML tags"""
        soup = BeautifulSoup(text, 'html.parser')
        return soup.get_text().strip()

class VietnameseNLPProcessor:
    """Xử lý ngôn ngữ tự nhiên tiếng Việt"""
    
    def __init__(self):
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="uitnlp/vietnamese-sentiment"
        )
        
    def tokenize_vietnamese(self, text: str) -> List[str]:
        """Tokenize tiếng Việt"""
        return ViTokenizer.tokenize(text).split()
    
    def extract_entities(self, text: str) -> List[str]:
        """Trích xuất entities từ text tiếng Việt"""
        try:
            entities = underthesea.ner(text)
            return [entity[0] for entity in entities if entity[1] != 'O']
        except:
            return []
    
    def classify_intent(self, text: str) -> str:
        """Phân loại intent của câu hỏi"""
        question_patterns = {
            'troubleshooting': ['bị lỗi', 'không hoạt động', 'sửa', 'fix', 'khắc phục'],
            'buying_advice': ['nên mua', 'tư vấn', 'chọn', 'recommend', 'budget'],
            'build_guide': ['lắp ráp', 'build', 'cấu hình', 'combo'],
            'upgrade': ['nâng cấp', 'upgrade', 'thay thế', 'replace'],
            'comparison': ['so sánh', 'compare', 'vs', 'khác nhau'],
            'explanation': ['là gì', 'what is', 'giải thích', 'hoạt động']
        }
        
        text_lower = text.lower()
        
        for intent, patterns in question_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    return intent
        
        return 'general'

class OnlineLearningSystem:
    """Hệ thống học online tự động"""
    
    def __init__(self, model_name="microsoft/DialoGPT-small"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.knowledge_db = "computer_knowledge.db"
        self.vectorizer = TfidfVectorizer(max_features=10000)
        self.knowledge_vectors = None
        self.nlp_processor = VietnameseNLPProcessor()
        
        self.init_knowledge_db()
        self.load_model()
    
    def init_knowledge_db(self):
        """Khởi tạo database kiến thức"""
        conn = sqlite3.connect(self.knowledge_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_base (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hash_id TEXT UNIQUE,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                source TEXT,
                language TEXT,
                topic TEXT,
                difficulty TEXT,
                confidence_score REAL DEFAULT 0.0,
                usage_count INTEGER DEFAULT 0,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                embedding BLOB
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_type TEXT,
                items_learned INTEGER,
                sources TEXT,
                model_version TEXT,
                performance_score REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_model(self):
        """Load model và tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
    
    async def auto_crawl_and_learn(self):
        """Tự động crawl và học kiến thức mới"""
        logger.info("Bắt đầu auto crawling...")
        
        crawler = ComputerKnowledgeCrawler()
        await crawler.init_session()
        
        try:
            # Crawl từ nhiều nguồn
            all_knowledge = []
            
            # StackOverflow
            logger.info("Crawling StackOverflow...")
            stackoverflow_data = await crawler.crawl_stackoverflow(limit=1000)
            all_knowledge.extend(stackoverflow_data)
            
            # Reddit
            logger.info("Crawling Reddit...")
            reddit_buildapc = await crawler.crawl_reddit('buildapc', limit=500)
            reddit_pcmr = await crawler.crawl_reddit('pcmasterrace', limit=300)
            all_knowledge.extend(reddit_buildapc)
            all_knowledge.extend(reddit_pcmr)
            
            # Vietnamese forums
            logger.info("Crawling Vietnamese forums...")
            vn_forums = await crawler.crawl_vietnamese_forums()
            all_knowledge.extend(vn_forums)
            
            # Process và lưu kiến thức
            logger.info(f"Xử lý {len(all_knowledge)} items...")
            processed_count = await self.process_and_store_knowledge(all_knowledge)
            
            # Auto training với dữ liệu mới
            if processed_count > 50:
                logger.info(f"Training với {processed_count} items mới...")
                await self.incremental_training()
            
            # Log learning session
            self.log_learning_session('auto_crawl', len(all_knowledge), 
                                    ['stackoverflow', 'reddit', 'vietnamese_forums'])
            
            logger.info(f"Hoàn thành auto learning: {processed_count} items mới")
            
        finally:
            await crawler.close_session()
    
    async def process_and_store_knowledge(self, knowledge_items: List[KnowledgeItem]) -> int:
        """Xử lý và lưu trữ kiến thức"""
        conn = sqlite3.connect(self.knowledge_db)
        cursor = conn.cursor()
        
        processed_count = 0
        
        for item in knowledge_items:
            try:
                # Preprocess text
                processed_question = self.preprocess_text(item.question)
                processed_answer = self.preprocess_text(item.answer)
                
                if len(processed_question) < 10 or len(processed_answer) < 10:
                    continue
                
                # Calculate confidence score
                confidence = self.calculate_confidence_score(processed_question, processed_answer)
                
                # Extract topic từ question
                topic = self.extract_topic(processed_question)
                
                # Classify difficulty
                difficulty = self.classify_difficulty(processed_question, processed_answer)
                
                # Insert vào database (ignore nếu đã tồn tại)
                cursor.execute('''
                    INSERT OR IGNORE INTO knowledge_base 
                    (hash_id, question, answer, source, language, topic, difficulty, confidence_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (item.hash_id, processed_question, processed_answer, 
                      item.source, item.language, topic, difficulty, confidence))
                
                if cursor.rowcount > 0:
                    processed_count += 1
                
                # Batch commit
                if processed_count % 100 == 0:
                    conn.commit()
                    logger.info(f"Processed {processed_count} items...")
            
            except Exception as e:
                logger.error(f"Error processing item: {e}")
        
        conn.commit()
        conn.close()
        
        return processed_count
    
    def preprocess_text(self, text: str) -> str:
        """Tiền xử lý text"""
        if not text:
            return ""
        
        # Remove HTML
        text = BeautifulSoup(text, 'html.parser').get_text()
        
        # Remove special characters
        text = re.sub(r'[^\w\s\u00C0-\u024F\u1E00-\u1EFF]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def calculate_confidence_score(self, question: str, answer: str) -> float:
        """Tính confidence score cho knowledge item"""
        score = 0.0
        
        # Length score
        if len(answer) > 100:
            score += 0.3
        elif len(answer) > 50:
            score += 0.2
        
        # Technical terms score
        tech_terms = ['CPU', 'GPU', 'RAM', 'SSD', 'motherboard', 'BIOS', 'driver']
        term_count = sum(1 for term in tech_terms if term.lower() in answer.lower())
        score += min(0.3, term_count * 0.1)
        
        # Vietnamese context score
        if any(word in question.lower() for word in ['máy tính', 'vi xử lý', 'bo mạch']):
            score += 0.2
        
        # Question quality score
        if '?' in question or any(word in question.lower() for word in ['how', 'what', 'why', 'làm sao', 'tại sao']):
            score += 0.2
        
        return min(1.0, score)
    
    def extract_topic(self, text: str) -> str:
        """Trích xuất topic từ text"""
        topic_keywords = {
            'cpu': ['cpu', 'processor', 'vi xử lý', 'intel', 'amd'],
            'gpu': ['gpu', 'graphics', 'card đồ họa', 'nvidia', 'rtx', 'gtx'],
            'ram': ['ram', 'memory', 'bộ nhớ', 'ddr4', 'ddr5'],
            'storage': ['ssd', 'hdd', 'nvme', 'ổ cứng', 'storage'],
            'motherboard': ['motherboard', 'mainboard', 'bo mạch chủ'],
            'psu': ['psu', 'power supply', 'nguồn', 'watt'],
            'cooling': ['cooling', 'fan', 'tản nhiệt', 'thermal'],
            'bios': ['bios', 'uefi', 'firmware'],
            'os': ['windows', 'linux', 'macos', 'operating system'],
            'software': ['software', 'driver', 'phần mềm', 'program']
        }
        
        text_lower = text.lower()
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return topic
        
        return 'general'
    
    def classify_difficulty(self, question: str, answer: str) -> str:
        """Phân loại độ khó"""
        # Simple heuristic based on content complexity
        tech_level_indicators = {
            'beginner': ['basic', 'simple', 'easy', 'cơ bản', 'đơn giản'],
            'intermediate': ['install', 'setup', 'configure', 'cài đặt', 'thiết lập'],
            'advanced': ['overclock', 'bios', 'registry', 'advanced', 'nâng cao', 'chuyên sâu']
        }
        
        combined_text = (question + ' ' + answer).lower()
        
        for level, indicators in tech_level_indicators.items():
            if any(indicator in combined_text for indicator in indicators):
                return level
        
        # Default based on answer length
        if len(answer) > 500:
            return 'advanced'
        elif len(answer) > 200:
            return 'intermediate'
        else:
            return 'beginner'
    
    async def incremental_training(self):
        """Training incremental với dữ liệu mới"""
        logger.info("Bắt đầu incremental training...")
        
        # Lấy dữ liệu training mới
        training_data = self.prepare_training_data()
        
        if len(training_data) < 50:
            logger.info("Không đủ dữ liệu mới để training")
            return
        
        try:
            # Create dataset
            dataset = Dataset.from_dict({"text": training_data})
            
            # Tokenize
            def tokenize_function(examples):
                return self.tokenizer(
                    examples["text"],
                    truncation=True,
                    padding=True,
                    max_length=512
                )
            
            tokenized_dataset = dataset.map(tokenize_function, batched=True)
            
            # Training arguments for incremental learning
            training_args = TrainingArguments(
                output_dir="./incremental_model",
                overwrite_output_dir=True,
                num_train_epochs=1,  # Single epoch cho incremental
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                learning_rate=1e-5,  # Lower learning rate
                warmup_steps=50,
                logging_steps=10,
                save_steps=100,
                fp16=True,
                dataloader_pin_memory=False,
            )
            
            # Trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_dataset,
                tokenizer=self.tokenizer,
            )
            
            # Train
            trainer.train()
            
            # Save updated model
            trainer.save_model("./fine_tuned_computer_model")
            
            logger.info("Incremental training hoàn thành!")
            
        except Exception as e:
            logger.error(f"Error during incremental training: {e}")
    
    def prepare_training_data(self) -> List[str]:
        """Chuẩn bị dữ liệu training từ knowledge base"""
        conn = sqlite3.connect(self.knowledge_db)
        cursor = conn.cursor()
        
        # Lấy high-quality knowledge items
        cursor.execute('''
            SELECT question, answer FROM knowledge_base 
            WHERE confidence_score > 0.5 
            AND length(answer) > 50
            ORDER BY confidence_score DESC, timestamp DESC
            LIMIT 1000
        ''')
        
        training_texts = []
        
        for question, answer in cursor.fetchall():
            # Format training text
            text = f"Human: {question}\nAssistant: {answer}{self.tokenizer.eos_token}"
            training_texts.append(text)
        
        conn.close()
        return training_texts
    
    def generate_smart_response(self, user_question: str) -> str:
        """Generate response thông minh với knowledge base"""
        try:
            # Tìm kiếm trong knowledge base
            similar_qa = self.find_similar_questions(user_question)
            
            if similar_qa:
                # Có knowledge tương tự, combine với generation
                context = f"Tham khảo: {similar_qa[0]['answer']}\n\n"
                prompt = f"{context}Human: {user_question}\nAssistant:"
            else:
                # Không có knowledge, dùng model thuần
                prompt = f"Human: {user_question}\nAssistant:"
            
            # Generate response
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 150,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only assistant part
            if "Assistant:" in response:
                ai_response = response.split("Assistant:")[-1].strip()
            else:
                ai_response = "Xin lỗi, tôi cần thêm thông tin để trả lời câu hỏi này."
            
            # Update usage statistics
            if similar_qa:
                self.update_knowledge_usage(similar_qa[0]['id'])
            
            return ai_response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Xin lỗi, tôi đang gặp sự cố khi xử lý câu hỏi của bạn."
    
    def find_similar_questions(self, question: str, top_k=3) -> List[Dict]:
        """Tìm câu hỏi tương tự trong knowledge base"""
        conn = sqlite3.connect(self.knowledge_db)
        cursor = conn.cursor()
        
        # Get all questions để so sánh
        cursor.execute('SELECT id, question, answer FROM knowledge_base WHERE confidence_score > 0.3')
        knowledge_items = cursor.fetchall()
        
        if not knowledge_items:
            return []
        
        # Calculate similarity
        questions = [item[1] for item in knowledge_items]
        questions.append(question)
        
        try:
            # TF-IDF similarity
            tfidf_matrix = self.vectorizer.fit_transform(questions)
            similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1]).flatten()
            
            # Get top similar questions
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            similar_qa = []
            for idx in top_indices:
                if similarities[idx] > 0.3:  # Threshold
                    similar_qa.append({
                        'id': knowledge_items[idx][0],
                        'question': knowledge_items[idx][1],
                        'answer': knowledge_items[idx][2],
                        'similarity': similarities[idx]
                    })
            
            conn.close()
            return similar_qa
            
        except Exception as e:
            logger.error(f"Error finding similar questions: {e}")
            conn.close()
            return []
    
    def update_knowledge_usage(self, knowledge_id: int):
        """Cập nhật usage statistics"""
        conn = sqlite3.connect(self.knowledge_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE knowledge_base 
            SET usage_count = usage_count + 1 
            WHERE id = ?
        ''', (knowledge_id,))
        
        conn.commit()
        conn.close()
    
    def log_learning_session(self, session_type: str, items_count: int, sources: List[str]):
        """Log learning session"""
        conn = sqlite3.connect(self.knowledge_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO learning_sessions 
            (session_type, items_learned, sources, model_version, performance_score)
            VALUES (?, ?, ?, ?, ?)
        ''', (session_type, items_count, ','.join(sources), 'v1.0', 0.8))
        
        conn.commit()
        conn.close()
    
    def get_learning_stats(self) -> Dict:
        """Lấy thống kê learning"""
        conn = sqlite3.connect(self.knowledge_db)
        cursor = conn.cursor()
        
        # Total knowledge items
        cursor.execute('SELECT COUNT(*) FROM knowledge_base')
        total_items = cursor.fetchone()[0]
        
        # By language
        cursor.execute('SELECT language, COUNT(*) FROM knowledge_base GROUP BY language')
        by_language = dict(cursor.fetchall())
        
        # By topic
        cursor.execute('SELECT topic, COUNT(*) FROM knowledge_base GROUP BY topic ORDER BY COUNT(*) DESC LIMIT 10')
        by_topic = dict(cursor.fetchall())
        
        # By source
        cursor.execute('SELECT source, COUNT(*) FROM knowledge_base GROUP BY source')
        by_source = dict(cursor.fetchall())
        
        # Learning sessions
        cursor.execute('SELECT COUNT(*) FROM learning_sessions')
        total_sessions = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_knowledge_items': total_items,
            'by_language': by_language,
            'by_topic': by_topic,
            'by_source': by_source,
            'total_learning_sessions': total_sessions
        }

async def main():
    """Main function để chạy auto learning"""
    learning_system = OnlineLearningSystem()
    
    # Schedule auto learning mỗi ngày
    while True:
        try:
            logger.info("=== BẮT ĐẦU AUTO LEARNING SESSION ===")
            
            # Auto crawl và học
            await learning_system.auto_crawl_and_learn()
            
            # In stats
            stats = learning_system.get_learning_stats()
            logger.info(f"Learning Stats: {stats}")
            
            # Nghỉ 24 giờ trước lần học tiếp theo
            logger.info("=== HOÀN THÀNH AUTO LEARNING SESSION ===")
            logger.info("Nghỉ 24 giờ trước session tiếp theo...")
            
            await asyncio.sleep(24 * 3600)  # 24 hours
            
        except Exception as e:
            logger.error(f"Error in auto learning: {e}")
            await asyncio.sleep(3600)  # Retry after 1 hour

if __name__ == "__main__":
    asyncio.run(main())