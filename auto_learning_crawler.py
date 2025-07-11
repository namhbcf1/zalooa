#!/usr/bin/env python3
"""
🧠 AUTO LEARNING CRAWLER - Tự học từ internet thật sự
Crawl data PC gaming từ các website Việt Nam và tự training
"""

import requests
import sqlite3
import time
import re
import json
import threading
import schedule
from datetime import datetime
from typing import List, Dict, Tuple
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutoLearningCrawler:
    """Auto Learning Crawler cho PC Gaming Knowledge"""
    
    def __init__(self, db_path="conversations.db"):
        self.db_path = db_path
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Vietnamese PC websites to crawl
        self.pc_websites = [
            {
                'name': 'Tinhte',
                'base_url': 'https://tinhte.vn',
                'search_urls': [
                    'https://tinhte.vn/search/?q=cpu+intel+i7',
                    'https://tinhte.vn/search/?q=gpu+rtx+4070',
                    'https://tinhte.vn/search/?q=pc+gaming',
                    'https://tinhte.vn/search/?q=build+pc'
                ]
            },
            {
                'name': 'Genk',
                'base_url': 'https://genk.vn',
                'search_urls': [
                    'https://genk.vn/tim-kiem.chn?q=cpu+gaming',
                    'https://genk.vn/tim-kiem.chn?q=gpu+rtx',
                    'https://genk.vn/tim-kiem.chn?q=pc+gaming'
                ]
            }
        ]
        
        # PC components keywords
        self.pc_keywords = [
            'cpu', 'processor', 'intel', 'amd', 'i5', 'i7', 'i9', 'ryzen',
            'gpu', 'vga', 'rtx', 'gtx', 'nvidia', 'radeon',
            'ram', 'memory', 'ddr4', 'ddr5',
            'ssd', 'nvme', 'hdd', 'storage',
            'mainboard', 'motherboard', 'chipset',
            'psu', 'power supply', 'nguồn',
            'gaming', 'pc', 'build', 'cấu hình'
        ]
        
        self.init_knowledge_db()
    
    def init_knowledge_db(self):
        """Initialize knowledge database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Auto-learned knowledge table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS auto_knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_url TEXT NOT NULL,
                title TEXT,
                content TEXT NOT NULL,
                category TEXT,
                keywords TEXT,
                confidence REAL DEFAULT 0.0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                processed BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Generated Q&A pairs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS generated_qa (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                source_id INTEGER,
                category TEXT,
                confidence REAL DEFAULT 0.0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                used_for_training BOOLEAN DEFAULT FALSE,
                FOREIGN KEY (source_id) REFERENCES auto_knowledge (id)
            )
        ''')
        
        # Learning sessions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_start DATETIME DEFAULT CURRENT_TIMESTAMP,
                session_end DATETIME,
                articles_crawled INTEGER DEFAULT 0,
                qa_pairs_generated INTEGER DEFAULT 0,
                training_triggered BOOLEAN DEFAULT FALSE,
                status TEXT DEFAULT 'running'
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Knowledge database initialized")
    
    def crawl_pc_content(self, max_articles=20) -> List[Dict]:
        """Crawl PC content from Vietnamese websites"""
        logger.info("🕷️ Starting PC content crawling...")
        
        crawled_content = []
        
        # Simple crawling from tech news sites
        tech_urls = [
            'https://genk.vn/cong-nghe.chn',
            'https://vnexpress.net/so-hoa',
        ]
        
        for url in tech_urls:
            try:
                logger.info(f"Crawling: {url}")
                response = self.session.get(url, timeout=10)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Find articles
                    articles = soup.find_all(['article', 'div'], class_=re.compile(r'(article|post|news|item)'))
                    
                    for article in articles[:5]:  # Limit per site
                        title_elem = article.find(['h1', 'h2', 'h3', 'a'])
                        content_elem = article.find(['p', 'div'], class_=re.compile(r'(content|summary|desc)'))
                        
                        if title_elem and content_elem:
                            title = title_elem.get_text(strip=True)
                            content = content_elem.get_text(strip=True)
                            
                            # Check if PC-related
                            if self.is_pc_related(title + " " + content):
                                crawled_content.append({
                                    'source_url': url,
                                    'title': title,
                                    'content': content,
                                    'category': self.categorize_content(title + " " + content)
                                })
                                
                                if len(crawled_content) >= max_articles:
                                    break
                
                time.sleep(2)  # Be respectful
                
            except Exception as e:
                logger.error(f"Error crawling {url}: {e}")
                continue
        
        # Add some manual PC knowledge if crawling fails
        if len(crawled_content) < 5:
            crawled_content.extend(self.get_manual_pc_knowledge())
        
        logger.info(f"✅ Crawled {len(crawled_content)} PC articles")
        return crawled_content
    
    def get_manual_pc_knowledge(self) -> List[Dict]:
        """Get manual PC knowledge as fallback"""
        return [
            {
                'source_url': 'manual_knowledge',
                'title': 'CPU Intel Core i7-13700F Review',
                'content': 'Intel Core i7-13700F là CPU gaming mạnh mẽ với 16 cores (8P+8E), 24 threads. Hiệu năng gaming vượt trội, phù hợp cho build PC cao cấp. Giá khoảng 9-10 triệu.',
                'category': 'CPU'
            },
            {
                'source_url': 'manual_knowledge', 
                'title': 'GPU RTX 4070 vs RTX 4060 So sánh',
                'content': 'RTX 4070 mạnh hơn RTX 4060 khoảng 20-25%. RTX 4070 có 12GB VRAM vs 8GB của 4060. Giá RTX 4070 khoảng 14-15 triệu, RTX 4060 khoảng 8-9 triệu.',
                'category': 'GPU'
            },
            {
                'source_url': 'manual_knowledge',
                'title': 'RAM DDR5 vs DDR4 cho Gaming',
                'content': 'DDR5 nhanh hơn DDR4 nhưng giá đắt hơn. Cho gaming hiện tại, DDR4-3200 16GB vẫn đủ. DDR5 phù hợp cho build cao cấp hoặc workstation.',
                'category': 'RAM'
            },
            {
                'source_url': 'manual_knowledge',
                'title': 'SSD NVMe vs SATA cho Gaming',
                'content': 'SSD NVMe nhanh hơn SATA 3-5 lần. Game load nhanh hơn, Windows boot nhanh hơn. Giá chênh lệch ít nên nên ưu tiên NVMe cho build mới.',
                'category': 'Storage'
            },
            {
                'source_url': 'manual_knowledge',
                'title': 'Build PC Gaming 25 triệu 2024',
                'content': 'PC Gaming 25 triệu: CPU i5-13400F (4.5tr), GPU RTX 4060 Ti (10tr), RAM 16GB DDR4 (1.2tr), SSD 500GB (1tr), Main B660 (2tr), PSU 650W (1.5tr), Case (1tr). Total: ~25tr.',
                'category': 'Build'
            }
        ]
    
    def is_pc_related(self, text: str) -> bool:
        """Check if content is PC-related"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.pc_keywords)
    
    def categorize_content(self, text: str) -> str:
        """Categorize PC content"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['cpu', 'processor', 'intel', 'amd', 'i5', 'i7', 'ryzen']):
            return 'CPU'
        elif any(word in text_lower for word in ['gpu', 'vga', 'rtx', 'gtx', 'nvidia']):
            return 'GPU'
        elif any(word in text_lower for word in ['ram', 'memory', 'ddr']):
            return 'RAM'
        elif any(word in text_lower for word in ['ssd', 'hdd', 'storage']):
            return 'Storage'
        elif any(word in text_lower for word in ['build', 'cấu hình', 'pc gaming']):
            return 'Build'
        else:
            return 'General'
    
    def save_crawled_content(self, content_list: List[Dict]) -> int:
        """Save crawled content to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        saved_count = 0
        for content in content_list:
            try:
                cursor.execute('''
                    INSERT INTO auto_knowledge (source_url, title, content, category, keywords)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    content['source_url'],
                    content['title'],
                    content['content'],
                    content['category'],
                    ','.join(self.extract_keywords(content['content']))
                ))
                saved_count += 1
            except Exception as e:
                logger.error(f"Error saving content: {e}")
        
        conn.commit()
        conn.close()
        
        logger.info(f"💾 Saved {saved_count} articles to database")
        return saved_count
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        keywords = []
        text_lower = text.lower()
        
        for keyword in self.pc_keywords:
            if keyword in text_lower:
                keywords.append(keyword)
        
        return keywords[:10]  # Limit keywords
    
    def generate_qa_pairs(self, content_id: int, title: str, content: str, category: str) -> List[Dict]:
        """Generate Q&A pairs from content"""
        qa_pairs = []
        
        # Simple rule-based Q&A generation
        if category == 'CPU':
            qa_pairs.extend([
                {
                    'question': f"{title.split()[0]} có tốt không?",
                    'answer': content[:200] + "...",
                    'category': category
                },
                {
                    'question': f"Giá {title.split()[0]} bao nhiêu?",
                    'answer': f"Theo thông tin mới nhất: {content[:150]}...",
                    'category': category
                }
            ])
        
        elif category == 'GPU':
            qa_pairs.extend([
                {
                    'question': f"{title.split()[0]} gaming được không?",
                    'answer': content[:200] + "...",
                    'category': category
                },
                {
                    'question': f"So sánh {title.split()[0]} với card khác?",
                    'answer': f"Về hiệu năng: {content[:150]}...",
                    'category': category
                }
            ])
        
        elif category == 'Build':
            qa_pairs.extend([
                {
                    'question': f"Build PC {category.lower()} như thế nào?",
                    'answer': content[:250] + "...",
                    'category': category
                },
                {
                    'question': f"Cấu hình PC {title.split()[-1] if title.split() else 'gaming'}?",
                    'answer': f"Gợi ý cấu hình: {content[:200]}...",
                    'category': category
                }
            ])
        
        else:
            # General Q&A
            qa_pairs.append({
                'question': f"Về {title.lower()}?",
                'answer': content[:200] + "...",
                'category': category
            })
        
        return qa_pairs
    
    def save_qa_pairs(self, qa_pairs: List[Dict], source_id: int) -> int:
        """Save generated Q&A pairs"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        saved_count = 0
        for qa in qa_pairs:
            try:
                cursor.execute('''
                    INSERT INTO generated_qa (question, answer, source_id, category, confidence)
                    VALUES (?, ?, ?, ?, ?)
                ''', (qa['question'], qa['answer'], source_id, qa['category'], 0.8))
                saved_count += 1
            except Exception as e:
                logger.error(f"Error saving Q&A: {e}")
        
        conn.commit()
        conn.close()
        
        return saved_count
    
    def process_unprocessed_content(self):
        """Process unprocessed content to generate Q&A"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, title, content, category 
            FROM auto_knowledge 
            WHERE processed = FALSE
        ''')
        
        unprocessed = cursor.fetchall()
        
        total_qa_generated = 0
        for content_id, title, content, category in unprocessed:
            qa_pairs = self.generate_qa_pairs(content_id, title, content, category)
            saved_qa = self.save_qa_pairs(qa_pairs, content_id)
            total_qa_generated += saved_qa
            
            # Mark as processed
            cursor.execute('''
                UPDATE auto_knowledge 
                SET processed = TRUE 
                WHERE id = ?
            ''', (content_id,))
        
        conn.commit()
        conn.close()
        
        logger.info(f"🧠 Generated {total_qa_generated} Q&A pairs from {len(unprocessed)} articles")
        return total_qa_generated
    
    def get_training_qa_data(self, limit: int = 50) -> List[Tuple[str, str]]:
        """Get Q&A data for training"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT question, answer 
            FROM generated_qa 
            WHERE used_for_training = FALSE 
            ORDER BY confidence DESC, created_at DESC 
            LIMIT ?
        ''', (limit,))
        
        qa_data = cursor.fetchall()
        
        # Mark as used for training
        if qa_data:
            qa_ids = [qa[0] for qa in cursor.execute('''
                SELECT id FROM generated_qa 
                WHERE used_for_training = FALSE 
                ORDER BY confidence DESC, created_at DESC 
                LIMIT ?
            ''', (limit,)).fetchall()]
            
            cursor.executemany('''
                UPDATE generated_qa 
                SET used_for_training = TRUE 
                WHERE id = ?
            ''', [(qa_id,) for qa_id in qa_ids])
        
        conn.commit()
        conn.close()
        
        return qa_data
    
    def run_learning_session(self):
        """Run a complete learning session"""
        logger.info("🚀 Starting auto learning session...")
        
        # Start session
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO learning_sessions (session_start, status)
            VALUES (?, ?)
        ''', (datetime.now(), 'running'))
        session_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        try:
            # 1. Crawl new content
            crawled_content = self.crawl_pc_content(max_articles=10)
            articles_crawled = self.save_crawled_content(crawled_content)
            
            # 2. Process content to generate Q&A
            qa_generated = self.process_unprocessed_content()
            
            # 3. Update session
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE learning_sessions 
                SET session_end = ?, articles_crawled = ?, qa_pairs_generated = ?, status = ?
                WHERE id = ?
            ''', (datetime.now(), articles_crawled, qa_generated, 'completed', session_id))
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Learning session completed: {articles_crawled} articles, {qa_generated} Q&A pairs")
            
            # 4. Trigger training if enough new data
            if qa_generated >= 5:
                self.trigger_ai_training()
            
        except Exception as e:
            logger.error(f"Learning session failed: {e}")
            
            # Update session as failed
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE learning_sessions 
                SET session_end = ?, status = ?
                WHERE id = ?
            ''', (datetime.now(), 'failed', session_id))
            conn.commit()
            conn.close()
    
    def trigger_ai_training(self):
        """Trigger AI training with new Q&A data"""
        try:
            logger.info("🧠 Triggering AI training with new Q&A data...")
            
            # Get new Q&A data
            qa_data = self.get_training_qa_data(limit=20)
            
            if qa_data:
                # Add to conversations table for training
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                for question, answer in qa_data:
                    cursor.execute('''
                        INSERT INTO conversations (user_id, message, response, confidence)
                        VALUES (?, ?, ?, ?)
                    ''', ('auto_learner', question, answer, 0.9))
                
                conn.commit()
                conn.close()
                
                # Trigger training via API
                try:
                    response = requests.post('http://localhost:8000/trigger-training', 
                                           json={'force': True}, timeout=5)
                    if response.status_code == 200:
                        logger.info("✅ AI training triggered successfully")
                    else:
                        logger.warning("⚠️ Failed to trigger AI training via API")
                except:
                    logger.warning("⚠️ AI server not available for training")
                
            else:
                logger.info("No new Q&A data for training")
                
        except Exception as e:
            logger.error(f"Error triggering training: {e}")
    
    def start_scheduled_learning(self):
        """Start scheduled learning sessions"""
        logger.info("📅 Starting scheduled auto learning...")
        
        # Schedule learning every 2 hours
        schedule.every(2).hours.do(self.run_learning_session)
        
        # Schedule immediate learning
        schedule.every().minute.do(self.run_learning_session).tag('immediate')
        
        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(60)
        
        # Run scheduler in background thread
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        
        logger.info("✅ Scheduled learning started")
        
        # Run immediate learning session
        schedule.clear('immediate')
        self.run_learning_session()

def main():
    """Main function for testing"""
    crawler = AutoLearningCrawler()
    
    # Run one learning session
    crawler.run_learning_session()
    
    # Show results
    conn = sqlite3.connect(crawler.db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM auto_knowledge")
    knowledge_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM generated_qa")
    qa_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM learning_sessions")
    session_count = cursor.fetchone()[0]
    
    conn.close()
    
    print(f"\n🎉 AUTO LEARNING RESULTS:")
    print(f"📚 Knowledge Articles: {knowledge_count}")
    print(f"❓ Q&A Pairs: {qa_count}")
    print(f"🧠 Learning Sessions: {session_count}")

if __name__ == "__main__":
    main()
