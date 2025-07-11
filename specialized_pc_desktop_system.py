# specialized_pc_desktop_system.py - SPECIALIZED PC DESKTOP AI TRAINING SYSTEM
import asyncio
import aiohttp
import sqlite3
import json
import re
import time
import random
from datetime import datetime, timedelta
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, asdict
import hashlib
import logging
from urllib.parse import urljoin, urlparse
import csv
from concurrent.futures import ThreadPoolExecutor
import threading
import schedule

import requests
from bs4 import BeautifulSoup
import pandas as pd
from fake_useragent import UserAgent
import price_parser

# Vietnamese NLP 
from underthesea import word_tokenize, pos_tag, ner
from pyvi import ViTokenizer

# ML/AI imports
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# FastAPI
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PCComponent:
    """PC Component với đầy đủ thông tin"""
    name: str
    category: str  # CPU, GPU, RAM, SSD, HDD, Motherboard, PSU, Case, Cooling
    brand: str
    model: str
    price: float
    currency: str
    specs: Dict[str, str]
    availability: str
    store: str
    url: str
    image_url: str
    description: str
    reviews: int
    rating: float
    timestamp: datetime
    hash_id: str

@dataclass
class PCTroubleshooting:
    """PC Troubleshooting với solution"""
    title: str
    problem_description: str
    symptoms: List[str]
    causes: List[str]
    solutions: List[str]
    components: List[str]
    difficulty: str  # easy, medium, hard
    votes: int
    source: str
    url: str
    timestamp: datetime
    hash_id: str

@dataclass
class PCQAPair:
    """Generated Q&A pair cho training"""
    question: str
    answer: str
    category: str
    intent: str  # price, specs, comparison, troubleshooting, build_advice
    confidence: float
    source_type: str  # component, troubleshooting, generated
    vietnamese_context: bool
    timestamp: datetime

class VietnamesePCTerminology:
    """Vietnamese PC terminology processor"""
    
    def __init__(self):
        self.vn_en_mapping = {
            # Components
            'máy tính để bàn': 'desktop pc',
            'máy tính bàn': 'desktop pc', 
            'pc gaming': 'gaming pc',
            'máy gaming': 'gaming pc',
            'vi xử lý': 'cpu',
            'bộ xử lý': 'cpu',
            'con chip': 'cpu',
            'card đồ họa': 'gpu',
            'vga': 'gpu',
            'card màn hình': 'gpu',
            'bộ nhớ': 'ram',
            'ram': 'ram',
            'memory': 'ram',
            'ổ cứng': 'storage',
            'ssd': 'ssd',
            'hdd': 'hdd',
            'bo mạch chủ': 'motherboard',
            'mainboard': 'motherboard',
            'main': 'motherboard',
            'nguồn máy tính': 'psu',
            'nguồn': 'psu',
            'psu': 'psu',
            'case máy tính': 'case',
            'vỏ máy': 'case',
            'thùng máy': 'case',
            'tản nhiệt': 'cooling',
            'quạt': 'fan',
            
            # Specs
            'core': 'core',
            'luồng': 'thread',
            'xung nhịp': 'clock speed',
            'tốc độ': 'speed',
            'dung lượng': 'capacity',
            'băng thông': 'bandwidth',
            'hiệu năng': 'performance',
            'fps': 'fps',
            'điểm': 'score',
            
            # Actions
            'lắp ráp': 'build',
            'build': 'build',
            'nâng cấp': 'upgrade',
            'thay thế': 'replace',
            'sửa chữa': 'repair',
            'bảo hành': 'warranty',
            'tư vấn': 'advice',
            'so sánh': 'compare',
            'đánh giá': 'review',
            
            # Problems
            'bị lỗi': 'error',
            'không hoạt động': 'not working',
            'hỏng': 'broken',
            'chậm': 'slow',
            'nóng': 'hot',
            'ồn': 'noisy',
            'blue screen': 'bsod',
            'màn hình xanh': 'bsod',
            'restart': 'restart',
            'tắt máy': 'shutdown'
        }
        
        self.pc_brands = {
            'intel', 'amd', 'nvidia', 'asus', 'msi', 'gigabyte', 'asrock',
            'corsair', 'gskill', 'kingston', 'samsung', 'wd', 'seagate',
            'thermaltake', 'cooler master', 'noctua', 'be quiet', 'evga'
        }
        
        self.price_patterns = [
            r'(\d{1,3}(?:[,.]?\d{3})*)\s*(?:đ|vnd|vnđ|triệu|tr|nghìn|k)',
            r'(\d{1,3}(?:[,.]?\d{3})*)\s*(?:usd|\$)',
            r'giá\s*:?\s*(\d{1,3}(?:[,.]?\d{3})*)',
            r'price\s*:?\s*(\d{1,3}(?:[,.]?\d{3})*)'
        ]
    
    def normalize_vietnamese_query(self, text: str) -> str:
        """Normalize Vietnamese query to standard terms"""
        text_lower = text.lower()
        
        for vn_term, en_term in self.vn_en_mapping.items():
            text_lower = text_lower.replace(vn_term, en_term)
        
        return text_lower
    
    def extract_pc_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract PC-related entities from Vietnamese text"""
        entities = {
            'components': [],
            'brands': [],
            'specs': [],
            'prices': [],
            'actions': []
        }
        
        text_lower = text.lower()
        
        # Extract components
        for vn_term, en_term in self.vn_en_mapping.items():
            if vn_term in text_lower:
                if en_term in ['cpu', 'gpu', 'ram', 'ssd', 'hdd', 'motherboard', 'psu', 'case']:
                    entities['components'].append(en_term)
        
        # Extract brands
        for brand in self.pc_brands:
            if brand in text_lower:
                entities['brands'].append(brand)
        
        # Extract prices
        for pattern in self.price_patterns:
            matches = re.findall(pattern, text_lower)
            entities['prices'].extend(matches)
        
        return entities
    
    def classify_intent(self, text: str) -> str:
        """Classify intent of Vietnamese PC query"""
        text_lower = text.lower()
        
        intent_patterns = {
            'price_inquiry': ['giá', 'bao nhiêu tiền', 'cost', 'price', 'giá bán'],
            'specs_inquiry': ['thông số', 'specs', 'cấu hình', 'chi tiết', 'đánh giá'],
            'comparison': ['so sánh', 'vs', 'compare', 'khác nhau', 'tốt hơn'],
            'build_advice': ['build', 'lắp ráp', 'tư vấn', 'cấu hình', 'combo'],
            'troubleshooting': ['lỗi', 'hỏng', 'không chạy', 'fix', 'sửa', 'problem'],
            'upgrade_advice': ['nâng cấp', 'upgrade', 'thay thế', 'replace'],
            'buying_advice': ['nên mua', 'recommend', 'tư vấn mua', 'chọn']
        }
        
        for intent, patterns in intent_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                return intent
        
        return 'general_inquiry'

class SpecializedPCCrawler:
    """Specialized crawler cho PC Desktop components và troubleshooting"""
    
    def __init__(self):
        self.ua = UserAgent()
        self.session = None
        self.terminology = VietnamesePCTerminology()
        
        # Vietnamese PC stores
        self.pc_stores = {
            'thegioididong': {
                'base_url': 'https://www.thegioididong.com',
                'pc_categories': [
                    '/may-tinh-de-ban',
                    '/linh-kien-may-tinh'
                ],
                'parser': self.parse_tgdd
            },
            'fpt_shop': {
                'base_url': 'https://fptshop.com.vn',
                'pc_categories': [
                    '/may-tinh-de-ban',
                    '/linh-kien-may-tinh'
                ],
                'parser': self.parse_fpt_shop
            },
            'cellphones': {
                'base_url': 'https://cellphones.com.vn',
                'pc_categories': [
                    '/pc-linh-kien-may-tinh.html'
                ],
                'parser': self.parse_cellphones
            },
            'phong_vu': {
                'base_url': 'https://phongvu.vn',
                'pc_categories': [
                    '/c/pc-gaming',
                    '/c/linh-kien-pc'
                ],
                'parser': self.parse_phong_vu
            },
            'gear_vn': {
                'base_url': 'https://gear.vn',
                'pc_categories': [
                    '/pc-gaming',
                    '/linh-kien'
                ],
                'parser': self.parse_gear_vn
            }
        }
        
        # Vietnamese PC forums for troubleshooting
        self.troubleshooting_sources = {
            'voz': {
                'base_url': 'https://voz.vn',
                'forums': ['/f/may-tinh.17/', '/f/cong-nghe.16/'],
                'parser': self.parse_voz_troubleshooting
            },
            'tinhte': {
                'base_url': 'https://tinhte.vn',
                'forums': ['/forums/may-tinh-pc.54/', '/forums/cong-nghe.12/'],
                'parser': self.parse_tinhte_troubleshooting
            }
        }
        
        # Component categories mapping
        self.component_categories = {
            'cpu': {
                'keywords': ['cpu', 'processor', 'vi xử lý', 'bộ xử lý', 'intel', 'amd'],
                'specs': ['cores', 'threads', 'base_clock', 'boost_clock', 'cache', 'tdp']
            },
            'gpu': {
                'keywords': ['gpu', 'vga', 'card đồ họa', 'graphics', 'nvidia', 'rtx', 'gtx'],
                'specs': ['vram', 'memory_type', 'memory_bus', 'core_clock', 'memory_clock']
            },
            'ram': {
                'keywords': ['ram', 'memory', 'bộ nhớ'],
                'specs': ['capacity', 'speed', 'latency', 'voltage', 'type']
            },
            'storage': {
                'keywords': ['ssd', 'hdd', 'ổ cứng', 'storage'],
                'specs': ['capacity', 'interface', 'read_speed', 'write_speed', 'form_factor']
            },
            'motherboard': {
                'keywords': ['motherboard', 'mainboard', 'bo mạch chủ'],
                'specs': ['socket', 'chipset', 'ram_slots', 'expansion_slots', 'form_factor']
            },
            'psu': {
                'keywords': ['psu', 'power supply', 'nguồn'],
                'specs': ['wattage', 'efficiency', 'modular', 'certification']
            },
            'case': {
                'keywords': ['case', 'vỏ máy', 'thùng máy'],
                'specs': ['form_factor', 'expansion_slots', 'drive_bays', 'cooling_support']
            },
            'cooling': {
                'keywords': ['cooling', 'tản nhiệt', 'fan', 'cooler'],
                'specs': ['fan_size', 'rpm', 'noise_level', 'socket_compatibility']
            }
        }
        
        self.db_path = "specialized_pc_knowledge.db"
        self.init_specialized_database()
    
    def init_specialized_database(self):
        """Initialize specialized PC database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # PC Components table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pc_components (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hash_id TEXT UNIQUE,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                brand TEXT,
                model TEXT,
                price REAL,
                currency TEXT DEFAULT 'VND',
                specs TEXT,  -- JSON
                availability TEXT,
                store TEXT,
                url TEXT,
                image_url TEXT,
                description TEXT,
                reviews INTEGER DEFAULT 0,
                rating REAL DEFAULT 0.0,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Price history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                component_id INTEGER,
                price REAL,
                store TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (component_id) REFERENCES pc_components (id)
            )
        ''')
        
        # PC Troubleshooting table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pc_troubleshooting (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hash_id TEXT UNIQUE,
                title TEXT NOT NULL,
                problem_description TEXT,
                symptoms TEXT,  -- JSON array
                causes TEXT,    -- JSON array
                solutions TEXT, -- JSON array
                components TEXT, -- JSON array
                difficulty TEXT,
                votes INTEGER DEFAULT 0,
                source TEXT,
                url TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Generated Q&A pairs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pc_qa_pairs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                category TEXT,
                intent TEXT,
                confidence REAL DEFAULT 0.0,
                source_type TEXT,
                vietnamese_context BOOLEAN DEFAULT TRUE,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                used_for_training BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Crawling sessions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS crawling_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_type TEXT, -- components, troubleshooting, combined
                source TEXT,
                items_crawled INTEGER DEFAULT 0,
                qa_pairs_generated INTEGER DEFAULT 0,
                session_start DATETIME DEFAULT CURRENT_TIMESTAMP,
                session_end DATETIME,
                status TEXT DEFAULT 'running'
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Specialized PC database initialized")
    
    async def init_session(self):
        """Initialize async session"""
        headers = {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'vi-VN,vi;q=0.8,en-US;q=0.5,en;q=0.3',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(
            headers=headers,
            timeout=timeout,
            connector=aiohttp.TCPConnector(limit=10)
        )
    
    async def close_session(self):
        if self.session:
            await self.session.close()
    
    async def crawl_all_pc_data(self):
        """Crawl tất cả PC data - components + troubleshooting"""
        logger.info("🚀 Starting comprehensive PC data crawling...")
        
        await self.init_session()
        
        try:
            # Start crawling session
            session_id = self.start_crawling_session('combined', 'all_sources')
            
            total_components = 0
            total_troubleshooting = 0
            
            # Crawl PC components from stores
            logger.info("💰 Crawling PC components from Vietnamese stores...")
            for store_name, config in self.pc_stores.items():
                try:
                    logger.info(f"🏪 Crawling {store_name}...")
                    components = await self.crawl_store_components(store_name, config)
                    
                    if components:
                        saved_count = await self.save_components(components)
                        total_components += saved_count
                        logger.info(f"✅ {store_name}: {saved_count} components")
                    
                    await asyncio.sleep(2)  # Rate limiting
                    
                except Exception as e:
                    logger.error(f"❌ Error crawling {store_name}: {e}")
            
            # Crawl troubleshooting from forums
            logger.info("🔧 Crawling troubleshooting from Vietnamese forums...")
            for forum_name, config in self.troubleshooting_sources.items():
                try:
                    logger.info(f"📋 Crawling {forum_name}...")
                    troubleshooting_data = await self.crawl_forum_troubleshooting(forum_name, config)
                    
                    if troubleshooting_data:
                        saved_count = await self.save_troubleshooting(troubleshooting_data)
                        total_troubleshooting += saved_count
                        logger.info(f"✅ {forum_name}: {saved_count} troubleshooting items")
                    
                    await asyncio.sleep(3)  # Longer delay for forums
                    
                except Exception as e:
                    logger.error(f"❌ Error crawling {forum_name}: {e}")
            
            # Generate Q&A pairs
            logger.info("🧠 Generating Q&A pairs...")
            qa_pairs_generated = await self.generate_comprehensive_qa_pairs()
            
            # End crawling session
            self.end_crawling_session(session_id, total_components + total_troubleshooting, qa_pairs_generated)
            
            logger.info(f"✅ Crawling completed: {total_components} components, {total_troubleshooting} troubleshooting, {qa_pairs_generated} Q&A pairs")
            
            return {
                'components': total_components,
                'troubleshooting': total_troubleshooting,
                'qa_pairs': qa_pairs_generated
            }
            
        finally:
            await self.close_session()
    
    async def crawl_store_components(self, store_name: str, config: Dict) -> List[PCComponent]:
        """Crawl components from specific store"""
        components = []
        
        # Mock implementation - tạo sample data cho demonstration
        # Trong thực tế sẽ parse HTML từ websites
        
        sample_components = [
            {
                'name': 'CPU Intel Core i7-13700F',
                'category': 'cpu',
                'brand': 'Intel',
                'model': 'i7-13700F',
                'price': 9500000,
                'specs': {
                    'cores': '16 (8P+8E)',
                    'threads': '24',
                    'base_clock': '2.1 GHz',
                    'boost_clock': '5.2 GHz',
                    'cache': '30MB',
                    'tdp': '65W'
                },
                'description': 'CPU gaming mạnh mẽ với 16 cores, phù hợp cho gaming và content creation',
                'availability': 'in_stock',
                'reviews': 150,
                'rating': 4.7
            },
            {
                'name': 'GPU RTX 4070 Ti SUPER',
                'category': 'gpu', 
                'brand': 'NVIDIA',
                'model': 'RTX 4070 Ti SUPER',
                'price': 20500000,
                'specs': {
                    'vram': '16GB GDDR6X',
                    'memory_bus': '256-bit',
                    'boost_clock': '2610 MHz',
                    'cuda_cores': '8448'
                },
                'description': 'Card đồ họa cao cấp cho gaming 4K và ray tracing',
                'availability': 'in_stock',
                'reviews': 89,
                'rating': 4.8
            },
            {
                'name': 'RAM G.Skill Trident Z5 32GB DDR5-6000',
                'category': 'ram',
                'brand': 'G.Skill',
                'model': 'Trident Z5',
                'price': 4200000,
                'specs': {
                    'capacity': '32GB (2x16GB)',
                    'speed': 'DDR5-6000',
                    'latency': 'CL30',
                    'voltage': '1.35V'
                },
                'description': 'RAM DDR5 hiệu năng cao cho gaming và workstation',
                'availability': 'in_stock',
                'reviews': 67,
                'rating': 4.6
            },
            {
                'name': 'SSD Samsung 980 PRO 1TB',
                'category': 'storage',
                'brand': 'Samsung',
                'model': '980 PRO',
                'price': 2800000,
                'specs': {
                    'capacity': '1TB',
                    'interface': 'PCIe 4.0 x4',
                    'read_speed': '7000 MB/s',
                    'write_speed': '5000 MB/s',
                    'form_factor': 'M.2 2280'
                },
                'description': 'SSD NVMe tốc độ cao cho gaming và professional work',
                'availability': 'in_stock',
                'reviews': 234,
                'rating': 4.9
            }
        ]
        
        for comp_data in sample_components:
            component = PCComponent(
                name=comp_data['name'],
                category=comp_data['category'],
                brand=comp_data['brand'],
                model=comp_data['model'],
                price=comp_data['price'],
                currency='VND',
                specs=comp_data['specs'],
                availability=comp_data['availability'],
                store=store_name,
                url=f"{config['base_url']}/sample-product",
                image_url=f"{config['base_url']}/sample-image.jpg",
                description=comp_data['description'],
                reviews=comp_data['reviews'],
                rating=comp_data['rating'],
                timestamp=datetime.now(),
                hash_id=hashlib.md5(f"{comp_data['name']}{store_name}".encode()).hexdigest()
            )
            components.append(component)
        
        return components
    
    async def crawl_forum_troubleshooting(self, forum_name: str, config: Dict) -> List[PCTroubleshooting]:
        """Crawl troubleshooting from forums"""
        troubleshooting_data = []
        
        # Sample troubleshooting data
        sample_troubleshooting = [
            {
                'title': 'Máy tính bị Blue Screen khi chơi game',
                'problem_description': 'Máy tính bị màn hình xanh chết máy khi chơi game nặng, thỉnh thoảng restart tự động',
                'symptoms': ['Blue Screen of Death (BSOD)', 'Restart tự động', 'Xảy ra khi gaming'],
                'causes': ['Driver GPU lỗi', 'PSU không đủ nguồn', 'RAM lỗi', 'GPU quá nóng'],
                'solutions': [
                    'Update driver GPU mới nhất',
                    'Kiểm tra và thay PSU mạnh hơn',
                    'Test RAM bằng MemTest86',
                    'Vệ sinh tản nhiệt GPU',
                    'Undervolting GPU để giảm nhiệt'
                ],
                'components': ['gpu', 'psu', 'ram', 'cooling'],
                'difficulty': 'medium',
                'votes': 45
            },
            {
                'title': 'CPU quá nóng khi render video',
                'problem_description': 'CPU i7 nóng lên 90°C khi render video, máy lag và chậm',
                'symptoms': ['Nhiệt độ CPU >85°C', 'Performance giảm', 'Fan chạy ầm ầm'],
                'causes': ['Tản nhiệt kém', 'Keo tản nhiệt khô', 'Airflow case không tốt'],
                'solutions': [
                    'Thay keo tản nhiệt thermal paste',
                    'Nâng cấp tản nhiệt CPU',
                    'Thêm fan case cho airflow',
                    'Undervolting CPU',
                    'Giới hạn power limit'
                ],
                'components': ['cpu', 'cooling', 'case'],
                'difficulty': 'easy',
                'votes': 67
            },
            {
                'title': 'RAM không nhận đủ dung lượng',
                'problem_description': 'Cắm 2 thanh RAM 16GB nhưng Windows chỉ nhận 16GB thay vì 32GB',
                'symptoms': ['Windows chỉ nhận 1 thanh RAM', 'BIOS không detect hết RAM'],
                'causes': ['RAM slot bị lỗi', 'RAM không compatible', 'Setting BIOS sai'],
                'solutions': [
                    'Thử đổi slot RAM',
                    'Update BIOS lên version mới',
                    'Enable XMP/DOCP trong BIOS',
                    'Test từng thanh RAM riêng lẻ',
                    'Kiểm tra QVL của motherboard'
                ],
                'components': ['ram', 'motherboard'],
                'difficulty': 'medium',
                'votes': 33
            }
        ]
        
        for troubleshoot_data in sample_troubleshooting:
            troubleshooting = PCTroubleshooting(
                title=troubleshoot_data['title'],
                problem_description=troubleshoot_data['problem_description'],
                symptoms=troubleshoot_data['symptoms'],
                causes=troubleshoot_data['causes'],
                solutions=troubleshoot_data['solutions'],
                components=troubleshoot_data['components'],
                difficulty=troubleshoot_data['difficulty'],
                votes=troubleshoot_data['votes'],
                source=forum_name,
                url=f"{config['base_url']}/sample-thread",
                timestamp=datetime.now(),
                hash_id=hashlib.md5(f"{troubleshoot_data['title']}{forum_name}".encode()).hexdigest()
            )
            troubleshooting_data.append(troubleshooting)
        
        return troubleshooting_data
    
    async def generate_comprehensive_qa_pairs(self) -> int:
        """Generate comprehensive Q&A pairs từ crawled data"""
        logger.info("🤖 Generating comprehensive Q&A pairs...")
        
        qa_pairs = []
        
        # Generate từ components
        qa_pairs.extend(await self.generate_component_qa_pairs())
        
        # Generate từ troubleshooting
        qa_pairs.extend(await self.generate_troubleshooting_qa_pairs())
        
        # Generate comparison questions
        qa_pairs.extend(await self.generate_comparison_qa_pairs())
        
        # Generate build advice questions
        qa_pairs.extend(await self.generate_build_advice_qa_pairs())
        
        # Save all Q&A pairs
        saved_count = await self.save_qa_pairs(qa_pairs)
        
        logger.info(f"✅ Generated and saved {saved_count} Q&A pairs")
        return saved_count
    
    async def generate_component_qa_pairs(self) -> List[PCQAPair]:
        """Generate Q&A pairs từ component data"""
        qa_pairs = []
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT name, category, brand, model, price, specs, description 
            FROM pc_components 
            ORDER BY timestamp DESC 
            LIMIT 100
        ''')
        
        components = cursor.fetchall()
        conn.close()
        
        for comp in components:
            name, category, brand, model, price, specs_json, description = comp
            
            try:
                specs = json.loads(specs_json) if specs_json else {}
            except:
                specs = {}
            
            # Price questions
            qa_pairs.extend([
                PCQAPair(
                    question=f"Giá {name} bao nhiêu?",
                    answer=f"{name} có giá {price:,.0f} VNĐ. {description}",
                    category=category,
                    intent='price_inquiry',
                    confidence=0.9,
                    source_type='component',
                    vietnamese_context=True,
                    timestamp=datetime.now()
                ),
                PCQAPair(
                    question=f"{brand} {model} giá bao nhiêu tiền?",
                    answer=f"{brand} {model} hiện tại có giá {price:,.0f} VNĐ. Đây là {description}",
                    category=category,
                    intent='price_inquiry',
                    confidence=0.85,
                    source_type='component',
                    vietnamese_context=True,
                    timestamp=datetime.now()
                )
            ])
            
            # Specs questions
            if specs:
                specs_text = ', '.join([f"{k}: {v}" for k, v in specs.items()])
                qa_pairs.extend([
                    PCQAPair(
                        question=f"Thông số {name} như thế nào?",
                        answer=f"Thông số kỹ thuật {name}: {specs_text}. {description}",
                        category=category,
                        intent='specs_inquiry',
                        confidence=0.9,
                        source_type='component',
                        vietnamese_context=True,
                        timestamp=datetime.now()
                    ),
                    PCQAPair(
                        question=f"{name} có tốt không?",
                        answer=f"{name} là sản phẩm tốt với {specs_text}. {description} Giá: {price:,.0f} VNĐ",
                        category=category,
                        intent='buying_advice',
                        confidence=0.8,
                        source_type='component',
                        vietnamese_context=True,
                        timestamp=datetime.now()
                    )
                ])
        
        return qa_pairs
    
    async def generate_troubleshooting_qa_pairs(self) -> List[PCQAPair]:
        """Generate Q&A pairs từ troubleshooting data"""
        qa_pairs = []
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT title, problem_description, symptoms, solutions, components 
            FROM pc_troubleshooting 
            ORDER BY votes DESC 
            LIMIT 50
        ''')
        
        troubleshooting_data = cursor.fetchall()
        conn.close()
        
        for data in troubleshooting_data:
            title, problem_desc, symptoms_json, solutions_json, components_json = data
            
            try:
                symptoms = json.loads(symptoms_json) if symptoms_json else []
                solutions = json.loads(solutions_json) if solutions_json else []
                components = json.loads(components_json) if components_json else []
            except:
                symptoms, solutions, components = [], [], []
            
            if solutions:
                solutions_text = '\n'.join([f"- {sol}" for sol in solutions])
                
                qa_pairs.extend([
                    PCQAPair(
                        question=title,
                        answer=f"Vấn đề: {problem_desc}\n\nCách khắc phục:\n{solutions_text}",
                        category='troubleshooting',
                        intent='troubleshooting',
                        confidence=0.9,
                        source_type='troubleshooting',
                        vietnamese_context=True,
                        timestamp=datetime.now()
                    ),
                    PCQAPair(
                        question=f"Cách fix {title.lower()}?",
                        answer=f"Để khắc phục vấn đề này:\n{solutions_text}",
                        category='troubleshooting',
                        intent='troubleshooting',
                        confidence=0.85,
                        source_type='troubleshooting',
                        vietnamese_context=True,
                        timestamp=datetime.now()
                    )
                ])
        
        return qa_pairs
    
    async def generate_comparison_qa_pairs(self) -> List[PCQAPair]:
        """Generate comparison Q&A pairs"""
        qa_pairs = []
        
        # Sample comparison data
        comparisons = [
            {
                'question': 'Intel i7 vs AMD Ryzen 7 nên chọn cái nào?',
                'answer': 'Intel i7: Tốt cho gaming, single-core mạnh, giá cao hơn. AMD Ryzen 7: Tốt cho đa nhiệm, multi-core mạnh, giá rẻ hơn. Chọn Intel nếu chủ yếu gaming, chọn AMD nếu cần render/stream.',
                'category': 'cpu'
            },
            {
                'question': 'RTX 4070 vs RTX 4060 Ti khác nhau gì?',
                'answer': 'RTX 4070: 12GB VRAM, mạnh hơn 20-25%, giá cao hơn. RTX 4060 Ti: 8GB/16GB VRAM, giá rẻ hơn, đủ cho 1440p. Chọn 4070 cho 4K gaming, 4060 Ti cho 1440p.',
                'category': 'gpu'
            },
            {
                'question': 'DDR4 vs DDR5 cho gaming có khác biệt không?',
                'answer': 'DDR5: Nhanh hơn, hỗ trợ tương lai, giá đắt. DDR4: Đủ cho gaming hiện tại, giá rẻ, stable. Gaming khác biệt nhỏ (5-10%), DDR4 vẫn đủ dùng.',
                'category': 'ram'
            }
        ]
        
        for comp in comparisons:
            qa_pairs.append(PCQAPair(
                question=comp['question'],
                answer=comp['answer'],
                category=comp['category'],
                intent='comparison',
                confidence=0.9,
                source_type='generated',
                vietnamese_context=True,
                timestamp=datetime.now()
            ))
        
        return qa_pairs
    
    async def generate_build_advice_qa_pairs(self) -> List[PCQAPair]:
        """Generate build advice Q&A pairs"""
        qa_pairs = []
        
        build_guides = [
            {
                'question': 'Build PC gaming 30 triệu cấu hình nào?',
                'answer': 'PC Gaming 30tr: CPU i5-13400F (5tr), GPU RTX 4060 Ti (12tr), RAM 16GB DDR4 (1.5tr), SSD 500GB (1tr), Main B660 (2.5tr), PSU 650W (1.5tr), Case (1.5tr), Cooling (1tr). Chạy mượt 1440p.',
                'category': 'build_guide'
            },
            {
                'question': 'Build PC render video 50 triệu như thế nào?',
                'answer': 'PC Render 50tr: CPU i7-13700F (9tr), GPU RTX 4070 (15tr), RAM 32GB DDR4 (3tr), SSD 1TB (2tr), Main B660 (3tr), PSU 750W (2tr), Case thoáng (2tr). Tối ưu cho Adobe Premiere, DaVinci.',
                'category': 'build_guide'
            },
            {
                'question': 'Nâng cấp từ GTX 1060 lên card nào?',
                'answer': 'Upgrade từ GTX 1060: RTX 4060 (8tr) cho 1080p, RTX 4060 Ti (12tr) cho 1440p, RTX 4070 (15tr) cho 4K. Kiểm tra PSU đủ mạnh (tối thiểu 500W), CPU không bottleneck.',
                'category': 'upgrade_advice'
            }
        ]
        
        for guide in build_guides:
            qa_pairs.append(PCQAPair(
                question=guide['question'],
                answer=guide['answer'],
                category=guide['category'],
                intent='build_advice',
                confidence=0.95,
                source_type='generated',
                vietnamese_context=True,
                timestamp=datetime.now()
            ))
        
        return qa_pairs
    
    async def save_components(self, components: List[PCComponent]) -> int:
        """Save components to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        saved_count = 0
        for component in components:
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO pc_components 
                    (hash_id, name, category, brand, model, price, currency, specs, 
                     availability, store, url, image_url, description, reviews, rating, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    component.hash_id, component.name, component.category, component.brand,
                    component.model, component.price, component.currency, json.dumps(component.specs),
                    component.availability, component.store, component.url, component.image_url,
                    component.description, component.reviews, component.rating, datetime.now()
                ))
                saved_count += 1
            except Exception as e:
                logger.error(f"Error saving component: {e}")
        
        conn.commit()
        conn.close()
        return saved_count
    
    async def save_troubleshooting(self, troubleshooting_data: List[PCTroubleshooting]) -> int:
        """Save troubleshooting data to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        saved_count = 0
        for data in troubleshooting_data:
            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO pc_troubleshooting 
                    (hash_id, title, problem_description, symptoms, causes, solutions, 
                     components, difficulty, votes, source, url)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data.hash_id, data.title, data.problem_description,
                    json.dumps(data.symptoms), json.dumps(data.causes), json.dumps(data.solutions),
                    json.dumps(data.components), data.difficulty, data.votes, data.source, data.url
                ))
                saved_count += 1
            except Exception as e:
                logger.error(f"Error saving troubleshooting: {e}")
        
        conn.commit()
        conn.close()
        return saved_count
    
    async def save_qa_pairs(self, qa_pairs: List[PCQAPair]) -> int:
        """Save Q&A pairs to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        saved_count = 0
        for qa in qa_pairs:
            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO pc_qa_pairs 
                    (question, answer, category, intent, confidence, source_type, vietnamese_context)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (qa.question, qa.answer, qa.category, qa.intent, qa.confidence, qa.source_type, qa.vietnamese_context))
                saved_count += 1
            except Exception as e:
                logger.error(f"Error saving Q&A pair: {e}")
        
        conn.commit()
        conn.close()
        return saved_count
    
    def start_crawling_session(self, session_type: str, source: str) -> int:
        """Start crawling session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO crawling_sessions (session_type, source, session_start, status)
            VALUES (?, ?, ?, ?)
        ''', (session_type, source, datetime.now(), 'running'))
        
        session_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return session_id
    
    def end_crawling_session(self, session_id: int, items_crawled: int, qa_pairs_generated: int):
        """End crawling session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE crawling_sessions 
            SET session_end = ?, items_crawled = ?, qa_pairs_generated = ?, status = ?
            WHERE id = ?
        ''', (datetime.now(), items_crawled, qa_pairs_generated, 'completed', session_id))
        
        conn.commit()
        conn.close()
    
    # Parser methods (simplified for demo)
    async def parse_tgdd(self, url: str) -> List[PCComponent]:
        """Parse thegioididong.com"""
        return []
    
    async def parse_fpt_shop(self, url: str) -> List[PCComponent]:
        """Parse fptshop.com.vn"""
        return []
    
    async def parse_cellphones(self, url: str) -> List[PCComponent]:
        """Parse cellphones.com.vn"""
        return []
    
    async def parse_phong_vu(self, url: str) -> List[PCComponent]:
        """Parse phongvu.vn"""
        return []
    
    async def parse_gear_vn(self, url: str) -> List[PCComponent]:
        """Parse gear.vn"""
        return []
    
    async def parse_voz_troubleshooting(self, url: str) -> List[PCTroubleshooting]:
        """Parse voz.vn troubleshooting"""
        return []
    
    async def parse_tinhte_troubleshooting(self, url: str) -> List[PCTroubleshooting]:
        """Parse tinhte.vn troubleshooting"""
        return []

class SpecializedPCTrainingSystem:
    """Specialized training system cho PC Desktop AI"""
    
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.crawler = SpecializedPCCrawler()
        self.terminology = VietnamesePCTerminology()
        
        self.load_model()
        logger.info(f"Specialized PC Training System initialized on {self.device}")
    
    def load_model(self):
        """Load specialized model"""
        logger.info(f"Loading {self.model_name} for PC domain...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        
        logger.info("✅ Model loaded successfully!")
    
    async def train_with_specialized_data(self):
        """Train với specialized PC data"""
        logger.info("🧠 Starting specialized PC training...")
        
        try:
            # Get specialized training data
            training_data = self.get_specialized_training_data()
            
            if len(training_data) < 100:
                logger.warning(f"Not enough specialized data: {len(training_data)}")
                return False
            
            # Create dataset
            texts = []
            for qa in training_data:
                question, answer, category, intent = qa
                # Format for Vietnamese PC context
                text = f"[{category.upper()}] Human: {question}\nAssistant: {answer}{self.tokenizer.eos_token}"
                texts.append(text)
            
            # Tokenize
            encodings = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=768,  # Longer for detailed PC responses
                return_tensors="pt"
            )
            
            dataset = Dataset.from_dict({
                "input_ids": encodings["input_ids"],
                "attention_mask": encodings["attention_mask"]
            })
            
            # Training arguments optimized for PC domain
            training_args = TrainingArguments(
                output_dir="./specialized_pc_model",
                overwrite_output_dir=True,
                num_train_epochs=3,
                per_device_train_batch_size=3,  # Adjusted for GTX 1070
                gradient_accumulation_steps=4,
                learning_rate=3e-5,  # Lower LR for domain adaptation
                warmup_steps=100,
                logging_steps=20,
                save_steps=200,
                save_total_limit=2,
                fp16=True,
                dataloader_pin_memory=False,
                evaluation_strategy="no",
                report_to=None
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
            )
            
            # Trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=dataset,
            )
            
            # Train
            logger.info(f"Training with {len(training_data)} specialized PC Q&A pairs...")
            trainer.train()
            
            # Save specialized model
            trainer.save_model("./specialized_pc_model")
            self.tokenizer.save_pretrained("./specialized_pc_model")
            
            logger.info("✅ Specialized PC training completed!")
            return True
            
        except Exception as e:
            logger.error(f"Specialized training failed: {e}")
            return False
    
    def get_specialized_training_data(self) -> List[Tuple[str, str, str, str]]:
        """Get specialized training data from PC database"""
        conn = sqlite3.connect(self.crawler.db_path)
        cursor = conn.cursor()
        
        # Get high-quality Q&A pairs
        cursor.execute('''
            SELECT question, answer, category, intent 
            FROM pc_qa_pairs 
            WHERE confidence > 0.7 AND vietnamese_context = 1
            ORDER BY confidence DESC, timestamp DESC
            LIMIT 1000
        ''')
        
        training_data = cursor.fetchall()
        conn.close()
        
        logger.info(f"Retrieved {len(training_data)} specialized training samples")
        return training_data
    
    def generate_specialized_response(self, user_query: str) -> Dict[str, any]:
        """Generate specialized response for PC queries"""
        try:
            # Preprocess Vietnamese query
            normalized_query = self.terminology.normalize_vietnamese_query(user_query)
            entities = self.terminology.extract_pc_entities(user_query)
            intent = self.terminology.classify_intent(user_query)
            
            # Find similar Q&A in database
            similar_qa = self.find_similar_pc_qa(user_query)
            
            # Generate response
            if similar_qa and similar_qa['similarity'] > 0.7:
                # Use high-confidence database answer
                response = similar_qa['answer']
                confidence = similar_qa['similarity']
                source = 'database'
            else:
                # Generate with model
                response, confidence = self.generate_with_model(user_query, intent)
                source = 'model'
            
            return {
                'response': response,
                'confidence': confidence,
                'source': source,
                'intent': intent,
                'entities': entities,
                'similar_qa': similar_qa
            }
            
        except Exception as e:
            logger.error(f"Error generating specialized response: {e}")
            return {
                'response': 'Xin lỗi, tôi đang gặp sự cố khi xử lý câu hỏi về PC. Bạn có thể hỏi lại không?',
                'confidence': 0.1,
                'source': 'error',
                'intent': 'unknown',
                'entities': {},
                'similar_qa': None
            }
    
    def find_similar_pc_qa(self, query: str, threshold: float = 0.5) -> Optional[Dict]:
        """Find similar Q&A in PC database"""
        conn = sqlite3.connect(self.crawler.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT question, answer, category, intent, confidence 
            FROM pc_qa_pairs 
            WHERE vietnamese_context = 1
            ORDER BY confidence DESC
            LIMIT 100
        ''')
        
        qa_data = cursor.fetchall()
        conn.close()
        
        if not qa_data:
            return None
        
        # Simple similarity using TF-IDF
        questions = [item[0] for item in qa_data]
        questions.append(query)
        
        try:
            vectorizer = TfidfVectorizer(max_features=1000)
            tfidf_matrix = vectorizer.fit_transform(questions)
            similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1]).flatten()
            
            best_idx = np.argmax(similarities)
            best_similarity = similarities[best_idx]
            
            if best_similarity > threshold:
                best_qa = qa_data[best_idx]
                return {
                    'question': best_qa[0],
                    'answer': best_qa[1],
                    'category': best_qa[2],
                    'intent': best_qa[3],
                    'confidence': best_qa[4],
                    'similarity': float(best_similarity)
                }
        except Exception as e:
            logger.error(f"Error finding similar Q&A: {e}")
        
        return None
    
    def generate_with_model(self, query: str, intent: str) -> Tuple[str, float]:
        """Generate response với specialized model"""
        try:
            # Add intent context to prompt
            prompt = f"[{intent.upper()}] Human: {query}\nAssistant:"
            
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 200,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2
                )
            
            response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            
            # Calculate confidence based on response quality
            confidence = min(0.9, len(response.strip()) / 100.0)
            
            return response.strip() or "Tôi cần thêm thông tin để trả lời câu hỏi về PC này.", confidence
            
        except Exception as e:
            logger.error(f"Error generating with model: {e}")
            return "Xin lỗi, tôi đang gặp sự cố khi generate response.", 0.1

# FastAPI Server with Specialized PC System
app = FastAPI(title="Specialized PC Desktop AI Server", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
crawler = None
training_system = None
is_crawling = False
is_training = False

class PCMessage(BaseModel):
    user_id: str
    message: str
    timestamp: int

class PCResponse(BaseModel):
    response: str
    confidence: float
    source: str
    intent: str
    entities: Dict
    crawling_status: str
    training_status: str

@app.on_event("startup")
async def startup_event():
    """Startup event"""
    global crawler, training_system
    
    logger.info("🚀 Starting Specialized PC Desktop AI Server...")
    
    crawler = SpecializedPCCrawler()
    training_system = SpecializedPCTrainingSystem()
    
    # Start background crawling
    asyncio.create_task(background_crawling())
    
    logger.info("✅ Specialized PC Server ready!")

async def background_crawling():
    """Background crawling task"""
    global is_crawling
    
    # Initial crawling
    await asyncio.sleep(5)  # Wait for startup
    
    while True:
        try:
            if not is_crawling:
                is_crawling = True
                logger.info("🕷️ Starting background PC data crawling...")
                
                result = await crawler.crawl_all_pc_data()
                logger.info(f"✅ Background crawling completed: {result}")
                
                # Trigger training if enough new data
                if result['qa_pairs'] > 50:
                    asyncio.create_task(background_training())
                
                is_crawling = False
            
            # Wait 2 hours before next crawling
            await asyncio.sleep(7200)
            
        except Exception as e:
            logger.error(f"Background crawling error: {e}")
            is_crawling = False
            await asyncio.sleep(3600)  # Wait 1 hour on error

async def background_training():
    """Background training task"""
    global is_training, training_system
    
    if is_training:
        return
    
    try:
        is_training = True
        logger.info("🧠 Starting background PC training...")
        
        success = await training_system.train_with_specialized_data()
        
        if success:
            logger.info("✅ Background training completed successfully!")
        else:
            logger.warning("⚠️ Background training failed or insufficient data")
        
    except Exception as e:
        logger.error(f"Background training error: {e}")
    finally:
        is_training = False

@app.post("/process-pc-message", response_model=PCResponse)
async def process_pc_message(message_data: PCMessage):
    """Process PC-related message với specialized AI"""
    global training_system
    
    try:
        user_id = message_data.user_id
        message = message_data.message
        
        logger.info(f"Processing PC message from {user_id}: {message}")
        
        # Generate specialized response
        result = training_system.generate_specialized_response(message)
        
        # Save conversation (implement if needed)
        # save_pc_conversation(user_id, message, result['response'])
        
        return PCResponse(
            response=result['response'],
            confidence=result['confidence'],
            source=result['source'],
            intent=result['intent'],
            entities=result['entities'],
            crawling_status="active" if is_crawling else "idle",
            training_status="active" if is_training else "idle"
        )
        
    except Exception as e:
        logger.error(f"Error processing PC message: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/trigger-pc-crawling")
async def trigger_pc_crawling(background_tasks: BackgroundTasks):
    """Manually trigger PC data crawling"""
    global is_crawling
    
    if is_crawling:
        return {"status": "Crawling already in progress"}
    
    background_tasks.add_task(manual_crawling)
    return {"status": "PC crawling started"}

async def manual_crawling():
    """Manual crawling task"""
    global is_crawling, crawler
    
    try:
        is_crawling = True
        result = await crawler.crawl_all_pc_data()
        logger.info(f"Manual crawling completed: {result}")
    except Exception as e:
        logger.error(f"Manual crawling error: {e}")
    finally:
        is_crawling = False

@app.post("/trigger-pc-training")
async def trigger_pc_training(background_tasks: BackgroundTasks):
    """Manually trigger PC training"""
    global is_training
    
    if is_training:
        return {"status": "Training already in progress"}
    
    background_tasks.add_task(manual_training)
    return {"status": "PC training started"}

async def manual_training():
    """Manual training task"""
    global is_training, training_system
    
    try:
        is_training = True
        success = await training_system.train_with_specialized_data()
        result = "completed" if success else "failed"
        logger.info(f"Manual training {result}")
    except Exception as e:
        logger.error(f"Manual training error: {e}")
    finally:
        is_training = False

@app.get("/pc-stats")
async def get_pc_stats():
    """Get PC system statistics"""
    try:
        conn = sqlite3.connect(crawler.db_path)
        cursor = conn.cursor()
        
        # Components stats
        cursor.execute("SELECT COUNT(*) FROM pc_components")
        total_components = cursor.fetchone()[0]
        
        cursor.execute("SELECT category, COUNT(*) FROM pc_components GROUP BY category")
        components_by_category = dict(cursor.fetchall())
        
        # Troubleshooting stats
        cursor.execute("SELECT COUNT(*) FROM pc_troubleshooting")
        total_troubleshooting = cursor.fetchone()[0]
        
        # Q&A stats
        cursor.execute("SELECT COUNT(*) FROM pc_qa_pairs")
        total_qa_pairs = cursor.fetchone()[0]
        
        cursor.execute("SELECT intent, COUNT(*) FROM pc_qa_pairs GROUP BY intent")
        qa_by_intent = dict(cursor.fetchall())
        
        # Crawling sessions
        cursor.execute("SELECT COUNT(*) FROM crawling_sessions WHERE status = 'completed'")
        completed_sessions = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_components": total_components,
            "components_by_category": components_by_category,
            "total_troubleshooting": total_troubleshooting,
            "total_qa_pairs": total_qa_pairs,
            "qa_by_intent": qa_by_intent,
            "completed_crawling_sessions": completed_sessions,
            "system_status": {
                "crawling_active": is_crawling,
                "training_active": is_training,
                "device": str(training_system.device)
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting PC stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "Specialized PC Desktop AI Server",
        "version": "3.0.0",
        "features": [
            "Specialized PC Desktop components crawling",
            "Vietnamese PC terminology processing", 
            "Real-time price tracking",
            "PC troubleshooting database",
            "Specialized AI training for PC domain",
            "Vietnamese context understanding"
        ],
        "status": "ready"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)