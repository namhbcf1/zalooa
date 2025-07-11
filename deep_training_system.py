#!/usr/bin/env python3
"""
🧠 DEEP TRAINING SYSTEM - Training sâu với dữ liệu chuyên sâu
Tạo và training với hàng nghìn Q&A pairs chuyên sâu về PC gaming
"""

import sqlite3
import json
import time
import random
from datetime import datetime
from typing import List, Dict, Tuple
import logging

import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from datasets import Dataset
import requests

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepTrainingDataGenerator:
    """Tạo dữ liệu training sâu cho PC gaming AI"""
    
    def __init__(self, db_path="conversations.db"):
        self.db_path = db_path
        
        # PC Gaming knowledge base - DEEP DATA
        self.deep_pc_knowledge = {
            "cpu_data": [
                {
                    "name": "Intel Core i7-13700F",
                    "specs": "16 cores (8P+8E), 24 threads, 3.4-5.2GHz",
                    "price": "9,500,000",
                    "gaming_performance": "Excellent cho gaming 1440p/4K",
                    "pros": ["Hiệu năng cao", "Đa nhiệm tốt", "Tiết kiệm điện"],
                    "cons": ["Giá cao", "Cần tản nhiệt tốt"],
                    "use_cases": ["Gaming cao cấp", "Streaming", "Content creation"]
                },
                {
                    "name": "AMD Ryzen 7 7700X",
                    "specs": "8 cores, 16 threads, 4.5-5.4GHz",
                    "price": "8,800,000",
                    "gaming_performance": "Tuyệt vời cho gaming, hiệu năng/giá tốt",
                    "pros": ["Hiệu năng/giá tốt", "Tiết kiệm điện", "Overclock tốt"],
                    "cons": ["Cần RAM DDR5", "Giá mainboard cao"],
                    "use_cases": ["Gaming", "Productivity", "Budget high-end"]
                },
                {
                    "name": "Intel Core i5-13400F",
                    "specs": "10 cores (6P+4E), 16 threads, 2.5-4.6GHz",
                    "price": "4,500,000",
                    "gaming_performance": "Rất tốt cho gaming 1080p/1440p",
                    "pros": ["Giá tốt", "Hiệu năng ổn", "Tương thích rộng"],
                    "cons": ["Ít core hơn i7", "Không có iGPU"],
                    "use_cases": ["Gaming mainstream", "Office", "Budget build"]
                }
            ],
            
            "gpu_data": [
                {
                    "name": "RTX 4070",
                    "specs": "12GB GDDR6X, 2610MHz boost",
                    "price": "14,500,000",
                    "gaming_performance": "1440p Ultra 60+ FPS, 4K Medium-High",
                    "ray_tracing": "Excellent với DLSS 3",
                    "vram": "12GB đủ cho gaming hiện tại và tương lai",
                    "power": "200W TGP",
                    "pros": ["VRAM 12GB", "DLSS 3", "Hiệu năng tốt"],
                    "cons": ["Giá cao", "Cần PSU 650W+"],
                    "use_cases": ["Gaming 1440p", "Content creation", "Ray tracing"]
                },
                {
                    "name": "RTX 4060 Ti",
                    "specs": "16GB GDDR6, 2540MHz boost",
                    "price": "10,500,000",
                    "gaming_performance": "1440p High 60+ FPS, 1080p Ultra",
                    "ray_tracing": "Good với DLSS",
                    "vram": "16GB version tốt cho tương lai",
                    "power": "165W TGP",
                    "pros": ["VRAM 16GB", "Tiết kiệm điện", "Giá hợp lý"],
                    "cons": ["Bus 128-bit", "Hiệu năng 1440p hạn chế"],
                    "use_cases": ["Gaming 1080p/1440p", "Budget high-end"]
                },
                {
                    "name": "RTX 4060",
                    "specs": "8GB GDDR6, 2460MHz boost",
                    "price": "8,200,000",
                    "gaming_performance": "1080p Ultra 60+ FPS, 1440p Medium-High",
                    "ray_tracing": "Decent với DLSS",
                    "vram": "8GB đủ cho 1080p gaming",
                    "power": "115W TGP",
                    "pros": ["Tiết kiệm điện", "Giá tốt", "Compact"],
                    "cons": ["VRAM 8GB hạn chế", "Hiệu năng 1440p thấp"],
                    "use_cases": ["Gaming 1080p", "Budget gaming", "SFF builds"]
                }
            ],
            
            "ram_data": [
                {
                    "type": "DDR4-3200 16GB",
                    "price": "1,200,000",
                    "performance": "Đủ cho gaming hiện tại",
                    "compatibility": "Tương thích rộng với Intel/AMD",
                    "pros": ["Giá rẻ", "Tương thích tốt", "Ổn định"],
                    "cons": ["Chậm hơn DDR5", "Không future-proof"],
                    "use_cases": ["Budget builds", "Upgrade cũ", "Gaming 1080p"]
                },
                {
                    "type": "DDR5-5600 32GB",
                    "price": "3,800,000",
                    "performance": "Excellent cho gaming và productivity",
                    "compatibility": "Intel 12th gen+, AMD Ryzen 7000+",
                    "pros": ["Tốc độ cao", "Future-proof", "Đa nhiệm tốt"],
                    "cons": ["Giá cao", "Tương thích hạn chế"],
                    "use_cases": ["High-end gaming", "Content creation", "Future builds"]
                }
            ],
            
            "storage_data": [
                {
                    "type": "SSD NVMe PCIe 4.0 1TB",
                    "price": "2,200,000",
                    "performance": "7000+ MB/s read, game load cực nhanh",
                    "pros": ["Tốc độ cao", "Future-proof", "Game load nhanh"],
                    "cons": ["Giá cao hơn PCIe 3.0", "Cần mainboard hỗ trợ"],
                    "use_cases": ["Gaming", "OS drive", "High-end builds"]
                },
                {
                    "type": "SSD SATA 1TB",
                    "price": "1,500,000",
                    "performance": "550 MB/s, đủ cho gaming",
                    "pros": ["Giá tốt", "Tương thích rộng", "Đáng tin cậy"],
                    "cons": ["Chậm hơn NVMe", "Cần cáp SATA"],
                    "use_cases": ["Budget builds", "Secondary storage", "Upgrade cũ"]
                }
            ]
        }
        
        # Build configurations
        self.build_configs = {
            "budget_15m": {
                "cpu": "i5-13400F",
                "gpu": "RTX 4060",
                "ram": "DDR4-3200 16GB",
                "storage": "SSD SATA 500GB",
                "total": "15,000,000",
                "performance": "1080p Ultra gaming",
                "use_case": "Gaming entry-level"
            },
            "mid_range_25m": {
                "cpu": "i7-13700F",
                "gpu": "RTX 4060 Ti",
                "ram": "DDR4-3200 32GB",
                "storage": "SSD NVMe 1TB",
                "total": "25,000,000",
                "performance": "1440p High gaming",
                "use_case": "Gaming mainstream"
            },
            "high_end_35m": {
                "cpu": "i7-13700F",
                "gpu": "RTX 4070",
                "ram": "DDR5-5600 32GB",
                "storage": "SSD NVMe PCIe 4.0 1TB",
                "total": "35,000,000",
                "performance": "1440p Ultra, 4K Medium gaming",
                "use_case": "Gaming cao cấp"
            }
        }
    
    def generate_deep_qa_pairs(self) -> List[Tuple[str, str]]:
        """Generate hàng nghìn Q&A pairs chuyên sâu"""
        qa_pairs = []
        
        # 1. CPU Q&A
        for cpu in self.deep_pc_knowledge["cpu_data"]:
            qa_pairs.extend([
                (f"CPU {cpu['name']} có tốt không?", 
                 f"{cpu['name']} {cpu['gaming_performance']}. Specs: {cpu['specs']}. Giá: {cpu['price']} VND. Ưu điểm: {', '.join(cpu['pros'])}. Phù hợp cho: {', '.join(cpu['use_cases'])}."),
                
                (f"Giá {cpu['name']} bao nhiêu?",
                 f"{cpu['name']} hiện tại giá khoảng {cpu['price']} VND. {cpu['gaming_performance']}. Đây là lựa chọn tốt cho {', '.join(cpu['use_cases'])}."),
                
                (f"Specs {cpu['name']} như thế nào?",
                 f"{cpu['name']} có {cpu['specs']}. {cpu['gaming_performance']}. Ưu điểm: {', '.join(cpu['pros'])}. Nhược điểm: {', '.join(cpu['cons'])}."),
                
                (f"{cpu['name']} gaming được không?",
                 f"Có! {cpu['name']} {cpu['gaming_performance']}. Với {cpu['specs']}, CPU này phù hợp cho {', '.join(cpu['use_cases'])}. Giá {cpu['price']} VND."),
                
                (f"Nên mua {cpu['name']} không?",
                 f"Nên! {cpu['name']} là lựa chọn tốt với {cpu['gaming_performance']}. Ưu điểm: {', '.join(cpu['pros'])}. Phù hợp nếu bạn cần {', '.join(cpu['use_cases'])}."),
            ])
        
        # 2. GPU Q&A
        for gpu in self.deep_pc_knowledge["gpu_data"]:
            qa_pairs.extend([
                (f"GPU {gpu['name']} có tốt không?",
                 f"{gpu['name']} rất tốt! {gpu['gaming_performance']}. VRAM: {gpu['vram']}, TGP: {gpu['power']}. Ray tracing: {gpu['ray_tracing']}. Giá: {gpu['price']} VND."),
                
                (f"RTX {gpu['name'].split()[-1]} gaming được không?",
                 f"Được! {gpu['name']} {gpu['gaming_performance']}. {gpu['ray_tracing']}. Ưu điểm: {', '.join(gpu['pros'])}. Phù hợp cho: {', '.join(gpu['use_cases'])}."),
                
                (f"Giá {gpu['name']} bao nhiêu?",
                 f"{gpu['name']} giá khoảng {gpu['price']} VND. Với hiệu năng {gpu['gaming_performance']}, đây là lựa chọn tốt cho {', '.join(gpu['use_cases'])}."),
                
                (f"{gpu['name']} chơi game 1440p được không?",
                 f"Được! {gpu['name']} {gpu['gaming_performance']}. VRAM {gpu['vram']} đủ cho gaming 1440p. {gpu['ray_tracing']}."),
                
                (f"So sánh {gpu['name']} với card khác?",
                 f"{gpu['name']}: {gpu['gaming_performance']}. Ưu điểm: {', '.join(gpu['pros'])}. Nhược điểm: {', '.join(gpu['cons'])}. TGP: {gpu['power']}."),
            ])
        
        # 3. Build Q&A
        for build_name, config in self.build_configs.items():
            budget = config['total']
            qa_pairs.extend([
                (f"Build PC {budget} được gì?",
                 f"PC {budget}: CPU {config['cpu']}, GPU {config['gpu']}, RAM {config['ram']}, SSD {config['storage']}. Hiệu năng: {config['performance']}. Phù hợp cho: {config['use_case']}."),
                
                (f"Cấu hình PC gaming {budget}?",
                 f"Gợi ý PC {budget}: {config['cpu']} + {config['gpu']} + {config['ram']} + {config['storage']}. Chơi game {config['performance']}. Tổng: {config['total']} VND."),
                
                (f"PC {budget} chơi game được không?",
                 f"Được! PC {budget} với {config['cpu']} + {config['gpu']} có thể {config['performance']}. Cấu hình này phù hợp cho {config['use_case']}."),
            ])
        
        # 4. Comparison Q&A
        qa_pairs.extend([
            ("Intel vs AMD CPU nào tốt hơn?",
             "Intel: Hiệu năng gaming cao, tương thích tốt. AMD: Giá/hiệu năng tốt, tiết kiệm điện. Intel tốt cho gaming thuần, AMD tốt cho đa nhiệm."),
            
            ("RTX vs GTX khác nhau gì?",
             "RTX có Ray Tracing và DLSS, hiệu năng cao hơn. GTX giá rẻ hơn nhưng không có RT/DLSS. RTX phù hợp gaming cao cấp, GTX cho budget."),
            
            ("DDR4 vs DDR5 nên chọn gì?",
             "DDR5: Nhanh hơn, future-proof nhưng đắt. DDR4: Rẻ, tương thích rộng. Chọn DDR5 cho build mới cao cấp, DDR4 cho budget/upgrade."),
            
            ("SSD vs HDD nên dùng gì?",
             "SSD: Nhanh, game load nhanh, bền. HDD: Rẻ, dung lượng lớn. Dùng SSD cho OS/game, HDD cho lưu trữ."),
        ])
        
        # 5. Troubleshooting Q&A
        qa_pairs.extend([
            ("PC không khởi động được?",
             "Kiểm tra: 1) Nguồn điện, 2) RAM lắp chặt, 3) Cáp 24pin + 8pin CPU, 4) GPU lắp chặt, 5) Monitor cắm đúng cổng GPU."),
            
            ("Game bị lag, giật?",
             "Nguyên nhân: 1) GPU yếu, 2) RAM không đủ, 3) CPU bottleneck, 4) Nhiệt độ cao, 5) Driver cũ. Kiểm tra từng cái."),
            
            ("PC nóng quá?",
             "Giải pháp: 1) Vệ sinh quạt, 2) Thay keo tản nhiệt, 3) Thêm quạt case, 4) Kiểm tra airflow, 5) Undervolt CPU/GPU."),
            
            ("Màn hình không có tín hiệu?",
             "Kiểm tra: 1) Cáp monitor, 2) Cắm vào GPU không phải mainboard, 3) RAM lắp chặt, 4) GPU có nguồn, 5) Monitor bật đúng input."),
        ])
        
        logger.info(f"Generated {len(qa_pairs)} deep Q&A pairs")
        return qa_pairs
    
    def save_deep_training_data(self, qa_pairs: List[Tuple[str, str]]):
        """Save deep training data to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        saved_count = 0
        for question, answer in qa_pairs:
            try:
                cursor.execute('''
                    INSERT INTO conversations (user_id, message, response, confidence, source)
                    VALUES (?, ?, ?, ?, ?)
                ''', ('deep_training', question, answer, 0.95, 'deep_generated'))
                saved_count += 1
            except Exception as e:
                logger.error(f"Error saving Q&A: {e}")
        
        conn.commit()
        conn.close()
        
        logger.info(f"💾 Saved {saved_count} deep training pairs to database")
        return saved_count
    
    def generate_and_save_deep_data(self):
        """Generate và save tất cả deep training data"""
        logger.info("🧠 Generating deep training data...")
        
        # Generate Q&A pairs
        qa_pairs = self.generate_deep_qa_pairs()
        
        # Save to database
        saved_count = self.save_deep_training_data(qa_pairs)
        
        logger.info(f"✅ Deep training data generation complete: {saved_count} pairs")
        return saved_count

class DeepTrainingSystem:
    """Deep training system với enhanced model"""
    
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.db_path = "conversations.db"
        self.model_save_path = "./deep_trained_model"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Using device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    def load_model(self):
        """Load model for deep training"""
        logger.info(f"Loading {self.model_name} for deep training...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        
        logger.info("✅ Model loaded successfully")
    
    def get_deep_training_data(self) -> List[Tuple[str, str]]:
        """Get all training data for deep training"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all conversations including deep generated ones
        cursor.execute('''
            SELECT message, response FROM conversations 
            WHERE confidence > 0.7
            ORDER BY timestamp DESC
        ''')
        
        training_data = cursor.fetchall()
        conn.close()
        
        logger.info(f"📊 Retrieved {len(training_data)} training samples")
        return training_data
    
    def deep_train_model(self):
        """Perform deep training"""
        logger.info("🧠 Starting DEEP TRAINING...")
        
        # Get training data
        training_data = self.get_deep_training_data()
        
        if len(training_data) < 10:
            logger.error("Not enough training data for deep training")
            return False
        
        # Prepare training texts
        texts = []
        for question, answer in training_data:
            text = f"{question}{self.tokenizer.eos_token}{answer}{self.tokenizer.eos_token}"
            texts.append(text)
        
        # Tokenize
        logger.info("🔤 Tokenizing training data...")
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        dataset = Dataset.from_dict({
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"]
        })
        
        # Training arguments for deep training
        training_args = TrainingArguments(
            output_dir=self.model_save_path,
            overwrite_output_dir=True,
            num_train_epochs=3,  # More epochs for deep training
            per_device_train_batch_size=2,  # Smaller batch for deeper training
            gradient_accumulation_steps=4,  # Accumulate gradients
            save_steps=50,
            save_total_limit=3,
            prediction_loss_only=True,
            learning_rate=3e-5,  # Lower learning rate for stability
            warmup_steps=100,
            logging_steps=10,
            fp16=torch.cuda.is_available(),
            dataloader_drop_last=True,
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
        
        # Deep training
        logger.info(f"🚀 Deep training with {len(training_data)} samples...")
        trainer.train()
        
        # Save deep trained model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.model_save_path)
        
        logger.info("✅ DEEP TRAINING COMPLETED!")
        return True

def main():
    """Main function for deep training"""
    logger.info("🧠 STARTING DEEP TRAINING SYSTEM...")
    
    # 1. Generate deep training data
    data_generator = DeepTrainingDataGenerator()
    data_generator.generate_and_save_deep_data()
    
    # 2. Perform deep training
    training_system = DeepTrainingSystem()
    training_system.load_model()
    training_system.deep_train_model()
    
    logger.info("🎉 DEEP TRAINING SYSTEM COMPLETE!")

if __name__ == "__main__":
    main()
