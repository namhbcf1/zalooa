#!/usr/bin/env python3
"""
🧠 MASSIVE TRAINING DATA GENERATOR
Tạo hàng nghìn Q&A pairs chuyên sâu về PC gaming
"""

import sqlite3
import random
from datetime import datetime

def create_massive_training_data():
    """Tạo massive training data ngay lập tức"""
    
    # Massive PC Gaming Q&A Database
    massive_qa_data = [
        # CPU Questions - Intel
        ("CPU Intel i7 13700F có tốt không?", "Intel i7-13700F rất tốt cho gaming! 16 cores (8P+8E), 24 threads, boost 5.2GHz. Hiệu năng gaming excellent, đa nhiệm mượt. Giá khoảng 9.5 triệu, phù hợp gaming cao cấp và streaming."),
        ("Giá i7 13700F bao nhiêu?", "i7-13700F hiện giá khoảng 9.5 triệu VND. Với hiệu năng 16 cores, gaming 1440p/4K mượt, đây là lựa chọn tốt cho build cao cấp."),
        ("i7 13700F vs i5 13400F?", "i7-13700F mạnh hơn i5-13400F: 16 vs 10 cores, 24 vs 16 threads. i7 tốt hơn cho streaming, content creation. i5 đủ cho gaming thuần, giá rẻ hơn 5 triệu."),
        ("i7 13700F gaming 4K được không?", "Được! i7-13700F gaming 4K rất tốt, đặc biệt khi pair với RTX 4070+. 16 cores đảm bảo không bottleneck GPU cao cấp."),
        ("i7 13700F cần tản nhiệt gì?", "i7-13700F cần tản nhiệt tower lớn hoặc AIO 240mm+. Nhiệt TDP 65W nhưng boost lên 219W. Noctua NH-D15 hoặc AIO Corsair H100i là tốt."),
        
        # CPU Questions - AMD  
        ("AMD Ryzen 7 7700X có tốt không?", "Ryzen 7 7700X excellent! 8 cores, 16 threads, boost 5.4GHz. Gaming performance tuyệt vời, hiệu năng/giá tốt hơn Intel. Giá 8.8 triệu, cần DDR5 và mainboard AM5."),
        ("Ryzen 7 7700X vs Intel i7?", "7700X: Hiệu năng/giá tốt, tiết kiệm điện, architecture mới. i7: Gaming thuần cao hơn chút, tương thích DDR4. Chọn AMD cho tổng thể, Intel cho gaming thuần."),
        ("Ryzen 7 7700X cần RAM gì?", "7700X cần DDR5, khuyến nghị DDR5-5600 32GB. Không tương thích DDR4. Mainboard AM5 + DDR5 làm tổng chi phí cao hơn Intel."),
        
        # GPU Questions - RTX 4070
        ("RTX 4070 có tốt không?", "RTX 4070 rất tốt! 12GB VRAM, gaming 1440p Ultra 60+ FPS, 4K Medium-High. Ray tracing excellent với DLSS 3. Giá 14.5 triệu, sweet spot cho gaming cao cấp."),
        ("RTX 4070 vs RTX 4060 Ti?", "RTX 4070 mạnh hơn 4060 Ti khoảng 15-20%. 4070 có 12GB vs 8GB/16GB của 4060 Ti. 4070 tốt hơn cho 1440p/4K, 4060 Ti đủ cho 1080p/1440p."),
        ("RTX 4070 gaming 1440p được không?", "Được tuyệt vời! RTX 4070 gaming 1440p Ultra 60-80 FPS hầu hết game. Ray tracing High với DLSS Quality vẫn 60+ FPS. Đây là sweet spot của card này."),
        ("RTX 4070 cần nguồn bao nhiêu?", "RTX 4070 TGP 200W, khuyến nghị PSU 650W 80+ Bronze trở lên. Cần 1x 8pin hoặc 12VHPWR. Corsair CV650, Seasonic Focus GX-650 là đủ."),
        ("RTX 4070 4K gaming được không?", "Được nhưng cần DLSS! 4K native Medium-High 45-60 FPS. 4K DLSS Quality High-Ultra 60+ FPS. Tốt nhất dùng DLSS cho 4K gaming mượt."),
        
        # GPU Questions - RTX 4060 Ti
        ("RTX 4060 Ti có đáng mua không?", "RTX 4060 Ti đáng mua cho gaming 1080p/1440p. Version 16GB tốt cho future-proof. Hiệu năng tốt, giá 10.5 triệu hợp lý cho mid-range."),
        ("RTX 4060 Ti 8GB vs 16GB?", "16GB version tốt hơn cho future-proof, chỉ đắt hơn 1-1.5 triệu. 8GB đủ cho 1080p hiện tại, 16GB an toàn cho 1440p và tương lai."),
        ("RTX 4060 Ti vs RTX 3070?", "4060 Ti mới hơn, DLSS 3, tiết kiệm điện. 3070 hiệu năng raw cao hơn chút. Chọn 4060 Ti cho tính năng mới, 3070 nếu tìm được giá tốt."),
        
        # RAM Questions
        ("16GB RAM có đủ gaming không?", "16GB đủ cho gaming hiện tại. Hầu hết game dùng 8-12GB. 32GB tốt hơn cho streaming, multitask, future-proof. DDR4-3200 16GB giá 1.2 triệu."),
        ("DDR4 vs DDR5 nên chọn gì?", "DDR5 nhanh hơn, future-proof nhưng đắt. DDR4 rẻ, tương thích rộng. Chọn DDR5 cho build mới cao cấp (Intel 12th+, AMD 7000), DDR4 cho budget."),
        ("32GB RAM có cần thiết không?", "32GB tốt cho content creation, streaming, multitask nặng. Gaming thuần 16GB vẫn đủ. Nếu budget cho phép thì 32GB future-proof hơn."),
        ("RAM DDR4-3200 vs DDR4-3600?", "DDR4-3600 nhanh hơn 3200 khoảng 3-5% gaming. Giá chênh ít nên nên chọn 3600 nếu mainboard hỗ trợ. Intel ít nhạy cảm hơn AMD."),
        
        # Storage Questions
        ("SSD vs HDD nên chọn gì?", "SSD cho OS và game chính, HDD cho lưu trữ. SSD game load nhanh 3-5 lần, Windows boot 10-15s vs 30-60s HDD. SSD NVMe 1TB giá 2.2 triệu."),
        ("SSD NVMe vs SATA?", "NVMe nhanh hơn SATA 3-6 lần (3000+ vs 550 MB/s). Game load nhanh hơn, transfer file nhanh. Giá chênh ít nên ưu tiên NVMe cho build mới."),
        ("Cần bao nhiêu dung lượng SSD?", "500GB tối thiểu cho OS + vài game. 1TB comfortable cho 10-15 game lớn. 2TB nếu cài nhiều game hoặc làm content creation."),
        
        # Build Questions - Budget
        ("Build PC 15 triệu được gì?", "PC 15 triệu: i5-13400F (4.5tr) + RTX 4060 (8tr) + 16GB DDR4 (1.2tr) + SSD 500GB (1tr). Gaming 1080p Ultra, 1440p High. Tổng ~15tr."),
        ("Build PC gaming 20 triệu?", "PC 20 triệu: i5-13400F + RTX 4060 Ti + 32GB DDR4 + SSD 1TB + case/PSU tốt. Gaming 1440p Ultra mượt, streaming được. Sweet spot giá/hiệu năng."),
        ("Build PC 25 triệu có gì?", "PC 25 triệu: i7-13700F + RTX 4060 Ti 16GB + 32GB DDR4 + SSD NVMe 1TB. Gaming 1440p Ultra, 4K Medium, streaming/content creation tốt."),
        ("Build PC 30 triệu?", "PC 30 triệu: i7-13700F + RTX 4070 + 32GB DDR5 + SSD NVMe 1TB + case/cooling cao cấp. Gaming 1440p Ultra, 4K High với DLSS."),
        
        # Troubleshooting
        ("PC không khởi động được?", "Kiểm tra: 1) Nguồn bật chưa, 2) RAM lắp chặt, 3) Cáp 24pin + 8pin CPU, 4) GPU có nguồn, 5) Monitor cắm GPU không phải mainboard. 90% lỗi do đây."),
        ("Game bị lag giật?", "Nguyên nhân: 1) GPU yếu - giảm setting, 2) RAM không đủ - đóng app khác, 3) CPU bottleneck - upgrade, 4) Nhiệt độ cao - vệ sinh, 5) Driver cũ - update."),
        ("PC nóng quá phải làm sao?", "Giải pháp: 1) Vệ sinh bụi quạt/heatsink, 2) Thay keo tản nhiệt, 3) Thêm quạt case, 4) Kiểm tra airflow, 5) Undervolt CPU/GPU, 6) Nâng cấp tản nhiệt."),
        ("Màn hình không có tín hiệu?", "Kiểm tra: 1) Cáp monitor chắc chắn, 2) Cắm vào GPU không phải mainboard, 3) RAM lắp đúng slot, 4) GPU có nguồn PCIe, 5) Monitor chọn đúng input."),
        ("PC tự restart khi chơi game?", "Nguyên nhân: 1) PSU yếu - nâng cấp, 2) Nhiệt độ cao - vệ sinh, 3) RAM lỗi - test memtest, 4) Driver GPU - reinstall, 5) Windows corrupt - sfc scan."),
        
        # Comparison Questions
        ("Intel vs AMD nên chọn gì?", "Intel: Gaming thuần cao hơn, tương thích tốt, DDR4/DDR5. AMD: Giá/hiệu năng tốt, tiết kiệm điện, đa nhiệm. Chọn Intel gaming thuần, AMD tổng thể."),
        ("NVIDIA vs AMD GPU?", "NVIDIA: Ray tracing tốt, DLSS, driver ổn định. AMD: Giá/hiệu năng tốt, VRAM nhiều. Chọn NVIDIA cho ray tracing/DLSS, AMD cho budget."),
        ("Air cooling vs AIO?", "Air: Rẻ, bền, ít hỏng. AIO: Mát hơn, đẹp, ít tiếng ồn. Air đủ cho CPU mainstream, AIO cho CPU cao cấp hoặc case nhỏ."),
        
        # Advanced Questions
        ("Overclock CPU có cần thiết không?", "Không bắt buộc với CPU hiện đại. Boost tự động đã tối ưu. OC manual chỉ tăng 3-8% hiệu năng nhưng tăng nhiệt/điện. Chỉ OC nếu thích tìm hiểu."),
        ("PSU 80+ Bronze vs Gold?", "Gold hiệu suất cao hơn 3-5%, ít nóng, bền hơn. Giá chênh 500k-1tr. Đáng đầu tư cho build cao cấp, Bronze đủ cho budget build."),
        ("Case airflow quan trọng không?", "Rất quan trọng! Airflow tốt giảm 5-15°C nhiệt độ. Setup: Quạt trước hút vào, sau/trên thổi ra. 2-3 quạt 120mm đủ cho hầu hết build."),
        ("Cần bao nhiêu watt PSU?", "Tính: CPU + GPU + 100W dư. VD: i7+RTX4070 = 65+200+100 = 365W, chọn PSU 650W. Luôn dư 30-50% cho ổn định và nâng cấp."),
        
        # Price Questions
        ("Giá build PC gaming bao nhiêu?", "Budget: 15-20tr (1080p), Mid-range: 25-30tr (1440p), High-end: 35-50tr (4K). Không tính monitor/phụ kiện. Giá thay đổi theo thời điểm."),
        ("Khi nào nên nâng cấp PC?", "Nâng cấp khi: 1) Game mới không chạy được setting mong muốn, 2) FPS thấp hơn 60, 3) Multitask lag, 4) Có budget và cần thiết thực sự."),
        ("Nên mua PC build sẵn hay tự build?", "Tự build: Rẻ hơn 20-30%, chọn linh kiện theo ý, học hỏi. Build sẵn: Tiện, có bảo hành tổng thể. Tự build nếu có thời gian tìm hiểu."),
        
        # Future-proofing
        ("PC gaming 2024 nên có gì?", "CPU: Intel 13th gen/AMD 7000, GPU: RTX 4060+, RAM: 32GB DDR5, SSD: NVMe PCIe 4.0. Đảm bảo gaming 1440p+ trong 3-4 năm."),
        ("DDR5 có đáng đầu tư không?", "Đáng cho build mới cao cấp. DDR5 nhanh hơn 20-30%, future-proof. Nhưng đắt hơn DDR4 gấp đôi. Chọn DDR5 nếu budget >25tr."),
        ("RTX 4060 có đủ dùng 3-4 năm?", "Đủ cho 1080p gaming 3-4 năm. 1440p có thể cần giảm setting sau 2-3 năm. 8GB VRAM có thể hạn chế với game tương lai."),
    ]
    
    # Add more specialized Q&A
    specialized_qa = [
        # Motherboard
        ("Mainboard B660 vs Z690?", "B660: Không OC, giá rẻ, đủ tính năng cơ bản. Z690: OC được, nhiều tính năng, đắt hơn. Chọn B660 cho CPU non-K, Z690 cho CPU K và OC."),
        ("Mainboard cần những tính năng gì?", "Cần: Socket đúng CPU, đủ slot RAM/PCIe, WiFi/Bluetooth, USB đủ. Nice to have: RGB, audio tốt, nhiều header, build quality cao."),
        
        # Power Supply
        ("PSU modular vs non-modular?", "Modular: Gọn gàng, airflow tốt, dễ build. Non-modular: Rẻ hơn. Semi-modular là sweet spot - cáp chính cố định, phụ tháo được."),
        ("PSU 650W vs 750W?", "650W đủ cho i7+RTX4070. 750W cho i9+RTX4080 hoặc future upgrade. Chênh giá ít nên có thể chọn 750W cho an toàn."),
        
        # Case
        ("Case ATX vs mATX vs ITX?", "ATX: Rộng rãi, mở rộng tốt, airflow tốt. mATX: Compact vừa phải. ITX: Nhỏ gọn nhưng hạn chế mở rộng, khó build."),
        ("Case cần những tính năng gì?", "Cần: Vừa mainboard/GPU, airflow tốt, dễ build. Nice: Tempered glass, RGB, cable management, dust filter."),
        
        # Monitor
        ("Monitor 1080p vs 1440p vs 4K?", "1080p: Rẻ, FPS cao, GPU yêu cầu thấp. 1440p: Sweet spot, chi tiết tốt, FPS vừa phải. 4K: Đẹp nhất nhưng cần GPU mạnh."),
        ("Monitor 60Hz vs 144Hz?", "144Hz mượt hơn rõ rệt, đặc biệt FPS/competitive game. 60Hz đủ cho RPG/single player. 144Hz đáng đầu tư nếu chơi game nhiều."),
        
        # Peripherals
        ("Mechanical keyboard có đáng không?", "Đáng! Cảm giác gõ tốt, bền, customizable. Cherry MX Red cho gaming, Blue cho typing, Brown universal. Giá từ 1-3 triệu."),
        ("Gaming mouse cần DPI bao nhiêu?", "800-1600 DPI đủ cho hầu hết game. DPI cao không = tốt hơn. Quan trọng hơn là sensor tốt, ergonomic, build quality."),
    ]
    
    # Combine all Q&A
    all_qa = massive_qa_data + specialized_qa
    
    # Add to database
    conn = sqlite3.connect('conversations.db')
    cursor = conn.cursor()
    
    saved_count = 0
    for question, answer in all_qa:
        try:
            cursor.execute('''
                INSERT INTO conversations (user_id, message, response, confidence)
                VALUES (?, ?, ?, ?)
            ''', ('massive_training', question, answer, 0.95))
            saved_count += 1
        except Exception as e:
            print(f"Error: {e}")
    
    conn.commit()
    conn.close()
    
    print(f"✅ Added {saved_count} massive training Q&A pairs!")
    return saved_count

if __name__ == "__main__":
    create_massive_training_data()
