# management.py - Tool quản lý hệ thống
import sqlite3
import json
import requests
import argparse
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ZaloAIManager:
    def __init__(self, db_path="conversations.db", server_url="http://localhost:8000"):
        self.db_path = db_path
        self.server_url = server_url
    
    def get_connection(self):
        return sqlite3.connect(self.db_path)
    
    def get_stats(self):
        """Lấy thống kê tổng quan"""
        with self.get_connection() as conn:
            # Thống kê conversations
            df_conv = pd.read_sql_query("""
                SELECT 
                    user_id,
                    COUNT(*) as message_count,
                    MIN(timestamp) as first_message,
                    MAX(timestamp) as last_message
                FROM conversations 
                GROUP BY user_id
            """, conn)
            
            # Thống kê theo ngày
            df_daily = pd.read_sql_query("""
                SELECT 
                    DATE(timestamp) as date,
                    COUNT(*) as messages,
                    COUNT(DISTINCT user_id) as unique_users
                FROM conversations 
                GROUP BY DATE(timestamp)
                ORDER BY date
            """, conn)
            
            print("=== THỐNG KÊ TỔNG QUAN ===")
            print(f"Tổng số người dùng: {len(df_conv)}")
            print(f"Tổng số tin nhắn: {df_conv['message_count'].sum()}")
            print(f"Trung bình tin nhắn/người: {df_conv['message_count'].mean():.1f}")
            print(f"Người dùng hoạt động nhất: {df_conv['message_count'].max()} tin nhắn")
            
            print("\n=== THỐNG KÊ 7 NGÀY GẦN NHẤT ===")
            recent_days = df_daily.tail(7)
            print(recent_days.to_string(index=False))
            
            return df_conv, df_daily
    
    def get_training_history(self):
        """Lấy lịch sử training"""
        with self.get_connection() as conn:
            df = pd.read_sql_query("""
                SELECT * FROM training_sessions 
                ORDER BY timestamp DESC
            """, conn)
            
            print("=== LỊCH SỬ TRAINING ===")
            print(df.to_string(index=False))
            return df
    
    def export_conversations(self, user_id=None, output_file="conversations_export.json"):
        """Export conversations"""
        with self.get_connection() as conn:
            if user_id:
                df = pd.read_sql_query("""
                    SELECT * FROM conversations 
                    WHERE user_id = ?
                    ORDER BY timestamp
                """, conn, params=(user_id,))
            else:
                df = pd.read_sql_query("""
                    SELECT * FROM conversations 
                    ORDER BY timestamp
                """, conn)
            
            # Convert to JSON
            conversations = df.to_dict('records')
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(conversations, f, ensure_ascii=False, indent=2, default=str)
            
            print(f"Đã export {len(conversations)} conversations vào {output_file}")
    
    def cleanup_old_data(self, days=30):
        """Dọn dẹp dữ liệu cũ"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Đếm số record sẽ xóa
            cursor.execute("""
                SELECT COUNT(*) FROM conversations 
                WHERE timestamp < ?
            """, (cutoff_date,))
            
            count = cursor.fetchone()[0]
            
            if count > 0:
                confirm = input(f"Sẽ xóa {count} conversations cũ hơn {days} ngày. Tiếp tục? (y/N): ")
                if confirm.lower() == 'y':
                    cursor.execute("""
                        DELETE FROM conversations 
                        WHERE timestamp < ?
                    """, (cutoff_date,))
                    conn.commit()
                    print(f"Đã xóa {count} conversations cũ")
                else:
                    print("Hủy bỏ")
            else:
                print("Không có dữ liệu cũ cần xóa")
    
    def trigger_training(self):
        """Trigger manual training"""
        try:
            response = requests.post(f"{self.server_url}/manual-training")
            result = response.json()
            print(f"Training status: {result['status']}")
        except Exception as e:
            print(f"Error triggering training: {e}")
    
    def get_server_stats(self):
        """Lấy stats từ server"""
        try:
            response = requests.get(f"{self.server_url}/stats")
            stats = response.json()
            
            print("=== SERVER STATS ===")
            for key, value in stats.items():
                print(f"{key}: {value}")
            
            return stats
        except Exception as e:
            print(f"Error getting server stats: {e}")
            return None
    
    def test_ai_response(self, message, user_id="test_user"):
        """Test AI response"""
        try:
            payload = {
                "user_id": user_id,
                "message": message,
                "timestamp": int(datetime.now().timestamp() * 1000)
            }
            
            response = requests.post(f"{self.server_url}/process-message", json=payload)
            result = response.json()
            
            print(f"User: {message}")
            print(f"AI: {result['response']}")
            print(f"Confidence: {result['confidence']:.2f}")
            
            return result
        except Exception as e:
            print(f"Error testing AI: {e}")
            return None
    
    def generate_report(self, output_file="ai_report.html"):
        """Tạo báo cáo HTML"""
        df_conv, df_daily = self.get_stats()
        
        # Tạo biểu đồ
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Biểu đồ 1: Messages per day
        df_daily.plot(x='date', y='messages', kind='line', ax=axes[0,0], title='Messages per Day')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Biểu đồ 2: Unique users per day
        df_daily.plot(x='date', y='unique_users', kind='line', ax=axes[0,1], title='Unique Users per Day', color='orange')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Biểu đồ 3: Message count distribution
        df_conv['message_count'].hist(bins=20, ax=axes[1,0], title='Message Count Distribution')
        axes[1,0].set_xlabel('Messages per User')
        
        # Biểu đồ 4: Top users
        top_users = df_conv.nlargest(10, 'message_count')
        top_users.plot(x='user_id', y='message_count', kind='bar', ax=axes[1,1], title='Top 10 Users')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('ai_stats.png', dpi=300, bbox_inches='tight')
        
        # Tạo HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Zalo OA AI Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #4CAF50; color: white; padding: 20px; border-radius: 5px; }}
                .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
                .stat-box {{ background: #f5f5f5; padding: 15px; border-radius: 5px; text-align: center; }}
                .chart {{ text-align: center; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Zalo OA AI System Report</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="stats">
                <div class="stat-box">
                    <h3>{len(df_conv)}</h3>
                    <p>Total Users</p>
                </div>
                <div class="stat-box">
                    <h3>{df_conv['message_count'].sum()}</h3>
                    <p>Total Messages</p>
                </div>
                <div class="stat-box">
                    <h3>{df_conv['message_count'].mean():.1f}</h3>
                    <p>Avg Messages/User</p>
                </div>
                <div class="stat-box">
                    <h3>{len(df_daily)}</h3>
                    <p>Active Days</p>
                </div>
            </div>
            
            <div class="chart">
                <h2>Statistics Charts</h2>
                <img src="ai_stats.png" style="max-width: 100%;">
            </div>
            
            <h2>Recent Activity</h2>
            <table border="1" style="width: 100%; border-collapse: collapse;">
                <tr style="background: #f0f0f0;">
                    <th>Date</th>
                    <th>Messages</th>
                    <th>Unique Users</th>
                </tr>
        """
        
        for _, row in df_daily.tail(10).iterrows():
            html_content += f"""
                <tr>
                    <td>{row['date']}</td>
                    <td>{row['messages']}</td>
                    <td>{row['unique_users']}</td>
                </tr>
            """
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Report saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Zalo OA AI Management Tool')
    parser.add_argument('action', choices=[
        'stats', 'training-history', 'export', 'cleanup', 
        'train', 'server-stats', 'test', 'report'
    ])
    parser.add_argument('--user-id', help='User ID for specific operations')
    parser.add_argument('--message', help='Test message for AI')
    parser.add_argument('--days', type=int, default=30, help='Days for cleanup')
    parser.add_argument('--output', help='Output file path')
    
    args = parser.parse_args()
    
    manager = ZaloAIManager()
    
    if args.action == 'stats':
        manager.get_stats()
    elif args.action == 'training-history':
        manager.get_training_history()
    elif args.action == 'export':
        output_file = args.output or "conversations_export.json"
        manager.export_conversations(args.user_id, output_file)
    elif args.action == 'cleanup':
        manager.cleanup_old_data(args.days)
    elif args.action == 'train':
        manager.trigger_training()
    elif args.action == 'server-stats':
        manager.get_server_stats()
    elif args.action == 'test':
        if not args.message:
            print("Cần --message để test")
            return
        manager.test_ai_response(args.message, args.user_id or "test_user")
    elif args.action == 'report':
        output_file = args.output or "ai_report.html"
        manager.generate_report(output_file)

if __name__ == "__main__":
    main()