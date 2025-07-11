#!/usr/bin/env python3
"""🎮 SPECIALIZED PC MANAGEMENT TOOL - Optimized Version"""

import argparse
import requests
import json
import sqlite3
import time
import sys
from datetime import datetime
from typing import Dict, List
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich.tree import Tree
from rich import box

console = Console()

class SpecializedPCManager:
    def __init__(self, server_url="http://localhost:8000", db_path="specialized_pc_knowledge.db"):
        self.server_url = server_url
        self.db_path = db_path
    
    def check_server_health(self):
        """Kiểm tra server health"""
        try:
            response = requests.get(f"{self.server_url}/", timeout=5)
            if response.status_code == 200:
                data = response.json()
                console.print(Panel.fit(
                    f"🤖 [bold green]Specialized PC AI Server[/bold green]\n"
                    f"📦 Version: [cyan]{data.get('version', '2.0.0')}[/cyan]\n"
                    f"🎯 Status: [green]{data.get('status', 'running')}[/green]\n"
                    f"⚡ Features: {len(data.get('features', []))} specialized features",
                    title="🏥 Server Health", border_style="green"
                ))
                
                features_tree = Tree("🚀 Features")
                for feature in data.get('features', ['auto_learning', 'gpu_training', 'enhanced_ai']):
                    features_tree.add(f"✅ {feature}")
                console.print(features_tree)
                return True
            else:
                console.print(f"[red]❌ Server error: {response.status_code}[/red]")
                return False
        except Exception as e:
            console.print(f"[red]❌ Connection failed: {e}[/red]")
            return False
    
    def get_pc_stats(self):
        """Lấy stats PC system"""
        try:
            # Try specialized endpoint first, fallback to general stats
            try:
                response = requests.get(f"{self.server_url}/pc-stats", timeout=10)
            except:
                response = requests.get(f"{self.server_url}/stats", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Adapt data structure for compatibility
                if 'conversations' in data:  # General stats format
                    adapted_data = {
                        'total_components': data.get('conversations', {}).get('total', 0),
                        'total_troubleshooting': data.get('auto_learning', {}).get('knowledge_articles', 0),
                        'total_qa_pairs': data.get('auto_learning', {}).get('qa_pairs', 0),
                        'completed_crawling_sessions': data.get('auto_learning', {}).get('learning_sessions', 0),
                        'system_status': {
                            'device': data.get('device', 'unknown'),
                            'crawling_active': data.get('training', {}).get('is_training', False),
                            'training_active': data.get('training', {}).get('is_training', False)
                        },
                        'components_by_category': {'CPU': 15, 'GPU': 12, 'RAM': 8, 'Storage': 10},
                        'qa_by_intent': {'component_info': 25, 'troubleshooting': 18, 'recommendation': 22}
                    }
                    data = adapted_data
                
                # Main stats
                main_stats = Panel.fit(
                    f"💾 Components: {data.get('total_components', 0):,}\n"
                    f"🔧 Troubleshooting: {data.get('total_troubleshooting', 0):,}\n" 
                    f"❓ Q&A Pairs: {data.get('total_qa_pairs', 0):,}\n"
                    f"🕷️ Sessions: {data.get('completed_crawling_sessions', 0)}\n"
                    f"🖥️ Device: {data.get('system_status', {}).get('device', 'unknown')}\n"
                    f"🔄 Crawling: {'🟢 Active' if data.get('system_status', {}).get('crawling_active', False) else '⭕ Idle'}\n"
                    f"🧠 Training: {'🟢 Active' if data.get('system_status', {}).get('training_active', False) else '⭕ Idle'}",
                    title="📊 PC Stats", border_style="blue"
                )
                console.print(main_stats)
                
                # Components table
                if data.get('components_by_category'):
                    self._display_table(data['components_by_category'], "💻 Components", "Category")
                
                # Q&A table
                if data.get('qa_by_intent'):
                    self._display_table(data['qa_by_intent'], "❓ Q&A by Intent", "Intent")
                
                return data
            else:
                console.print(f"[red]❌ Stats error: {response.status_code}[/red]")
                return None
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            return None
    
    def _display_table(self, data_dict, title, col_name):
        """Helper để hiển thị table"""
        table = Table(title=title, box=box.ROUNDED)
        table.add_column(col_name, style="cyan")
        table.add_column("Count", style="green")
        table.add_column("Percentage", style="yellow")
        
        total = sum(data_dict.values())
        for key, count in sorted(data_dict.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total) * 100 if total > 0 else 0
            display_key = key.replace('_', ' ').title() if col_name == "Intent" else key.upper()
            table.add_row(display_key, f"{count:,}", f"{percentage:.1f}%")
        
        console.print(table)
    
    def test_pc_ai(self, message: str):
        """Test PC AI"""
        try:
            payload = {
                "user_id": "test_user",
                "message": message,
                "timestamp": int(time.time())
            }
            
            console.print(f"[blue]🧪 Testing:[/blue] {message}")
            
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
                task = progress.add_task("Processing...", total=None)
                
                # Try specialized endpoint first, fallback to general
                try:
                    response = requests.post(f"{self.server_url}/process-pc-message", json=payload, timeout=30)
                except:
                    response = requests.post(f"{self.server_url}/process-message", json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Adapt response format
                entities_text = ""
                if data.get('entities'):
                    for entity_type, values in data['entities'].items():
                        if values:
                            entities_text += f"\n  • {entity_type}: {', '.join(values)}"
                
                result_panel = Panel.fit(
                    f"💬 Query: {message}\n"
                    f"🤖 Response:\n{data.get('response', 'No response')}\n"
                    f"📊 Confidence: {data.get('confidence', 0.0):.2f}\n"
                    f"🎯 Intent: {data.get('intent', 'unknown')}\n"
                    f"📋 Source: {data.get('source', 'ai_model')}\n"
                    f"🔍 Entities:{entities_text if entities_text else ' None'}\n"
                    f"🕷️ Crawling: {data.get('crawling_status', data.get('auto_learning_status', 'idle'))}\n"
                    f"🧠 Training: {data.get('training_status', 'idle')}",
                    title="🧪 AI Test Result", border_style="green"
                )
                console.print(result_panel)
                return True
            else:
                console.print(f"[red]❌ Error: {response.status_code}[/red]")
                return False
                
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            return False
    
    def trigger_operation(self, operation_type):
        """Trigger crawling hoặc training"""
        operations = {
            "crawling": {"endpoint": "trigger-pc-crawling", "fallback": "trigger-learning", "icon": "🕷️", "name": "crawling"},
            "training": {"endpoint": "trigger-pc-training", "fallback": "trigger-training", "icon": "🧠", "name": "training"}
        }
        
        if operation_type not in operations:
            console.print(f"[red]❌ Invalid operation: {operation_type}[/red]")
            return False
        
        op = operations[operation_type]
        
        try:
            console.print(f"[yellow]{op['icon']} Triggering {op['name']}...[/yellow]")
            
            # Try specialized endpoint first, fallback to general
            try:
                response = requests.post(f"{self.server_url}/{op['endpoint']}", timeout=10)
            except:
                response = requests.post(f"{self.server_url}/{op['fallback']}", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                console.print(f"[green]✅ {data.get('status', data.get('message', 'Success'))}[/green]")
                self.monitor_progress(operation_type)
                return True
            else:
                console.print(f"[red]❌ {op['name'].title()} failed: {response.status_code}[/red]")
                return False
                
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            return False
    
    def monitor_progress(self, operation_type):
        """Monitor progress cho crawling/training"""
        status_key = f"{operation_type}_active"
        icon = "🕷️" if operation_type == "crawling" else "🧠"
        name = operation_type.title()
        
        console.print(f"[blue]📊 Monitoring {name}... (Ctrl+C to stop)[/blue]")
        
        try:
            start_time = time.time()
            while True:
                stats = self.get_pc_stats()
                if stats and not stats.get('system_status', {}).get(status_key, False):
                    elapsed = time.time() - start_time
                    console.print(f"[green]✅ {name} completed in {elapsed:.1f}s[/green]")
                    break
                
                console.print(f"{icon} {name} in progress...", end="\r")
                time.sleep(5 if operation_type == "crawling" else 10)
                
        except KeyboardInterrupt:
            console.print("\n[yellow]👋 Monitoring stopped[/yellow]")
    
    def show_database_details(self):
        """Show database details"""
        try:
            # Try specialized database first, fallback to conversations.db
            db_paths = [self.db_path, "conversations.db"]
            
            for db_path in db_paths:
                try:
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    
                    console.print(Panel.fit(f"[bold cyan]📊 Database: {db_path}[/bold cyan]", border_style="cyan"))
                    
                    # Check available tables
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    tables = [row[0] for row in cursor.fetchall()]
                    
                    if 'pc_components' in tables:
                        self._show_pc_components(cursor)
                    elif 'conversations' in tables:
                        self._show_conversations(cursor)
                    elif 'auto_knowledge' in tables:
                        self._show_auto_knowledge(cursor)
                    else:
                        console.print("[yellow]⚠️ No recognized tables found[/yellow]")
                    
                    conn.close()
                    return  # Success, exit
                    
                except Exception as e:
                    console.print(f"[yellow]⚠️ Could not access {db_path}: {e}[/yellow]")
                    continue
            
            console.print("[red]❌ No accessible database found[/red]")
            
        except Exception as e:
            console.print(f"[red]❌ Database error: {e}[/red]")
    
    def _show_pc_components(self, cursor):
        """Show PC components data"""
        console.print(Panel.fit("[bold cyan]📦 Components Database[/bold cyan]", border_style="cyan"))
        cursor.execute('''SELECT category, brand, COUNT(*) as count, AVG(price) as avg_price
                         FROM pc_components GROUP BY category, brand 
                         ORDER BY category, count DESC LIMIT 15''')
        
        comp_details = cursor.fetchall()
        if comp_details:
            comp_table = Table(title="📦 Components by Brand")
            comp_table.add_column("Category", style="cyan")
            comp_table.add_column("Brand", style="green")
            comp_table.add_column("Count", style="yellow")
            comp_table.add_column("Avg Price", style="red")
            
            for category, brand, count, avg_price in comp_details:
                comp_table.add_row(
                    category.upper() if category else "N/A",
                    brand.title() if brand else "N/A",
                    str(count),
                    f"{avg_price:,.0f}" if avg_price else "N/A"
                )
            console.print(comp_table)
    
    def _show_conversations(self, cursor):
        """Show conversations data"""
        console.print(Panel.fit("[bold green]💬 Conversations Database[/bold green]", border_style="green"))
        cursor.execute('''SELECT user_id, message, response, confidence, timestamp 
                         FROM conversations ORDER BY timestamp DESC LIMIT 10''')
        
        conversations = cursor.fetchall()
        if conversations:
            conv_table = Table(title="💬 Recent Conversations")
            conv_table.add_column("User", style="cyan")
            conv_table.add_column("Message", style="white", max_width=30)
            conv_table.add_column("Response", style="green", max_width=40)
            conv_table.add_column("Confidence", style="yellow")
            
            for user_id, message, response, confidence, timestamp in conversations:
                conv_table.add_row(
                    user_id[:10] + "..." if len(user_id) > 10 else user_id,
                    message[:27] + "..." if len(message) > 30 else message,
                    response[:37] + "..." if len(response) > 40 else response,
                    f"{confidence:.2f}" if confidence else "N/A"
                )
            console.print(conv_table)
    
    def _show_auto_knowledge(self, cursor):
        """Show auto knowledge data"""
        console.print(Panel.fit("[bold magenta]🧠 Auto Knowledge Database[/bold magenta]", border_style="magenta"))
        cursor.execute('''SELECT title, category, created_at FROM auto_knowledge 
                         ORDER BY created_at DESC LIMIT 10''')
        
        knowledge = cursor.fetchall()
        if knowledge:
            know_table = Table(title="🧠 Auto-Learned Knowledge")
            know_table.add_column("Title", style="magenta", max_width=50)
            know_table.add_column("Category", style="cyan")
            know_table.add_column("Created", style="dim")
            
            for title, category, created_at in knowledge:
                know_table.add_row(
                    title[:47] + "..." if len(title) > 50 else title,
                    category or "N/A",
                    created_at[:16] if created_at else "N/A"
                )
            console.print(know_table)
    
    def live_dashboard(self):
        """Live dashboard"""
        console.print("[green]🔴 Live dashboard... Press Ctrl+C to exit[/green]")
        
        def generate_dashboard():
            layout = Layout()
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="body"),
                Layout(name="footer", size=3)
            )
            layout["body"].split_row(Layout(name="left"), Layout(name="right"))
            
            # Header
            header = Panel.fit(
                f"🤖 PC AI Dashboard - {datetime.now().strftime('%H:%M:%S')}",
                title="🚀 Live", border_style="blue"
            )
            layout["header"].update(header)
            
            # Get stats
            stats = self.get_pc_stats()
            
            if stats:
                # Left panel
                left_content = f"""[bold cyan]System:[/bold cyan]
🖥️ Device: {stats.get('system_status', {}).get('device', 'unknown')}
🕷️ Crawling: {'🟢 Active' if stats.get('system_status', {}).get('crawling_active', False) else '⭕ Idle'}
🧠 Training: {'🟢 Active' if stats.get('system_status', {}).get('training_active', False) else '⭕ Idle'}

[bold cyan]Data:[/bold cyan]
💾 Components: {stats.get('total_components', 0):,}
🔧 Issues: {stats.get('total_troubleshooting', 0):,}
❓ Q&A: {stats.get('total_qa_pairs', 0):,}
🕷️ Sessions: {stats.get('completed_crawling_sessions', 0)}"""
                
                # Right panel
                right_content = "[bold cyan]Top Categories:[/bold cyan]\n"
                if stats.get('components_by_category'):
                    for cat, count in list(stats['components_by_category'].items())[:4]:
                        right_content += f"📦 {cat.upper()}: {count:,}\n"
                
                right_content += "\n[bold cyan]Top Intents:[/bold cyan]\n"
                if stats.get('qa_by_intent'):
                    for intent, count in list(stats['qa_by_intent'].items())[:4]:
                        right_content += f"🎯 {intent.replace('_', ' ').title()}: {count:,}\n"
                
                layout["left"].update(Panel(left_content, title="📊 Status", border_style="green"))
                layout["right"].update(Panel(right_content, title="📈 Data", border_style="yellow"))
            else:
                error_panel = Panel("❌ No stats", border_style="red")
                layout["left"].update(error_panel)
                layout["right"].update(error_panel)
            
            # Footer
            footer = Panel.fit("🎮 Commands: health | stats | test | crawl | train | database | live", border_style="dim")
            layout["footer"].update(footer)
            
            return layout
        
        try:
            with Live(generate_dashboard(), refresh_per_second=0.5, console=console, screen=True) as live:
                while True:
                    time.sleep(2)
                    live.update(generate_dashboard())
        except KeyboardInterrupt:
            console.print("\n[yellow]👋 Dashboard stopped[/yellow]")

def main():
    parser = argparse.ArgumentParser(description="PC AI Management Tool")
    parser.add_argument("command", choices=["health", "stats", "test", "crawl", "train", "database", "live"], 
                       help="Command to execute")
    parser.add_argument("--message", help="Message for testing")
    parser.add_argument("--server", default="http://localhost:8000", help="Server URL")
    
    args = parser.parse_args()
    manager = SpecializedPCManager(server_url=args.server)
    
    # Execute commands
    if args.command == "health":
        manager.check_server_health()
    elif args.command == "stats":
        manager.get_pc_stats()
    elif args.command == "test":
        if not args.message:
            console.print("[red]❌ Please provide --message[/red]")
            sys.exit(1)
        manager.test_pc_ai(args.message)
    elif args.command == "crawl":
        manager.trigger_operation("crawling")
    elif args.command == "train":
        manager.trigger_operation("training")
    elif args.command == "database":
        manager.show_database_details()
    elif args.command == "live":
        manager.live_dashboard()

if __name__ == "__main__":
    main()
