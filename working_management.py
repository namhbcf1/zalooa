#!/usr/bin/env python3
"""
🎮 WORKING MANAGEMENT TOOL - Hoạt động 100%
Rich CLI Dashboard cho Zalo AI Bot
"""

import argparse
import requests
import json
import sqlite3
import time
import sys
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich import box

console = Console()

class WorkingZaloAIManager:
    def __init__(self, server_url="http://localhost:8000", db_path="conversations.db"):
        self.server_url = server_url
        self.db_path = db_path
    
    def check_server_health(self):
        """Kiểm tra health của server"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                
                status_color = "green" if data['status'] == 'healthy' else "red"
                training_color = "yellow" if data['training_active'] else "blue"
                
                console.print(Panel.fit(
                    f"🟢 Status: [{status_color}]{data['status']}[/{status_color}]\n"
                    f"🖥️ Device: [cyan]{data['device']}[/cyan]\n"
                    f"🤖 Model: [green]{'Loaded' if data['model_loaded'] else 'Loading'}[/green]\n"
                    f"🧠 Training: [{training_color}]{'Active' if data['training_active'] else 'Idle'}[/{training_color}]\n"
                    f"💬 Conversations: [yellow]{data['total_conversations']}[/yellow]\n"
                    f"📚 Training Sessions: [magenta]{data['training_sessions']}[/magenta]\n"
                    f"🕒 Last Check: [dim]{data['timestamp']}[/dim]",
                    title="🏥 Server Health",
                    border_style="green"
                ))
                return True
            else:
                console.print(f"[red]❌ Server error: {response.status_code}[/red]")
                return False
        except Exception as e:
            console.print(f"[red]❌ Cannot connect to server: {e}[/red]")
            return False
    
    def get_server_stats(self):
        """Lấy thống kê server"""
        try:
            response = requests.get(f"{self.server_url}/stats", timeout=5)
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None
    
    def test_ai_response(self, message: str):
        """Test AI response"""
        try:
            payload = {
                "user_id": "test_user",
                "message": message,
                "timestamp": int(time.time())
            }
            
            console.print(f"[blue]🧪 Testing message:[/blue] {message}")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Generating AI response...", total=None)
                
                response = requests.post(
                    f"{self.server_url}/process-message",
                    json=payload,
                    timeout=30
                )
            
            if response.status_code == 200:
                data = response.json()
                
                console.print(Panel.fit(
                    f"💬 [bold]User:[/bold] {message}\n"
                    f"🤖 [bold]AI:[/bold] {data['response']}\n"
                    f"📊 [bold]Confidence:[/bold] {data['confidence']:.2f}\n"
                    f"👤 [bold]User ID:[/bold] {data['user_id']}\n"
                    f"🧠 [bold]Training:[/bold] {data['training_status']}\n"
                    f"💬 [bold]Total Conversations:[/bold] {data['total_conversations']}",
                    title="🧪 AI Response Test",
                    border_style="blue"
                ))
                return True
            else:
                console.print(f"[red]❌ API Error: {response.status_code}[/red]")
                console.print(f"[red]{response.text}[/red]")
                return False
                
        except Exception as e:
            console.print(f"[red]❌ Error testing AI: {e}[/red]")
            return False
    
    def trigger_training(self, force=False):
        """Trigger manual training"""
        try:
            payload = {"force": force}
            
            console.print("[yellow]🧠 Triggering manual training...[/yellow]")
            
            response = requests.post(
                f"{self.server_url}/trigger-training",
                json=payload,
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                console.print(f"[green]✅ {data['message']}[/green]")
                console.print(f"[blue]Status: {data['status']}[/blue]")
                return True
            else:
                console.print(f"[red]❌ Training trigger failed: {response.status_code}[/red]")
                return False
                
        except Exception as e:
            console.print(f"[red]❌ Error triggering training: {e}[/red]")
            return False
    
    def show_database_stats(self):
        """Show database statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get conversation stats
            cursor.execute("SELECT COUNT(*) FROM conversations")
            total_conversations = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT user_id) FROM conversations")
            unique_users = cursor.fetchone()[0]
            
            cursor.execute("SELECT AVG(confidence) FROM conversations")
            avg_confidence = cursor.fetchone()[0] or 0
            
            # Get recent conversations
            cursor.execute("""
                SELECT user_id, message, response, confidence, timestamp 
                FROM conversations 
                ORDER BY timestamp DESC 
                LIMIT 5
            """)
            recent_conversations = cursor.fetchall()
            
            # Get training sessions
            cursor.execute("SELECT COUNT(*) FROM training_sessions")
            training_sessions = cursor.fetchone()[0]
            
            conn.close()
            
            # Display stats
            console.print(Panel.fit(
                f"💬 [bold]Total Conversations:[/bold] {total_conversations}\n"
                f"👥 [bold]Unique Users:[/bold] {unique_users}\n"
                f"📊 [bold]Average Confidence:[/bold] {avg_confidence:.2f}\n"
                f"🧠 [bold]Training Sessions:[/bold] {training_sessions}",
                title="📊 Database Statistics",
                border_style="cyan"
            ))
            
            # Recent conversations table
            if recent_conversations:
                table = Table(title="💬 Recent Conversations", box=box.ROUNDED)
                table.add_column("User ID", style="cyan")
                table.add_column("Message", style="white")
                table.add_column("Response", style="green")
                table.add_column("Confidence", style="yellow")
                table.add_column("Time", style="dim")
                
                for conv in recent_conversations:
                    user_id, message, response, confidence, timestamp = conv
                    table.add_row(
                        user_id[:10] + "..." if len(user_id) > 10 else user_id,
                        message[:30] + "..." if len(message) > 30 else message,
                        response[:40] + "..." if len(response) > 40 else response,
                        f"{confidence:.2f}",
                        timestamp[:16] if timestamp else "N/A"
                    )
                
                console.print(table)
            
        except Exception as e:
            console.print(f"[red]❌ Database error: {e}[/red]")
    
    def live_dashboard(self):
        """Live dashboard với auto-refresh"""
        console.print("[green]🔴 Starting live dashboard... Press Ctrl+C to exit[/green]")
        
        def generate_dashboard():
            layout = Layout()
            
            # Get data
            server_stats = self.get_server_stats()
            
            if server_stats:
                # Server status panel
                status_panel = Panel.fit(
                    f"🖥️ Device: [cyan]{server_stats.get('device', 'Unknown')}[/cyan]\n"
                    f"💬 Conversations: [yellow]{server_stats.get('total_conversations', 0)}[/yellow]\n"
                    f"👥 Users: [blue]{server_stats.get('unique_users', 0)}[/blue]\n"
                    f"🧠 Training Sessions: [magenta]{server_stats.get('training_sessions', 0)}[/magenta]\n"
                    f"📈 Model Status: [green]{server_stats.get('model_status', 'Unknown')}[/green]\n"
                    f"🔄 Training: [red]{'Active' if server_stats.get('is_training') else 'Idle'}[/red]\n"
                    f"🎯 Auto Training: [yellow]{server_stats.get('auto_training_threshold', 10)}[/yellow]",
                    title="📊 Server Status",
                    border_style="green"
                )
            else:
                status_panel = Panel.fit(
                    "[red]❌ Server not responding[/red]",
                    title="📊 Server Status",
                    border_style="red"
                )
            
            # Time panel
            time_panel = Panel.fit(
                f"🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                title="⏰ Current Time",
                border_style="blue"
            )
            
            layout.split_column(
                Layout(status_panel, name="status"),
                Layout(time_panel, name="time")
            )
            
            return layout
        
        try:
            with Live(generate_dashboard(), refresh_per_second=1, console=console) as live:
                while True:
                    time.sleep(1)
                    live.update(generate_dashboard())
        except KeyboardInterrupt:
            console.print("\n[yellow]👋 Dashboard stopped[/yellow]")

def main():
    parser = argparse.ArgumentParser(description="Working Zalo AI Management Tool")
    parser.add_argument("command", choices=[
        "health", "stats", "test", "train", "dashboard", "live"
    ], help="Command to execute")
    parser.add_argument("--message", help="Message for testing")
    parser.add_argument("--force", action="store_true", help="Force training")
    parser.add_argument("--server", default="http://localhost:8000", help="Server URL")
    
    args = parser.parse_args()
    
    manager = WorkingZaloAIManager(server_url=args.server)
    
    if args.command == "health":
        manager.check_server_health()
    
    elif args.command == "stats":
        manager.show_database_stats()
    
    elif args.command == "test":
        if not args.message:
            console.print("[red]❌ Please provide --message for testing[/red]")
            sys.exit(1)
        manager.test_ai_response(args.message)
    
    elif args.command == "train":
        manager.trigger_training(force=args.force)
    
    elif args.command == "dashboard":
        manager.show_database_stats()
        manager.check_server_health()
    
    elif args.command == "live":
        manager.live_dashboard()

if __name__ == "__main__":
    main()
