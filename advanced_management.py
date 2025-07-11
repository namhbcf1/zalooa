# advanced_management.py - Advanced management tool cho auto learning system
import asyncio
import argparse
import requests
import json
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import time
import sys
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout

console = Console()

class AdvancedZaloAIManager:
    def __init__(self, server_url="http://localhost:8000", db_path="conversations.db", knowledge_db="computer_knowledge.db"):
        self.server_url = server_url
        self.db_path = db_path
        self.knowledge_db = knowledge_db
    
    def check_server_health(self):
        """Kiểm tra health của server"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                
                console.print(Panel.fit(
                    f"🟢 Server Status: [green]{data['status']}[/green]\n"
                    f"Learning Active: [blue]{data['learning_active']}[/blue]\n"
                    f"Total Knowledge: [yellow]{data['total_knowledge']}[/yellow]\n"
                    f"Last Check: [dim]{data['timestamp']}[/dim]",
                    title="Server Health"
                ))
                return True
            else:
                console.print(f"[red]❌ Server error: {response.status_code}[/red]")
                return False
        except Exception as e:
            console.print(f"[red]❌ Cannot connect to server: {e}[/red]")
            return False
    
    def get_learning_dashboard(self):
        """Hiển thị dashboard learning real-time"""
        try:
            # Get learning status
            response = requests.get(f"{self.server_url}/learning-status")
            learning_data = response.json()
            
            # Get knowledge stats
            response = requests.get(f"{self.server_url}/knowledge-stats")
            knowledge_data = response.json()
            
            # Create rich dashboard
            layout = Layout()
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="body"),
                Layout(name="footer", size=3)
            )
            
            layout["body"].split_row(
                Layout(name="left"),
                Layout(name="right")
            )
            
            # Header
            header_text = f"🤖 AI Auto Learning Dashboard - {datetime.now().strftime('%H:%M:%S')}"
            layout["header"].update(Panel(header_text, style="bold blue"))
            
            # Left panel - Learning Status
            learning_status = "🔥 ACTIVE" if learning_data['is_active'] else "💤 IDLE"
            left_content = f"""
[bold]Learning Status:[/bold] {learning_status}
[bold]Total Knowledge:[/bold] {learning_data['total_knowledge']:,}
[bold]Sources:[/bold] {len(learning_data['stats']['by_source'])}
[bold]Languages:[/bold] {', '.join(learning_data['stats']['by_language'].keys())}

[bold]Top Topics:[/bold]
"""
            for topic, count in list(learning_data['stats']['by_topic'].items())[:5]:
                left_content += f"  • {topic}: {count}\n"
            
            layout["left"].update(Panel(left_content, title="Learning Status"))
            
            # Right panel - Recent Activity
            right_content = "[bold]Recent Learning Sessions:[/bold]\n"
            for session in knowledge_data['recent_learning_sessions'][:5]:
                right_content += f"  • {session['type']}: {session['items']} items\n"
            
            right_content += "\n[bold]Most Used Knowledge:[/bold]\n"
            for knowledge in knowledge_data['top_used_knowledge'][:5]:
                question = knowledge['question'][:50] + "..." if len(knowledge['question']) > 50 else knowledge['question']
                right_content += f"  • {question} ({knowledge['usage_count']} uses)\n"
            
            layout["right"].update(Panel(right_content, title="Activity"))
            
            # Footer
            footer_text = f"📊 Press Ctrl+C to exit | Server: {self.server_url}"
            layout["footer"].update(Panel(footer_text, style="dim"))
            
            return layout
            
        except Exception as e:
            console.print(f"[red]Error creating dashboard: {e}[/red]")
            return None
    
    def live_dashboard(self):
        """Live dashboard với auto refresh"""
        console.print("[green]Starting live dashboard... Press Ctrl+C to exit[/green]")
        
        try:
            with Live(self.get_learning_dashboard(), refresh_per_second=1, screen=True) as live:
                while True:
                    time.sleep(2)
                    new_layout = self.get_learning_dashboard()
                    if new_layout:
                        live.update(new_layout)
        except KeyboardInterrupt:
            console.print("\n[yellow]Dashboard stopped[/yellow]")
    
    def trigger_learning(self, learning_type="manual"):
        """Trigger learning session"""
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
            ) as progress:
                task = progress.add_task("Starting learning session...", total=None)
                
                response = requests.post(f"{self.server_url}/manual-learning")
                result = response.json()
                
                progress.update(task, description=f"Learning status: {result['status']}")
                time.sleep(2)
                
                # Monitor learning progress
                while True:
                    status_response = requests.get(f"{self.server_url}/learning-status")
                    status_data = status_response.json()
                    
                    if status_data['is_active']:
                        progress.update(task, description="Learning in progress...")
                        time.sleep(5)
                    else:
                        progress.update(task, description="Learning completed!")
                        break
                
            console.print("[green]✅ Learning session completed![/green]")
            
        except Exception as e:
            console.print(f"[red]❌ Error triggering learning: {e}[/red]")
    
    def search_knowledge(self, query, limit=10):
        """Tìm kiếm trong knowledge base"""
        try:
            params = {"query": query, "limit": limit}
            response = requests.get(f"{self.server_url}/search-knowledge", params=params)
            data = response.json()
            
            if data['results']:
                table = Table(title=f"Search Results for: '{query}'")
                table.add_column("Question", style="cyan")
                table.add_column("Similarity", style="green")
                table.add_column("Answer Preview", style="dim")
                
                for result in data['results']:
                    question = result['question'][:60] + "..." if len(result['question']) > 60 else result['question']
                    answer = result['answer'][:80] + "..." if len(result['answer']) > 80 else result['answer']
                    similarity = f"{result['similarity']:.2f}"
                    
                    table.add_row(question, similarity, answer)
                
                console.print(table)
            else:
                console.print(f"[yellow]No results found for '{query}'[/yellow]")
                
        except Exception as e:
            console.print(f"[red]Error searching: {e}[/red]")
    
    def test_ai_advanced(self, message):
        """Test AI với thông tin chi tiết"""
        try:
            payload = {
                "user_id": "test_user",
                "message": message,
                "timestamp": int(datetime.now().timestamp() * 1000)
            }
            
            start_time = time.time()
            response = requests.post(f"{self.server_url}/process-message", json=payload)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                # Create response panel
                response_content = f"""
[bold cyan]User:[/bold cyan] {message}

[bold green]AI Response:[/bold green]
{data['response']}

[bold]Metadata:[/bold]
• Confidence: {data['confidence']:.2f}
• Source: {data['source']}
• Learning Status: {data['learning_status']}
• Response Time: {response_time:.2f}s
"""
                
                console.print(Panel(response_content, title="AI Test Result", border_style="green"))
            else:
                console.print(f"[red]Error: {response.status_code}[/red]")
                
        except Exception as e:
            console.print(f"[red]Error testing AI: {e}[/red]")
    
    def add_knowledge_interactive(self):
        """Thêm knowledge theo interactive mode"""
        try:
            console.print("[bold blue]Add Knowledge Interactive Mode[/bold blue]")
            
            question = console.input("Enter question: ")
            answer = console.input("Enter answer: ")
            topic = console.input("Enter topic (optional): ") or "general"
            
            response = requests.post(
                f"{self.server_url}/add-knowledge",
                params={"question": question, "answer": answer, "topic": topic}
            )
            
            if response.status_code == 200:
                data = response.json()
                console.print(f"[green]✅ Added {data['items_added']} knowledge item(s)[/green]")
            else:
                console.print(f"[red]❌ Error adding knowledge: {response.status_code}[/red]")
                
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    def optimize_system(self):
        """Tối ưu hóa hệ thống"""
        try:
            with Progress() as progress:
                task1 = progress.add_task("Optimizing knowledge base...", total=100)
                
                # Optimize knowledge
                response = requests.post(f"{self.server_url}/optimize-knowledge")
                progress.update(task1, advance=50)
                
                if response.status_code == 200:
                    data = response.json()
                    progress.update(task1, advance=50)
                    
                    console.print(Panel(
                        f"[green]✅ Optimization completed![/green]\n"
                        f"• Deleted duplicates: {data['deleted_duplicates']}\n"
                        f"• Deleted low quality: {data['deleted_low_quality']}",
                        title="Optimization Results"
                    ))
                else:
                    console.print(f"[red]❌ Optimization failed: {response.status_code}[/red]")
                    
        except Exception as e:
            console.print(f"[red]Error optimizing: {e}[/red]")
    
    def generate_advanced_report(self):
        """Tạo báo cáo nâng cao"""
        try:
            console.print("[blue]Generating advanced report...[/blue]")
            
            # Get comprehensive stats
            response = requests.get(f"{self.server_url}/knowledge-stats")
            data = response.json()
            
            # Create visualizations
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle('Zalo OA AI - Advanced Analytics Report', fontsize=16, fontweight='bold')
            
            # 1. Knowledge by source
            sources = data['overview']['by_source']
            if sources:
                axes[0,0].pie(sources.values(), labels=sources.keys(), autopct='%1.1f%%')
                axes[0,0].set_title('Knowledge by Source')
            
            # 2. Knowledge by topic
            topics = data['overview']['by_topic']
            if topics:
                topic_items = list(topics.items())[:10]
                axes[0,1].barh([item[0] for item in topic_items], [item[1] for item in topic_items])
                axes[0,1].set_title('Top 10 Topics')
                axes[0,1].set_xlabel('Count')
            
            # 3. Usage distribution
            if data['top_used_knowledge']:
                usage_counts = [item['usage_count'] for item in data['top_used_knowledge']]
                axes[0,2].hist(usage_counts, bins=10, alpha=0.7)
                axes[0,2].set_title('Usage Distribution')
                axes[0,2].set_xlabel('Usage Count')
            
            # 4. Learning sessions over time
            if data['recent_learning_sessions']:
                sessions_df = pd.DataFrame(data['recent_learning_sessions'])
                sessions_df['timestamp'] = pd.to_datetime(sessions_df['timestamp'])
                sessions_by_day = sessions_df.groupby(sessions_df['timestamp'].dt.date)['items'].sum()
                
                axes[1,0].plot(sessions_by_day.index, sessions_by_day.values, marker='o')
                axes[1,0].set_title('Learning Progress Over Time')
                axes[1,0].set_ylabel('Items Learned')
                axes[1,0].tick_params(axis='x', rotation=45)
            
            # 5. Language distribution
            languages = data['overview']['by_language']
            if languages:
                axes[1,1].bar(languages.keys(), languages.values())
                axes[1,1].set_title('Knowledge by Language')
                axes[1,1].set_ylabel('Count')
            
            # 6. System performance summary
            total_knowledge = data['overview']['total_knowledge_items']
            total_sessions = data['overview']['total_learning_sessions']
            
            performance_text = f"""
Total Knowledge Items: {total_knowledge:,}
Total Learning Sessions: {total_sessions}
Average Items/Session: {total_knowledge/max(total_sessions, 1):.1f}
Active Sources: {len(data['overview']['by_source'])}
Active Topics: {len(data['overview']['by_topic'])}
            """
            
            axes[1,2].text(0.1, 0.5, performance_text, fontsize=12, 
                          verticalalignment='center', transform=axes[1,2].transAxes)
            axes[1,2].set_title('System Summary')
            axes[1,2].axis('off')
            
            plt.tight_layout()
            plt.savefig('advanced_ai_report.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Generate HTML report
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Zalo OA AI - Advanced Analytics Report</title>
                <style>
                    body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
                    .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                              color: white; padding: 30px; border-radius: 10px; text-align: center; }}
                    .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
                                  gap: 20px; margin: 30px 0; }}
                    .stat-card {{ background: white; padding: 25px; border-radius: 10px; 
                                 box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; }}
                    .stat-number {{ font-size: 2.5em; font-weight: bold; color: #667eea; }}
                    .chart-container {{ background: white; padding: 30px; border-radius: 10px; 
                                       box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 20px 0; }}
                    .table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                    .table th, .table td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                    .table th {{ background: #f8f9fa; font-weight: 600; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🤖 Zalo OA AI - Advanced Analytics Report</h1>
                    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-number">{total_knowledge:,}</div>
                        <p>Total Knowledge Items</p>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{total_sessions}</div>
                        <p>Learning Sessions</p>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{len(data['overview']['by_source'])}</div>
                        <p>Active Sources</p>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{len(data['overview']['by_topic'])}</div>
                        <p>Topics Covered</p>
                    </div>
                </div>
                
                <div class="chart-container">
                    <h2>📊 Analytics Visualization</h2>
                    <img src="advanced_ai_report.png" style="width: 100%; max-width: 1200px;">
                </div>
                
                <div class="chart-container">
                    <h2>🔥 Most Used Knowledge</h2>
                    <table class="table">
                        <thead>
                            <tr><th>Question</th><th>Usage Count</th></tr>
                        </thead>
                        <tbody>
            """
            
            for item in data['top_used_knowledge'][:10]:
                html_content += f"""
                            <tr>
                                <td>{item['question'][:100]}...</td>
                                <td>{item['usage_count']}</td>
                            </tr>
                """
            
            html_content += """
                        </tbody>
                    </table>
                </div>
                
                <div class="chart-container">
                    <h2>📈 Recent Learning Activity</h2>
                    <table class="table">
                        <thead>
                            <tr><th>Session Type</th><th>Items Learned</th><th>Timestamp</th></tr>
                        </thead>
                        <tbody>
            """
            
            for session in data['recent_learning_sessions'][:10]:
                html_content += f"""
                            <tr>
                                <td>{session['type']}</td>
                                <td>{session['items']}</td>
                                <td>{session['timestamp']}</td>
                            </tr>
                """
            
            html_content += """
                        </tbody>
                    </table>
                </div>
            </body>
            </html>
            """
            
            with open('advanced_ai_report.html', 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            console.print("[green]✅ Advanced report generated: advanced_ai_report.html[/green]")
            
        except Exception as e:
            console.print(f"[red]Error generating report: {e}[/red]")

def main():
    parser = argparse.ArgumentParser(description='Advanced Zalo OA AI Management Tool')
    parser.add_argument('action', choices=[
        'health', 'dashboard', 'learn', 'search', 'test', 'add-knowledge',
        'optimize', 'report', 'live'
    ])
    parser.add_argument('--query', help='Search query or test message')
    parser.add_argument('--limit', type=int, default=10, help='Limit for search results')
    parser.add_argument('--server', default='http://localhost:8000', help='Server URL')
    
    args = parser.parse_args()
    
    manager = AdvancedZaloAIManager(server_url=args.server)
    
    if args.action == 'health':
        manager.check_server_health()
    elif args.action == 'dashboard':
        dashboard = manager.get_learning_dashboard()
        if dashboard:
            console.print(dashboard)
    elif args.action == 'live':
        manager.live_dashboard()
    elif args.action == 'learn':
        manager.trigger_learning()
    elif args.action == 'search':
        if not args.query:
            console.print("[red]--query required for search[/red]")
            return
        manager.search_knowledge(args.query, args.limit)
    elif args.action == 'test':
        if not args.query:
            console.print("[red]--query required for test[/red]")
            return
        manager.test_ai_advanced(args.query)
    elif args.action == 'add-knowledge':
        manager.add_knowledge_interactive()
    elif args.action == 'optimize':
        manager.optimize_system()
    elif args.action == 'report':
        manager.generate_advanced_report()

if __name__ == "__main__":
    main()