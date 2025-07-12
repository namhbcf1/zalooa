#!/usr/bin/env python3
"""
🚀 AUTO DEPLOY TO CLOUDFLARE - FINAL VERSION
Tự động deploy worker lên Cloudflare Workers
"""

import requests
import json
import subprocess
import tempfile
import os
import time
from datetime import datetime

def deploy_with_wrangler():
    """Deploy using Wrangler CLI"""
    print("🔧 Method 1: Wrangler CLI Deployment")
    
    try:
        # Create wrangler.toml
        wrangler_config = """
name = "zaloapi"
main = "cloudflare_worker_final.js"
compatibility_date = "2023-12-01"

[env.production]
account_id = "5b62d10947844251d23e0eac532531dd"
"""
        
        with open('wrangler.toml', 'w') as f:
            f.write(wrangler_config)
        
        print("📝 Created wrangler.toml")
        
        # Try to install wrangler if not exists
        try:
            subprocess.run(['npm', 'install', '-g', 'wrangler'], 
                          capture_output=True, timeout=60)
            print("📦 Installed Wrangler CLI")
        except:
            print("⚠️ Could not install Wrangler")
        
        # Deploy
        result = subprocess.run(['wrangler', 'deploy'], 
                              capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ Wrangler deployment successful!")
            print(f"Output: {result.stdout}")
            return True
        else:
            print(f"❌ Wrangler error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Wrangler deployment failed: {e}")
        return False

def deploy_with_curl():
    """Deploy using curl command"""
    print("\n🔧 Method 2: Curl Deployment")
    
    try:
        # Read worker code
        with open('cloudflare_worker_final.js', 'r', encoding='utf-8') as f:
            worker_code = f.read()
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False, encoding='utf-8') as f:
            f.write(worker_code)
            temp_file = f.name
        
        # Cloudflare API endpoint
        url = "https://api.cloudflare.com/client/v4/accounts/5b62d10947844251d23e0eac532531dd/workers/scripts/zaloapi"
        
        # Curl command
        curl_cmd = [
            'curl', '-X', 'PUT', url,
            '-H', 'Content-Type: application/javascript',
            '-H', 'User-Agent: Auto-Deploy/1.0',
            '--data-binary', f'@{temp_file}',
            '--max-time', '60',
            '--silent'
        ]
        
        print("🌐 Executing curl deployment...")
        
        result = subprocess.run(curl_cmd, capture_output=True, text=True, timeout=90)
        
        # Clean up temp file
        os.unlink(temp_file)
        
        print(f"📊 Curl exit code: {result.returncode}")
        print(f"📊 Response: {result.stdout[:300]}...")
        
        if result.returncode == 0:
            try:
                response_data = json.loads(result.stdout)
                if response_data.get('success'):
                    print("✅ Curl deployment successful!")
                    return True
                else:
                    print(f"❌ API error: {response_data.get('errors', 'Unknown error')}")
                    return False
            except:
                # Check if response contains success indicators
                if 'success' in result.stdout.lower() or 'deployed' in result.stdout.lower():
                    print("✅ Curl deployment likely successful!")
                    return True
                else:
                    print("❌ Curl deployment failed")
                    return False
        else:
            print(f"❌ Curl command failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Curl deployment failed: {e}")
        return False

def deploy_with_requests():
    """Deploy using Python requests"""
    print("\n🔧 Method 3: Python Requests Deployment")
    
    try:
        # Read worker code
        with open('cloudflare_worker_final.js', 'r', encoding='utf-8') as f:
            worker_code = f.read()
        
        # Cloudflare API endpoint
        url = "https://api.cloudflare.com/client/v4/accounts/5b62d10947844251d23e0eac532531dd/workers/scripts/zaloapi"
        
        headers = {
            'Content-Type': 'application/javascript',
            'User-Agent': 'Auto-Deploy-Python/1.0'
        }
        
        print("📡 Sending deployment request...")
        
        response = requests.put(url, headers=headers, data=worker_code.encode('utf-8'), timeout=60)
        
        print(f"📊 Response status: {response.status_code}")
        print(f"📊 Response: {response.text[:300]}...")
        
        if response.status_code == 200:
            try:
                data = response.json()
                if data.get('success'):
                    print("✅ Python requests deployment successful!")
                    return True
                else:
                    print(f"❌ API error: {data.get('errors', 'Unknown error')}")
                    return False
            except:
                print("✅ Deployment likely successful (non-JSON response)")
                return True
        else:
            print(f"❌ HTTP error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Python requests deployment failed: {e}")
        return False

def verify_deployment():
    """Verify deployment worked"""
    print("\n🔍 VERIFYING DEPLOYMENT...")
    print("=" * 50)
    
    test_url = "https://zaloapi.bangachieu2.workers.dev/"
    
    try:
        print(f"📍 Testing: {test_url}")
        response = requests.get(test_url, timeout=15)
        
        print(f"📊 Status: {response.status_code}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                message = data.get('message', '')
                version = data.get('version', '')
                
                print(f"📊 Message: {message}")
                print(f"📊 Version: {version}")
                
                if "Production Ready" in message:
                    print("✅ NEW WORKER DEPLOYED SUCCESSFULLY!")
                    print("✅ Production version detected!")
                    return True
                elif "TỰ HỌC" in message:
                    print("❌ Still old worker")
                    return False
                else:
                    print("⚠️ Unknown worker version")
                    return False
                    
            except:
                print(f"📊 Raw response: {response.text[:200]}...")
                if "Production Ready" in response.text:
                    print("✅ NEW WORKER DEPLOYED!")
                    return True
                else:
                    print("❌ Still old worker")
                    return False
        else:
            print(f"❌ HTTP error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def main():
    """Main deployment function"""
    print("🚀 CLOUDFLARE AUTO DEPLOYMENT SYSTEM")
    print("=" * 60)
    print(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if worker file exists
    if not os.path.exists('cloudflare_worker_final.js'):
        print("❌ Worker file not found: cloudflare_worker_final.js")
        return False
    
    with open('cloudflare_worker_final.js', 'r', encoding='utf-8') as f:
        code = f.read()
    
    print(f"📊 Worker code: {len(code):,} characters")
    print(f"📊 Worker lines: {len(code.splitlines()):,}")
    
    # Try multiple deployment methods
    deployment_methods = [
        ("Wrangler CLI", deploy_with_wrangler),
        ("Curl Command", deploy_with_curl),
        ("Python Requests", deploy_with_requests)
    ]
    
    deployment_success = False
    
    for method_name, method_func in deployment_methods:
        print(f"\n{'='*60}")
        print(f"🚀 TRYING: {method_name}")
        print(f"{'='*60}")
        
        try:
            success = method_func()
            if success:
                print(f"✅ {method_name} deployment successful!")
                deployment_success = True
                break
            else:
                print(f"❌ {method_name} deployment failed")
        except Exception as e:
            print(f"❌ {method_name} exception: {e}")
    
    # Wait for propagation
    if deployment_success:
        print("\n⏳ Waiting 15 seconds for deployment propagation...")
        time.sleep(15)
    
    # Verify deployment
    verification_success = verify_deployment()
    
    # Final status
    print("\n" + "=" * 60)
    print("🎯 FINAL DEPLOYMENT STATUS")
    print("=" * 60)
    
    if deployment_success and verification_success:
        print("🎉 DEPLOYMENT COMPLETELY SUCCESSFUL!")
        print("✅ New worker is live and verified!")
        print("✅ Zalo OA now uses PC Gaming Database!")
        print("\n🌐 Live URL: https://zaloapi.bangachieu2.workers.dev/")
        print("📊 Dashboard: https://dash.cloudflare.com/5b62d10947844251d23e0eac532531dd/workers/services/view/zaloapi/production/metrics")
        return True
    elif deployment_success:
        print("⚠️ DEPLOYMENT COMPLETED BUT VERIFICATION PENDING")
        print("🔄 Worker may still be propagating...")
        print("🔍 Check manually in 1-2 minutes")
        return True
    else:
        print("❌ ALL AUTO DEPLOYMENT METHODS FAILED")
        print("\n📋 MANUAL DEPLOYMENT REQUIRED:")
        print("1. Go to: https://dash.cloudflare.com/5b62d10947844251d23e0eac532531dd/workers/services/view/zaloapi/production/metrics")
        print("2. Click 'Quick Edit'")
        print("3. Delete all existing code")
        print("4. Copy & paste cloudflare_worker_final.js")
        print("5. Update IP address on line 12")
        print("6. Click 'Save and Deploy'")
        return False

if __name__ == "__main__":
    main()
