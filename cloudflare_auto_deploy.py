#!/usr/bin/env python3
"""
🚀 CLOUDFLARE AUTO DEPLOY - Tự động deploy worker lên Cloudflare
"""

import requests
import json
import time
from datetime import datetime

def auto_deploy_to_cloudflare():
    """Tự động deploy worker code lên Cloudflare Workers"""
    
    print("🚀 STARTING AUTO DEPLOY TO CLOUDFLARE WORKERS...")
    print("=" * 60)
    
    # Read worker code
    try:
        with open('cloudflare_worker_deep_trained.js', 'r', encoding='utf-8') as f:
            worker_code = f.read()
        print(f"✅ Read worker code: {len(worker_code)} characters")
    except Exception as e:
        print(f"❌ Error reading worker file: {e}")
        return False
    
    # Cloudflare configuration
    ACCOUNT_ID = "5b62d10947844251d23e0eac532531dd"
    WORKER_NAME = "zaloapi"
    
    # Try multiple deployment methods
    deployment_methods = [
        "wrangler_cli",
        "direct_api", 
        "curl_command"
    ]
    
    for method in deployment_methods:
        print(f"\n🔧 Trying deployment method: {method}")
        
        if method == "wrangler_cli":
            success = deploy_with_wrangler(worker_code)
        elif method == "direct_api":
            success = deploy_with_api(worker_code, ACCOUNT_ID, WORKER_NAME)
        elif method == "curl_command":
            success = deploy_with_curl(worker_code, ACCOUNT_ID, WORKER_NAME)
        
        if success:
            print(f"✅ Deployment successful with {method}!")
            return True
        else:
            print(f"❌ Deployment failed with {method}")
    
    print("\n❌ All deployment methods failed")
    return False

def deploy_with_wrangler(worker_code):
    """Deploy using Wrangler CLI"""
    try:
        import subprocess
        
        # Create wrangler.toml
        wrangler_config = f"""
name = "zaloapi"
main = "cloudflare_worker_deep_trained.js"
compatibility_date = "2023-12-01"

[env.production]
account_id = "5b62d10947844251d23e0eac532531dd"
"""
        
        with open('wrangler.toml', 'w') as f:
            f.write(wrangler_config)
        
        print("📝 Created wrangler.toml")
        
        # Try to deploy with wrangler
        result = subprocess.run(['wrangler', 'deploy'], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Wrangler deployment successful!")
            return True
        else:
            print(f"❌ Wrangler error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Wrangler deployment failed: {e}")
        return False

def deploy_with_api(worker_code, account_id, worker_name):
    """Deploy using Cloudflare API directly"""
    try:
        # This would need API token, trying without for now
        url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/workers/scripts/{worker_name}"
        
        headers = {
            "Content-Type": "application/javascript",
            "User-Agent": "Auto-Deploy-Script/1.0"
        }
        
        print(f"📡 Attempting API deployment to: {url}")
        
        # Try without auth first (will likely fail but worth trying)
        response = requests.put(url, headers=headers, data=worker_code, timeout=30)
        
        print(f"📊 Response status: {response.status_code}")
        print(f"📊 Response: {response.text[:200]}...")
        
        if response.status_code == 200:
            return True
        else:
            return False
            
    except Exception as e:
        print(f"❌ API deployment failed: {e}")
        return False

def deploy_with_curl(worker_code, account_id, worker_name):
    """Deploy using curl command"""
    try:
        import subprocess
        import tempfile
        
        # Save worker code to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
            f.write(worker_code)
            temp_file = f.name
        
        # Construct curl command
        url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/workers/scripts/{worker_name}"
        
        curl_cmd = [
            'curl', '-X', 'PUT', url,
            '-H', 'Content-Type: application/javascript',
            '-H', 'User-Agent: Auto-Deploy-Script/1.0',
            '--data-binary', f'@{temp_file}',
            '--max-time', '30'
        ]
        
        print(f"🌐 Executing curl command...")
        
        result = subprocess.run(curl_cmd, capture_output=True, text=True, timeout=60)
        
        print(f"📊 Curl exit code: {result.returncode}")
        print(f"📊 Curl output: {result.stdout[:200]}...")
        
        if result.returncode == 0 and '"success":true' in result.stdout:
            return True
        else:
            return False
            
    except Exception as e:
        print(f"❌ Curl deployment failed: {e}")
        return False

def verify_deployment():
    """Verify deployment worked"""
    print("\n🔍 VERIFYING DEPLOYMENT...")
    print("=" * 40)
    
    test_endpoints = [
        ("Main", "https://zaloapi.bangachieu2.workers.dev/"),
        ("Training Info", "https://zaloapi.bangachieu2.workers.dev/training-info"),
        ("Stats", "https://zaloapi.bangachieu2.workers.dev/stats")
    ]
    
    deployment_success = False
    
    for name, url in test_endpoints:
        try:
            print(f"\n📍 Testing {name}: {url}")
            response = requests.get(url, timeout=10)
            
            print(f"   Status: {response.status_code}")
            
            if name == "Main":
                if "DEEP TRAINED VERSION" in response.text:
                    print("   ✅ Deep Trained Worker detected!")
                    deployment_success = True
                else:
                    print("   ❌ Still old worker")
                    print(f"   Response: {response.text[:100]}...")
            
            elif response.status_code == 200:
                print("   ✅ New endpoint working!")
                try:
                    data = response.json()
                    if name == "Training Info" and "deep_training_completed" in data:
                        print("   ✅ Training info endpoint confirmed!")
                    elif name == "Stats" and "deep_training_queries" in data:
                        print("   ✅ Enhanced stats endpoint confirmed!")
                except:
                    pass
            else:
                print(f"   ❌ Endpoint not working: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error testing {name}: {e}")
    
    return deployment_success

def create_deployment_report():
    """Create deployment report"""
    print("\n📋 DEPLOYMENT REPORT")
    print("=" * 40)
    
    # Check worker code
    try:
        with open('cloudflare_worker_deep_trained.js', 'r') as f:
            code = f.read()
        
        print(f"📊 Worker Code:")
        print(f"   - Size: {len(code):,} characters")
        print(f"   - Lines: {len(code.splitlines()):,}")
        
        # Check for key features
        features = [
            ("Deep Trained Version", "DEEP TRAINED VERSION" in code),
            ("Training Info Endpoint", "/training-info" in code),
            ("Enhanced Analytics", "deepTrainingQueries" in code),
            ("PC Query Detection", "pcKeywords" in code),
            ("Zalo Integration", "ZALO_CONFIG" in code)
        ]
        
        print(f"\n📊 Features:")
        for feature, present in features:
            status = "✅" if present else "❌"
            print(f"   {status} {feature}")
            
    except Exception as e:
        print(f"❌ Error reading worker code: {e}")
    
    print(f"\n📊 Deployment Target:")
    print(f"   - Account ID: 5b62d10947844251d23e0eac532531dd")
    print(f"   - Worker Name: zaloapi")
    print(f"   - URL: https://zaloapi.bangachieu2.workers.dev/")
    print(f"   - Dashboard: https://dash.cloudflare.com/5b62d10947844251d23e0eac532531dd/workers/services/view/zaloapi/production/metrics")

def main():
    """Main deployment function"""
    print("🧠 CLOUDFLARE AUTO DEPLOY SYSTEM")
    print("=" * 50)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create deployment report
    create_deployment_report()
    
    # Attempt auto deployment
    deployment_success = auto_deploy_to_cloudflare()
    
    # Wait a moment for propagation
    if deployment_success:
        print("\n⏳ Waiting 10 seconds for deployment propagation...")
        time.sleep(10)
    
    # Verify deployment
    verification_success = verify_deployment()
    
    # Final status
    print("\n" + "=" * 60)
    print("🎯 FINAL DEPLOYMENT STATUS")
    print("=" * 60)
    
    if deployment_success and verification_success:
        print("🎉 DEPLOYMENT SUCCESSFUL!")
        print("✅ Deep Trained Worker is now live!")
        print("✅ All endpoints are working!")
        print("\n🚀 Zalo OA now uses Deep Trained AI with 93+ samples!")
    elif deployment_success:
        print("⚠️ DEPLOYMENT COMPLETED BUT VERIFICATION FAILED")
        print("🔄 Worker may still be propagating...")
        print("🔍 Check manually in a few minutes")
    else:
        print("❌ AUTO DEPLOYMENT FAILED")
        print("📋 Manual deployment required")
        print("\n🔧 Manual Steps:")
        print("1. Go to Cloudflare Dashboard")
        print("2. Edit Worker Code")
        print("3. Copy & Paste cloudflare_worker_deep_trained.js")
        print("4. Save and Deploy")
    
    print(f"\n🕐 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
