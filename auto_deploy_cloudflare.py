#!/usr/bin/env python3
"""
🚀 AUTO DEPLOY TO CLOUDFLARE WORKERS
Tự động deploy deep trained worker code lên Cloudflare
"""

import requests
import json
import os
from datetime import datetime

def deploy_to_cloudflare():
    """Deploy worker code to Cloudflare Workers"""
    
    print("🚀 STARTING AUTO DEPLOY TO CLOUDFLARE...")
    
    # Read the deep trained worker code
    try:
        with open('cloudflare_worker_deep_trained.js', 'r', encoding='utf-8') as f:
            worker_code = f.read()
        print("✅ Read cloudflare_worker_deep_trained.js successfully")
    except Exception as e:
        print(f"❌ Error reading worker file: {e}")
        return False
    
    # Cloudflare API credentials (you need to add these)
    CLOUDFLARE_API_TOKEN = "YOUR_API_TOKEN_HERE"  # Get from Cloudflare dashboard
    ACCOUNT_ID = "5b62d10947844251d23e0eac532531dd"
    WORKER_NAME = "zaloapi"
    
    # API endpoint
    url = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/workers/scripts/{WORKER_NAME}"
    
    headers = {
        "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
        "Content-Type": "application/javascript"
    }
    
    print("🌐 Deploying to Cloudflare Workers...")
    print(f"📍 URL: {url}")
    print(f"📝 Code size: {len(worker_code)} characters")
    
    # Deploy the worker
    try:
        response = requests.put(url, headers=headers, data=worker_code)
        
        if response.status_code == 200:
            print("✅ DEPLOYMENT SUCCESSFUL!")
            print("🎉 Deep Trained Worker deployed to Cloudflare!")
            return True
        else:
            print(f"❌ Deployment failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error deploying: {e}")
        return False

def verify_deployment():
    """Verify the deployment worked"""
    print("\n🔍 VERIFYING DEPLOYMENT...")
    
    endpoints_to_test = [
        "https://zaloapi.bangachieu2.workers.dev/",
        "https://zaloapi.bangachieu2.workers.dev/training-info",
        "https://zaloapi.bangachieu2.workers.dev/stats"
    ]
    
    for endpoint in endpoints_to_test:
        try:
            response = requests.get(endpoint, timeout=10)
            print(f"📍 {endpoint}")
            print(f"   Status: {response.status_code}")
            
            if endpoint.endswith('/'):
                # Check if it's the new deep trained response
                if "DEEP TRAINED VERSION" in response.text:
                    print("   ✅ Deep Trained Worker detected!")
                else:
                    print("   ❌ Still old worker")
            elif response.status_code == 200:
                print("   ✅ New endpoint working!")
            else:
                print(f"   ❌ Not working: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n🎯 VERIFICATION COMPLETE")

def manual_deploy_instructions():
    """Show manual deployment instructions"""
    print("\n" + "="*60)
    print("📋 MANUAL DEPLOYMENT INSTRUCTIONS")
    print("="*60)
    
    print("\n🔧 STEP 1: Open Cloudflare Dashboard")
    print("URL: https://dash.cloudflare.com/5b62d10947844251d23e0eac532531dd/workers/services/view/zaloapi/production/metrics")
    
    print("\n🔧 STEP 2: Edit Worker")
    print("1. Click 'Quick Edit' or 'Edit Code'")
    print("2. DELETE ALL existing code")
    print("3. COPY & PASTE code from 'cloudflare_worker_deep_trained.js'")
    
    print("\n🔧 STEP 3: Update IP Address")
    print("Line 11: Change 'http://192.168.1.8:8000' to your actual IP")
    
    print("\n🔧 STEP 4: Deploy")
    print("1. Click 'Save and Deploy'")
    print("2. Wait for deployment to complete")
    
    print("\n🔧 STEP 5: Verify")
    print("Test these URLs:")
    print("- https://zaloapi.bangachieu2.workers.dev/")
    print("- https://zaloapi.bangachieu2.workers.dev/training-info")
    print("- https://zaloapi.bangachieu2.workers.dev/stats")
    
    print("\n✅ Expected Response:")
    print("Main endpoint should return: '🧠 Zalo OA AI Bot - DEEP TRAINED VERSION'")
    print("Training-info should return: Deep training statistics")
    print("Stats should return: Enhanced analytics")

def show_worker_code_preview():
    """Show preview of worker code to deploy"""
    print("\n" + "="*60)
    print("📝 WORKER CODE PREVIEW")
    print("="*60)
    
    try:
        with open('cloudflare_worker_deep_trained.js', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"📊 Total lines: {len(lines)}")
        print(f"📊 File size: {sum(len(line) for line in lines)} characters")
        
        print("\n🔍 First 20 lines:")
        for i, line in enumerate(lines[:20], 1):
            print(f"{i:2d}: {line.rstrip()}")
        
        print("\n🔍 Key features detected:")
        content = ''.join(lines)
        
        features = [
            ("Deep Trained Version", "DEEP TRAINED VERSION" in content),
            ("Training Info Endpoint", "/training-info" in content),
            ("Enhanced Stats", "deep_training_queries" in content),
            ("PC Query Detection", "pcKeywords" in content),
            ("Training Analytics", "training_stats" in content)
        ]
        
        for feature, detected in features:
            status = "✅" if detected else "❌"
            print(f"   {status} {feature}")
            
    except Exception as e:
        print(f"❌ Error reading worker file: {e}")

def main():
    """Main function"""
    print("🧠 CLOUDFLARE DEPLOYMENT CHECKER & DEPLOYER")
    print("=" * 50)
    
    # Show current status
    print("\n📊 CURRENT WORKER STATUS:")
    verify_deployment()
    
    # Show worker code info
    show_worker_code_preview()
    
    # Show manual instructions
    manual_deploy_instructions()
    
    print("\n" + "="*60)
    print("🎯 DEPLOYMENT READY!")
    print("="*60)
    print("✅ Worker code is ready in 'cloudflare_worker_deep_trained.js'")
    print("✅ Manual deployment instructions provided above")
    print("✅ Verification endpoints ready for testing")
    print("\n🚀 DEPLOY NOW USING MANUAL INSTRUCTIONS!")

if __name__ == "__main__":
    main()
