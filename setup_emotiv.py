#!/usr/bin/env python3
"""
Emotiv EEG Setup Assistant
==========================

This script helps you configure your Emotiv EEG device for the Consciousness App.

Steps to get your credentials:
1. Go to https://www.emotiv.com/developer/
2. Create a developer account or log in
3. Create a new application to get your client_id and client_secret
4. Get your license key from your Emotiv account

Once you have the credentials, run this script to set them up.
"""

import os
import json
import yaml

def setup_emotiv_credentials():
    """Interactive setup for Emotiv credentials"""
    print("🧠 Emotiv EEG Credentials Setup")
    print("=" * 40)
    print()
    
    print("Before proceeding, make sure you have:")
    print("1. ✅ Emotiv Pro or EPOC Connect software installed and running")
    print("2. ✅ Your EEG headset connected and working in Emotiv software")
    print("3. ✅ Valid Emotiv API credentials (see instructions above)")
    print()
    
    setup_choice = input("Do you have your Emotiv API credentials ready? (y/n): ").lower().strip()
    
    if setup_choice != 'y':
        print("\n📋 To get your credentials:")
        print("1. Visit: https://www.emotiv.com/developer/")
        print("2. Create a developer account")
        print("3. Create a new application")
        print("4. Copy your Client ID, Client Secret, and License Key")
        print("\nRun this script again when you have your credentials.")
        return False
    
    print("\n🔐 Enter your Emotiv credentials:")
    client_id = input("Client ID: ").strip()
    client_secret = input("Client Secret: ").strip()
    license_key = input("License Key: ").strip()
    
    if not all([client_id, client_secret, license_key]):
        print("❌ Error: All credentials are required!")
        return False
    
    # Create environment variables setup
    env_content = f"""# Emotiv EEG Credentials
# Add these to your environment variables or run this script
set EMOTIV_CLIENT_ID={client_id}
set EMOTIV_CLIENT_SECRET={client_secret}
set EMOTIV_LICENSE_KEY={license_key}

# For PowerShell:
# $env:EMOTIV_CLIENT_ID="{client_id}"
# $env:EMOTIV_CLIENT_SECRET="{client_secret}"
# $env:EMOTIV_LICENSE_KEY="{license_key}"
"""
    
    with open("emotiv_credentials.bat", "w") as f:
        f.write(env_content)
    
    # Update config files
    try:
        # Update app_config.yaml
        with open("config/app_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        config["hardware"]["emotiv"]["client_id"] = client_id
        config["hardware"]["emotiv"]["client_secret"] = client_secret
        config["hardware"]["emotiv"]["license"] = license_key
        
        with open("config/app_config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Update eeg_config.yaml
        with open("config/eeg_config.yaml", "r") as f:
            eeg_config = yaml.safe_load(f)
        
        eeg_config["cortex"]["client_id"] = client_id
        eeg_config["cortex"]["client_secret"] = client_secret
        eeg_config["cortex"]["license_key"] = license_key
        
        with open("config/eeg_config.yaml", "w") as f:
            yaml.dump(eeg_config, f, default_flow_style=False)
            
        print("\n✅ Credentials saved successfully!")
        print(f"📁 Configuration updated in config files")
        print(f"📄 Environment variables saved to: emotiv_credentials.bat")
        
    except Exception as e:
        print(f"❌ Error updating config files: {e}")
        return False
    
    return True

def test_emotiv_connection():
    """Test connection to Emotiv Cortex"""
    print("\n🔍 Testing Emotiv connection...")
    
    import subprocess
    try:
        # Run a quick test
        result = subprocess.run([
            "python", "run.py", "--eeg-source", "cortex", "--test-rng", "--debug"
        ], capture_output=True, text=True, timeout=30)
        
        if "Successfully connected and streaming from cortex" in result.stdout:
            print("✅ EEG connection successful!")
            return True
        elif "Missing Cortex credentials" in result.stdout:
            print("❌ Credentials still not working. Check your API keys.")
            return False
        else:
            print("⚠️  Connection test completed. Check the output above.")
            print("If you see 'cortex' in the connection info, it's working!")
            return True
            
    except subprocess.TimeoutExpired:
        print("⚠️  Test timed out. You may need to manually verify the connection.")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🧠 Emotiv EEG Setup for Consciousness App")
    print("==========================================")
    
    # Check if Cortex is running
    print("\n🔍 Checking if Emotiv Cortex service is running...")
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', 6868))
        sock.close()
        
        if result == 0:
            print("✅ Emotiv Cortex service is running on port 6868")
        else:
            print("❌ Emotiv Cortex service not detected!")
            print("Please start Emotiv Pro or EPOC Connect software first.")
            return
            
    except Exception as e:
        print(f"❌ Error checking Cortex service: {e}")
        return
    
    # Setup credentials
    if setup_emotiv_credentials():
        print("\n🧪 Testing connection...")
        test_emotiv_connection()
        
        print("\n🎉 Setup Complete!")
        print("\nNext steps:")
        print("1. Test with: python run.py --eeg-source cortex --test-rng")
        print("2. Use real EEG in Oracle: python oracle_369_launcher.py")
        print("3. Try inference mode: python run.py --mode inference --test-rng")
        
    else:
        print("\n❌ Setup failed. Please try again with valid credentials.")

if __name__ == "__main__":
    main()