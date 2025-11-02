#!/usr/bin/env python3
"""
Security Check Script - Run before pushing to GitHub
Verifies no sensitive data is in the git repository
"""

import subprocess
import sys
import re

def run_git_command(cmd):
    """Run a git command and return output"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd="."
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return 1, "", str(e)

def check_sensitive_files():
    """Check if sensitive files are tracked"""
    print("🔍 Checking for sensitive files...")
    
    sensitive_patterns = [
        "app_config.yaml",
        "eeg_config.yaml",
        "configure_eeg.py",
        r"\.env$",
        r".*secret.*",
        r".*\.log$",
        r"data/.*\.h5$",
        r"data/.*\.json$",
        r"models/.*\.pth$"
    ]
    
    issues = []
    
    for pattern in sensitive_patterns:
        code, stdout, _ = run_git_command(f'git ls-files | grep -E "{pattern}"')
        if stdout.strip():
            issues.append(f"❌ FOUND: {pattern}")
            print(f"   Found files matching: {pattern}")
            for line in stdout.strip().split('\n'):
                print(f"      {line}")
    
    if not issues:
        print("✅ No sensitive files found in repository")
        return True
    else:
        print("\n⚠️  SENSITIVE FILES DETECTED!")
        return False

def check_secret_content():
    """Check for hardcoded secrets in content"""
    print("\n🔍 Checking for secret content...")
    
    # Known secret patterns (partial, for detection)
    secret_patterns = [
        "6rWkJx8PJUz1",  # From configure_eeg.py
        "4dhU7ZO60BFB",  # Client ID
        "XusOebdM72vH",  # Another client ID
        "Zr4R60IO4czF",  # Another secret
    ]
    
    issues = []
    
    for pattern in secret_patterns:
        code, stdout, _ = run_git_command(f'git grep -i "{pattern}"')
        if code == 0 and stdout.strip():
            # Exclude example files, documentation, and the security check script itself
            lines = stdout.strip().split('\n')
            real_issues = [
                line for line in lines 
                if 'example' not in line.lower() 
                and '.md' not in line.lower()
                and 'security_check.py' not in line.lower()
            ]
            if real_issues:
                issues.append(f"❌ FOUND SECRET: {pattern[:10]}...")
                for line in real_issues:
                    print(f"   {line}")
    
    if not issues:
        print("✅ No hardcoded secrets found")
        return True
    else:
        print("\n⚠️  HARDCODED SECRETS DETECTED!")
        return False

def check_example_files():
    """Verify example files have placeholders"""
    print("\n🔍 Checking example files...")
    
    example_files = [
        ("config/app_config.example.yaml", ["YOUR_CLIENT_ID_HERE", "YOUR_CLIENT_SECRET_HERE"]),
        (".env.example", ["your_client_id_here", "your_client_secret_here"])
    ]
    
    all_good = True
    
    for filename, required_placeholders in example_files:
        code, stdout, _ = run_git_command(f'git show HEAD:{filename}')
        if code != 0:
            print(f"⚠️  Could not find {filename}")
            all_good = False
            continue
        
        for placeholder in required_placeholders:
            if placeholder in stdout:
                print(f"✅ {filename} has placeholder: {placeholder}")
            else:
                print(f"❌ {filename} missing placeholder: {placeholder}")
                all_good = False
    
    return all_good

def check_gitignore():
    """Verify .gitignore is comprehensive"""
    print("\n🔍 Checking .gitignore...")
    
    required_patterns = [
        "config/app_config.yaml",
        "configure_eeg.py",
        ".env",
        "*.log",
        "data/*.h5",
        "models/**/*.pth"
    ]
    
    code, stdout, _ = run_git_command('git show HEAD:.gitignore')
    
    if code != 0:
        print("❌ .gitignore not found!")
        return False
    
    all_good = True
    for pattern in required_patterns:
        if pattern in stdout:
            print(f"✅ .gitignore includes: {pattern}")
        else:
            print(f"⚠️  .gitignore missing: {pattern}")
            all_good = False
    
    return all_good

def main():
    """Run all security checks"""
    print("=" * 60)
    print("🔐 GitHub Security Check")
    print("=" * 60)
    
    checks = [
        ("Sensitive Files", check_sensitive_files),
        ("Secret Content", check_secret_content),
        ("Example Files", check_example_files),
        (".gitignore", check_gitignore)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Error checking {name}: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("📊 RESULTS")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("✅ ALL CHECKS PASSED!")
        print("🚀 Safe to push to GitHub")
        return 0
    else:
        print("❌ SOME CHECKS FAILED!")
        print("⚠️  DO NOT PUSH until issues are fixed")
        print("\nTo fix:")
        print("1. Review failed checks above")
        print("2. Update .gitignore if needed")
        print("3. Remove sensitive files: git rm --cached <file>")
        print("4. Re-run this script")
        return 1

if __name__ == "__main__":
    sys.exit(main())
