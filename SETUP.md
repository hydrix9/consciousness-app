# 🧠 Consciousness App - Setup Guide

## 🔐 Security & Credentials Setup

This project uses hardware APIs that require credentials. **Never commit your actual credentials to git!**

### First Time Setup

1. **Copy the example config files:**
   ```bash
   cp config/app_config.example.yaml config/app_config.yaml
   cp .env.example .env
   ```

2. **Get your Emotiv API credentials:**
   - Sign up at https://www.emotiv.com/
   - Go to https://www.emotiv.com/developer/
   - Create a new application
   - Copy your `client_id` and `client_secret`

3. **Update your config file:**
   Edit `config/app_config.yaml` and replace:
   - `YOUR_CLIENT_ID_HERE` with your actual client ID
   - `YOUR_CLIENT_SECRET_HERE` with your actual client secret
   - `YOUR_LICENSE_KEY_HERE` with your license key (if you have one)

4. **Or use environment variables:**
   Edit `.env` file and add your credentials there instead.

### Important Files (Already in .gitignore)

These files contain sensitive data and are **never committed**:
- `config/app_config.yaml` - Your actual config with credentials
- `config/eeg_config.yaml` - EEG-specific credentials
- `configure_eeg.py` - Script with hardcoded credentials
- `.env` - Environment variables
- `data/*.h5`, `data/*.json` - Your personal session data
- `models/*.pth` - Large trained model files

### What Gets Committed

These template files **are safe to commit**:
- `config/app_config.example.yaml` - Template with placeholder values
- `.env.example` - Template for environment variables
- `README.md` - Documentation
- All source code in `src/`

## 🚀 Installation

See main [README.md](README.md) for full installation and usage instructions.

## ⚠️ Security Checklist

Before pushing to GitHub:
- [ ] Config files contain only placeholder credentials
- [ ] `.env` file is in `.gitignore`
- [ ] No API keys in source code
- [ ] No personal session data in `data/` folder
- [ ] Large model files excluded from git

## 📝 Notes

- The `.gitignore` file is configured to protect your sensitive data
- Example config files use `YOUR_*_HERE` placeholders
- Always use `app_config.example.yaml` as template, never commit `app_config.yaml`
