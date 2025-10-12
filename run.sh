#!/usr/bin/env bash
# ============================================================
#  Local Dash App Launcher for macOS / Linux
#  Creates virtual environment, installs dependencies, runs app
# ============================================================

set -e

echo "🚀 Starting setup for Dash dashboard..."

# 1. Create and activate virtual environment
if [ ! -d ".venv" ]; then
  echo "🔧 Creating virtual environment..."
  python3 -m venv .venv
fi
source .venv/bin/activate

# 2. Install / update dependencies
echo "📦 Installing Python packages..."
pip install -U pip
pip install -r requirements.txt

# 3. Run the app (disable hot reload to avoid reload loops)
export DEV_TOOLS_HOT_RELOAD=False
export DASH_IGNORE_FILES="student_hook_local.*"

echo "✅ Setup complete. Launching the dashboard..."
python app_core.py

