#!/bin/bash

# ==========================================
# 🔧 KONFIGURASI SERVER & PROYEK
# ==========================================
SERVER="YOUR_VPS_IP"           # Ganti dengan IP VPS kamu
USER="YOUR_USER"               # Ganti dengan username VPS
KEY_PATH="$HOME/.ssh/id_rsa"   # Path ke SSH key
PROJECT_DIR="/home/$USER/blackhatseo"
APP_NAME="blackhatseo-api"
PORT=5001

# Warna untuk output
GREEN='\033[0;32m'
CYAN='\033[0;36m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${CYAN}🚀 Memulai Deploy BlackHat SEO Detector ke VPS ($SERVER)...${NC}"

# Cek koneksi ke server
if ! ping -c 1 $SERVER &> /dev/null; then
    echo -e "${RED}❌ Gagal konek ke $SERVER.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Koneksi ke server berhasil${NC}"

ssh -i "$KEY_PATH" "$USER@$SERVER" << 'EOF'
    set -e
    
    echo "📂 Setup direktori project..."
    PROJECT_DIR="/home/$USER/blackhatseo"
    APP_NAME="blackhatseo-api"
    PORT=5001
    
    # Clone atau pull repo
    if [ ! -d "$PROJECT_DIR" ]; then
        echo "📥 Cloning repository..."
        git clone https://github.com/akhmadzaqiriyadi/blackhatseo.git "$PROJECT_DIR"
    else
        echo "⬇️  Pull update..."
        cd "$PROJECT_DIR"
        git pull origin main
    fi
    
    cd "$PROJECT_DIR"
    
    # Setup Python virtual environment
    echo "🐍 Setup Python virtual environment..."
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi
    source venv/bin/activate
    
    # Install dependencies
    echo "📦 Install Dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    
    # Train model jika belum ada
    echo "🤖 Cek model..."
    if [ ! -f "models/model.pkl" ] || [ ! -f "models/vectorizer.pkl" ]; then
        echo "🏋️ Training model..."
        python3 -m src.main train --train-urls data/train_data_balanced_corrected.csv \
            --model-output models/model.pkl \
            --vectorizer-output models/vectorizer.pkl
    else
        echo "✅ Model sudah ada"
    fi
    
    # Buat ecosystem.config.cjs untuk PM2
    echo "⚙️  Buat PM2 ecosystem config..."
    cat > ecosystem.config.cjs <<INNER_EOF
module.exports = {
  apps: [{
    name: '$APP_NAME',
    script: 'venv/bin/python',
    args: 'app.py',
    cwd: '$PROJECT_DIR',
    instances: 1,
    exec_mode: 'fork',
    env: {
      NODE_ENV: 'production',
      FLASK_ENV: 'production',
      PORT: $PORT
    },
    merge_logs: true,
    autorestart: true,
    max_restarts: 10,
    restart_delay: 1000
  }]
};
INNER_EOF

    # Restart PM2
    echo "🔄 Restart PM2..."
    pm2 delete $APP_NAME 2>/dev/null || true
    pm2 start ecosystem.config.cjs
    pm2 save
    
    # Tampilkan status
    echo ""
    echo "📊 Status PM2:"
    pm2 status
    
    echo ""
    echo "✅ Deploy selesai!"
    echo "🌐 API berjalan di: http://$(hostname -I | awk '{print $1}'):$PORT"
    echo "📌 Health check: curl http://localhost:$PORT/api/health"
EOF

echo -e "${GREEN}✅ Deploy ke VPS selesai!${NC}"
