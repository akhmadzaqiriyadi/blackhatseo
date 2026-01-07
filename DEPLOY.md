# BlackHat SEO Detector - VPS Deployment Guide

## 📋 Prasyarat di VPS

Pastikan VPS sudah terinstall:

```bash
# Update sistem
sudo apt update && sudo apt upgrade -y

# Install Python 3.11+
sudo apt install python3 python3-pip python3-venv -y

# Install PM2 (Node.js process manager)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install nodejs -y
sudo npm install -g pm2

# Install Git
sudo apt install git -y

# (Opsional) Install Nginx sebagai reverse proxy
sudo apt install nginx -y
```

## 🚀 Cara Deploy

### 1. Edit Konfigurasi deploy.sh

Edit file `deploy.sh` dan ubah:
- `SERVER` = IP VPS kamu
- `USER` = Username SSH
- `KEY_PATH` = Path ke SSH private key

### 2. Jalankan Deploy

```bash
# Berikan permission executable
chmod +x deploy.sh

# Jalankan deploy
./deploy.sh
```

## 🌐 Setup Nginx (Opsional tapi Direkomendasikan)

Buat file konfigurasi Nginx di VPS:

```bash
sudo nano /etc/nginx/sites-available/blackhatseo
```

Isi dengan:

```nginx
server {
    listen 80;
    server_name your-domain.com;  # Atau IP VPS

    location / {
        proxy_pass http://127.0.0.1:5001;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        
        # Timeout untuk scraping yang lama
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
}
```

Aktifkan site:

```bash
sudo ln -s /etc/nginx/sites-available/blackhatseo /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## 🔒 Setup SSL dengan Let's Encrypt

```bash
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d your-domain.com
```

## 📊 Monitoring

```bash
# Lihat status PM2
pm2 status

# Lihat logs
pm2 logs blackhatseo-api

# Restart aplikasi
pm2 restart blackhatseo-api

# Stop aplikasi
pm2 stop blackhatseo-api
```

## 🧪 Test API

```bash
# Health check
curl http://YOUR_VPS_IP:5001/api/health

# Predict URL
curl -X POST http://YOUR_VPS_IP:5001/api/predict \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://ugm.ac.id"]}'
```

## 🔧 Troubleshooting

### Port 5001 tidak bisa diakses

```bash
# Buka firewall
sudo ufw allow 5001/tcp
sudo ufw reload
```

### Model tidak ditemukan

```bash
cd /home/$USER/blackhatseo
source venv/bin/activate
python3 -m src.main train --train-urls data/train_data_balanced_corrected.csv \
    --model-output models/model.pkl \
    --vectorizer-output models/vectorizer.pkl
```

### Dependency error

```bash
cd /home/$USER/blackhatseo
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```
