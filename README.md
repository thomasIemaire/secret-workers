python3.11 -m venv venv

venv\Scripts\activate ou source venv/bin/activate

pip install -r requirements.txt

ou 

mkdir -p /data/tmp /data/pip-cache

# utilise le gros disque pour les fichiers temporaires et le cache
env TMPDIR=/data/tmp PIP_CACHE_DIR=/data/pip-cache \
    pip install --no-cache-dir -r requirements.txt