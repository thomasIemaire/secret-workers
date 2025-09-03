python3.11 -m venv venv

venv\Scripts\activate ou source venv/bin/activate

pip install -r requirements.txt

ou 

mkdir -p /data/tmp /data/pip-cache

TMPDIR=/data/tmp PIP_CACHE_DIR=/data/pip-cache XDG_CACHE_HOME=/data/.cache \
pip install -r requirements.txt
