python3.11 -m venv venv

venv\Scripts\activate ou source venv/bin/activate

pip install -r requirements.txt

ou

mkdir -p /data/tmp /data/pip-cache

TMPDIR=/data/tmp PIP_CACHE_DIR=/data/pip-cache XDG_CACHE_HOME=/data/.cache \
pip install -r requirements.txt

## Limitations connues

- L'entraînement NER repose sur une classification de tokens IOB : les entités qui se
  chevauchent ou sont imbriquées dans le texte ne peuvent donc pas être encodées.
  Le prétraitement déclenche désormais une erreur explicite si de telles entités sont
  détectées dans le jeu d'entraînement.
