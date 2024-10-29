web: apt-get update && apt-get install -y ffmpeg && gunicorn app:app --workers 4 --bind 0.0.0.0:$PORT --log-level debug
