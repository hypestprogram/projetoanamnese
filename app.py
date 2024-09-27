import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import io

load_dotenv()

app = Flask(__name__)
CORS(app)

openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route('/transcrever', methods=['POST'])
def transcrever_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "Nenhum arquivo de áudio enviado"}), 400

    audio_file = request.files['audio']
    try:
        audio_bytes = audio_file.read()
        audio_stream = io.BytesIO(audio_bytes)
        audio_stream.name = audio_file.filename

        # O Whisper precisa de um arquivo com a extensão correta para funcionar
        if not audio_stream.name.endswith(('.webm', '.wav', '.ogg', '.mp3')):
            return jsonify({"error": "Formato de arquivo não suportado"}), 400

        transcript = openai.Audio.transcribe("whisper-1", audio_stream)
        return jsonify({"transcricao": transcript['text']})
    except Exception as e:
        print(f"Erro na transcrição: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
