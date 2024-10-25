import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
from dotenv import load_dotenv
import io
from pydub import AudioSegment
from openai.error import OpenAIError

# Configurar caminho do ffmpeg
AudioSegment.converter = "/usr/bin/ffmpeg"

# Carregar variáveis de ambiente
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

openai.api_key = os.getenv("OPENAI_API_KEY")

SUPPORTED_FORMATS = ['audio/webm', 'audio/ogg', 'audio/mpeg', 'audio/wav', 'audio/mp4']

def convert_audio(audio_bytes, target_format='wav'):
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        audio_io = io.BytesIO()
        audio.export(audio_io, format=target_format)
        audio_io.seek(0)
        return audio_io
    except Exception as e:
        print(f"Erro na conversão de áudio: {str(e)}")
        raise

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "API ativa e funcionando"}), 200

@app.route('/transcrever', methods=['POST'])
def transcrever_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "Nenhum arquivo de áudio enviado"}), 400

    audio_file = request.files['audio']
    try:
        audio_bytes = audio_file.read()
        if len(audio_bytes) == 0:
            return jsonify({"error": "Arquivo de áudio vazio ou inválido."}), 400

        mime_type = audio_file.mimetype
        print(f"Tipo de arquivo recebido: {mime_type}")

        if mime_type not in SUPPORTED_FORMATS:
            return jsonify({
                "error": f"Formato não suportado: {mime_type}. "
                         f"Formatos suportados: {SUPPORTED_FORMATS}"
            }), 400

        audio_stream = convert_audio(audio_bytes, target_format='wav')
        audio_stream.name = audio_file.filename or 'audio.wav'

        transcript = openai.Audio.transcribe("whisper-1", audio_stream, timeout=60)

        return jsonify({"transcricao": transcript['text']})

    except OpenAIError as e:
        print(f"Erro na API OpenAI: {str(e)}")
        return jsonify({"error": f"Erro na API: {str(e)}"}), 500

    except Exception as e:
        print(f"Erro inesperado: {str(e)}")
        return jsonify({"error": f"Erro inesperado: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
