import os
import io
import openai
from flask import Flask, request, jsonify
from flask_cors import CORS
from pydub import AudioSegment
from google.cloud import speech_v1p1beta1 as speech
from dotenv import load_dotenv

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Inicializar o app Flask
app = Flask(__name__)
CORS(app)

SUPPORTED_FORMATS = ['audio/webm', 'audio/ogg', 'audio/mpeg', 'audio/wav']

def convert_audio(audio_bytes, target_format='wav'):
    """Converte áudio para WAV e garante amostras de 16 bits."""
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        audio = audio.set_sample_width(2)  # Define para 16-bit PCM
        audio_io = io.BytesIO()
        audio.export(audio_io, format=target_format)
        sample_rate = audio.frame_rate
        audio_io.seek(0)

        print(f"Áudio convertido para: {target_format} com taxa de {sample_rate} Hz")
        return audio_io, sample_rate
    except Exception as e:
        print(f"Erro na conversão de áudio: {str(e)}")
        raise

@app.route('/transcrever', methods=['POST'])
def transcrever_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "Nenhum arquivo de áudio enviado"}), 400

    audio_file = request.files['audio']
    try:
        audio_bytes = audio_file.read()
        mime_type = audio_file.mimetype

        if mime_type not in SUPPORTED_FORMATS:
            return jsonify({"error": f"Formato não suportado: {mime_type}."}), 400

        audio_stream, sample_rate = convert_audio(audio_bytes)

        client = speech.SpeechClient()
        audio = speech.RecognitionAudio(content=audio_stream.read())
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            language_code="pt-BR"
        )

        response = client.recognize(config=config, audio=audio)
        transcript = " ".join([result.alternatives[0].transcript for result in response.results])

        return jsonify({"transcricao": transcript})
    except Exception as e:
        print(f"Erro inesperado: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/anamnese', methods=['POST'])
def anamnese_texto():
    data = request.get_json()
    texto = data.get('texto', '')

    if not texto:
        return jsonify({"error": "Nenhum texto de anamnese enviado"}), 400

    try:
        resumo_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "Resuma o seguinte texto:"}, {"role": "user", "content": texto}],
            max_tokens=150
        )
        resumo = resumo_response['choices'][0]['message']['content'].strip()

        return jsonify({"resumo": resumo})
    except Exception as e:
        print(f"Erro na anamnese: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
