import os
import io
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from pydub import AudioSegment
import openai
from google.cloud import speech_v1p1beta1 as speech
from openai.error import OpenAIError

# Configuração do OpenAI e Google Speech
openai.api_key = os.getenv("OPENAI_API_KEY")

# Salvar credenciais do Google Cloud em /tmp no servidor
GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
if GOOGLE_CREDENTIALS_JSON:
    with open("/tmp/credentials.json", "w") as f:
        f.write(GOOGLE_CREDENTIALS_JSON)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/credentials.json"

# Inicialização do Flask
app = Flask(__name__)
CORS(app)

SUPPORTED_FORMATS = ['audio/webm', 'audio/ogg', 'audio/mpeg', 'audio/wav']

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "API ativa e funcionando"}), 200

# Função para converter áudio para WAV
def convert_audio(audio_bytes):
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    audio = audio.set_sample_width(2)  # 16 bits por amostra
    audio_io = io.BytesIO()
    audio.export(audio_io, format="wav")
    audio_io.seek(0)
    return audio_io, audio.frame_rate

# Endpoint para transcrever áudio
@app.route('/transcrever', methods=['POST'])
def transcrever_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "Nenhum arquivo de áudio enviado"}), 400

    audio_file = request.files['audio']
    audio_bytes = audio_file.read()

    if not audio_bytes:
        return jsonify({"error": "Áudio inválido ou vazio"}), 400

    try:
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
        return jsonify({"error": str(e)}), 500

# Endpoint para anamnese usando OpenAI
@app.route('/anamnese', methods=['POST'])
def anamnese_texto():
    data = request.get_json()
    texto = data.get('texto', '')

    if not texto:
        return jsonify({"error": "Texto de anamnese não enviado"}), 400

    try:
        resumo = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Resuma o seguinte texto: {texto}"}],
            max_tokens=150
        )
        topicos = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Liste os tópicos principais: {texto}"}],
            max_tokens=100
        )
        tratamentos = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Liste exames e medicamentos apropriados: {texto}"}],
            max_tokens=100
        )

        return jsonify({
            "resumo": resumo.choices[0].message['content'].strip(),
            "topicos": topicos.choices[0].message['content'].strip(),
            "tratamentos": tratamentos.choices[0].message['content'].strip()
        })

    except OpenAIError as e:
        return jsonify({"error": f"Erro na API OpenAI: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Erro inesperado: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
