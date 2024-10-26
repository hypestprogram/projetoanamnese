import os
import io
import subprocess
from flask import Flask, request, jsonify
from flask_cors import CORS
from pydub import AudioSegment
from google.cloud import speech_v1p1beta1 as speech
import openai
from dotenv import load_dotenv
from openai.error import APIError, InvalidRequestError

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Configuração da API OpenAI e Google Speech
openai.api_key = os.getenv("OPENAI_API_KEY")
GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")

if GOOGLE_CREDENTIALS_JSON:
    with open("/tmp/credentials.json", "w") as f:
        f.write(GOOGLE_CREDENTIALS_JSON)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/credentials.json"

# Inicializando o Flask
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Formatos de áudio suportados
SUPPORTED_FORMATS = ['audio/webm', 'audio/ogg', 'audio/mpeg', 'audio/wav', 'audio/mp4']

def verificar_ffmpeg():
    """Verifica se o FFmpeg está disponível."""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, check=True)
        print(f"FFmpeg encontrado: {result.stdout.splitlines()[0]}")
    except subprocess.CalledProcessError as e:
        print(f"Erro ao verificar FFmpeg: {e.stderr}")
        raise RuntimeError("FFmpeg não está disponível. Verifique a instalação.")

verificar_ffmpeg()

def convert_audio(audio_bytes, target_format='wav'):
    """Converte áudio para WAV com 16-bit PCM."""
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        audio = audio.set_sample_width(2)  # 16 bits por amostra
        audio_io = io.BytesIO()
        audio.export(audio_io, format=target_format)
        audio_io.seek(0)
        print(f"Áudio convertido para {target_format} com taxa {audio.frame_rate} Hz")
        return audio_io, audio.frame_rate
    except Exception as e:
        print(f"Erro na conversão de áudio: {str(e)}")
        raise

@app.route('/', methods=['GET'])
def health_check():
    """Verifica o status da API."""
    return jsonify({"status": "API ativa e funcionando"}), 200

@app.route('/transcrever', methods=['POST'])
def transcrever_audio():
    """Transcrição de áudio usando Google Speech-to-Text."""
    if 'audio' not in request.files:
        return jsonify({"error": "Nenhum arquivo de áudio enviado."}), 400

    audio_file = request.files['audio']
    try:
        audio_bytes = audio_file.read()
        if not audio_bytes:
            return jsonify({"error": "Arquivo de áudio vazio ou inválido."}), 400

        mime_type = audio_file.mimetype
        print(f"Arquivo recebido: {mime_type}")

        if mime_type not in SUPPORTED_FORMATS:
            return jsonify({
                "error": f"Formato não suportado: {mime_type}. "
                         f"Suportados: {SUPPORTED_FORMATS}"
            }), 400

        audio_stream, sample_rate = convert_audio(audio_bytes)

        # Configurar Google Speech-to-Text
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
        print(f"Erro na transcrição: {str(e)}")
        return jsonify({"error": f"Erro inesperado: {str(e)}"}), 500

@app.route('/anamnese', methods=['POST'])
def anamnese_texto():
    """Gera anamnese utilizando a API OpenAI."""
    data = request.get_json()
    texto = data.get('texto', '')

    if not texto:
        return jsonify({"error": "Texto de anamnese não enviado."}), 400

    try:
        resumo_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Resuma o seguinte texto: {texto}"}],
            max_tokens=150
        )
        resumo = resumo_response['choices'][0]['message']['content'].strip()

        topicos_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Liste os tópicos principais: {texto}"}],
            max_tokens=100
        )
        topicos = topicos_response['choices'][0]['message']['content'].strip()

        tratamentos_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Liste exames ou medicamentos apropriados: {texto}"}],
            max_tokens=100
        )
        tratamentos = tratamentos_response['choices'][0]['message']['content'].strip()

        return jsonify({
            "resumo": resumo,
            "topicos": topicos,
            "tratamentos": tratamentos
        })

    except (APIError, InvalidRequestError) as e:
        print(f"Erro na API OpenAI: {str(e)}")
        return jsonify({"error": f"Erro na API OpenAI: {str(e)}"}), 500

    except Exception as e:
        print(f"Erro inesperado: {str(e)}")
        return jsonify({"error": f"Erro inesperado: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
