import os
import io
import json
import subprocess
from flask import Flask, request, jsonify
from flask_cors import CORS
from pydub import AudioSegment
from google.cloud import speech_v1p1beta1 as speech
import openai

# Carregar variáveis de ambiente e configurar as credenciais do Google Cloud
GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Escreve o JSON de credenciais em um arquivo temporário
if GOOGLE_CREDENTIALS_JSON:
    with open("/tmp/credentials.json", "w") as f:
        f.write(GOOGLE_CREDENTIALS_JSON)

# Configurar a variável de ambiente para apontar para o arquivo temporário
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/credentials.json"

# Inicializar o app Flask
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Inicializar a API da OpenAI
openai.api_key = OPENAI_API_KEY

# Formatos suportados
SUPPORTED_FORMATS = ['audio/webm', 'audio/ogg', 'audio/mpeg', 'audio/wav', 'audio/mp4']

def verificar_ffmpeg():
    """Verifica se o FFmpeg está instalado e disponível."""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, check=True)
        print(f"Versão do FFmpeg: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Erro ao verificar FFmpeg: {e.stderr}")

# Verificar o FFmpeg no início do serviço
verificar_ffmpeg()

def convert_audio(audio_bytes, target_format='wav'):
    """Converte áudio para o formato especificado (WAV por padrão)."""
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        audio_io = io.BytesIO()
        audio.export(audio_io, format=target_format)
        audio_io.seek(0)
        print(f"Áudio convertido para: {target_format}")
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

        # Converter o áudio para WAV
        audio_stream = convert_audio(audio_bytes, target_format='wav')
        audio_content = audio_stream.read()

        # Configurar o cliente de Speech-to-Text do Google Cloud
        client = speech.SpeechClient()

        # Configurar a requisição de reconhecimento de áudio
        audio = speech.RecognitionAudio(content=audio_content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="pt-BR"
        )

        # Realizar a transcrição
        response = client.recognize(config=config, audio=audio)

        # Extrair o texto transcrito
        transcript = " ".join([result.alternatives[0].transcript for result in response.results])
        print(f"Texto transcrito: {transcript}")

        # Enviar o texto transcrito para a API da OpenAI
        openai_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Você é um assistente que ajuda a processar transcrições."},
                {"role": "user", "content": transcript}
            ],
            max_tokens=150
        )

        # Extrair a resposta da OpenAI
        processed_text = openai_response['choices'][0]['message']['content'].strip()

        return jsonify({
            "transcricao": transcript,
            "processado": processed_text
        })

    except Exception as e:
        print(f"Erro inesperado: {str(e)}")
        return jsonify({"error": f"Erro inesperado: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
