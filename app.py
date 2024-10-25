import os
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
from google.cloud import speech_v1p1beta1 as speech
from pydub import AudioSegment
from dotenv import load_dotenv
from openai.error import OpenAIError

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Configurar as chaves das APIs
openai.api_key = os.getenv("OPENAI_API_KEY")
GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")

# Salvar as credenciais do Google Cloud em um arquivo temporário, se fornecido
if GOOGLE_CREDENTIALS_JSON:
    with open("/tmp/credentials.json", "w") as f:
        f.write(GOOGLE_CREDENTIALS_JSON)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/credentials.json"

# Inicializar o app Flask
app = Flask(__name__)
CORS(app)

SUPPORTED_FORMATS = ['audio/webm', 'audio/ogg', 'audio/mpeg', 'audio/wav']

def convert_audio_to_wav(audio_bytes):
    """Converte áudio para WAV e ajusta para 16 bits por amostra."""
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        audio = audio.set_sample_width(2)  # Força para 16 bits por amostra
        audio = audio.set_frame_rate(16000)  # Ajusta para 16kHz

        audio_io = io.BytesIO()
        audio.export(audio_io, format="wav")
        audio_io.seek(0)
        print("Áudio convertido para WAV com taxa de 16kHz e 16 bits.")
        return audio_io
    except Exception as e:
        print(f"Erro na conversão de áudio: {e}")
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

        audio_stream = convert_audio_to_wav(audio_bytes)

        # Configurar o cliente de Speech-to-Text do Google
        client = speech.SpeechClient()
        audio = speech.RecognitionAudio(content=audio_stream.read())
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="pt-BR"
        )

        # Executar a transcrição
        response = client.recognize(config=config, audio=audio)
        transcript = " ".join([result.alternatives[0].transcript for result in response.results])

        return jsonify({"transcricao": transcript})

    except Exception as e:
        print(f"Erro na transcrição: {e}")
        return jsonify({"error": f"Erro inesperado: {e}"}), 500

@app.route('/anamnese', methods=['POST'])
def anamnese_texto():
    data = request.get_json()
    texto = data.get('texto', '')

    if not texto:
        return jsonify({"error": "Nenhum texto de anamnese enviado"}), 400

    try:
        # Chamada para gerar o resumo com OpenAI
        resumo_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Resuma o seguinte texto:"},
                {"role": "user", "content": texto}
            ],
            max_tokens=150
        )
        # Chamada para listar os tópicos principais
        topicos_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Liste os tópicos principais do texto:"},
                {"role": "user", "content": texto}
            ],
            max_tokens=100
        )
        # Chamada para listar exames ou tratamentos sugeridos
        tratamentos_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Liste exames ou medicamentos apropriados:"},
                {"role": "user", "content": texto}
            ],
            max_tokens=100
        )

        # Extrair as respostas da API
        resumo = resumo_response['choices'][0]['message']['content'].strip()
        topicos = topicos_response['choices'][0]['message']['content'].strip()
        tratamentos = tratamentos_response['choices'][0]['message']['content'].strip()

        return jsonify({
            "resumo": resumo,
            "topicos": topicos,
            "tratamentos": tratamentos
        })

    except OpenAIError as e:
        print(f"Erro na API OpenAI: {e}")
        return jsonify({"error": f"Erro na API: {e}"}), 500
    except Exception as e:
        print(f"Erro na anamnese: {e}")
        return jsonify({"error": f"Erro inesperado: {e}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
