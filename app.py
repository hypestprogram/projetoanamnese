import os
import io
import json
import subprocess
from flask import Flask, request, jsonify
from flask_cors import CORS
from pydub import AudioSegment
from google.cloud import speech_v1p1beta1 as speech
from dotenv import load_dotenv
import openai

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Configurar chaves de API
openai.api_key = os.getenv("OPENAI_API_KEY")
GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")

if GOOGLE_CREDENTIALS_JSON:
    with open("/tmp/credentials.json", "w") as f:
        f.write(GOOGLE_CREDENTIALS_JSON)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/credentials.json"

# Inicializar o app Flask
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

SUPPORTED_FORMATS = ['audio/webm', 'audio/ogg', 'audio/mpeg', 'audio/wav', 'audio/mp4']

def verificar_ffmpeg():
    """Verifica se o FFmpeg está instalado e disponível."""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, check=True)
        print(f"Versão do FFmpeg: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Erro ao verificar FFmpeg: {e.stderr}")

verificar_ffmpeg()

def convert_audio(audio_bytes, target_format='wav'):
    """Converte áudio para WAV e ajusta para 16-bit PCM."""
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        audio = audio.set_sample_width(2)  # 2 bytes = 16 bits por amostra
        sample_rate = audio.frame_rate

        audio_io = io.BytesIO()
        audio.export(audio_io, format=target_format)
        audio_io.seek(0)

        print(f"Áudio convertido para: {target_format} com taxa de {sample_rate} Hz")
        return audio_io, sample_rate
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
                "error": f"Formato não suportado: {mime_type}. Formatos suportados: {SUPPORTED_FORMATS}"
            }), 400

        # Converter áudio e detectar taxa de amostragem
        audio_stream, sample_rate = convert_audio(audio_bytes)

        # Configurar o cliente do Google Speech-to-Text
        client = speech.SpeechClient()
        audio = speech.RecognitionAudio(content=audio_stream.read())
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            language_code="pt-BR"
        )

        # Realizar a transcrição
        response = client.recognize(config=config, audio=audio)
        transcript = " ".join([result.alternatives[0].transcript for result in response.results])

        return jsonify({"transcricao": transcript})

    except Exception as e:
        print(f"Erro na transcrição: {str(e)}")
        return jsonify({"error": f"Erro inesperado: {str(e)}"}), 500

@app.route('/anamnese', methods=['POST'])
def anamnese_texto():
    data = request.get_json()
    texto = data.get('texto', '')

    if not texto:
        return jsonify({"error": "Nenhum texto de anamnese enviado"}), 400

    try:
        resumo_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "Organize e resuma o seguinte texto em no máximo 150 tokens, "
                "focando nas principais seções da anamnese:"
                "\n1. Identificação do paciente (iniciais, idade, sexo)"
                "\n2. Queixa principal e duração dos sintomas"
                "\n3. História da doença atual (início, evolução, fatores agravantes ou de alívio, sintomas associados)"
                "\n4. Histórico médico, cirúrgico e medicamentoso"
                "\n5. Histórico familiar e social (doenças hereditárias, hábitos de vida)"
                "\n6. Exame físico (sinais vitais e achados relevantes)"
                "\n7. Hipóteses diagnósticas e plano terapêutico" }, {"role": "user", "content": texto}],
            max_tokens=150
        )
        topicos_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "Liste os principais tópicos identificados na anamnese em no maximo 150 tokens , incluindo se houver:"
                "\n- Queixa principal"
                "\n- Evolução dos sintomas"
                "\n- Fatores agravantes e de alívio"
                "\n- Histórico médico e familiar"
                "\n- Achados do exame físico"
                "\n- Hipóteses diagnósticas e plano terapêutico"}, {"role": "user", "content": texto}],
            max_tokens=150
        )
        tratamentos_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "Com base nas informações fornecidas, sugira um plano diagnóstico e terapêutico adequado para o paciente em no maximo 200 tokens. "
                "Inclua as seguintes seções:"
                "\n- Hipóteses Diagnósticas: Liste possíveis diagnósticos diferenciais."
                "\n- Exames Complementares Solicitados: Informe quais exames são necessários."
                "\n- Medicações Prescritas: Liste as medicações recomendadas."
                "\n- Orientações ao Paciente: Descreva orientações e recomendações ao paciente."
                "\n- Seguimento e Reavaliação: Informe sobre o plano de seguimento e necessidade de reavaliações futuras."}, {"role": "user", "content": texto}],
            max_tokens=200
        )

        resumo = resumo_response['choices'][0]['message']['content'].strip()
        topicos = topicos_response['choices'][0]['message']['content'].strip()
        tratamentos = tratamentos_response['choices'][0]['message']['content'].strip()

        return jsonify({"resumo": resumo, "topicos": topicos, "tratamentos": tratamentos})

    except Exception as e:
        print(f"Erro na anamnese: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
