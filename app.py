import os
import io
import subprocess
from flask import Flask, request, jsonify
from flask_cors import CORS
from pydub import AudioSegment
from google.cloud import speech_v1p1beta1 as speech
import openai
from dotenv import load_dotenv

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Configurar as chaves de API
openai.api_key = os.getenv("OPENAI_API_KEY")
GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")

if GOOGLE_CREDENTIALS_JSON:
    with open("/tmp/credentials.json", "w") as f:
        f.write(GOOGLE_CREDENTIALS_JSON)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/credentials.json"

# Inicializar o app Flask com CORS
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

SUPPORTED_FORMATS = ['audio/webm', 'audio/ogg', 'audio/mpeg', 'audio/wav', 'audio/mp4']

def verificar_ffmpeg():
    """Verifica se o FFmpeg está instalado e disponível."""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, check=True)
        print(f"Versão do FFmpeg detectada:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Erro ao verificar FFmpeg: {e.stderr}")
        raise RuntimeError("FFmpeg não está instalado ou disponível no PATH.")

verificar_ffmpeg()

@app.route('/anamnese', methods=['POST'])
def anamnese_texto():
    data = request.get_json()
    texto = data.get('texto', '')

    if not texto:
        return jsonify({"error": "Nenhum texto de anamnese enviado"}), 400

    try:
        # Nova API de chat do OpenAI
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Resuma o seguinte texto:"},
                {"role": "user", "content": texto}
            ]
        )
        resumo = completion.choices[0].message["content"].strip()

        # Gerar tópicos
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Liste os tópicos principais do texto:"},
                {"role": "user", "content": texto}
            ]
        )
        topicos = completion.choices[0].message["content"].strip()

        # Gerar tratamentos sugeridos
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Liste exames ou medicamentos apropriados:"},
                {"role": "user", "content": texto}
            ]
        )
        tratamentos = completion.choices[0].message["content"].strip()

        return jsonify({
            "resumo": resumo,
            "topicos": topicos,
            "tratamentos": tratamentos
        })

    except openai.OpenAIError as e:
        print(f"Erro na API OpenAI: {str(e)}")
        return jsonify({"error": f"Erro na API: {str(e)}"}), 500

    except Exception as e:
        print(f"Erro inesperado: {str(e)}")
        return jsonify({"error": f"Erro inesperado: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
