import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
from dotenv import load_dotenv
import io
from openai.error import OpenAIError

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

app = Flask(__name__)
CORS(app)  # Habilitar CORS

openai.api_key = os.getenv("OPENAI_API_KEY")

SUPPORTED_FORMATS = ['audio/webm', 'audio/ogg', 'audio/mpeg', 'audio/wav']

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "API ativa e funcionando"}), 200

# Endpoint para transcrição de áudio
@app.route('/transcrever', methods=['POST'])
def transcrever_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "Nenhum arquivo de áudio enviado"}), 400

    audio_file = request.files['audio']
    try:
        audio_bytes = audio_file.read()
        if len(audio_bytes) == 0:
            return jsonify({"error": "Arquivo de áudio vazio ou inválido."}), 400

        audio_stream = io.BytesIO(audio_bytes)
        mime_type = audio_file.mimetype
        print(f"Tipo de arquivo recebido: {mime_type}")

        if mime_type not in SUPPORTED_FORMATS:
            return jsonify({
                "error": f"Formato de arquivo não suportado: {mime_type}. "
                         f"Formatos suportados: {SUPPORTED_FORMATS}"
            }), 400

        audio_stream.name = audio_file.filename or 'audio.webm'
        transcript = openai.Audio.transcribe("whisper-1", audio_stream, timeout=15)
        return jsonify({"transcricao": transcript['text']})

    except OpenAIError as e:
        print(f"Erro na API OpenAI: {str(e)}")
        return jsonify({"error": f"Erro na API: {str(e)}"}), 500
    except Exception as e:
        print(f"Erro inesperado: {str(e)}")
        return jsonify({"error": f"Erro inesperado: {str(e)}"}), 500

# Endpoint para processar o texto de anamnese
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
        topicos_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "Liste os tópicos principais do texto:"}, {"role": "user", "content": texto}],
            max_tokens=100
        )
        tratamentos_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "Liste exames ou medicamentos apropriados:"}, {"role": "user", "content": texto}],
            max_tokens=100
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
