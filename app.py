import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
from dotenv import load_dotenv  # Carregar dotenv para carregar variáveis de ambiente
import io

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

app = Flask(__name__)
CORS(app)  # Habilitar CORS

# Configurar a chave da API da OpenAI usando variável de ambiente
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route('/transcrever', methods=['POST'])
def transcrever_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "Nenhum arquivo de áudio enviado"}), 400

    audio_file = request.files['audio']
    try:
        # Ler o arquivo de áudio e criar um objeto BytesIO
        audio_bytes = audio_file.read()
        audio_stream = io.BytesIO(audio_bytes)
        # Definir o atributo 'name' com a extensão apropriada
        audio_stream.name = audio_file.filename or 'audio.webm'

        # Transcrever o áudio usando o modelo Whisper da OpenAI
        transcript = openai.Audio.transcribe("whisper-1", audio_stream)
        return jsonify({"transcricao": transcript['text']})
    except Exception as e:
        # Imprimir o erro nos logs do servidor para depuração
        error_message = str(e)
        print(f"Erro na transcrição: {error_message}")
        return jsonify({"error": error_message}), 500

@app.route('/anamnese', methods=['POST'])
def anamnese_texto():
    data = request.get_json()
    texto = data.get('texto', '')

    if not texto:
        return jsonify({"error": "Nenhum texto de anamnese enviado"}), 400

    try:
        resumo_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Resuma o seguinte texto:"},
                {"role": "user", "content": texto}
            ],
            max_tokens=150
        )
        topicos_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Liste os tópicos principais do seguinte texto:"},
                {"role": "user", "content": texto}
            ],
            max_tokens=100
        )
        resumo = resumo_response['choices'][0]['message']['content'].strip()
        topicos = topicos_response['choices'][0]['message']['content'].strip()
        return jsonify({
            "resumo": resumo,
            "topicos": topicos
        })
    except Exception as e:
        error_message = str(e)
        print(f"Erro na anamnese: {error_message}")
        return jsonify({"error": error_message}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
