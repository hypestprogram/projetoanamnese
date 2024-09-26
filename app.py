import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
from io import BytesIO
from dotenv import load_dotenv

# Carregar vari�veis de ambiente
load_dotenv()

app = Flask(__name__)
CORS(app)  # Habilitar CORS

# Configurar a chave da API da OpenAI
openai.api_key = os.getenv("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")

@app.route('/transcrever', methods=['POST'])
def transcrever_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "Nenhum arquivo de �udio enviado"}), 400

    audio_file = request.files['audio']
    audio_bytes = BytesIO(audio_file.read())
    
    try:
        transcript = openai.Audio.transcribe("whisper-1", audio_bytes)
        return jsonify({"transcricao": transcript['text']})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/anamnese', methods=['POST'])
def anamnese_texto():
    data = request.json
    texto = data.get('texto', '')

    if not texto:
        return jsonify({"error": "Nenhum texto de anamnese enviado"}), 400

    try:
        resumo = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"Resuma o seguinte texto: {texto}",
            max_tokens=150
        )
        topicos = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"Liste os t�picos principais: {texto}",
            max_tokens=100
        )
        return {
            "resumo": resumo['choices'][0]['text'].strip(),
            "topicos": topicos['choices'][0]['text'].strip()
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
