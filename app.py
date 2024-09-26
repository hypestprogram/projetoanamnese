import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
from io import BytesIO

app = Flask(__name__)
CORS(app)  # Habilitar CORS

# Configurar a chave da API da OpenAI diretamente no código
openai.api_key = "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"

@app.route('/transcrever', methods=['POST'])
def transcrever_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "Nenhum arquivo de áudio enviado"}), 400

    audio_file = request.files['audio']
    audio_bytes = BytesIO(audio_file.read())
    audio_bytes.name = "audio.wav"  # Defina um nome fictício para o arquivo

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
            prompt=f"Liste os tópicos principais do seguinte texto: {texto}",
            max_tokens=100
        )
        return {
            "resumo": resumo['choices'][0]['text'].strip(),
            "topicos": topicos['choices'][0]['text'].strip()
        }
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
