import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import assemblyai as aai
from io import BytesIO

app = Flask(__name__)
CORS(app)

# Configurar a chave da API da AssemblyAI diretamente no código
aai.settings.api_key = "065ae935676a4a3db93a4d1ac0d6b1c6"  # Substitua pela sua chave da API

@app.route('/transcrever', methods=['POST'])
def transcrever_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "Nenhum arquivo de áudio enviado"}), 400

    audio_file = request.files['audio']
    audio_bytes = BytesIO(audio_file.read())
    
    try:
        # Enviar o áudio para transcrição usando AssemblyAI
        transcript = aai.Transcriber().transcribe(audio_bytes)
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
        # Usar AssemblyAI ou qualquer outro modelo para gerar resumo e tópicos
        resumo = aai.Completion.create(
            model="text-davinci-003",
            prompt=f"Resuma o seguinte texto: {texto}",
            max_tokens=150
        )
        topicos = aai.Completion.create(
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
