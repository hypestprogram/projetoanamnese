import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
from io import BytesIO

app = Flask(__name__)
CORS(app)  # Habilitar CORS

# Configurar a chave da API da OpenAI diretamente no código
openai.api_key = "sk-gQ6-dduUm-nd_DTcw0_f9L7j_MEY3EMD6wrFBxpEvoT3BlbkFJFpV5DhAdK89dMX0AjCkSxKuvpK-ButPlZN5vOwAMYA"

@app.route('/transcrever', methods=['POST'])
def transcrever_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "Nenhum arquivo de áudio enviado"}), 400

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
            prompt=f"Liste os tópicos principais do seguinte texto: {texto}",
            max_tokens=100
        )
        return jsonify({
            "resumo": resumo['choices'][0]['text'].strip(),
            "topicos": topicos['choices'][0]['text'].strip()  # Adiciona '.strip()' para remover espaços extras
        })
    except Exception as e:
        return jsonify({"error": f"Erro ao processar o texto de anamnese: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
