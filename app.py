import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
from io import BytesIO

# Inicializa a aplicação Flask
app = Flask(__name__)
CORS(app)  # Habilita CORS para permitir requisições cross-origin

# Configura a chave da API da OpenAI diretamente no código
openai.api_key = "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"  # Substitua pela sua chave da OpenAI

# Rota para transcrição de áudio
@app.route('/transcrever', methods=['POST'])
def transcrever_audio():
    # Verifica se o áudio foi enviado na requisição
    if 'audio' not in request.files:
        return jsonify({"error": "Nenhum arquivo de áudio foi enviado"}), 400

    # Lê o arquivo de áudio e o converte para bytes
    audio_file = request.files['audio']
    audio_bytes = BytesIO(audio_file.read())

    try:
        # Faz a transcrição do áudio usando a API do Whisper da OpenAI
        transcript = openai.Audio.transcribe("whisper-1", audio_bytes)
        return jsonify({"transcricao": transcript['text']})
    except Exception as e:
        # Retorna uma mensagem de erro caso algo dê errado
        return jsonify({"error": f"Erro ao transcrever o áudio: {str(e)}"}), 500

# Rota para processar texto de anamnese
@app.route('/anamnese', methods=['POST'])
def anamnese_texto():
    # Tenta extrair o texto da requisição JSON
    data = request.json
    texto = data.get('texto', '')

    if not texto:
        return jsonify({"error": "Nenhum texto de anamnese foi enviado"}), 400

    try:
        # Faz um resumo do texto enviado usando a API da OpenAI
        resumo = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"Resuma o seguinte texto: {texto}",
            max_tokens=150
        )
        # Gera tópicos principais do texto enviado
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
        # Retorna uma mensagem de erro caso algo dê errado
        return jsonify({"error": f"Erro ao processar o texto de anamnese: {str(e)}"}), 500

# Inicializa o servidor
if __name__ == '__main__':
    # Executa o app na porta configurada no ambiente ou 5000
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
