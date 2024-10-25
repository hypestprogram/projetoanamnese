import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
from dotenv import load_dotenv
import io

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

app = Flask(__name__)
CORS(app)  # Habilitar CORS

# Configurar a chave da API da OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route('/anamnese', methods=['POST'])
def anamnese_texto():
    data = request.get_json()
    texto = data.get('texto', '')

    if not texto:
        return jsonify({"error": "Nenhum texto de anamnese enviado"}), 400

    try:
        # Resumo
        resumo_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Resuma a seguinte anamnese:"},
                {"role": "user", "content": texto}
            ],
            max_tokens=200
        )

        # Tópicos principais
        topicos_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Liste os tópicos principais encontrados na anamnese:"},
                {"role": "user", "content": texto}
            ],
            max_tokens=100
        )

        # Tratamentos sugeridos
        tratamentos_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Sugira tratamentos e medicamentos adequados para a anamnese. Liste por relevância:"},
                {"role": "user", "content": texto}
            ],
            max_tokens=150
        )

        # Extrair conteúdo das respostas da API
        resumo = resumo_response['choices'][0]['message']['content'].strip()
        topicos = topicos_response['choices'][0]['message']['content'].strip()
        tratamentos = tratamentos_response['choices'][0]['message']['content'].strip()

        # Verificar se as respostas estão vazias e fornecer valores padrão
        topicos = topicos if topicos else "Nenhum tópico identificado"
        tratamentos = tratamentos if tratamentos else "Nenhum tratamento sugerido"

        # Retornar a resposta no formato JSON
        return jsonify({
            "resumo": resumo,
            "topicos": topicos,
            "tratamentos": tratamentos
        })

    except Exception as e:
        error_message = str(e)
        print(f"Erro na anamnese: {error_message}")
        return jsonify({"error": error_message}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
