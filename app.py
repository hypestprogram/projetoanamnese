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

# Configurar a chave da API da OpenAI usando variável de ambiente
openai.api_key = os.getenv("OPENAI_API_KEY")

# Função para validar texto da anamnese
def validar_texto(texto):
    if not texto or len(texto.strip()) < 10:
        return "Texto da anamnese muito curto ou vazio."
    return None

# Endpoint para processar o texto de anamnese
@app.route('/anamnese', methods=['POST'])
def anamnese_texto():
    data = request.get_json()
    texto = data.get('texto', '')

    # Validar texto
    error = validar_texto(texto)
    if error:
        return jsonify({"error": error}), 400

    try:
        # Novo prompt para maior precisão e relevância
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": (
                    "Você é um assistente médico especializado. Com base na anamnese fornecida, "
                    "sua tarefa é realizar as seguintes ações:"
                    "\n1. Resuma a anamnese em até 5 linhas."
                    "\n2. Destaque possíveis fatores de risco (ex.: tabagismo, hipertensão)."
                    "\n3. Sugira tratamentos e medicamentos apropriados, listados por ordem de relevância."
                    "\n4. Caso algum sintoma exija ação imediata, adicione o alerta: 'ATENÇÃO: Urgente'."
                )},
                {"role": "user", "content": texto}
            ],
            max_tokens=500
        )

        # Extração das partes da resposta
        resposta = response['choices'][0]['message']['content'].strip()
        partes = resposta.split("\n\n")

        resumo = partes[0] if len(partes) > 0 else "Resumo não disponível"
        fatores_risco = partes[1] if len(partes) > 1 else "Nenhum fator de risco identificado"
        tratamentos = partes[2] if len(partes) > 2 else "Nenhum tratamento sugerido"
        alerta = partes[3] if len(partes) > 3 else ""  # Alerta se houver

        return jsonify({
            "resumo": resumo,
            "fatores_de_risco": fatores_risco,
            "tratamentos": tratamentos,
            "alerta": alerta
        })

    except Exception as e:
        # Tratamento de erro e log para depuração
        print(f"Erro na anamnese: {str(e)}")
        return jsonify({"error": "Erro ao processar a anamnese. Tente novamente mais tarde."}), 500

# Inicializar o servidor
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
