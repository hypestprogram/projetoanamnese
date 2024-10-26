import openai
from openai.error import APIError, InvalidRequestError

@app.route('/anamnese', methods=['POST'])
def anamnese_texto():
    data = request.get_json()
    texto = data.get('texto', '')

    if not texto:
        return jsonify({"error": "Texto de anamnese não enviado"}), 400

    try:
        # Chamada para resumir o texto
        resumo_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Resuma o seguinte texto: {texto}"}],
            max_tokens=150
        )
        resumo = resumo_response['choices'][0]['message']['content'].strip()

        # Chamada para listar tópicos principais
        topicos_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Liste os tópicos principais: {texto}"}],
            max_tokens=100
        )
        topicos = topicos_response['choices'][0]['message']['content'].strip()

        # Chamada para listar exames e medicamentos
        tratamentos_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Liste exames ou medicamentos apropriados: {texto}"}],
            max_tokens=100
        )
        tratamentos = tratamentos_response['choices'][0]['message']['content'].strip()

        return jsonify({
            "resumo": resumo,
            "topicos": topicos,
            "tratamentos": tratamentos
        })

    except (APIError, InvalidRequestError) as e:
        print(f"Erro na API OpenAI: {str(e)}")
        return jsonify({"error": f"Erro na API OpenAI: {str(e)}"}), 500

    except Exception as e:
        print(f"Erro inesperado: {str(e)}")
        return jsonify({"error": f"Erro inesperado: {str(e)}"}), 500
