@app.route('/anamnese', methods=['POST'])
def anamnese_texto():
    data = request.get_json()
    texto = data.get('texto', '')

    if not texto:
        return jsonify({"error": "Nenhum texto de anamnese enviado"}), 400

    try:
        # Fazer a chamada correta para ChatCompletion
        resumo_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "Resuma o seguinte texto:"}, 
                      {"role": "user", "content": texto}],
            max_tokens=150
        )
        topicos_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "Liste os tópicos principais do texto:"}, 
                      {"role": "user", "content": texto}],
            max_tokens=100
        )
        tratamentos_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "Liste exames ou medicamentos apropriados:"}, 
                      {"role": "user", "content": texto}],
            max_tokens=100
        )

        resumo = resumo_response['choices'][0]['message']['content'].strip()
        topicos = topicos_response['choices'][0]['message']['content'].strip()
        tratamentos = tratamentos_response['choices'][0]['message']['content'].strip()

        return jsonify({
            "resumo": resumo,
            "topicos": topicos,
            "tratamentos": tratamentos
        })

    except openai.OpenAIError as e:
        print(f"Erro na API OpenAI: {str(e)}")
        return jsonify({"error": f"Erro na API: {str(e)}"}), 500

    except Exception as e:
        print(f"Erro inesperado: {str(e)}")
        return jsonify({"error": f"Erro inesperado: {str(e)}"}), 500
