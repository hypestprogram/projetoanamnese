# Endpoint para processar o texto de anamnese médica
@app.route('/anamnese', methods=['POST'])
def anamnese_texto():
    data = request.get_json()
    texto = data.get('texto', '')

    if not texto:
        return jsonify({"error": "Nenhum texto de anamnese enviado"}), 400

    try:
        # Resumo clínico da anamnese
        resumo_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Elabore um resumo conciso da anamnese abaixo, destacando sintomas principais, sinais relevantes, condições preexistentes e possíveis fatores de risco. O resumo deve ser claro e objetivo, com até 5 linhas:"},
                {"role": "user", "content": texto}
            ],
            max_tokens=200
        )

        # Identificação de tópicos clínicos e fatores de risco
        topicos_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Identifique e liste os principais tópicos clínicos encontrados na anamnese abaixo, incluindo sintomas, sinais de alerta, histórico familiar relevante, condições preexistentes e fatores de risco. Organize os tópicos por ordem de relevância clínica:"},
                {"role": "user", "content": texto}
            ],
            max_tokens=100
        )

        # Sugestões de tratamento e medicamentos
        tratamentos_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Com base na anamnese abaixo, sugira tratamentos e medicamentos adequados, priorizando aqueles que se alinham com os sintomas apresentados e o histórico do paciente. Inclua uma breve justificativa clínica (entre parênteses). Se houver sinais que indiquem urgência, adicione o alerta: 'ATENÇÃO: Urgente - Encaminhar imediatamente para atendimento emergencial.' Liste os tratamentos por ordem de relevância clínica:"},
                {"role": "user", "content": texto}
            ],
            max_tokens=150
        )

        # Extração das respostas do modelo
        resumo = resumo_response['choices'][0]['message']['content'].strip()
        topicos = topicos_response['choices'][0]['message']['content'].strip()
        tratamentos = tratamentos_response['choices'][0]['message']['content'].strip()

        topicos = topicos if topicos else "Nenhum tópico identificado"
        tratamentos = tratamentos if tratamentos else "Nenhum tratamento sugerido"

        # Retorno da resposta como JSON
        return jsonify({
            "resumo": resumo,
            "topicos": topicos,
            "tratamentos": tratamentos
        })

    except Exception as e:
        error_message = str(e)
        print(f"Erro na anamnese: {error_message}")
        return jsonify({"error": error_message}), 500
