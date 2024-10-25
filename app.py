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

openai.api_key = os.getenv("OPENAI_API_KEY")

SUPPORTED_FORMATS = ['audio/webm', 'audio/ogg', 'audio/mpeg', 'audio/wav']

# Endpoint para transcrição de áudio
@app.route('/transcrever', methods=['POST'])
def transcrever_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "Nenhum arquivo de áudio enviado"}), 400

    audio_file = request.files['audio']
    try:
        audio_bytes = audio_file.read()
        audio_stream = io.BytesIO(audio_bytes)

        mime_type = audio_file.mimetype
        print(f"Tipo de arquivo recebido: {mime_type}")

        if mime_type not in SUPPORTED_FORMATS:
            return jsonify({
                "error": f"Formato de arquivo não suportado: {mime_type}. "
                         f"Formatos suportados: {SUPPORTED_FORMATS}"
            }), 400

        audio_stream.name = audio_file.filename or 'audio.webm'
        transcript = openai.Audio.transcribe("whisper-1", audio_stream)
        return jsonify({"transcricao": transcript['text']})

    except Exception as e:
        print(f"Erro na transcrição: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Endpoint para processar o texto de anamnese
@app.route('/anamnese', methods=['POST'])
def anamnese_texto():
    data = request.get_json()
    texto = data.get('texto', '')

    if not texto:
        return jsonify({"error": "Nenhum texto de anamnese enviado"}), 400

    try:
        # Criar o resumo da anamnese
        resumo_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Elabore um resumo conciso da anamnese abaixo, destacando sintomas principais, "
                        "sinais relevantes, condições preexistentes e possíveis fatores de risco. "
                        "O resumo deve ser claro e objetivo, com até 5 linhas:"
                    )
                },
                {"role": "user", "content": texto}
            ],
            max_tokens=200
        )

        # Identificar tópicos clínicos e fatores de risco
        topicos_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Identifique e liste os principais tópicos clínicos encontrados na anamnese abaixo, "
                        "incluindo sintomas, sinais de alerta, histórico familiar relevante, condições preexistentes "
                        "e fatores de risco. Organize os tópicos por ordem de relevância clínica:"
                    )
                },
                {"role": "user", "content": texto}
            ],
            max_tokens=150
        )

        # Sugestões de exames e medicamentos
        tratamentos_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Com base na anamnese abaixo, liste apenas os **nomes de exames** relevantes ou "
                        "**medicações** apropriadas para o quadro clínico apresentado. Não inclua justificativas ou "
                        "detalhes adicionais, apenas nomes claros e específicos. Organize por ordem de relevância clínica. "
                        "Se houver algum exame ou medicação que indique urgência, inclua o aviso: "
                        "'ATENÇÃO: Urgente - Encaminhar imediatamente para atendimento emergencial.'"
                    )
                },
                {"role": "user", "content": texto}
            ],
            max_tokens=150
        )

        # Extrair e organizar as respostas
        resumo = resumo_response['choices'][0]['message']['content'].strip()
        topicos = topicos_response['choices'][0]['message']['content'].strip()
        tratamentos = tratamentos_response['choices'][0]['message']['content'].strip()

        return jsonify({
            "resumo": resumo,
            "topicos": topicos,
            "tratamentos": tratamentos
        })

    except Exception as e:
        print(f"Erro na anamnese: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
