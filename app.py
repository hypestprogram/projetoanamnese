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

# Endpoint para transcrição de áudio
@app.route('/transcrever', methods=['POST'])
def transcrever_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "Nenhum arquivo de áudio enviado"}), 400

    audio_file = request.files['audio']
    try:
        # Ler o arquivo de áudio e garantir que seja um formato suportado
        audio_bytes = audio_file.read()
        audio_stream = io.BytesIO(audio_bytes)

        # Verificar o tipo de arquivo enviado
        mime_type = audio_file.mimetype
        print(f"Tipo de arquivo recebido: {mime_type}")  # Log para depuração

        # Certificar que o arquivo é de um dos formatos suportados
        supported_formats = ['audio/webm', 'audio/ogg', 'audio/mpeg', 'audio/wav']
        if mime_type not in supported_formats:
            return jsonify({"error": f"Formato de arquivo não suportado: {mime_type}. Formatos suportados: {supported_formats}"}), 400

        # Definir o nome do arquivo como requerido pela API Whisper
        audio_stream.name = audio_file.filename or 'audio.webm'

        # Transcrever o áudio usando o modelo Whisper da OpenAI
        transcript = openai.Audio.transcribe("whisper-1", audio_stream)
        return jsonify({"transcricao": transcript['text']})
    except Exception as e:
        error_message = str(e)
        print(f"Erro na transcrição: {error_message}")
        return jsonify({"error": error_message}), 500

# Endpoint para processar o texto de anamnese
@app.route('/anamnese', methods=['POST'])
def anamnese_texto():
    data = request.get_json()
    texto = data.get('texto', '')

    if not texto:
        return jsonify({"error": "Nenhum texto de anamnese enviado"}), 400

    try:
        # Criar a solicitação para gerar um resumo
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

        # Identificação de tópicos clínicos e fatores de risco
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

        # Sugestões de tratamento e medicamentos
        tratamentos_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Com base na anamnese abaixo, sugira tratamentos e medicamentos apropriados, "
                        "priorizando aqueles que estejam diretamente relacionados aos sintomas apresentados "
                        "e ao histórico clínico do paciente. Cada sugestão deve incluir uma breve justificativa clínica "
                        "(entre parênteses) explicando sua indicação e relevância. Se forem identificados sinais que "
                        "indiquem urgência, adicione o alerta: 'ATENÇÃO: Urgente - Encaminhar imediatamente para "
                        "atendimento emergencial.' Liste os tratamentos e intervenções em ordem de relevância e impacto clínico."
                    )
                },
                {"role": "user", "content": texto}
            ],
            max_tokens=200
        )

        resumo = resumo_response['choices'][0]['message']['content'].strip()
        topicos = topicos_response['choices'][0]['message']['content'].strip()
        tratamentos = tratamentos_response['choices'][0]['message']['content'].strip()

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
