import os
import io
import json
import subprocess
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from pydub import AudioSegment
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import storage
from google.oauth2 import service_account
from dotenv import load_dotenv
import openai

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Configurar chaves de API para o Speech-to-Text (usando a variável padrão)
openai.api_key = os.getenv("OPENAI_API_KEY")
GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
if GOOGLE_CREDENTIALS_JSON:
    with open("/tmp/credentials.json", "w") as f:
        f.write(GOOGLE_CREDENTIALS_JSON)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/credentials.json"

# Nome do bucket do Cloud Storage para áudios longos
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

# Inicializar o app Flask
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

SUPPORTED_FORMATS = ['audio/webm', 'audio/ogg', 'audio/mpeg', 'audio/wav', 'audio/mp4']

def verificar_ffmpeg():
    """Verifica se o FFmpeg está instalado e disponível."""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, check=True)
        print(f"Versão do FFmpeg: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Erro ao verificar FFmpeg: {e.stderr}")

verificar_ffmpeg()

def convert_audio(audio_bytes, target_format='wav'):
    """
    Converte o áudio para WAV com 16-bit PCM, detecta a taxa de amostragem e
    calcula a duração (em segundos). Retorna um objeto BytesIO, a taxa e a duração.
    """
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        audio = audio.set_sample_width(2)  # 2 bytes = 16 bits por amostra
        sample_rate = audio.frame_rate
        duration = len(audio) / 1000.0  # duração em segundos

        audio_io = io.BytesIO()
        audio.export(audio_io, format=target_format)
        audio_io.seek(0)

        print(f"Áudio convertido para {target_format} com taxa de {sample_rate} Hz e duração de {duration} segundos")
        return audio_io, sample_rate, duration
    except Exception as e:
        print(f"Erro na conversão de áudio: {str(e)}")
        raise

def upload_to_gcs(audio_io, bucket_name, destination_blob_name):
    """
    Faz o upload do áudio (BytesIO) para o GCS utilizando as credenciais definidas na variável
    GOOGLE_APPLICATION_STORAGE_CREDENTIALS_JSON (se configurada) e retorna o GCS URI.
    """
    try:
        audio_io.seek(0)
        # Verifica se existe uma chave exclusiva para o Cloud Storage
        storage_key = os.getenv("GOOGLE_APPLICATION_STORAGE_CREDENTIALS_JSON")
        if storage_key:
            temp_storage_path = "/tmp/storage_credentials.json"
            with open(temp_storage_path, "w") as f:
                f.write(storage_key)
            storage_credentials = service_account.Credentials.from_service_account_file(temp_storage_path)
            storage_client = storage.Client(credentials=storage_credentials)
        else:
            storage_client = storage.Client()
            
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_file(audio_io, content_type='audio/wav')
        gcs_uri = f"gs://{bucket_name}/{destination_blob_name}"
        print(f"Áudio enviado para o GCS: {gcs_uri}")
        return gcs_uri
    except Exception as e:
        print(f"Erro ao enviar áudio para o GCS: {str(e)}")
        raise

def delete_from_gcs(bucket_name, blob_name):
    """
    Exclui o objeto (arquivo) especificado no bucket do GCS.
    """
    try:
        # Utiliza as mesmas credenciais configuradas para o Storage
        storage_key = os.getenv("GOOGLE_APPLICATION_STORAGE_CREDENTIALS_JSON")
        if storage_key:
            temp_storage_path = "/tmp/storage_credentials.json"
            with open(temp_storage_path, "w") as f:
                f.write(storage_key)
            storage_credentials = service_account.Credentials.from_service_account_file(temp_storage_path)
            storage_client = storage.Client(credentials=storage_credentials)
        else:
            storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.delete()
        print(f"Arquivo '{blob_name}' deletado do bucket '{bucket_name}'.")
    except Exception as e:
        print(f"Erro ao deletar o arquivo do GCS: {str(e)}")

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "API ativa e funcionando"}), 200

@app.route('/transcrever', methods=['POST'])
def transcrever_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "Nenhum arquivo de áudio enviado"}), 400

    audio_file = request.files['audio']
    try:
        audio_bytes = audio_file.read()
        if len(audio_bytes) == 0:
            return jsonify({"error": "Arquivo de áudio vazio ou inválido."}), 400

        mime_type = audio_file.mimetype
        print(f"Tipo de arquivo recebido: {mime_type}")
        if mime_type not in SUPPORTED_FORMATS:
            return jsonify({"error": f"Formato não suportado: {mime_type}. Formatos suportados: {SUPPORTED_FORMATS}"}), 400

        # Converter áudio e obter taxa e duração
        audio_stream, sample_rate, duration = convert_audio(audio_bytes)

        client = speech.SpeechClient()
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            language_code="pt-BR"
        )

        if duration > 60:
    if not GCS_BUCKET_NAME:
        return jsonify({"error": "GCS_BUCKET_NAME não configurado para áudios longos."}), 500

    destination_blob_name = f"temp_audio_{uuid.uuid4().hex}.wav"
    gcs_uri = upload_to_gcs(audio_stream, GCS_BUCKET_NAME, destination_blob_name)
    recognition_audio = speech.RecognitionAudio(uri=gcs_uri)

    # 🔹 Configuração aprimorada para transcrição mais rápida
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,  # 🔹 Reduz taxa de amostragem para otimizar velocidade
        language_code="pt-BR",
        alternative_language_codes=["pt-PT"],  # 🔹 Suporte a variações do português
        enable_automatic_punctuation=True,  # 🔹 Adiciona pontuação automática
        enable_speaker_diarization=True,  # 🔹 Separa falas de diferentes pessoas
        diarization_speaker_count=2,  # 🔹 Estima 2 participantes na conversa
        use_enhanced=True,  # 🔹 Usa modelo otimizado para maior performance
        model="video"  # 🔹 Modelo mais rápido para áudio longo
    )

    print("🔹 Iniciando transcrição assíncrona no Cloud Speech-to-Text...")
    operation = client.long_running_recognize(config=config, audio=recognition_audio)

    # 🔹 Aguarda o processamento sem bloquear completamente
    import time
    while not operation.done():
        print("⏳ Processando transcrição no Google Cloud...")
        time.sleep(5)  # 🔹 Aguarda 5 segundos antes de verificar o progresso

    response = operation.result(timeout=300)  # 🔹 Tempo máximo de espera: 5 minutos
    delete_from_gcs(GCS_BUCKET_NAME, destination_blob_name)  # 🔹 Deleta após processar

    transcript = " ".join([result.alternatives[0].transcript for result in response.results])
    return jsonify({"transcricao": transcript})

    except Exception as e:
        print(f"Erro na transcrição: {str(e)}")
        return jsonify({"error": f"Erro inesperado: {str(e)}"}), 500

@app.route('/anamnese', methods=['POST'])
def anamnese_texto():
    data = request.get_json()
    texto = data.get('texto', '')
    if not texto:
        return jsonify({"error": "Nenhum texto de anamnese enviado"}), 400
    try:
        resumo_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": 
                 "Com base exclusivamente na transcrição abaixo, gere uma anamnese no formato SOAP, seguindo estas regras:"
                 "\n\n*1. Fidelidade ao texto:* Não complemente informações não mencionadas. Se algo não for relatado, escreva 'Não relatado'."
                 "\n\n*2. Formato SOAP:*"
                 "\n- *S (Subjetivo):* Dados relatados pelo paciente."
                 "\n  - Identificação: [Iniciais], [Idade], [Sexo]."
                 "\n  - Queixa Principal (QP): [Descreva sintomas e duração apenas se mencionados]."
                 "\n  - História da Doença Atual (HDA): [Relate início, evolução, fatores agravantes/alívio, sintomas associados apenas como descrito]."
                 "\n  - Antecedentes: [Inclua histórico médico, cirúrgico, medicamentoso, familiar e social somente se mencionados]."
                 "\n\n- *O (Objetivo):* Dados observáveis ou registrados."
                 "\n  - Exame Físico: [Sinais vitais e achados explicitamente descritos]."
                 "\n  - Exames Complementares: [Resuma resultados conforme as regras abaixo]."
                 "\n\n*Regras para Exames:*"
                 "\n- Formato: Data | Resultados resumidos com abreviações e números absolutos (Ex.: HB: 14.5 | HT: 42% | Troponina: 3.2)."
                 "\n- Use abreviações: Hemoglobina → HB | Hematócrito → HT | Troponina → TRP."
                 "\n- Exclua: Hemácias, valores normais ou unidades."
                 "\n\n- *A (Avaliação):* Liste hipóteses diagnósticas com base no relato. Não justifique além do descrito."
                 "\n\n- *P (Plano):* Proposta terapêutica com base nos dados."
                 "\n  - Conduta: [Exames solicitados, medicações e orientações mencionadas]."
                },
                {"role": "user", "content": texto}
            ],
            max_tokens=150
        )
        topicos_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": 
                 "Com base exclusivamente na transcrição abaixo, identifique os principais tópicos da anamnese em no máximo 150 tokens, seguindo rigorosamente estas regras:"
                 "\n\n- *Queixa Principal (QP):* [Descreva apenas se mencionado]."
                 "\n- *Evolução dos Sintomas:* [Inclua detalhes relevantes se relatados]."
                 "\n- *Fatores Agravantes e de Alívio:* [Informe conforme descrito]."
                 "\n- *Histórico Médico e Familiar:* [Detalhes apenas mencionados]."
                 "\n- *Achados do Exame Físico:* [Sinais vitais e achados relevantes explicitados]."
                 "\n- *Hipóteses Diagnósticas e Plano Terapêutico:* [Baseadas no relato]."
                },
                {"role": "user", "content": texto}
            ],
            max_tokens=150
        )
        tratamentos_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": 
                 "Com base nas informações fornecidas, sugira um plano diagnóstico e terapêutico adequado para o paciente em no maximo 200 tokens. "
                 "Inclua as seguintes seções:"
                 "\n- Hipóteses Diagnósticas: Liste possíveis diagnósticos diferenciais."
                 "\n- Exames Complementares Solicitados: Informe quais exames são necessários."
                 "\n- Medicações Prescritas: Liste as medicações recomendadas."
                 "\n- Orientações ao Paciente: Descreva orientações e recomendações ao paciente."
                 "\n- Seguimento e Reavaliação: Informe sobre o plano de seguimento e necessidade de reavaliações futuras."
                },
                {"role": "user", "content": texto}
            ],
            max_tokens=200
        )
        resumo = resumo_response['choices'][0]['message']['content'].strip()
        topicos = topicos_response['choices'][0]['message']['content'].strip()
        tratamentos = tratamentos_response['choices'][0]['message']['content'].strip()
        return jsonify({"resumo": resumo, "topicos": topicos, "tratamentos": tratamentos})
    except Exception as e:
        print(f"Erro na anamnese: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
