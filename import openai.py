import openai
import tkinter as tk
from io import BytesIO
from threading import Thread
import speech_recognition as sr

# Configurar API da OpenAI com a chave fornecida
openai.api_key = "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"

recognizer = sr.Recognizer()

# Variável global para armazenar o áudio gravado
audio_data = None

# Função para gravar áudio
def grava_audio():
    global audio_data
    with sr.Microphone() as source:
        print("Ouvindo...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio_data = recognizer.listen(source)

# Função para transcrever áudio
def transcrever_audio(audio):
    wav_data = BytesIO(audio.get_wav_data())
    wav_data.name = 'audio.wav'
    transcricao = openai.Audio.transcribe("whisper-1", wav_data)
    return transcricao['text']

# Função executada quando o botão é pressionado
def on_button_press():
    global audio_thread
    audio_thread = Thread(target=grava_audio_thread)
    audio_thread.start()

# Função para gravar o áudio em uma thread separada
def grava_audio_thread():
    grava_audio()

# Função executada quando o botão é liberado
def on_button_release():
    audio_thread.join()
    texto_transcrito = transcrever_audio(audio_data)
    resultado_label.config(text=f"Texto Transcrito: {texto_transcrito}")

# Criar interface gráfica
root = tk.Tk()
root.title("Gravação de Áudio para Texto")

# Botão para gravar o áudio
gravar_button = tk.Button(root, text="Segure para Gravar", width=25, height=2)
gravar_button.bind("<ButtonPress>", lambda event: on_button_press())
gravar_button.bind("<ButtonRelease>", lambda event: on_button_release())
gravar_button.pack(pady=20)

# Label para exibir a transcrição
resultado_label = tk.Label(root, text="Texto Transcrito aparecerá aqui")
resultado_label.pack(pady=20)

# Iniciar a interface gráfica
root.mainloop()
