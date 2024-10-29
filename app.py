from flask import Flask, request, jsonify, send_file, session, redirect, url_for
from flask_cors import CORS
from openpyxl import Workbook
import os

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Necessário para sessões
CORS(app)  # Permitir que o frontend na AwardSpace se conecte

@app.route('/calcular', methods=['POST'])
def calcular():
    data = request.json  # Recebe JSON do frontend
    corte = data['corte']
    peso = float(data['peso'])
    preco = float(data['preco'])
    quantidade = int(data['quantidade'])

    total = peso * preco * quantidade

    # Criar o Excel
    filename = f'calculo_carnes.xlsx'
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(['Corte de Carne', 'Peso (kg)', 'Preço por Kg', 'Quantidade', 'Total'])
    sheet.append([corte, peso, preco, quantidade, total])
    workbook.save(filename)

    # Retornar o resumo e o link para download
    return jsonify({
        'corte': corte,
        'peso': peso,
        'preco': preco,
        'quantidade': quantidade,
        'total': total,
        'download_url': f'/download/{filename}'
    })

@app.route('/download/<filename>', methods=['GET'])
def download(filename):
    if os.path.exists(filename):
        return send_file(filename, as_attachment=True)
    else:
        return "Arquivo não encontrado.", 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
