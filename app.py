from flask import Flask, jsonify, request
import joblib
import sklearn

#curl -d "{\"Medidas\":[[1,2,3,4]]}" -H "Content-Type: applicacion/json" -X POST http://127.0.0.1:5000/predecir

app = Flask(__name__)

@app.route("/")
def home():
    return "hola mundo"

@app.route("/predecir", methods=["POST"])
def predecir():
    json= request.get_json(force=True)
    Medidas= json['Medidas']
    clf= joblib.load('modelo_entrenado.pkl')
    prediction=clf.predict(Medidas)
    return "Tiene probabilidades de ganar de un {0}: ".format(prediction)

if __name__ == '__main__':
    app.run()