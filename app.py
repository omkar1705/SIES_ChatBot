from flask import Flask, render_template, request, jsonify
from chat import get_response 

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("base.html")

@app.route("/predict", methods=["POST"])
def predict():
    user_message = request.json.get("message")
    response = get_response(user_message)
    return jsonify({"answer": response}) 
if __name__ == "__main__":
    app.run(debug=True)
