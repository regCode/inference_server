from flask import Flask
from flask import request, jsonify, make_response
import numpy as np
import pickle


def load_model():
    return pickle.load(open(model_file, 'rb'))

app = Flask(__name__)
model_file = 'random_forest_regresor_prostate_cancer.pickle'
model = load_model()


@app.route("/")
def home():
    page = """
        <h1>Hello this is the API to consume the cancer lpa prediction model</h1>
        <p>You can access to any of our two endpoints</p>
        <ul>
            <li>/predict_single</li>
            <li>/predict_lpsa</li>
        </ul>
    """
    return page


@app.route("/predict_single")
def predict_single():
    """ Predict a single record
    :return: string of the predicted value
    """
    lcavol = request.args.get('lcavol', 0)
    lweight = request.args.get('lweight', 0)
    age = request.args.get('age', 0)
    lbph = request.args.get('lbph', 0)
    svi = request.args.get('svi', 0)
    lcp = request.args.get('lcp', 0)
    gleason = request.args.get('gleason', 0)
    pgg45 = request.args.get('pgg45', 0)

    register = np.array([lcavol, lweight, age, lbph, svi, lcp, gleason, pgg45], dtype='f').reshape(1, -1)

    return str(model.predict(register)[0])


@app.route("/predict_lpsa", methods=["POST"])
def predict_lpsa():
    """ Predict to multiple registers in a JSON load
    :return: JSON with the predicted values
    """
    if request.is_json:
        req = request.get_json()
        values = np.array(req['values'])

        predictions = model.predict(values)

        response_body = {
            "target": "lpsa",
            "predicted": predictions.tolist()
        }

        return make_response(jsonify(response_body), 200)

    else:
        return make_response(jsonify({"message": "Request body must be JSON"}), 400)


if __name__ == '__main__':
    app.run(debug=True)
