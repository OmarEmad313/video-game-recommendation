from flask import Flask, request, jsonify
import my_model  # Assuming the code you provided is saved in my_model.py
import my_model_firebase
app = Flask(__name__)
# curl -X GET  
#curl - X POST  http://127.0.0.1:5000//hi?
# curl -X GET http://127.0.0.1:5000//hi
# curl - X POST http://127.0.0.1:5000/recommendMobile?user_id=zXHem93J32u5q5Fgb2NV
#curl "http://127.0.0.1:5000/recommendMobile?user_id=zXHem93J32u5q5Fgb2NV"
# curl - X GET "http://127.0.0.1:5000/recommendMobile?user_id=zXHem93J32u5q5Fgb2NV"

@app.route('/retrain', methods=['POST'])
def retrain():
    my_model.retrain()
    return jsonify({"message": "Model retrained successfully"}), 200


@app.route('/hi', methods=['POST'])
def hi():
    return jsonify({"message": "HI there"}), 200


@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = int(request.args.get('user_id'))
    recommendations = my_model.recommend(user_id)
    return jsonify(recommendations), 200


@app.route('/retrainMobile', methods=['POST'])
def retrainMobile():
    my_model_firebase.retrainMobile()
    return jsonify({"message": "Model retrained successfully"}), 200


@app.route('/recommendMobile', methods=['GET'])
def recommendMobile():
    user_id = request.args.get('user_id')
    recommendations = my_model_firebase.recommendMobile(user_id)
    return jsonify(recommendations), 200


if __name__ == '__main__':
    app.run(debug=True)
