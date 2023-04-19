from flask import Flask, request, jsonify
import my_model  # Assuming the code you provided is saved in my_model.py

app = Flask(__name__)


@app.route('/retrain', methods=['POST'])
def retrain():
    my_model.retrain()
    return jsonify({"message": "Model retrained successfully"}), 200


@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = int(request.args.get('user_id'))
    recommendations = my_model.recommend(user_id)
    return jsonify(recommendations), 200


if __name__ == '__main__':
    app.run(debug=True)
