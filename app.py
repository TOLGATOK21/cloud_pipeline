from flask import Flask, request, jsonify
import joblib
import numpy as np

# Flask uygulamasını başlat
app = Flask(__name__)

# Modeli yükle
model = joblib.load('stock_model.pkl')

# Özellik sırası – eğitimde hangi sırayla kullanıldıysa aynısını belirtin
feature_order = ['MA20', 'MA50', 'Lag1', 'Lag2', 'Lag3']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_dict = data['features']

        # Özellikleri sıraya göre diz
        features = np.array([input_dict[feature] for feature in feature_order]).reshape(1, -1)

        # Tahmin yap
        prediction = model.predict(features)

        return jsonify({
            'prediction': int(prediction[0])
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/')
def home():
    return 'Stock Price Prediction API is running.'

# Ana fonksiyon
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
