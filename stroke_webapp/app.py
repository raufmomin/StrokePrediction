from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model('./stroke.h5')
model.summary()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = np.array(features).reshape(1, -1)
    prediction = model.predict(final_features)

    if prediction[0][0] > 0.5:
        output = 'You\'re likely to have stroke in future\n Please consult your doctor ASAP.' + \
            '\n Stroke probability is ' + \
            str(round(prediction[0][0]*100, 2)) + '%'
    else:
        output = 'Cheers no need to worry! You\'re doing well.' + \
            '\n Stroke Probability is ' + \
            str(round(prediction[0][0]*100, 2)) + '%'
    return render_template('index.html', prediction_text='{}'.format(output))


if __name__ == "__main__":
    app.run(host='localhost', port='8000', debug=True)
