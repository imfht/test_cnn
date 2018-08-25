from __future__ import division, print_function

import os

# Flask utils
from flask import Flask, request, render_template
from gevent.pywsgi import WSGIServer
# Keras
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from werkzeug.utils import secure_filename

basepath = os.path.dirname(__file__)

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = basepath + '/models/trained.model'
from predict import *

model = load_model(MODEL_PATH)


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        try:
            # Get the file from post request
            f = request.files['file']

            # Save the file to ./uploads
            file_path = os.path.join(
                basepath, 'uploads', secure_filename(f.filename))
            f.save(file_path)
            predict = model.predict(file2xtrain(file_name=file_path))
            img_name = vector_to_code(predict[0])
            return img_name
        except Exception as err:
            return str(err)
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
