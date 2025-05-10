from tensorflow.keras.preprocessing.image import img_to_array, load_img
from utils.explainer import GradCAM, GuidedBackprop, FeatureExtractor
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.models import load_model, Model
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from utils.visualization import explain
import pandas as pd
import numpy as np
import joblib
import os

import warnings
warnings.filterwarnings('ignore')

# Load model
resnet = load_model('models/resnet50v2.h5')
knn = joblib.load('models/knn_chp_model.pkl')
model_logit = Model(resnet.input, resnet.layers[-1].output)
gradCAM = GradCAM(model_logit, layerName='conv5_block3_out')
guidedBP = GuidedBackprop(resnet, layerName='conv5_block3_out')
extractor = FeatureExtractor(resnet, layerName='dense_feature')

# Load data
train_df = pd.read_csv('data/train.csv')
filenames = train_df.filenames
true_labels = train_df.labels
classes = {0: 'cat', 1: 'dog'}

# Create app
app = Flask(__name__, template_folder='website/templates', static_folder='website/static')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/explain', methods=['GET', 'POST'])
def explain_page():
    if request.method == 'POST':
        file = request.files.get('upload-image')

        if file:
            filename = secure_filename(file.filename)
            save_path = os.path.join('website/static/uploads', filename)
            file.save(save_path)

            im = img_to_array(load_img(save_path, target_size=(224, 224)))
            im = np.expand_dims(im, axis=0)
            im = preprocess_input(im)
            prediction = resnet.predict(im)
            y_pred = prediction.argmax()

            c_hp = extractor.compute_c_hp(im, y_pred)
            knn_index = knn.kneighbors(c_hp.reshape(1, -1), n_neighbors=1, return_distance=False)[0, 0]
            retrieved_path = filenames[knn_index]
            true_label = true_labels[knn_index]

            paths = explain(resnet, gradCAM, guidedBP, filename, retrieved_path, im, y_pred)
            
            return render_template('explain.html',
                                   test_image=paths[0],
                                   prob=prediction.max(), 
                                   label=classes[y_pred],
                                   true_label=true_label,
                                   result_paths=paths[1:])
    
    return render_template('explain.html')

if __name__ == '__main__':
    app.run(debug=True)
