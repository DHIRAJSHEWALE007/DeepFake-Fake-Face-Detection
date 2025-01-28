from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# Load your model

model_path = r"model\final_model_91.5.keras"
model = tf.keras.models.load_model(model_path)
labels = ["Fake", "Real"]



@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image
    try:
        file = request.files['image']
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
            
        if not file.filename.lower().endswith(('jpg', 'jpeg', 'png')):
            return jsonify({'error': 'Unsupported file format'}), 400


        file_bytes = BytesIO(file.read())
        
        img = tf.keras.utils.load_img(file_bytes, target_size=(256, 256))
        img = tf.keras.utils.img_to_array(img)
        img = tf.expand_dims(img, axis=0)
        # normalized_image = tf.image.convert_image_dtype(img, tf.float32)

        # Make prediction
        yhat = model.predict(img)
        result = labels[int(yhat[0] > 0.5)]

        confidence = f"{float(1-yhat[0]):.5f} %" if result.lower()=="fake" else f"{float(yhat[0]):.5f} %"
        
        return render_template('index.html', prediction=result)
    
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
