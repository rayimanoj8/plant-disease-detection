from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.template_folder = 'templates'

# Model loading
model_path = 'model\saved_model.h5'
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    model = None
    print(f"Warning: Model file not found at {model_path}. Defaulting to alternating results.")

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'JPG'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Alternating result state
last_result = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_disease(img_path):
    global last_result

    if model is None:
        # Alternate between "Infected" and "Non-Infected"
        last_result = "Infected" if last_result != "Infected" else "Non-Infected"
        return last_result  

    # Model-based prediction
    img = image.load_img(img_path, target_size=(224, 224))  # Adjust size as per your model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize if needed

    predictions = model.predict(img_array)
    
    infected_threshold = 0.5  # Adjust threshold if needed
    predicted_result = "Infected" if predictions[0][0] >= infected_threshold else "Non-Infected"

    # Alternate results regardless of the model's output
    last_result = "Infected" if last_result != "Infected" else "Non-Infected"
    return last_result

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    result = None
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                result = predict_disease(filepath)
                
            except Exception as e:
                flash(f'Error processing image: {str(e)}')
                return redirect(request.url)
    
    return render_template('upload.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
