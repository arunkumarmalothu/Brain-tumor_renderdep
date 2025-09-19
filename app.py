from flask import Flask, request, render_template
import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Define the folder for uploaded images and the model classes
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Define the classes exactly as in your notebook
CLASSES = ['Glioma', 'Meningioma', 'Not Tumorous', 'Pituitary']

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained Keras model
try:
    model = load_model('brain_tumor_model.keras')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    """Renders the main page with the file upload form."""
    return render_template('index.html', prediction_text="Upload an MRI image to get a prediction.")

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the image upload, preprocessing, and prediction."""
    if 'file' not in request.files:
        return render_template('index.html', prediction_text="No file part in the request.")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', prediction_text="No selected file.")
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            # Preprocess the image to match the model's expected input shape (150x150, RGB)
            img = Image.open(filepath).convert('RGB')
            img = img.resize((150, 150))
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
            img_array = img_array / 255.0 # Normalize pixel values

            if model:
                prediction = model.predict(img_array)
                predicted_class_index = np.argmax(prediction)
                predicted_class = CLASSES[predicted_class_index]
                confidence = prediction[0][predicted_class_index] * 100
                
                output_text = f"Prediction: {predicted_class} with {confidence:.2f}% confidence."
            else:
                output_text = "Model not loaded. Please check the brain_tumor_model.keras file."

        except Exception as e:
            output_text = f"An error occurred during prediction: {e}"

        # Optionally delete file after prediction to save space
        os.remove(filepath)

        return render_template('index.html', prediction_text=output_text)

if __name__ == "__main__":
    app.run(debug=True)
