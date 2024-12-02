from flask import Flask, render_template, request, redirect, url_for
import os
import zipfile
import shutil
from retrain_model import retrain_model

# Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'uploaded_data'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_folder():
    if 'file' not in request.files:
        return "No file part in the request.", 400

    file = request.files['file']

    # Check if the file is a zip archive
    if not file.filename.endswith('.zip'):
        return "Please upload a ZIP file containing the dataset.", 400

    # Save the uploaded ZIP file
    zip_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(zip_path)

    # Extract the ZIP file
    extract_path = os.path.join(app.config['UPLOAD_FOLDER'], 'dataset')

    # Clear or create the extraction path
    if os.path.exists(extract_path):
        shutil.rmtree(extract_path)  # Remove the directory and its contents
    os.makedirs(extract_path)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    # Adjust for an extra folder layer (e.g., `retrain/` inside `uploaded_data/dataset`)
    subfolders = [d for d in os.listdir(
        extract_path) if os.path.isdir(os.path.join(extract_path, d))]
    if len(subfolders) == 1:  # If a single folder exists, move its contents up
        top_level_folder = os.path.join(extract_path, subfolders[0])
        for item in os.listdir(top_level_folder):
            item_path = os.path.join(top_level_folder, item)
            shutil.move(item_path, extract_path)
        # Remove the now-empty top-level folder
        shutil.rmtree(top_level_folder)

    # Verify dataset structure
    categories = [d for d in os.listdir(
        extract_path) if os.path.isdir(os.path.join(extract_path, d))]
    if not categories:
        return "No valid categories (subdirectories) found in the dataset. Ensure the dataset is structured correctly.", 400

    # Log extracted directories
    print(f"Extracted categories: {categories}")

    # Check if categories contain valid images
    for category in categories:
        category_path = os.path.join(extract_path, category)
        if not os.listdir(category_path):
            return f"Category '{category}' is empty. Please add images to this folder.", 400

    # Retrain the model
    try:
        model = retrain_model(extract_path, categories)
    except Exception as e:
        return f"An error occurred during retraining: {str(e)}", 500

    return render_template('success.html', message="Model retrained successfully!")


if __name__ == "__main__":
    app.run(debug=True)
