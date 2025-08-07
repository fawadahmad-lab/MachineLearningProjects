from flask import Blueprint, request, jsonify, send_from_directory
from app.enhancer import enhance_image
from app.background_remover import remove_background
import os

# Create a Blueprint for your routes
main = Blueprint('main', __name__)

# Define the path where you'll store the images temporarily
UPLOAD_FOLDER = 'temp'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Route for image enhancement
@main.route('/enhance', methods=['POST'])
def enhance():
    # Check if the file is in the request
    if 'image' not in request.files:
        return jsonify({"error": "No image file found in the request"}), 400
    
    file = request.files['image']

    # Check if the file has a valid image extension
    if file and file.filename.endswith(('.png', '.jpg', '.jpeg')):
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Enhance the image using your model (assumes enhance_image is a function in enhancer.py)
        enhanced_image = enhance_image(file_path)

        # Save the enhanced image
        enhanced_image_path = os.path.join(OUTPUT_FOLDER, f"enhanced_{file.filename}")
        enhanced_image.save(enhanced_image_path)

        return send_from_directory(OUTPUT_FOLDER, f"enhanced_{file.filename}", as_attachment=True)

    return jsonify({"error": "Invalid image file format"}), 400

# Route for background removal
@main.route('/remove-background', methods=['POST'])
def remove_bg():
    # Check if the file is in the request
    if 'image' not in request.files:
        return jsonify({"error": "No image file found in the request"}), 400
    
    file = request.files['image']

    # Check if the file has a valid image extension
    if file and file.filename.endswith(('.png', '.jpg', '.jpeg')):
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Remove the background using your model (assumes remove_background is a function in background_remover.py)
        output_image = remove_background(file_path)

        # Save the output image
        output_image_path = os.path.join(OUTPUT_FOLDER, f"bg_removed_{file.filename}")
        output_image.save(output_image_path)

        return send_from_directory(OUTPUT_FOLDER, f"bg_removed_{file.filename}", as_attachment=True)

    return jsonify({"error": "Invalid image file format"}), 400
