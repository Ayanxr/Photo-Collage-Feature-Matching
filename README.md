# 📸 PhotoMatch — Photo Collage Using Feature Matching

PhotoMatch is a Computer Vision application that automatically combines multiple overlapping photographs into one aligned panoramic image.

The project uses feature detection, feature matching, RANSAC, homography estimation, and perspective warping to align the images.

## ✨ Features

- Upload multiple photographs
- Detect important features using SIFT
- Match features between images
- Filter incorrect matches using Lowe's Ratio Test
- Estimate image transformation using Homography
- Remove incorrect matches using RANSAC
- Align and stitch photographs
- Automatically crop unwanted black areas
- Generate a final panoramic image
- Simple Streamlit web interface

## 🔄 How It Works

```text
Input Images
     ↓
Image Preprocessing
     ↓
SIFT Feature Detection
     ↓
Feature Matching
     ↓
Lowe's Ratio Test
     ↓
RANSAC
     ↓
Homography Estimation
     ↓
Perspective Warping
     ↓
Image Stitching
     ↓
Black Area Cropping
     ↓
Final Panorama

🧠 Computer Vision Pipeline
1. SIFT Feature Detection

SIFT (Scale-Invariant Feature Transform) identifies distinctive points and patterns in each photograph.

2. Feature Matching

Features from different images are compared to find corresponding points between photographs.

3. Lowe's Ratio Test

The ratio test removes weak or ambiguous feature matches and keeps more reliable correspondences.

4. RANSAC

RANSAC removes incorrect matches (outliers) and finds a reliable transformation between the images.

5. Homography

A homography matrix represents the geometric transformation required to align one photograph with another.

6. Perspective Warping

The images are transformed according to the calculated homography so that overlapping areas line up.

7. Panorama Creation

The aligned images are combined into one larger panoramic image.

8. Black Area Cropping

Unwanted black regions created during image warping are automatically detected and cropped.

🛠️ Technologies Used
Python
OpenCV
NumPy
Streamlit
SIFT
RANSAC
Homography
Perspective Transformation
📁 Project Structure
Photo-Collage-Feature-Matching/
│
├── input/
│   ├── nature1.jpg
│   ├── nature2.jpg
│   └── nature3.jpg
│
├── output/
│
├── src/
│   ├── __init__.py
│   ├── feature_matching.py
│   ├── image_stitching.py
│   └── utils.py
│
├── app.py
├── test_app.py
├── requirements.txt
├── .gitignore
└── README.md
🚀 Installation

Clone the repository:

git clone https://github.com/Ayanxr/Photo-Collage-Feature-Matching.git

Move into the project folder:

cd Photo-Collage-Feature-Matching

Create a virtual environment:

python -m venv venv

Activate the virtual environment on Windows:

venv\Scripts\activate

Install the required packages:

pip install -r requirements.txt
▶️ Run the Application

Start the Streamlit application:

streamlit run app.py

The application will open in your browser.

Upload multiple overlapping photographs and let PhotoMatch automatically create the panorama.

🧪 Testing

Run the test file using:

python test_app.py

You can also verify that the Computer Vision modules are working with:

python -c "from src.feature_matching import *; from src.image_stitching import *; from src.utils import *; print('ALL CV MODULES WORKING')"
📷 Example Input

The project includes three sample nature photographs:

nature1.jpg
nature2.jpg
nature3.jpg

For best results, photographs should contain some overlapping visual regions.

🎯 Project Objective
The objective of PhotoMatch is to demonstrate how traditional Computer Vision techniques can be combined to automatically align and merge multiple photographs without manually positioning each image.
