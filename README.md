
Zero-Shot Object Detection using Grounding DINO

Overview
This project demonstrates Zero-Shot Object Detection (ZSD) using Grounding DINO, a state-of-the-art Vision-Language Model.
It detects and localizes unseen object categories in images using natural language prompts, without requiring any retraining or labeled data.

The base implementation is adapted from the official Grounding DINO repository by IDEA Research, with custom modifications for dataset integration, zero-shot inference, and evaluation.

Key Features
- Zero-shot detection on unseen object classes using text prompts
- Integration with Open Images Dataset V6 (Foods) from Roboflow
- COCO-format annotation generation for evaluation
- Automatic bounding box prediction and visualization
- Custom scripts for inference, evaluation, and visualization
- Pre-trained Grounding DINO weights for high-quality detections

Folder Structure
zero-shot-object-detection-grounding-dino/
- inference/
- dataset/
- results/
- utils/
- requirements.txt
- README.md
- report/

Installation & Setup
git clone https://github.com/<your-username>/zero-shot-object-detection-grounding-dino.git
cd zero-shot-object-detection-grounding-dino

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate   # Windows

pip install -r requirements.txt

How It Works
1. Model Loading – Loads the pre-trained Grounding DINO model weights.
2. Prompt Input – Accepts natural language text (e.g., "apple", "pizza", "bottle").
3. Zero-Shot Inference – Predicts bounding boxes for objects matching the text.
4. Evaluation – Generates predictions in COCO format and calculates mAP.
5. Visualization – Draws bounding boxes and labels on images.

Tech Stack
- Python 3.10+
- PyTorch
- Grounding DINO (IDEA Research)
- COCO Evaluation Toolkit
- Open Images Dataset V6 (Foods)

Dataset
Dataset used:
Open Images Dataset V6 – Foods (Roboflow)
- Contains 54 food categories
- Used for zero-shot evaluation and visualization

Mentor
Richa Thakur – Guided and reviewed the project implementation.

Acknowledgements
- Base implementation and pretrained weights from Grounding DINO (IDEA Research)
- Dataset from Roboflow Universe
- Supported by guidance from Richa Thakur

Author
Ajay Rajput
Data Science & Machine Learning Enthusiast
- Pune, India
- LinkedIn: https://linkedin.com/in/ajayrajput
- GitHub: https://github.com/ajayrajput
