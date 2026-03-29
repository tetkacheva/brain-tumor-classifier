# <img src="icons/brain.png" width="18"> Brain Scan AI
#### Kent Hack Enough · 2026 

⚠️ This tool is made for the educational and demonstration purposes only. It is **not** a medical device and should not be used for clinical diagnosis. If you have any concerns, please contact a medical professional.

### Team
Tatiana Tkacheva — solo project

### Inspiration
At 18, I was diagnosed with *"You have something in your brain, but we don't know what it is."* It took 6 months of endless doctor appointments across different clinics to figure out what tumor I had. I was fortunate to have the resources to get opinions from the best neurosurgeons in Europe — but not everyone has that opportunity. That's where BrainScan AI comes in.

### What It Does
BrainScan AI analyzes brain MRI scans and within seconds tells you the probability of a scan showing signs of a tumor. When a tumor is detected, it applies a **Grad-CAM heatmap** overlay to highlight the regions the model flagged — giving medical professionals a visual starting point rather than just a black box verdict.

### How It's Built
- Model: EfficientNet-B0 with transfer learning, fine-tuned in two phases — first training only the classification head, then unfreezing the full model
- Dataset: Brain MRI dataset from Kaggle (binary: tumor / no tumor)
- Explainability: Grad-CAM heatmap overlay on detected tumor scans
- UI: Chat-style Streamlit web app
- Stack: PyTorch · Torchvision · Streamlit · OpenCV · Pillow

### Tools & Frameworks
| Tool | Purpose |
|---|---|
| [PyTorch](https://pytorch.org/) | Model training and inference | 
| [Tourchvision](https://pytorch.org/vision/) | EfficientNet-B0 pretrained weights and transforms |
| [Streamlit](https://streamlit.io/) | Web app UI |
| [OpenCV](https://opencv.org/) | Grad-CAM heatmap generation |
| [Pillow](https://python-pillow.org/) | Image processing |
| [Kaggle Dataset](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection) | Brain MRI images for tumor detection |
| [Flaticon](https://www.flaticon.com/) | UI icons (brain, dinosaur, meteor, clip) |
| [Claude (Anthropic)](https://claude.ai/logout) | AI coding assistant used during development |


### Project Structure
```
brain-tumor-classifier/
├── icons/               # UI icons
├── data/                 # Dataset (not tracked)
│   ├── brain_tumor_dataset/
│   ├── no/               # Halthy scans
│   └── yes/              # Tumor scans
├── models/               # Saved weights (not tracked)
├── source/
│   ├── model.py          # EfficientNet-B0 architecture
│   ├── train.py          # Training script
│   ├── predict.py        # CLI inference
│   └── gradcam.py        # Grad-CAM implementation
├── test_images/          # Test MRI scans (not tracked)
├── app.py                # Streamlit web app
├── requirements.txt
└── README.md
```

### Setup
**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/brain-tumor-classifier.git
cd brain-tumor-classifier
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download the dataset**
```bash
kaggle datasets download -d navoneel/brain-mri-images-for-brain-tumor-detection
Expand-Archive brain-mri-images-for-brain-tumor-detection.zip -DestinationPath data/
```

### Usage
**Train the model**
```bash
python source/train.py
```

**Run CLI inference**
```bash
python source/predict.py test_images/scan.jpg
```

**Launch the web app**
```bash
streamlit run app.py
```

### Model Performance
| Metric | Value |
|---|---|
| Validation Accuracy | 94% | 
| Architecture | EfficientNet-B0 |
| Training Epochs | 10 (2-phase) |
| Dataset Size | 253 MRI Scans |

### Explainability
When a tumor is detected, BrainScan AI generates a **Grad-CAM heatmap** that highlights which regions of the MRI the model focused on. This makes the AI's decision transparent and interpretable — critical in a medical context.
 
### What's Next
- Tumor type classification (glioma, meningioma, pituitary)
- Larger, more diverse training dataset for better Grad-CAM spatial precision
- Integration with DICOM medical imaging format