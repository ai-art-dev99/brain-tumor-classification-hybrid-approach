# 🧠 Brain Tumor Classification – Hybrid Deep Learning Model

A deep learning project for automatic **brain tumor classification** from MRI images using a **hybrid model** that combines the strengths of multiple architectures for improved accuracy and generalization.  
This project was developed and trained in **Google Colab**, and can be run end-to-end in the provided `.ipynb` notebook.

---

## ✨ Features

- 🏥 **Medical Imaging Focus** – Designed for classifying brain MRI scans into tumor categories.
- 🔬 **Hybrid Deep Learning Architecture** – Combines convolutional neural networks (CNNs) with additional machine learning layers for improved performance.
- 📊 **Training and Evaluation** – Includes preprocessing, model training, validation, and performance metrics.
- ☁ **Google Colab Ready** – Fully runnable without local setup.
- 📈 **Visualization Tools** – Plots for accuracy, loss curves, and confusion matrices.

---

## 📂 Project Structure

```

brain-tumor-classification/
│
├── brain-tumor-classification-hybrid-model.ipynb   # Main Google Colab notebook
└── README.md                                       # Project documentation

````

---

## 🚀 Getting Started

### 1. Open in Google Colab

Click the badge below to open the notebook in Colab:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/your-repo-name/blob/main/brain_tumor_classification_hybrid_model.ipynb)

---

### 2. Dataset

You can use publicly available datasets such as:

- [Brain MRI Images for Brain Tumor Detection (Kaggle)](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)
- [Figshare Brain Tumor Dataset](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427)

Place the dataset in the appropriate directory or mount Google Drive in Colab.

---

### 3. Install Dependencies


If using Colab, dependencies are installed within the notebook automatically.

---

### 4. Run the Notebook

Execute all cells in the `brain-tumor-classification-hybrid-model.ipynb` notebook to train and evaluate the model.

---

## 🧠 Model Architecture

* **Feature Extractor** – A CNN backbone for capturing spatial features from MRI images.
* **Hybrid Layer** – Combines CNN output with a secondary model (e.g., Dense layers, classical ML classifier) for better classification accuracy.
* **Output Layer** – Softmax activation for multi-class classification.

---

## 📊 Results

The notebook includes:

* Accuracy and loss curves over training epochs
* Confusion matrix for detailed class-wise performance
* Example predictions with true vs predicted labels

---

## 📌 Future Improvements

* Experiment with different CNN backbones (ResNet, EfficientNet, etc.)
* Implement data augmentation for robustness
* Deploy the model via a web app (e.g., Streamlit or Flask)

---

## 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgements

* Dataset providers (Kaggle, Figshare, etc.)
* TensorFlow / PyTorch community
* Google Colab for cloud-based development

---

*Built to assist in medical image classification research and education.*
