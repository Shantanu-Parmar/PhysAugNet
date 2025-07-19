
# 🔧 PhysAugNet

**PhysAugNet** is a research-grade Python toolkit designed to enhance **few-shot** or **low-data** industrial defect segmentation tasks by integrating:

- 🌐 **Vector-Quantized Variational Autoencoding (VQ-VAE)** for **generative reconstructions**
- 🔥 **Physically-inspired augmentations** including **thermal distortion** and **sensor grain noise**

This augmentation pipeline improves generalization and robustness of deep segmentation models in **industrial inspection** workflows.

---

## 🚀 Key Features

- ✅ **Compact and efficient VQ-VAE** architecture with fast convergence  
- ✅ Dual-mode augmentation: **thermal distortion** + **sensor grain noise**  
- ✅ **PhysAugNet Fusion**: Combines VQ-VAE reconstructions with physics-inspired augmentations  
- ✅ CLI-based experiment control via **YAML configuration**  
- ✅ Fully **modular** and easy to plug into PyTorch pipelines  
- ✅ Lightweight design tailored for **few-shot learning** and **resource-constrained environments**

---

## 🧩 Computational Pipeline

| Module                      | Operation       | Description                                                    |
|----------------------------|------------------|----------------------------------------------------------------|
| `physaug/vqvae/train.py`   | Training         | Trains VQ-VAE to learn latent quantized space                  |
| `physaug/vqvae/infer.py`   | Inference        | Reconstructs defect images for augmentation                    |
| `physaug/augment/thermal.py` | Augmentation   | Applies thermal distortion to images                           |
| `physaug/augment/grain.py`   | Augmentation   | Applies synthetic sensor grain noise                           |
| `physaug/augment/combined.py`| Augmentation   | Fuses VQ-VAE reconstructions with thermal + grain noise        |
| `infer_video.py`           | Video Inference  | Performs VQ-VAE reconstructions on video frames                |
| `main.py`                  | CLI Launcher     | Unified command-line interface with `argparse` routing         |
| `configs/default.yaml`     | Config           | Centralized configuration for all training and inference tasks |

---

## 🗂 Project Structure

```
PhysAugNet/
├── physaug/
│   ├── __init__.py
│   ├── vqvae/
│   │   ├── __init__.py
│   │   ├── vqvae.py
│   │   ├── train.py
│   │   └── infer.py
│   ├── augment/
│   │   ├── __init__.py
│   │   ├── thermal.py
│   │   ├── grain.py
│   │   └── combined.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       ├── io.py
│       └── logger.py
├── configs/
│   └── default.yaml
├── examples/
│   └── notebook_demo.ipynb
├── physaugnet.egg-info/
├── main.py
├── train_vqvae.py
├── gen_vqvae.py
├── augment_thermal.py
├── augment_combined.py
├── infer_video.py
├── setup.py
├── requirements.txt
└── README.md
```

---

## 📦 Installation

### Option 1: Install from PyPI (Recommended)

```bash
pip install physaugnet
```

### Option 2: Clone the Repository

```bash
git clone https://github.com/Shantanu-Parmar/PhysAugNet
cd PhysAugNet
```

Create a virtual environment:
```bash
# Linux/Mac
python -m venv Physaug
source Physaug/bin/activate

# Windows
python -m venv Physaug
Physaug\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Or install the package:
```bash
python setup.py install
```

---

## ⚙️ Setup

### 📁 Dataset Preparation

- Place **training images** under:  
  `images/train/<class_name>/`
  
- Place **test images** under:  
  `images/test/`

### 📁 Directory Creation

```bash
mkdir -p checkpoints/vqvae outputs/reconstructed outputs/augmented outputs/combined logs
```

### 🛠 Config File

Edit `configs/default.yaml` to ensure all paths and parameters are correctly set for your environment.

---

## 🖥 Usage

Run from the root `PhysAugNet/` directory using CLI:

### 🔧 VQ-VAE Training

```bash
python -m physaug.main train_vqvae --config configs/default.yaml
```

### 🧠 Image Reconstruction (VQ-VAE)

```bash
python -m physaug.main reconstruct --config configs/default.yaml
```

### 🌡 Thermal + Grain Augmentation

```bash
python -m physaug.main augment_tg --config configs/default.yaml
```

### 🔁 Combined VQ-VAE + Physical Augmentations

```bash
python -m physaug.main augment_combined --config configs/default.yaml
```

### 🎞 Video Frame Reconstruction (VQ-VAE)

```bash
python infer_video.py   --video_path images/DEMO_INFERENCE.mp4   --output_path outputs/reconstructed_video.mp4   --checkpoint checkpoints/vqvae.pth   --config configs/default.yaml
```

---

## 📓 Notebook Demonstration

Open the demo notebook to:

- Train the VQ-VAE
- Reconstruct images
- Apply augmentations
- Combine and visualize outputs

```bash
jupyter notebook examples/notebook_demo.ipynb
```

---

## 📤 Output Structure

| Type                | Path                        |
|---------------------|-----------------------------|
| Logs                | `logs/` (e.g., `vqvae_trainer.log`) |
| Checkpoints         | `checkpoints/vqvae/`        |
| Reconstructed Images| `outputs/reconstructed/`    |
| Augmented Images    | `outputs/augmented/`        |
| Combined Outputs    | `outputs/combined/`         |
| Video Outputs       | `outputs/reconstructed_video.mp4` |

---

## 🧪 Applications

- Few-shot segmentation in manufacturing
- Synthetic data generation for metal defect detection
- Robustness to physical variations in sensor input
- Domain adaptation for industrial computer vision

---

## 📚 Citation

If you use PhysAugNet in your research, please cite:

```bibtex
@misc{parmar2025physaugnet,
  author       = {Shantanu Parmar},
  title        = {PhysAugNet: VQ-VAE and Physically-Inspired Augmentations for Metal Defect Segmentation},
  year         = {2025},
  howpublished = {\url{https://github.com/Shantanu-Parmar/PhysAugNet}},
  note         = {GitHub repository}
}
```

---

## 📄 License

**MIT License** — You are free to use, modify, and distribute this software with proper attribution.

© 2025 Shantanu Parmar. All rights reserved.
