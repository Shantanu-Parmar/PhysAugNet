# PhysAugNet

**PhysAugNet** is a research-focused Python toolkit designed to improve segmentation performance in few-shot or low-data industrial defect scenarios by leveraging:

- Vector-quantized variational autoencoding (VQ-VAE) for **generative reconstructions**  
- Physically inspired augmentations such as **thermal distortion** and **sensor noise simulation**

This augmentation strategy aims to produce synthetic data that enhances generalization and robustness for deep segmentation models in industrial inspection tasks.

## Key Contributions

- ✅ Compact and trainable **VQ-VAE** architecture with fast convergence  
- ✅ Dual-mode augmentation pipeline using **thermal distortion** + **sensor grain noise**  
- ✅ PhysAugNet fusion: combines reconstruction and physics-based augmentations  
- ✅ CLI with YAML-configurable experiment control  
- ✅ Modular for easy integration in PyTorch segmentation pipelines  
- ✅ Lightweight design suitable for few-shot learning and resource-constrained environments

## Computational Pipeline

| Module | Operation | Description |
|--------|-----------|-------------|
| `vqvae/train.py` | Training | Learn latent quantized space for image reconstructions |
| `vqvae/infer.py` | Inference | Reconstruct metal defect images for data augmentation |
| `augment/thermal_grain.py` | Augmentation | Apply thermal distortion + grain-based noise |
| `augment/combined.py` | Augmentation | Combine VQ-VAE reconstructions with physics-based distortion |
| `main.py` | CLI | Unified CLI with `argparse`-based command routing |
| `configs/default.yml` | Config | Controls all aspects of training, augmentation, and I/O |

## 📁 Project Structure

```
physaugnet/
├── physaug/
│   ├── vqvae/
│   ├── augment/
│   ├── config/
│   ├── utils/
│   └── main.py
├── configs/
│   └── default.yml
├── notebooks/
│   └── demo_physaugnet.ipynb
├── requirements.txt
├── setup.py
└── README.md
```

## Sample Commands

```bash
# Train the VQ-VAE model
python -m physaug.main train_vqvae --config configs/default.yml

# Reconstruct training set using VQ-VAE
python -m physaug.main reconstruct     --input_dir images/train     --output_dir outputs/reconstructed     --model_path checkpoints/vqvae.pth

# Apply thermal + grain augmentation
python -m physaug.main augment_tg     --input_dir images/train     --output_dir outputs/thermal_grain

# Combine both for hybrid augmentation
python -m physaug.main augment_combined     --vqvae_dir outputs/reconstructed     --output_dir outputs/combined
```

## Configuration (configs/default.yml)

```yaml
mode: train_vqvae
vqvae_path: checkpoints/vqvae.pth
input_dir: images/train
output_dir: outputs/reconstructed
image_size: 128
device: cuda
batch_size: 16
num_epochs: 20
learning_rate: 0.0002
```

## Notebook Demonstration

Open `notebooks/demo_physaugnet.ipynb` to:

- Visualize thermal & grain effects
- Plot VQ-VAE reconstructions
- Compare before/after augmentation
- Integrate with segmentation pipeline

## Applications

- Few-shot segmentation in manufacturing
- Data expansion for metal defect detection
- Domain generalization for industrial AI
- Simulation of real-world sensor variation

## 📌 Citation

If you use **PhysAugNet** in your research, please cite it as:

```bibtex
@misc{parmar2025physaugnet,
  author       = {Shantanu Parmar},
  title        = {PhysAugNet: VQ-VAE and Physically-Inspired Augmentations for Metal Defect Segmentation},
  year         = {2025},
  howpublished = {\url{https://github.com/Shantanu-Parmar/PhysAugNet}},
  note         = {GitHub repository}
}
```
## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).  
You are free to use, modify, and distribute this software with proper attribution.

© 2025 Shantanu Parmar. All rights reserved.

