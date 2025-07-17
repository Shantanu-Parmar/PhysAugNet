from setuptools import setup, find_packages

# Read dependencies from requirements.txt
with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="physaugnet",
    version="0.1.1",
    author="Shantanusinh Parmar",
    description="PhysAugNet: VQ-VAE and physics-inspired (thermal + grain) augmentation pipeline for metal defect segmentation",
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.10',
)
