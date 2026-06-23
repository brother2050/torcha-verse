from setuptools import setup, find_packages

setup(
    name="torcha_verse",
    version="0.1.0",
    description="TorchaVerse: A pure PyTorch all-modal generative AI framework",
    author="TorchaVerse Team",
    license="Apache-2.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "torchaudio>=2.1.0",
        "PyYAML>=6.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "sentencepiece>=0.1.99",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "click>=8.1.0",
        "tqdm>=4.66.0",
        "safetensors>=0.4.0",
        "rich>=13.0.0",
    ],
    entry_points={
        "console_scripts": [
            "torcha=serving.cli:main",
        ],
    },
)
