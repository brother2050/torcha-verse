from setuptools import setup, find_packages

setup(
    name="torcha_verse",
    version="0.3.1",
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
        # 数据校验与配置模型(AssetStore / 配置中心等依赖)
        "pydantic>=2.0.0",
        # Web UI / 演示界面依赖
        "gradio>=4.0.0",
        # RAG 向量检索后端(FAISS CPU 版)
        "faiss-cpu>=1.7.4",
    ],
    extras_require={
        # 量化推理可选依赖(INT4/INT8/NF4 量化)
        "quantization": [
            "bitsandbytes>=0.41.0",
        ],
        # 开发与测试依赖
        "dev": [
            "pytest>=7.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "torcha=serving.cli:main",
        ],
    },
)
