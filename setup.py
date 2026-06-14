from setuptools import setup, find_packages

setup(
    name="lite_llama",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "triton>=2.1.0",
        "safetensors>=0.4.1",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pytest>=7.0.0"
    ],
)
