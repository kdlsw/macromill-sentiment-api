from setuptools import setup, find_packages

setup(
    name="macromill-sentiment",
    version="1.0.0",
    description="Sentiment Classification for Movie Reviews",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.26.0",
        "scikit-learn>=1.7.0",
        "joblib>=1.5.0",
        "matplotlib>=3.10.0",
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "tokenizers>=0.14.0",
        "accelerate>=0.20.0",
        "datasets>=2.14.0",
        "tqdm>=4.66.0",
        "fastapi>=0.110.0",
        "uvicorn>=0.27.0",
        "pydantic>=2.6.0",
        "python-dateutil>=2.9.0",
    ],
)
