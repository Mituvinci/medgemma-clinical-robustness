"""
Setup script for MedGemma Clinical Robustness Assistant.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="medgemma-assistant",
    version="0.1.0",
    description="Multi-agent clinical decision support system for dermatology",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Halima",
    author_email="",
    url="https://www.kaggle.com/competitions/med-gemma-impact-challenge",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "python-dotenv>=1.0.0",
        "pydantic>=2.5.3",
        "pydantic-settings>=2.1.0",
        "google-cloud-aiplatform>=1.38.0",
        "google-genai>=0.2.0",
        "transformers>=4.36.2",
        "sentence-transformers>=2.2.2",
        "torch>=2.1.2",
        "chromadb>=0.4.22",
        "gradio>=4.12.0",
        "pandas>=2.1.4",
        "numpy>=1.26.3",
        "pillow>=10.1.0",
        "beautifulsoup4>=4.12.2",
        "requests>=2.31.0",
        "tenacity>=8.2.3",
        "tqdm>=4.66.1",
        "python-json-logger>=2.0.7",
        "pyyaml>=6.0.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "pytest-mock>=3.12.0",
            "black>=23.12.1",
            "flake8>=7.0.0",
            "mypy>=1.8.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "medgemma=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="medical AI dermatology clinical-decision-support multi-agent RAG",
)
