from setuptools import find_packages, setup

setup(
    name="xai-ids",
    version="1.0.0",
    description="Explainable AI Intrusion Detection System",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="AI Security Research Team",
    license="MIT",
    packages=find_packages(exclude=["tests*", "scripts*", "docker*", "docs*", "notebooks*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "scikit-learn>=1.4.0",
        "fastapi>=0.110.0",
        "uvicorn[standard]>=0.27.0",
        "pydantic>=2.0.0",
        "shap>=0.45.0",
        "matplotlib>=3.8.0",
        "plotly>=5.18.0",
        "requests>=2.31.0",
        "pyyaml>=6.0.1",
        "python-jose[cryptography]>=3.3.0",
        "passlib[bcrypt]>=1.7.4",
        "slowapi>=0.1.9",
        # torch is listed separately to allow CPU-only vs GPU installs
        # CI installs it with: pip install torch --index-url .../cpu
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.23.0",
            "httpx>=0.27.0",
            "ruff>=0.3.0",
            "black>=24.0.0",
            "isort>=5.13.0",
            "bandit>=1.7.7",
            "pip-audit>=2.7.0",
        ],
        "qiskit": [
            "qiskit>=1.0.0",
            "qiskit-aer>=0.14.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
