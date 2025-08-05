"""
Setup configuration for APF_TD3 package.
"""

from setuptools import setup, find_packages
import os
import re

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read version from __init__.py
def get_version():
    """Extract version from __init__.py"""
    init_py = os.path.join(this_directory, 'apf_td3', '__init__.py')
    if os.path.exists(init_py):
        with open(init_py, 'r') as f:
            content = f.read()
            version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", content, re.M)
            if version_match:
                return version_match.group(1)
    return "1.0.0"  # Default version

# Core dependencies
install_requires = [
    "numpy>=1.19.0,<2.0.0",
    "torch>=1.9.0,<3.0.0",
    "torchvision>=0.10.0,<1.0.0",
    "matplotlib>=3.3.0,<4.0.0",
    "seaborn>=0.11.0,<1.0.0",
    "PyYAML>=5.4.0,<7.0.0",
    "scipy>=1.7.0,<2.0.0",
]

# Optional dependencies
extras_require = {
    'dev': [
        "pytest>=6.2.0,<8.0.0",
        "pytest-cov>=2.12.0,<5.0.0",
        "black>=21.0.0,<24.0.0",
        "flake8>=3.9.0,<7.0.0",
        "mypy>=0.910,<2.0.0",
        "pre-commit>=2.15.0,<4.0.0",
        "isort>=5.9.0,<6.0.0",
    ],
    'docs': [
        "sphinx>=4.0.0,<8.0.0",
        "sphinx-rtd-theme>=0.5.0,<3.0.0",
        "myst-parser>=0.15.0,<3.0.0",
    ],
    'optimization': [
        "optuna>=2.10.0,<4.0.0",
        "plotly>=5.0.0,<6.0.0",
    ],
    'integration': [
        "gym>=0.21.0,<1.0.0",
        "stable-baselines3>=1.6.0,<3.0.0",
    ],
    'visualization': [
        "plotly>=5.0.0,<6.0.0",
        "seaborn>=0.11.0,<1.0.0",
    ],
}

# Add 'all' option that includes all extras
extras_require['all'] = list(set(
    dep for extra_deps in extras_require.values() for dep in extra_deps
))

setup(
    name="apf-td3",
    version=get_version(),
    author="APF_TD3 Contributors",
    author_email="support@apf-td3.com",
    description="Artificial Potential Field with Twin Delayed Deep Deterministic Policy Gradient for autonomous navigation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/APF_TD3",
    project_urls={
        "Bug Reports": "https://github.com/your-username/APF_TD3/issues",
        "Source": "https://github.com/your-username/APF_TD3",
        "Documentation": "https://apf-td3.readthedocs.io",
        "Changelog": "https://github.com/your-username/APF_TD3/blob/main/CHANGELOG.md",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Hardware :: Hardware Drivers",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    include_package_data=True,
    package_data={
        "apf_td3": [
            "configs/*.yaml",
            "configs/*.yml",
        ],
    },
    entry_points={
        "console_scripts": [
            "apf-td3-train=apf_td3.cli:train_command",
            "apf-td3-eval=apf_td3.cli:eval_command",
            "apf-td3-demo=apf_td3.cli:demo_command",
        ],
    },
    keywords=[
        "reinforcement learning",
        "artificial potential fields",
        "TD3",
        "autonomous navigation",
        "robotics",
        "path planning",
        "obstacle avoidance",
        "deep learning",
        "pytorch",
    ],
    zip_safe=False,
    # Additional metadata
    platforms=["any"],
    license="MIT",
    # Test configuration
    test_suite="tests",
    tests_require=extras_require['dev'],
)