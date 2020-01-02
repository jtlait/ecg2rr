"""Setup for the ecg2rr package."""

import setuptools

setuptools.setup(
    name="ecg2rr",
    version="0.1.0",
    description="ECG R-peak detection with LSTM network",
    long_description=open('README.md').read(),
    url="https://github.com/jtlait/ecg2rr",
    author="Juho Laitala",
    author_email="jtlait@utu.fi",
    license="MIT",
    packages=setuptools.find_packages(),
    install_requires=[
        "tensorflow",
        "scipy",
        "wfdb"
    ],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
    ]
)
