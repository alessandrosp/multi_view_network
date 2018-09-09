import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="multi_view_network",
    version="1.0",
    author="Alessandro Scoccia Pappagallo",
    author_email="aless@ndro.xyz",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    description="Keras implementation of Multi-View Network by Guo et al.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=['tensorflow', 'keras'],
    keywords='keras tensorflow machine-learning NLP research',
    packages=setuptools.find_packages(),
    url="https://github.com/annoys-parrot/multi_view_network",
)
