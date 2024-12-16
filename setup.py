from setuptools import setup, find_packages

setup(
    name='cjeu-ali',  # Your project name
    version='0.1',
    packages=find_packages(where='scripts'),  # Automatically find packages inside the 'scripts' directory
    install_requires=[
        'faiss-cpu',  # Include any other dependencies here
        'pytest',     # Make sure pytest is a dependency too
        'transformers',  # Include any other libraries you might need
    ],
)
