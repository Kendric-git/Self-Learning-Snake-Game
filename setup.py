from setuptools import setup, find_packages

setup(
    name='ai_snakegame',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pygame>=2.1.0',
        'numpy>=1.24.0',
        'torch>=1.13.0',
    ],
    entry_points={
        'console_scripts': [
            'snake-ai=snake_ai:main',  # if you wrap your train() in a main()
        ],
    },
)
