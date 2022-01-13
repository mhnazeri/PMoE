from setuptools import setup

# extract version from __init__.py
with open('PMoE/__init__.py', 'r') as f:
    VERSION_LINE = [l for l in f if l.startswith('__version__')][0]
    VERSION = VERSION_LINE.split('=')[1].strip()[1:-1]

setup(
    name='PMoE',
    version=VERSION,
    packages=[
        'PMoE',
        'PMoE.model',
        'PMoE.eval',
        'PMoE.trainer',
        'PMoE.utils'
    ],
    license='GNU AGPLv3',
    description='PMoE: Predictive Mixture of Experts for Urban Driving',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Mohammad Nazeri',
    author_email='m.nazeri@iasbs.ac.ir',
    url='https://github.com/mhnazeri/pmoe',
    zip_safe=False,

    install_requires=[
        'Pillow>=8.3.2',
        'numpy~=1.19.4',
        'carla~=0.9.6',
        'Pillow>=8.0.1,<9.1.0',
        'tqdm~=4.55.1',
        'opencv-python~=4.4.0.46',
        'torch~=1.8.1',
        'thop~=0.0.31.post2005241907',
        'matplotlib~=3.3.3',
        'torchvision~=0.9.1',
        'omegaconf~=2.0.5',
        'pandas~=1.2.1',
        'imgaug~=0.4.0',
    ],
    extras_require={
        'test': [
            'pylint<=2.4.2',
        ],
        'prep': [

        ],
    },
)
