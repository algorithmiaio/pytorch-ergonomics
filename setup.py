from setuptools import setup
from os.path import abspath, dirname


python_requirements = [
    'dill>=0.2.7.1',
    'torch>=0.3.1'
]

setup(
    name='ergo-pytorch',
    version='1.0.1',
    description='making torch even better.',
    author='Algorithmia',
    maintainer='Algorithmia',
    license='MIT',
    autho_email='support@algorithmia.com',
    packages=['ergonomics'],
    install_requires=python_requirements,
    include_package_data=True,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ]
)