from setuptools import setup, find_packages

setup(
    name='image_utilities',
    version='0.1',
    description='Miscellaneous image utilities',
    url='https://github.com/jmilou/image_utilities',
    author='Julien Milli',
    author_email='jmilli@eso.org',
    license='MIT',
    keywords='image processing data analysis',
    packages=find_packages(),
    install_requires=[
        'numpy', 'scipy', 'astropy', 'pandas', 'matplotlib','pandas','datetime'
    ],
    zip_safe=False
)
