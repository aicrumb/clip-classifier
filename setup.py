import setuptools

setuptools.setup(
    name='anyclass',
    version='0.0.1',
    author='aicrumb',
    author_email='aicrumbmail@gmail.com',
    description='Simple Classifiers built on top of CLIP',
    long_description='i was bored of writing the code myself every time lol',
    long_description_content_type="text/markdown",
    url='https://github.com/aicrumb/anyclass',
    project_urls = {
        "Bug Tracker": "https://github.com/aicrumb/anyclass/issues"
    },
    license='GNU GPLv3',
    packages=['anyclass'],
    install_requires=['tqdm'],
)
