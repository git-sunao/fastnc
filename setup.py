import re,os,sys

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

def get_requirements():
    fname = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    with open(fname, 'r') as f:
        requirements = [line.rstrip() for line in f if not line.startswith('#')]
    return requirements

def find_from_doc(what='version'):
    f = open(os.path.join(os.path.dirname(__file__), 'fastnc/__init__.py')).read()
    match = re.search(r"^__%s__ = ['\"]([^'\"]*)['\"]"%(what), f, re.M)
    if match:
        return match.group(1)
    raise RuntimeError("Unable to find %s string."%what)

if __name__ == '__main__':
    setup(
        name='fastnc',
        version=find_from_doc('version'),
        description='fastnc package',
        long_description=open('README.md').read(),
        long_description_content_type='text/markdown',
        url=find_from_doc('url'),
        author=find_from_doc('author'),
        author_email='sunaosugiyama@gmail.com',
        keywords=['cosmology', 'large scale structure', 'three-point correlation function'],
        packages=['fastnc'],
        install_requires=get_requirements(),
        classifiers=['Programming Language :: Python :: 3'],
    )
