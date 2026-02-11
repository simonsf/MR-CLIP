from setuptools import setup, find_packages

setup(
  name = 'mr-clip',
  packages = find_packages(exclude=[]),
  include_package_data = True,
  version = '1.0',
  description = 'MR-CLIP',
  install_requires=[
    'beartype',
    'einops>=0.6',
    'regex',
    'torch',
    # 'torchvision',
    "XlsxWriter",
    "h5py",
    "matplotlib",
    "seaborn",
    'ImageNetV2_pytorch @ git+https://github.com/modestyachts/ImageNetV2_pytorch.git',
    # "exit",
    "ftfy",
    "appdirs",
    "attr",
    "wilds",
    "nltk"
      ],
)
