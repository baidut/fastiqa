from distutils.core import setup

# note: version is maintained inside fastai/version.py
exec(open('fastiqa/version.py').read())

with open('README.md') as readme_file: readme = readme_file.read()

setup(
  name = 'fastiqa',
  version = __version__,
  packages = ['fastiqa'], 

  license='CC BY-NC-ND 4.0',
  description = "fastiqa makes deep learning for image quality assessment faster and easier",
  long_description = readme,
  long_description_content_type = 'text/markdown',

  author = 'Zhenqiang Ying', 
  author_email = 'yingzhenqiang@gmail.com',
  url = 'https://github.com/fastiqa/fastiqa', 

  keywords = 'fastiqa, deep learning, machine learning', 
  install_requires=[
          'fastai>=1.0.57',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    #'License :: OSI Approved :: MIT License',
    'License :: Other/Proprietary License',
    'Natural Language :: English',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.6',
  ],
)
