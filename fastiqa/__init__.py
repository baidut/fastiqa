# import logging
# logging.basicConfig()
# logging.basicConfig(level=logging.ERROR) # supress debug messages
# supress info
# logging.getLogger(__name__).addHandler(logging.NullHandler())

from functools import partial
import sys; print (sys.version)
import fastai; print(f'fastai.__version__(>= 2.5.3): {fastai.__version__}')
import fastcore; print(f'fastcore.__version__: {fastcore.__version__}')
import torch; print(f"torch.__version__(>= 1.9.1): {torch.__version__} {'w/' if torch.cuda.is_available() else 'w/o'} cuda ")
import torchvision; print(f'torchvision.__version__(>= 0.10.1): {torchvision.__version__}')
