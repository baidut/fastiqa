from fastai.vision.all import *
# from ..log import *
from loguru import logger
import timm
from packaging import version
import torchvision

if version.parse(torchvision.__version__) < version.parse("0.11.0"):
    from torchvision.models.utils import load_state_dict_from_url
else:
    from torch.hub import load_state_dict_from_url

# class IqaData():
#     db = None
#
#     def __init__(self, db, filter=None, **kwargs):
#         self.label = db() if isinstance(db, type) else db
#         self.name = f'{self.db.name}_{self.__class__.__name__}'
#         self.__dict__.update(kwargs)
#
#     def __getattr__(self, k: str):
#         return getattr(self.db, k)


def get_timm_model(name):
  f = partial(timm.create_model, name)
  f.__name__ = name
  return f





class IqaModel(Module):
    __name__ = None
    n_out = 1
    model_state_url = None

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        super().__init__()
        if self.__name__ is None: self.__name__ = self.__class__.__name__.split('__')[0]

    def __post_init__(self, *args,**kwargs):
        if self.model_state_url:
            logger.info('load finetuned model...')
            model_state = load_state_dict_from_url(self.model_state_url)
            # torch.load(self.PATH_TO_MODEL_STATE, map_location=lambda storage, loc: storage)
            self.load_state_dict(model_state) # ["model"]

    def bunch(self, dls):
        # assert not isinstance(dls, (tuple, list)), "do dls.bunch() first"
        # logging.info(f'bunching ... {self.__name__}@{dls.__name__}')
        #
        # if isinstance(dls.label_col, (list, tuple)):
        #     if len(dls.label_col) != self.n_out:
        #         dls.label_col = dls.label_col[:self.n_out]
        #         print(f'Changed dls.label_col to ({dls.label_col}) to fit model.n_out ({dls.__name__})')

        return dls
