from fastai.vision.all import *
import logging
# from ..log import *
import timm

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

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        super().__init__()
        if self.__name__ is None: self.__name__ = self.__class__.__name__.split('__')[0]

    def bunch(self, dls):
        # assert not isinstance(dls, (tuple, list)), "do dls.bunch() first"
        # logging.info(f'bunching ... {self.__name__}@{dls.__name__}')
        #
        # if isinstance(dls.label_col, (list, tuple)):
        #     if len(dls.label_col) != self.n_out:
        #         dls.label_col = dls.label_col[:self.n_out]
        #         print(f'Changed dls.label_col to ({dls.label_col}) to fit model.n_out ({dls.__name__})')

        return dls


# https://stackoverflow.com/questions/279561/what-is-the-python-equivalent-of-static-variables-inside-a-function
def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate
