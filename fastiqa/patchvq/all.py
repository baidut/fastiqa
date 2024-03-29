"""
================================================================================
# %%
%load_ext autoreload
%autoreload 2

from fastiqa.models.patchvq.main import PatchVQ
from fastai.vision.all import * # resnet18
from torchvision.models.video.resnet import * # r3d_18

dbinfo = {
  "__name__": "LIVE_VQC_tiny",
  "dir": "/home/zq/FB8T/db/LIVE_VQC_tiny/", #"/content/PatchVQ/data/LIVE_VQC_tiny/",
  "csv_labels": "labels.csv",
  "fn_col": "name",
  "label_col": "mos"
  }
# extractFeatures
PatchVQ().roipool('paq2piq', dbinfo, backbone=resnet18, path_to_model_state='/home/zq/FB8T/pth/RoIPoolModel-fit.10.bs.120.pth')
PatchVQ().roipool('r3d_18', dbinfo, backbone=r3d_18)

# %%

================================================================================
"""
# from fastiqa.vqa import *
# from torchvision.models.video.resnet import *
# from tqdm import tqdm
from .. import *
from ..basics import *
from ..model import *

import torch
from tqdm.auto import tqdm

from .model.inceptiontime import *
from .model import resnet3d
from .data.single_vid2mos import *
from .data.feat2mos import *

# extractFeatures
from fastai.vision.all import * # resnet18
# from torchvision.models.video.resnet import * # r3d_18

# roipool
from ..paq2piq._roi_pool import *
from ..paq2piq.model import P2P_RM
from ..learn import TestLearner, IqaLearner
from ..model import IqaModel
from ..iqa_exp import IqaExp

# soipool
from .model.soi_pool import pool_features

def get_features(x, name, bs, vid_id):
    x.dls.set_vid_id(vid_id)
    tmp_bs = bs
    while True:
        try:
            x.dls.bs = tmp_bs
            return x.extract_features(name=name, skip_exist=True)
        except RuntimeError as e:
            if tmp_bs == 1:
                logger.error("Batch size has been reduced to 1. Check if there is something wrong or clear GPU memory")
                raise e
            else:
                tmp_bs //= 2
                logger.warning(f'RuntimeError Encountered. Reduce bs from {bs} to {tmp_bs}.')
                continue

def load_feature(vid, feat, path):
    npy_file = Path(path)/feat/(str(vid) + '.npy')
    with open(npy_file, 'rb') as f:
        features = np.load(f)
    return torch.Tensor(features)

def save_feature(x, vid, feat, path):
    npy_file =  Path(path)/feat/(str(vid) + '.npy')
    Path(npy_file).parent.mkdir(parents=True, exist_ok=True)
    with open(npy_file, 'wb') as f:
        np.save(f, x)



class InceptionTimeModel(IqaModel):
    """
    c_out inception_time output
    n_out model output
    """
    siamese = False
    c_in = -1 # undefined

    def __init__(self, bottleneck=32,ks=40,nb_filters=32,residual=True,depth=6, **kwargs):
        super().__init__(**kwargs)
        c_out = 1 if self.siamese else self.n_out
        self.inception_time = InceptionTime(c_in=self.c_in,c_out=c_out, bottleneck=bottleneck,ks=ks,nb_filters=nb_filters,residual=residual,depth=depth)

    @classmethod
    def from_dls(cls, dls, n_out=None, **kwargs):
        if n_out is None: n_out = dls.c
        return cls(c_in = dls.vars, n_out=n_out, **kwargs)

    def forward(self, x, x2=None): # more features
        # if self.training == False:
        #     self.siamese = False
        #     self.n_out = 1

        if x2 is not None:
            x = torch.cat([x, x2], dim=-1)  # 4096 features
        if self.siamese:
            x = x.view(self.n_out*x.shape[0], -1, x.shape[-1])  # *x.shape[2:]
            # [bs, n_out*length, features] -->  [bs*n_out, length, features]
        # [bs*n_out, length, features] -->  [bs*n_out, features, length ]
        y = self.inception_time(x.transpose(1, 2))
        return y.view(-1, self.n_out) if self.siamese else y

    def input_sois(self, clip_num=16):
        raise NotImplementedError

def r3d18_K_200ep(pretrained=False, **kwargs):
    model = resnet3d.generate_model(model_depth=18, n_classes=700, **kwargs)
    if pretrained:
        # model_state = torch.load("/home/zq/FB8T/pth/fastai-r3d18_K_200ep.pth")
        # print('loading... local weights')
        # delete the cached weights
        model_state = load_state_dict_from_url('https://github.com/baidut/PatchVQ/releases/download/v0.1/fastai-r3d18_K_200ep.pth')
        model.load_state_dict(model_state)
    else:
        print('WARNING: pretrained r3d18 not loaded')

    model.eval()
    return model

class PatchVQ(InceptionTimeModel):
    siamese=True
    c_in=2048+2048
    n_out=4
    model_state_url = 'https://github.com/baidut/PatchVQ/releases/download/v0.1/PatchVQ.pth'

    def bunch(self, dls, bs=128):
        if isinstance(dls, dict):
            # download extracted features or extract by your own
            feats = FeatureBlock('paq2piq_pooled', roi_index=None, clip_num=None, clip_size=None), \
            FeatureBlock('r3d18_pooled', roi_index=None, clip_num=None, clip_size=None)
            dls = Feat2MOS.from_dict(dls, bs=bs, feats=feats)

        if hasattr(dls, 'label_col'):
          if not isinstance(dls.label_col, (list, tuple)): # database contains no patch labels
            self.n_out = 1
        return dls

    @staticmethod
    def extractFeatures(dbinfo):
        logger.info('Extract 2d features')
        model.roipool(LegacyRoIPoolModel(), dbinfo, featname='paq2piq', bs=32, clip_size=1)
        model.soipool('paq2piq', dbinfo)

        logger.info('Extract 3d features')
        model.roipool(RoIPoolModel(backbone=r3d18_K_200ep), dbinfo, featname='r3d18', bs=1, clip_num=40, clip_size=16)
        model.soipool('r3d18', dbinfo)

    @staticmethod
    def soipool(featname, dbinfo, input_suffix="", output_suffix="_pooled", pool_size=16):
        pool_features(dbinfo, featname, input_suffix=input_suffix, output_suffix=output_suffix, pool_size=pool_size)

    @staticmethod
    def roipool(model, dbinfo, featname, bs=16, clip_size=1, clip_num=None, roi_col=None, model_state=None, vid_id=None):
        dls = SingleVideo2MOS.from_dict(dbinfo,
            use_nan_label=True, clip_num=clip_num, clip_size=clip_size,
            bs=bs)
        dls.roi_col = roi_col

        # The following version is trained with new version, don't use it
        # path_to_model_state = '/home/zq/FB8T/pth/P2P-RM.pth' # RoIPoolModel-fit.10.bs.120 change the location accordingly
        # backbone + roipool
        # model = LegacyRoIPoolModel(backbone=backbone, pool_size=(2,2)) # try RoIPoolModel and see

        learn = TestLearner(dls, model, metrics=[])
        vid_list = dls.video_list.index.tolist()
        learn.dls.set_vid_id(vid_list[0])

        def process(x):
          bar = tqdm(vid_list)
          for vid_id in bar:
            bar.set_description('Processing: ' + str(vid_id))
            get_features(x, featname, bs=bs, vid_id=vid_id)
        if vid_id:
          learn.dls.set_vid_id(vid_id)
          return learn.extract_features(name=featname)
        else:
          e = IqaExp('exp_features', gpu=0, seed=None)
          e[featname] = learn
          e.run(process)
