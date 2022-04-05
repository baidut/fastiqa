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
from tqdm import tqdm
from ._inceptiontime import *
from ._io_single_vid2mos import *

# extractFeatures
from fastai.vision.all import * # resnet18
from torchvision.models.video.resnet import * # r3d_18

# roipool
from ..paq2piq._roi_pool import LegacyRoIPoolModel
from ...learn import TestLearner
from ...iqa_exp import IqaExp

# soipool
from ._soi_pool import pool_features



def get_features(x, name, bs, vid_id):
    try:
        x.dls.bs = bs
        x.dls.set_vid_id(vid_id)
        x.extract_features(name=name, skip_exist=True)
    except RuntimeError:
        tmp_bs = bs
        while True:
            tmp_bs //= 2
            try:
                x.dls.bs = tmp_bs
                x.extract_features(name=name, skip_exist=True)
                break
            except RuntimeError as e:
                if tmp_bs == 1:
                    print("Batch size has reduced to 1. Check if there is something wrong or clear GPU memory")
                    raise e
                else:
                    print(f'CUDA out of memory. Reduce bs from {bs} to {tmp_bs}.')
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


class PatchVQ(InceptionTimeModel):
    siamese=True
    c_in=2048+2048
    n_out=4
    path_to_paq2piq_model_state = '/home/zq/FB8T/pth/RoIPoolModel-fit.10.bs.120.pth'

    def bunch(self, dls):
        if isinstance(dls, dict):
            # download extracted features or extract by your own
            feats = FeatureBlock('paq2piq_pooled', roi_index=None, clip_num=None, clip_size=None), \
            FeatureBlock('r3d_18_pooled', roi_index=None, clip_num=None, clip_size=None)
            dls = Feat2MOS.from_dict(dls, bs=bs, feats=feats)

        if not isinstance(dls.label_col, (list, tuple)): # database contains no patch labels
            print(f'set self.n_out = 1')
            self.n_out = 1
        return dls

    def extractFeatures(self, featname, dbinfo):
        self.roipool('paq2piq', dbinfo, backbone=resnet18, path_to_model_state=self.path_to_paq2piq_model_state)
        self.soipool('paq2piq', dbinfo)

        self.roipool('r3d_18', dbinfo, backbone=r3d_18)
        self.soipool('r3d_18', dbinfo)


    def soipool(self, featname, dbinfo, input_suffix="", output_suffix="_pooled", pool_size=16):
        pool_features(dbinfo, featname, input_suffix=input_suffix, output_suffix=output_suffix, pool_size=pool_size)


    def roipool(self, featname, dbinfo, backbone, path_to_model_state=None):
        if '3d' in featname:
            bs, clip_num, clip_size = 8, 40, 8
        else:
            bs, clip_num, clip_size = 128, None, 1


        roi_col = None # depending on database

        dls = SingleVideo2MOS.from_dict(dbinfo,
            use_nan_label=True, clip_num=clip_num, clip_size=clip_size,
            bs=bs)
        dls.roi_col = roi_col

        # The following version is trained with new version, don't use it
        # path_to_model_state = '/home/zq/FB8T/pth/P2P-RM.pth' # RoIPoolModel-fit.10.bs.120 change the location accordingly
        # backbone + roipool
        model = LegacyRoIPoolModel(backbone=backbone, pool_size=(2,2))

        if path_to_model_state:
            model_state = torch.load(path_to_model_state, map_location=lambda storage, loc: storage)
            model.load_state_dict(model_state["model"]) # model.load_state_dict(model_state["model"])

        learn = TestLearner(dls, model)
        vid_list = dls.video_list.index.tolist()
        learn.dls.set_vid_id(vid_list[0])

        e = IqaExp('exp_features', gpu=0, seed=None)
        e[featname] = learn
        e.run(lambda x: [get_features(x, featname, bs=bs, vid_id=vid_id) for vid_id in tqdm(vid_list)])


    @staticmethod
    def demo(cls):
        # n_epoch = 10
        # bs = 128
        # modify the json files to point to your database locations
      	LSVQ = load_dbinfo('/home/zq/FB8T/db/LSVQ/dbinfo.json')
      	LIVE_VQC = load_dbinfo('/home/zq/FB8T/db/LIVE_VQC/dbinfo.json')

        e = IqaExp('release', gpu=0)
        e += IqaLearner(dls=LSVQ, model = PVQ(c_in=2048+2048, n_out=4), loss_func=L1LossFlat())

        # train the model if pretrained model is not available
        # e.fit_one_cycle(n_epoch)

        # load the trained model
        e.load();

        # cross database validation
        e.valid([All(LIVE_VQC)])
