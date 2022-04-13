import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
import math
from ...basics import * # logger

def takespread_idx(N, num, clip_size=16):
    length = float(N-clip_size)
    for i in range(num):
        start = int(math.ceil(i * length / (num-1)))
        yield start, start+clip_size

def get_features(x, name, bs, vid_id):
    try:
        x.dls.bs = bs
        x.dls.set_vid_id(vid_id)
        x.extract_features(name=name, skip_exist=True)
    except RuntimeError:
        tmp_bs = bs
        while tmp_bs > 1:
            tmp_bs //= 2
            try:
                x.dls.bs = tmp_bs
                x.extract_features(name=name, skip_exist=True)
                break
            except RuntimeError:
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

class FeatPooler():
    fn_col = 'name'
    frame_num_col = 'frame_number'

    def get_df(self, path):
        return  pd.read_csv(path).set_index(self.fn_col)

    def __init__(self, d, name, input_suffix, output_suffix, avg_pool=True, pool_size = 16, clip_num=16):
        self.path = Path(d['dir'])
        self.fn_col = d['fn_col']
        if 'frame_num_col' in d.keys():
            self.frame_num_col = d['frame_num_col']
        self.roi_index = [0, 1, 0, 2]
        self.m = nn.AdaptiveAvgPool1d(pool_size) if avg_pool else nn.AdaptiveMaxPool1d(pool_size)
        self.df = self.get_df(self.path/d['csv_labels'])
        self.input_feat = name + input_suffix
        self.output_feat = name + output_suffix
        self.clip_num = clip_num

    @classmethod
    def from_dict(cls, d, **kwargs):
        return cls(d, **kwargs)

    def prepare_feat(self, vid):
        logger.info(f'pooling : {vid}')
        df = self.df
        _feats = load_feature(vid, self.input_feat, self.path/'features')
        if _feats.dim() == 4:
            _feats = _feats.unsqueeze(dim=1)

        feats = _feats.refine_names('N', 'roi', 'C', 'H', 'W')
        feats = feats.flatten(['C', 'H', 'W'], 'features')
        feats = feats.align_to('roi', 'features', 'N') # N C L

        logger.debug(f'preparing:   {_feats.shape} to {feats.shape}')
        print(f'preparing:   {_feats.shape} to {feats.shape}')
        return feats

    def __call__(self, vid): # pool_feat
        # feats = self.prepare_feat(vid)
        # n_frames = feats.shape[-1]
        # pooled_feat_list = [] # 16 clips, indexes
        # for start, end in takespread_idx(n_frames, self.clip_num):  #n_chunks_idx(n_frames, self.clip_num):
        #     feat = feats[ :,  :, start:end].rename(None)
        #     pooled_feat = self.m(feat) # [1, features, frame_num] --> [1, features, clip_num]
        #     pooled_feat_list.append(pooled_feat.permute(0,2,1)) # --> [1, clip_num, features]
        #
        # # clip_num x -1  x  features
        # pooled_feats = torch.cat(pooled_feat_list, dim=0).view(-1, pooled_feat_list[0].shape[-1])
        # logger.debug(f'%cat {pooled_feat_list[0].shape} with length {len(pooled_feat_list)} to {pooled_feats.shape}')
        # print(f'%cat {pooled_feat_list[0].shape} with length {len(pooled_feat_list)} to {pooled_feats.shape}')
        # save_feature(pooled_feats, vid, self.output_feat, self.path/'features')  # --> [1, clip_num, features]
        # logger.debug(f'done:   {pooled_feats.shape} ')
        # return pooled_feats

        feats = self.prepare_feat(vid)
        feat = feats.rename(None)
        pooled_feat = self.m(feat) # [1, features, frame_num] --> [1, features, clip_num]
        pooled_feat = pooled_feat.permute(0,2,1) # --> [1, clip_num, features]
        pooled_feat = pooled_feat.view(-1, pooled_feat[0].shape[-1])
        save_feature(pooled_feat, vid, self.output_feat, self.path/'features')  # --> [1, clip_num, features]
        print(f'done:   {pooled_feat.shape} ')
        return pooled_feat


def pool_features(database, name, input_suffix="", output_suffix="_pooled", pool_size=16, avg_pool=True):
  pooler = FeatPooler.from_dict(database, name=name, input_suffix=input_suffix, output_suffix=output_suffix, pool_size=pool_size, avg_pool=avg_pool)

  vids = pooler.df.index.tolist()
  vids_todo = [vid for vid in vids if not (pooler.path/'features'/(pooler.output_feat + '/' + str(vid) + '.npy')).exists()]

  desc = f'{name}({input_suffix} --> {output_suffix})'

  with tqdm(desc=desc, total=len(vids_todo)) as progress_bar:
      [pooler(vid) for vid in vids_todo]

  # parallel:
  # num_cores = multiprocessing.cpu_count()
  # with tqdm_joblib(tqdm(desc=desc, total=len(vids_todo))) as progress_bar:
  #     Parallel(n_jobs=num_cores)(delayed(pooler)(vid) for vid in vids_todo)
