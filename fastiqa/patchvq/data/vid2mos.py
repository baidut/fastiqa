__all__ = ['Vid2MOS', 'clip2image']

"""
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from fastiqa.bunches.vqa.vid2mos import *
# dls = Vid2MOS.from_json('json/LIVE_VQC.json', bs=3, clip_num=3, clip_size=2)
dls = Vid2MOS.from_json('json/KoNViD.json', bs=3, clip_num=3, clip_size=2)
# dls = Vid2MOS.from_json('json/LIVE_FB_VQA_pad500.json', item_tfms=CropPad(500), bs=3, clip_num=3, clip_size=2)
dls.show_batch()
dls.bs
'_data' in dls.__dict__
del dls.__dict__['_data']
dls.bs = 2
dir(dls)
dls._data.bs
dls = dls.reset(bs=2)
dls.show_batch()
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

from fastai.vision.all import *
from ._clip import *
from ...bunch import IqaDataBunch

"""
Note here, a video means a collection of jpg files
"""

def clip2image(t, vertical=False):
    if t.dim() == 3: # [3, H, W] only 1 image in a clip
        return t
    if vertical:# vertical concat # [n, 3, H, W] --> [3, n, H, W] --> [3, n*H, W]
        return t.transpose(0, 1).reshape(3, -1, t.size(-1))
    else:
        # [n, 3, H, W] --> [3, H, n, W] --> [3, H, n*W]
        return t.transpose(0, 1).transpose(1, 2).reshape(3, t.size(-2), -1)

class Vid2MOS(IqaDataBunch):
    clip_size = 1
    clip_num = 8
    bs = 8
    fn_last_frame_col = "fn_last_frame"
    folder = "jpg"
    frame_num_col = 'frame_number'

    def get_df(self):
        df = super().get_df()
        # add_fn_last_frame col if not exists
        if self.frame_num_col not in df.columns:
            print('add frame_num_col')
            frame_numbers = []
            for folder in df[self.fn_col].tolist():
                n_max = 0
                # .split('.')[0]
                for file in (self.path/self.folder/folder).glob('*.jpg'):
                    n = int(str(file)[:-4].split('_')[-1])
                    if n > n_max:
                        n_max = n;
                frame_numbers.append(n_max)
            df[self.frame_num_col] = frame_numbers
            df.to_csv(self.path/self.csv_labels, index=False)
        if self.fn_last_frame_col not in df.columns:
            print('add fn_last_frame_col')
            df[self.fn_last_frame_col] = df[self.fn_col] + '/image_' + df[self.frame_num_col].astype(str).str.zfill(5)
            df.to_csv(self.path/self.csv_labels, index=False)
        return df

    def get_block(self):
        df = self.get_df()
        VideoBlock = partial(MultiClipBlock, clip_size=self.clip_size, clip_num=self.clip_num)
        return DataBlock(
            blocks     = (VideoBlock, RegressionBlock),
            getters = [
               ColReader(self.fn_last_frame_col, pref=self.path/(self.folder + '/')  ),
               ColReader(self.label_col), # mos_vid
            ],
            item_tfms = self.item_tfms,
            # batch_tfms=[Normalize.from_stats(*imagenet_stats)],
            splitter   = self.get_splitter(),
        )


    def create_batch(self, data):
        # cannot call self in this function
        return create_sequence_batch(data)
        # if self.clip_size > 1 else None
