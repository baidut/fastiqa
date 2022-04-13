from ._roi_pool import *
from torchvision.ops import RoIPool, RoIAlign
from ..basics import *
from .data import *
from ._fastai1 import create_paq2piq_head, bn_drop_lin, my_create_head

__all__ = ['P2P_BM', 'P2P_RM', 'P2P_FM']


class P2P_BM(BodyHeadModel):
    # also use padded image
    # bunch opt
    # label_
    model_state_url = 'https://github.com/baidut/fastiqa/releases/download/v2.0.0/P2P-BM.pth'
    def __init__(self, *args, **kwargs):
        super().__init__(*args, backbone = resnet18, **kwargs)
        self.__dict__.update(kwargs)

    def bunch(self, dls):
        if isinstance(dls, dict):
            dls = Im2MOS() << dls
        return dls

    def create_head(self):
        return create_paq2piq_head()


class P2P_RM(RoIPoolModel): # old version, use to load CVPR2020 public model
    model_state_url = 'https://github.com/baidut/fastiqa/releases/download/v2.0.0/P2P-RM.pth'
    def create_head(self):
        return create_paq2piq_head()


    def bunch(self, dls):
        if isinstance(dls, dict):
            dls = ImRoI2MOS() << dls
        return dls


    def __init__(self, *args, **kwargs):
        super().__init__(*args, backbone = resnet18, pool_size = (2,2), **kwargs)
        self.__dict__.update(kwargs)

    def demo_on_image(self, file):
        def open_image(fname):
            img = PILImage.create(fname)
            # PIL.Image.open(fname).convert('RGB')
            # img = img.resize((size, size))
            t = torch.Tensor(np.array(img))
            return t.permute(2,0,1).float()/255.0

        model.eval() ### important (otherwise the output is random and noisy)
        img = PILImage.create(file)
        sample =open_image(file).unsqueeze(0) #.cuda()
        blk_size = [[20,20]]
        model.input_block_rois(blk_size, [sample.shape[-2], sample.shape[-1]], device=sample.device)  # self.dls.img_raw_size
        t = model(sample).cpu()
        QualityMap(t[0].data[1:].reshape(blk_size[0]), img, t[0].data[0])


class P2P_FM(P2P_RM):
    contain_image_roi = True
    refine_image_score = True
    model_state_url = 'https://github.com/baidut/fastiqa/releases/download/v2.0.0/P2P-FM.pth'

    @staticmethod
    def split_on(m):
        return [[m.body], [m.patch_head, m.image_head]]

    def __init__(self, finetuned=True, drop=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_pool = nn.Sequential(*([AdaptiveConcatPool2d(), Flatten()]))
        self.patch_pool = RoIPool((2, 2), 1 / 32)
        nf = num_features_model(self.body) * 2

        n_output = 1  # 1 image score
        self.patch_head = my_create_head(nf, 1)

        n_rois = 4 if self.contain_image_roi else 3
        self.image_head = nn.Sequential(*(
                bn_drop_lin(nf + n_rois, 512, bn=True, p=drop, actn=nn.ReLU(inplace=True))
                + bn_drop_lin(512, n_output, bn=True, p=drop, actn=None))
                                        )
        self.__dict__.update(kwargs)


    def create_head(self):
        return None  # no param for old head

    def forward(self, im_data: Tensor, rois_data: Tensor=None, rois_type=None, **kwargs) -> Tensor:
        # rois_type: im_bbox2mos  followed by TensorBBox TensorMultiCategory
        use_fixed_rois = rois_data is None or rois_data.size(1) == 4

        batch_size = im_data.size(0)  # may change (last batch)
        idx = get_idx(batch_size, 4)

        if use_fixed_rois:
            self.input_fixed_rois(img_size=[im_data.size(-2), im_data.size(-1)], batch_size=batch_size)
            rois_data = self.rois.cuda()

        base_feat = self.body(im_data)

        if self.contain_image_roi:
            rois_image = rois_data.view(batch_size, -1).clone()[:, :4].reshape(-1, 4)
            image_idx = torch.arange(batch_size, dtype=torch.float).view(1, -1).t().cuda()
            rois_image = torch.cat((image_idx, rois_image), 1)

        rois_data = rois_data.view(-1, 4)
        rois = torch.cat((idx.cuda(), rois_data), 1)

        patch_feat = self.patch_pool(base_feat, rois)

        if self.contain_image_roi:
            pool_feat = self.patch_pool(base_feat, rois_image)
            image_feat = self.image_pool(pool_feat)
        else:
            image_feat = self.image_pool(base_feat)

        patch_pred = self.patch_head(patch_feat).view(batch_size, -1)

        if not self.contain_image_roi:
            patch_pred = patch_pred[:, 1:]

        concat_feat = torch.cat((image_feat, patch_pred), 1)
        image_pred = self.image_head(concat_feat)

        if use_fixed_rois:
            return image_pred
        else:
            if self.contain_image_roi:
                # update
                patch_pred[:, ::4] = image_pred
                return patch_pred
            else:
                return torch.cat((image_pred, patch_pred), 1)
