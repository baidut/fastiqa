from ._roi_pool import *
from torchvision.ops import RoIPool, RoIAlign

def bn_drop_lin(n_in:int, n_out:int, bn:bool=True, p:float=0., actn:Optional[nn.Module]=None):
    "Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`."
    layers = [nn.BatchNorm1d(n_in)] if bn else []
    if p != 0: layers.append(nn.Dropout(p))
    layers.append(nn.Linear(n_in, n_out, bias=True))
    if actn is not None: layers.append(actn)
    return layers

def my_create_head(nf:int, nc:int, lin_ftrs=None, ps=0.5,
                concat_pool:bool=True, bn_final:bool=False):
    "Model head that takes `nf` features, runs through `lin_ftrs`, and about `nc` classes."
    lin_ftrs = [nf, 512, nc] if lin_ftrs is None else [nf] + lin_ftrs + [nc]
    ps = listify(ps)
    if len(ps) == 1: ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps
    actns = [nn.ReLU(inplace=True)] * (len(lin_ftrs)-2) + [None]
    pool = AdaptiveConcatPool2d() if concat_pool else nn.AdaptiveAvgPool2d(1)
    layers = [pool, Flatten()]
    for ni,no,p,actn in zip(lin_ftrs[:-1], lin_ftrs[1:], ps, actns):
        layers += bn_drop_lin(ni, no, True, p, actn)
    if bn_final: layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
    return nn.Sequential(*layers)


class P2P_RM(LegacyRoIPoolModel): # old version, use to load CVPR2020 public model
    PATH_TO_MODEL_STATE = '/media/zq/FB8T/pth/P2P-RM.pth'
    def __init__(self, finetuned=True, *args, **kwargs):
        super().__init__(*args, backbone = resnet18, pool_size = (2,2), **kwargs)
        self.__dict__.update(kwargs)
        if finetuned:
            print('load finetuned model...')
            model_state = torch.load(self.PATH_TO_MODEL_STATE, map_location=lambda storage, loc: storage)
            self.load_state_dict(model_state) # ["model"]

class P2P_FM(LegacyRoIPoolModel):
    contain_image_roi = True
    refine_image_score = True
    PATH_TO_MODEL_STATE = '/media/zq/FB8T/pth/P2P-FM.pth'

    @staticmethod
    def split_on(m):
        return [[m.body], [m.patch_head, m.image_head]]

    def __init__(self, finetuned=True, drop=0.5, *args, **kwargs):
        super().__init__(*args, backbone = resnet18, pool_size = (2,2), **kwargs)
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
        if finetuned:
            print('load finetuned model...')
            model_state = torch.load(self.PATH_TO_MODEL_STATE, map_location=lambda storage, loc: storage)
            self.load_state_dict(model_state) # ["model"]

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
