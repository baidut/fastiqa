__all__ = ['BodyHeadModel']

from fastai.vision.all import *
import fastai
from .. import * # IqaModel
from ._3d import *
from packaging import version
# from ..utils.cached_property import cached_property

def abbreviate(x):
    abbreviations = ["", "K", "M", "B", "T", "Qd", "Qn", "Sx", "Sp", "O", "N",
                     "De", "Ud", "DD"]
    thing = "1"
    a = 0
    while len(thing) < len(str(x)) - 3:
        thing += "000"
        a += 1
    b = int(thing)
    thing = round(x / b, 2)
    return str(thing) + " " + abbreviations[a]
#
from fastai.vision.learner import _get_first_layer, _load_pretrained_weights, _update_first_layer

def _my_update_first_layer(model, n_in, pretrained):
    "Change first layer based on number of input channels"
    if n_in == 3: return
    first_layer, parent, name = _get_first_layer(model)
    if n_in == 1 and getattr(first_layer, 'in_channels') == 1: return  # audio network: yamnet
    _update_first_layer(model, n_in, pretrained)

def my_create_body(arch, n_in=3, pretrained=True, cut=None):
    "Cut off the body of a typically pretrained `arch` as determined by `cut`"
    model = arch(pretrained=pretrained)
    # AssertionError: Change of input channels only supported with Conv2d, found Conv3d
    if '3d' not in arch.__name__:
      _my_update_first_layer(model, n_in, pretrained)

    if 'x3d' in arch.__name__:
      layers = list(model.children())
      # return nn.Sequential(*layers[0][:-1], layers[0][-1].pool, layers[0][-1].output_pool)
      # keep the size
      return nn.Sequential(*layers[0][:-1], layers[0][-1].pool, nn.AdaptiveAvgPool3d(output_size=(1, None, None)))
    if 'movinet3dA' in arch.__name__:
      print('custom cut for movinet3dAx')
      layers = list(model.children())
      return nn.Sequential(*layers[:-2], nn.AdaptiveAvgPool3d(output_size=(1, None, None)))

    #cut = ifnone(cut, cnn_config(arch)['cut'])
    if cut is None:
        ll = list(enumerate(model.children()))
        if len(ll) == 1:
          print('len(ll) == 1, cut is set to -1')
          cut = -1
        else:
          cut = next(i for i,o in reversed(ll) if has_pool_type(o))
    if   isinstance(cut, int):
      ##########
      layers = list(model.children())
      if len(layers) == 1:
        print('len(layers) == 1')
        layers = layers[0]
      return nn.Sequential(*layers[:cut])
    elif callable(cut): return cut(model)
    else: raise NameError("cut must be either integer or a function")

class BodyHeadModel(IqaModel):
    backbone = None
    is_3d = None
    clip_num = 1
    head_num = 1
    _num_features_model = None
    num_features_backbone = None
    _create_head = None
    cut = None # https://docs.fast.ai/vision.learner#Cut-a-pretrained-model
    n_out_per_roi = 1
    n_in=3
    map_score = None

    @staticmethod
    def split_on(m):
        return [[m.body], [m.head]]

    @property
    def num_features(self):
        # '2.2.5' don't need  * 2, 2.0.18 need it
        # 128 sometime too small for model
        # 512
        base_feat = self.body(torch.empty(1, self.n_in, 16,640,640) if self.is_3d else torch.empty(1, self.n_in,128,128))
        num_features_backbone = base_feat.shape[1]
        t = 2 if version.parse(fastai.__version__) < version.parse("2.1.0") else 1
        return num_features_backbone * t #self._num_features_model(self.body) * t

    def create_body(self, pretrained=True):
      if self.cut is not None:
          return my_create_body(self.backbone, n_in=self.n_in, pretrained=pretrained, cut=self.cut)
      try:
          return my_create_body(self.backbone, n_in=self.n_in, pretrained=pretrained)
      except StopIteration:
          logging.warning('Cut pretraiend {self.backbone.__name__} at -1')
          return my_create_body(self.backbone, n_in=self.n_in, pretrained=pretrained, cut=-1)

    def create_head(self):
        return self._create_head(self.num_features * self.clip_num * self.head_num, self.n_out_per_roi, **self.head_kws)
        # output 1 score per image/video location
        # to learn distributions, set it to 5

    def __init__(self, backbone=None, pretrained=True, **kwargs):
        # remove simply fc
        # one could try only modify the last layer
        super().__init__(**kwargs)
        if backbone is not None:
            # self.__name__ += f' ({backbone.__name__})'
            self.backbone = backbone

        if self.is_3d is None:
            name = self.backbone.__name__
            self.is_3d = '3d' in name or '2p1d' in name or '2plus1d' in name or 'mc3' in name

        self._create_head = create_head_3d if self.is_3d else create_head
        self.body = self.create_body(pretrained=pretrained)
        #self._num_features_model = num_features_model_3d if self.is_3d else num_features_model
        # _num_features_model may not work for custom model, so we simply compute it
        # body[0].weight # AttributeError: 'ConvBlock3D' object has no attribute 'weight'
        self.head = self.create_head()
        if self.map_score:
          self.map = nn.Sigmoid()

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        # multi clip:
        #       bs x clip_num x clip_size x 3 x  H x W
        # -->   bs x clip_num x 3 x clip_size x H x W
        if self.is_3d and x.size()[-4] != 3 and x.size()[-3] == 3:
            x = x.transpose(-4,-3)
        # network.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        batch_size = x.size(0)
        # video data
        # if x.size()[1] != 3 and x.size()[2] == 3:
        #     x = x.transpose(1,2)
        base_feat = self.body(x)
        # if self.output_features:
        #     sz = (batch_size, -1, pooled_feat.size(1)) + self.pool_size
        #     return pooled_feat.view(*sz)
        # print('base_feat:',  base_feat.size()) # torch.Size([64, 8192, 128])
        pred = self.head(base_feat)
        score = pred.view(batch_size, -1)
        if self.map_score:
          return 100*self.map(score)

    @staticmethod
    def compare_paramters(backbones):
        body_params = []
        head_params = []
        total = []

        if type(backbones) is dict:
            labels = backbones.keys()
            backbones = backbones.values()
        elif type(backbones) is list or tuple:
            labels = [backbone.__name__ for backbone in backbones]
        else:
            raise TypeError('backbones must be a list, tuple or dict')

        for backbone in backbones:
            model = BodyHeadModel(backbone=backbone, n_in=1 if backbone.__name__=='yamnet' else 3, pretrained=False)
            # Give the number of parameters of a module ([0]) and if it's trainable or not ([1])
            body_param = total_params(model.body)[0]
            head_param = total_params(model.head)[0]

            body_params.append(body_param)
            head_params.append(head_param)
            total.append(body_param+head_param)

        width = 0.35       # the width of the bars: can also be len(x) sequence
        ind = np.arange(len(backbones))

        p1 = plt.barh(ind, body_params, width)
        p2 = plt.barh(ind, head_params, width, left=body_params) # botoom for bar, left for barh

        plt.xlabel('#parameters')
        plt.title('Model parameters')
        plt.yticks(ind, labels)
        #plt.yticks(np.arange(0, 81, 10))
        plt.legend((p1[0], p2[0]), ('backbone', 'head'))

        plt.show()

        # dataframe
        df = pd.DataFrame.from_dict({
          'name': labels,
          'body_params': body_params,
          'head_params': head_params,
          'total_params': total,
        })
        df.apply
        return df


# if fc:
#     return nn.Sequential(
#             nn.AdaptiveAvgPool2d(output_size=(1, 1)),
#             nn.Linear(in_features=num_features, out_features=self.n_out, bias=True)
#         )
        # if fc:
        #     return nn.Sequential(
        #             nn.AdaptiveAvgPool3d(output_size=(1, 1, 1)),
        #             nn.Linear(in_features=num_features, out_features=self.n_out, bias=True)
        #         )
