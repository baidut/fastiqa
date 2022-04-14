"""
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from fastiqa.models.all import *
from fastiqa.bunches.im_roi2mos import *
from fastiqa.bunches.base import to_json
import torchvision.models as models
from fastiqa.learn import *
dls = ImRoI2MOS.from_json('json/LIVE_FB_IQA.json', bs=3)
model = BodyHeadModel(backbone=models.resnet18)
learn = IqaLearner(dls, model)
to_json(learn, 'json/example_learner.json')
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from .resnet_3d import *
from .resnet3d.resnext import *
from .ts import *
"""

# from .baiscs import to_json
from . import *
from fastai.vision.all import *
# from .basics import IqaModel
from .bunch import * # All, IqaDataBunch
from .metrics import *
# from dataclasses import dataclass
import seaborn as sns
from loguru import logger


def create_sequence_batch(data):
    xs, ys = [], []
    for d in data:
        xs.append(d[0])
        ys.append(d[1])

    try:
        xs_ = torch.cat([TensorImage(torch.cat([im[None] for im in x], dim=0))[None] for x in xs], dim=0)
        ys_ = torch.cat([y[None] for y in ys], dim=0)
    except:
        print('Error@create_sequence_batch')
        for idx, x in enumerate(xs):
            print(idx, x.size()) # torch.Size([3, 500, 500])
        xs_ = torch.cat([TensorImage(torch.cat([im[None] for im in x], dim=0))[None] for x in xs], dim=0)
        print('xs_ is ok')
        print(xs_)
        raise

    return TensorImage(xs_), TensorCategory(ys_)
#
class IqaLearner(Learner):
    monitor = 'valid_loss'
    #log = False

    @property
    def abbr(self):
        assert hasattr(self.model, '__name__')
        assert hasattr(self.dls, '__name__')
        return self.model.__name__ + ' @' + self.dls.__name__

    # def to_json(self):
    #     return {'dls':self.dls, 'model':self.model}

    def __init__(self, dls, model, *args, splitter=None, metrics=[SRCC(), LCC()],
            cbs=None, add_cbs=[], loss_func=MSELossFlat(), **kwargs):
        # create a new instance each time!!!
        # otherwise it will use the same object

        if hasattr(model, 'bunch'):
            if isinstance(dls, IqaDataBunch):
                dls._bunching = True
            # dls being a dict or string?
            # no __name__ attribute
            # add __str__ ?
            # TODO: show changed properties when bunching
            logger.info(f'Bunching ... {model.__name__}')
            dls = model.bunch(dls)
            logger.info(f'Bunched ... {model.__name__}')
            if isinstance(dls, IqaDataBunch):
                dls._bunching = False
        else:
            logger.warning(f'model {model.__name__} did not implement bunch(dls) method')

        if hasattr(model, 'split_on'):
            logger.info('Discriminative layer training is enabled')
            self.splitter=model.split_on

        #if self.log: wandb.init()
        if cbs is None:
            # the first metric
            if metrics is not None and len(metrics) >= 1:
                # callable(metrics[0])
                self.monitor = metrics[0].__name__ if hasattr(metrics[0], '__name__') else metrics[0].name # getattr('name', metrics[0].__class__.__name__)
            cbs = [CSVLogger(append=True), ShowGraphCallback(), SaveModelCallback(monitor=self.monitor)]
            #if self.log: cbs += [WandbCallback(log_preds=False)] # too slow
            logger.info(f'monitoring {self.monitor}')
        super().__init__(dls, model, *args, metrics=metrics, cbs=cbs+add_cbs, loss_func=loss_func, **kwargs)

        # [Discriminative layer training](https://docs.fast.ai/basic_train.html#Discriminative-layer-training)
        # if hasattr(self.model, 'splitter'):
        #     #self.split(self.model.split_on)
        #     self.splitter=model.splitter

        # ShowGraphCallback()
        # SaveModelCallback()

        # self.callback_fns += [ShowGraph, partial(CSVLogger, append=True),
        #                       partial(SaveModelCallback, every='improvement',
        #                               monitor=self.metrics[0].name)]

        # predict on database (cached) and get numpy predictions

    def get_np_preds(self, db=None, cache=True, jointplot=False, **kwargs):
        """
        get numpy predictions on a bunched database
        TODO: check bunched
        IqaLearner assumes that the output is only a scalar number
            output = preds[0]
            target = preds[1]
        """
        on = db #rename

        if db is None:
            db = self.dls

        # if we don't flatten it, then we cannot store it.
        # so we need to only get the valid output
        # rois_learner gives three output csv

        # load dls here...... TOFIX
        # if isinstance(self.model, IqaModel):
        #
        if hasattr(self.model, 'bunch'):
            db = self.model.bunch(db)

        metric_idx = db.metric_idx
        # suffixes = ['', '_patch_1', '_patch_2', '_patch_3']
        suffixes = ['', '_p1', '_p2', '_p3']

        csv_file = self.path / ('valid@' + db.__name__ + suffixes[metric_idx] + '.csv')
        if os.path.isfile(csv_file) and cache:
            logger.warning(f'load cache {csv_file}')
            df = pd.read_csv(csv_file)
            output = np.array(df['output'].tolist())
            target = np.array(df['target'].tolist())
        else:
            c = db.c if type(db.c) == int else db.c[-1]
            logger.debug(f'validating... {self.model.__name__}@{db.__name__} (c={c})')
            # db.c = 1
            # TODO fuse duplicate code with rois_learner
            current_data = self.dls
            self.dls = db
            preds = self.get_preds()
            self.dls = current_data

            output, target = preds

            if not isinstance(output,(np.ndarray)):
                output = output.flatten().numpy()
                target = target.flatten().numpy()

            # print(len(output))
            # print(len(target))
            """
            preds is a list [output_tensor, target_tensor]
            torch.Size([8073, 4])
            """
            # don't call self.data.c to avoid unnecessary data loading
            # n_output = self.data.c  # only consider image score
            # no need since metric will take care of it
            # print(np.array(output).shape, np.array(target).shape) # (233, 1) (233,)
            if cache:
                # # we already loaded the data, so feel free to call data.c?
                if c == 4:
                    logger.debug('db.c==4')
                    if len(output) == len(target):
                        for n in [0, 1, 2, 3]:
                            df = pd.DataFrame({'output': output[n::c], 'target': target[n::c]})
                            csv_file = self.path / ('valid@' + db.__name__ + suffixes[n] + '.csv')
                            df.to_csv(csv_file, index=False)

                    elif c*len(output) == len(target):
                        df = pd.DataFrame({'output': output, 'target': target[0::c]})
                        csv_file = self.path / ('valid@' + db.__name__  + '.csv')
                        df.to_csv(csv_file, index=False)
                    else:
                        raise
                elif c == 2:
                    logger.debug('db.c==2')
                    for n, roi_index in enumerate(db.feats[0].roi_index):
                        df = pd.DataFrame({'output': output[n::c], 'target': target[n::c]})
                        csv_file = self.path / ('valid@' + db.__name__ + suffixes[roi_index] + '.csv')
                        df.to_csv(csv_file, index=False)
                elif c == 1:
                    logger.debug('db.c==1')
                    df = pd.DataFrame({'output': output, 'target': target})
                    df.to_csv(csv_file, index=False)
                else:
                    raise NotImplementedError

            if c*len(output)!=len(target): # dirty fix
                output = output[db.metric_idx::c]

            target = target[db.metric_idx::c]

        if cache and jointplot:
            # p = sns.jointplot(x="output", y="target", data=df)
            #plt.subplots_adjust(top=0.9)

            # size: 30k 2   1k 5
            g = sns.jointplot(x="output", y="target", data=df, kind="reg", marker = '.', scatter_kws={"s": 5},
                  xlim=(0, 100), ylim=(0, 100)) # color="r",
            plt.suptitle(f"{self.model.__name__}@{db.__name__}")
            #g.fig.suptitle(f"{self.model.__name__}@{db.__name__}") # https://stackoverflow.com/questions/60358228/how-to-set-title-on-seaborn-jointplot
            #g.annotate(stats.pearsonr)
            # g = sns.JointGrid(x="output", y="target", data=df) # ratio=100
            # g.plot_joint(sns.regplot)
            # g.annotate(stats.pearsonr)
            # g.ax_marg_x.set_axis_off()
            # g.ax_marg_y.set_axis_off()
        return output, target

    def valid(self, db=None, metrics=None, cache=True, jointplot=True, all_items=False, **kwargs):
        """
        all_items/ True: test on all items, False: test on items in valid subset.
        """
        def valid_one(data):
            # logger.debug(f'validating... {self.model.__name__}@{data.__name__}')
            logger.debug(f'validating... {self.model.__name__}@{data.__name__}')
            output, target = self.get_np_preds(on=data, cache=cache, jointplot=jointplot, **kwargs)  # TODO note here only output 1 scores
            output, target = torch.from_numpy(output), torch.from_numpy(target)
            return {metric.name: metric(output, target) for metric in metrics}

        if metrics is None: metrics = self.metrics

        # avoid changing self.data
        if db is None:
            db = self.dls

        if not isinstance(db,  (list, tuple)  ):
            db = [db]

        # call model.bunch after dls.bunch
        # if don't bunch, name is not available
        # bunch won't take time !!!! just update attributes, change svr model to allow that. only cache X when needed !!!

        # assert self.dls != None
        # automatically convert to all data? no, leave it for the users
        # All(x) if x['__name__'] != self.dls.__name__ else x
        # on = [self.dls.bunch(x, **kwargs) if isinstance(x, (str,dict)) else x for x in on]

        # call model.bunch instead
        db = [self.model.bunch(x, **kwargs) if isinstance(x, (str,dict)) else x for x in db]
        #don't change c
        # must call model.bunch
        # on = [self.model.bunch(x, **kwargs) for x in on]
        records = [valid_one(data) for data in db]
        return pd.DataFrame(records, index=[data.__name__ for data in db]) # abbr

    # on=None, if on is None: on = self.dls
    # Keep it simple: by default, extract self.dls
    def extract_features(self, name=None, create_batch=None, cache=True, skip_exist=False):
        # Learner.get_preds will get preds on valid set without shuffling and drop_last_batch
        # clip/frame features combined into one feature
        # output numpy features (192, 512, 2, 2)
        if name is None:
          if hasattr(self.model, 'backbone'):
            name = self.model.backbone.__name__
          else:
            name = self.model.__class__.__name__
        npy_file = self.dls.path / (f'features/{name}') / (self.dls.__name__ + '.npy')
        # there might be / in on.__name__ to help create sub folders
        npy_file.parent.mkdir(parents=True, exist_ok=True)
        # self.path / (f'{name}@' + on.__name__ + '.npy')
        if cache and npy_file.absolute().exists():
            if skip_exist: return None
            with open(npy_file, 'rb') as f:
                features = np.load(f)
        else:
            # try:
            old_setting = self.model.output_features
            self.model.output_features = True # TODO:  put it outside
            # reset roi
            if hasattr(self.model, 'rois'):
              self.model.rois = None

            ds_idx = 1 # valid
            dl = self.dls.get_data()[ds_idx].new(shuffle=False, drop_last=False)
            #dl.create_batch = create_sequence_batch
            # # AttributeError: 'MultiClip' object has no attribute 'transpose'
            if self.model.is_3d:
              # raise NotImplementedError('use create_sequence_batch')
              dl.create_batch = create_sequence_batch # self.dls.create_batch if create_batch is None else create_batch
            preds = self.get_preds(dl=dl) # get preds on one video with several batches
            self.model.output_features = old_setting
            features = preds[0]
            if cache:
                with open(npy_file, 'wb') as f:
                    np.save(f, features)

            # except: # memory issue
            #     return None
        return features

    def extract_scores(self, name=None, create_batch=None, cache=True, skip_exist=False):
        # Learner.get_preds will get preds on valid set without shuffling and drop_last_batch
        # clip/frame features combined into one feature
        # output numpy features (192, 512, 2, 2)
        if name is None: name = self.model.backbone.__name__
        npy_file = self.dls.path / (f'features/{name}#scores') / (self.dls.__name__ + '.npy')
        # there might be / in on.__name__ to help create sub folders
        npy_file.parent.mkdir(parents=True, exist_ok=True)
        # self.path / (f'{name}@' + on.__name__ + '.npy')
        if cache and npy_file.absolute().exists():
            if skip_exist: return None
            with open(npy_file, 'rb') as f:
                features = np.load(f)
        else:
            # try:
            ds_idx = 1 # valid
            dl = self.dls.get_data()[ds_idx].new(shuffle=False, drop_last=False)

            # reset roi
            sample = self.dls.one_batch()[0]
            blk_size = [[16,16], [8,8], [4,4], [2,2]]
            # each sample different shape
            self.model.input_block_rois(blk_size, [sample.shape[-2], sample.shape[-1]], device=sample.device)
            if self.model.is_3d:
              dl.create_batch = create_sequence_batch # self.dls.create_batch if create_batch is None else create_batch
            preds = self.get_preds(dl=dl) # get preds on one video with several batches
            features = preds[0]
            if cache:
                with open(npy_file, 'wb') as f:
                    np.save(f, features)

            # except: # memory issue
            #     return None
        return features




 # TestLearner(dls, model, metrics=[SRCC(), LCC()])
class TestLearner(IqaLearner):
    # don't use None, it will be filled in by IqaLearner
    # assert self.monitor in self.recorder.metric_names[1:]

    # metrics=()
    def __init__(self, dls=None, model=None, cbs=[], loss_func=DummyLoss,
            **kwargs):
        super().__init__(dls, model, cbs=cbs, loss_func=loss_func, **kwargs)
