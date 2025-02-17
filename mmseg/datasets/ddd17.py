from .builder import DATASETS
from .event_DDD17 import EventDDD17Dataset
from IPython import embed

@DATASETS.register_module()
class DDD17Dataset(EventDDD17Dataset):
    """dsec dataset.
    """
    nclass = 6
    CLASSES = ('road', 'background', 'object', 'nature', 'human', 'vehicle')

    # random generated color
    PALETTE = [[128, 64, 128], [70, 70, 70], [220,220,  0], [107,142, 35], [220, 20, 60], [  0,  0,142]]


    assert len(CLASSES) == len(PALETTE)

    def __init__(self, **kwargs):
        super(DDD17Dataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            height = 260,
            width= 346,
            **kwargs)