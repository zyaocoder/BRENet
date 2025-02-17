from .builder import DATASETS
from .event_DSEC import EventDSECDataset
from IPython import embed

@DATASETS.register_module()
class DSECDataset(EventDSECDataset):
    """dsec dataset.
    """
    nclass = 11
    CLASSES = ('background', 'building', 'fence', 'person', 'pole', 'road', 'sidewalk',
            'vegetation', 'car', 'wall', 'traffic sign')

    # random generated color
    PALETTE = [[0, 0, 0], [70, 70, 70], [190,153,153], [220, 20, 60], [153,153,153], [128, 64,128], [244, 35,232],
               [107, 142, 35], [0, 0, 142], [102,102,156], [220,220, 0]]

    assert len(CLASSES) == len(PALETTE)

    def __init__(self, **kwargs):
        super(DSECDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            height = 480,
            width= 640,
            **kwargs)