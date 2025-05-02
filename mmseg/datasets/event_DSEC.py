import os
import os.path as osp
from functools import reduce
import weakref
import h5py
import torch

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable
from torch.utils.data import Dataset
from pathlib import Path

from mmseg.core import eval_metrics
from mmseg.utils import get_root_logger, EventSlicer, VoxelGrid
from .builder import DATASETS
from .pipelines import Compose

@DATASETS.register_module()
class EventDSECDataset(Dataset):
    """Custom dataset for semantic segmentation. An example of file structure
    is as followed.

    ./BRENet                                 # current (project) directory
    └── data                                 # various datasets
      └── DSEC
          ├── train               
          │   ├── zurich_city_00_a           # various sequencese
          │   │   ├── 11classes              # segmentation labels
          │   │   │   ├── 000000.png
          │   │   │   └── ...
          │   │   ├── 19classes
          │   │   │   ├── 000000.png
          │   │   │   └── ...
          │   │   ├── events                 # event data
          │   │   |   └── left
          │   │   |       ├── events.h5
          │   │   |       └── rectify_map.h5
          │   │   └── images                 # images
          │   │       ├── left
          │   │       |   └── ev_inf
          │   │       |       ├── 000000.png
          │   │       |       └── ...
          │   │       └── timestamps.txt     # event timestamp data
          │   ├── zurich_city_01_a
          │   ├── zurich_city_02_a
          │   ├── zurich_city_04_a
          │   ├── zurich_city_05_a
          │   ├── zurich_city_06_a
          │   ├── zurich_city_07_a
          │   └── zurich_city_08_a
          └── test
              ├── zurich_city_13_a
              ├── zurich_city_14_a
              └── zurich_city_15_a
              
    Args:
        pipeline (list[dict]): Processing pipeline
        img_dir (str): Path to image directory
        img_suffix (str): Suffix of images. Default: '.jpg'
        ann_dir (str, optional): Path to annotation directory. Default: None
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
        split (str, optional): Split txt file. If split is specified, only
            file with suffix in the splits will be loaded. Otherwise, all
            images in img_dir/ann_dir will be loaded. Default: None
        data_root (str, optional): Data root for img_dir/ann_dir. Default:
            None.
        test_mode (bool): If test_mode=True, gt wouldn't be loaded.
        ignore_index (int): The label index to be ignored. Default: 255
        reduce_zero_label (bool): Whether to mark label zero as ignored.
            Default: False
        classes (str | Sequence[str], optional): Specify classes to load.
            If is None, ``cls.CLASSES`` will be used. Default: None.
        palette (Sequence[Sequence[int]]] | np.ndarray | None):
            The palette of segmentation map. If None is given, and
            self.PALETTE is None, random palette will be generated.
            Default: None
    """

    CLASSES = None

    PALETTE = None

    def __init__(self,
                 pipeline,
                 img_dir,
                 img_suffix='.png',
                 ann_dir=None,
                 seg_map_suffix='.png',
                 height = 480,
                 width = 640,
                 num_bins = 15,
                 split='train',
                 data_root=None,
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None):
        if split=='train':
            self.pipeline_load = Compose(pipeline[:2])
            self.pipeline_process = Compose(pipeline[2:])
        else:
            self.pipeline_load = Compose(pipeline[:1])
            self.pipeline_process = Compose(pipeline[1:])
        
        self.delta_t_ms = 50
        self.delta_t_us = self.delta_t_ms * 1000
        self.events_per_data = 40000
        self.events = self.events_per_data * num_bins
        self.fixed_interval = True

        self.pipeline = Compose(pipeline)
        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.ann_dir = ann_dir
        self.seg_map_suffix = seg_map_suffix
        self.height = height
        self.width = width
        self.num_bins = num_bins
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes, palette)
        self.voxel_grid = VoxelGrid((self.num_bins, self.height, self.width), normalize=True)
        self.train_sequences_namelist = ['zurich_city_00_a', 'zurich_city_01_a', 'zurich_city_02_a',
                                        'zurich_city_04_a', 'zurich_city_05_a', 'zurich_city_06_a',
                                        'zurich_city_07_a', 'zurich_city_08_a']
        self.val_sequences_namelist = ['zurich_city_13_a', 'zurich_city_14_c', 'zurich_city_15_a']
        self.remove_time_window = 250
        self.start_frame = (self.remove_time_window // 100 + 1) * 2

        # join paths if data_root is specified
        assert str(self.data_root)

        self.timestamps_flow = {}
        self.event_slicer = {}
        self.rectify_ev_map = {}

        # load data
        self.img_infos = self.load_data(self.data_root, self.img_suffix, self.split)
        self.load_event(self.data_root, self.split)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)

    def events_to_voxel_grid(self, p, t, x, y):
        t = (t - t[0]).astype('float32')
        t = (t/t[-1])
        x = x.astype('float32')
        y = y.astype('float32')
        pol = p.astype('float32')
        event_data_torch = {
            'p': torch.from_numpy(pol),
            't': torch.from_numpy(t),
            'x': torch.from_numpy(x),
            'y': torch.from_numpy(y),
        }
        return self.voxel_grid.convert(event_data_torch)

    def rectify_events(self, seq_name, x: np.ndarray, y: np.ndarray):
        # assert location in self.locations
        # From distorted to undistorted
        rectify_map = self.rectify_ev_map[seq_name]
        assert rectify_map.shape == (self.height, self.width, 2), rectify_map.shape
        assert x.max() < self.width
        assert y.max() < self.height
        return rectify_map[y, x]

    @staticmethod
    def close_callback(h5f):
        h5f.close()

    def load_event(self, dataroot, split):
        data_path = Path(dataroot)

        if split == 'train':
            main_path = data_path / 'train'
            sequences_namelist = self.train_sequences_namelist
        elif split == 'val':
            main_path = data_path / 'test'
            sequences_namelist = self.val_sequences_namelist

        for child in main_path.iterdir():
            if any(k in str(child) for k in sequences_namelist):
                img_dir = child / 'images' / 'left' / 'ev_inf'
                event_dir = child / 'events' / 'left'
                sequences_name = osp.split(child)[1]

                #Load and compute timestamps and indices
                timestamp_prefix = Path(str(img_dir).split('/left')[0])
                timestamps_images = np.loadtxt(timestamp_prefix / 'timestamps.txt', dtype='int64')
                # But only use every second one because we train at 10 Hz, and we leave away the 1st & last one
                
                self.timestamps_flow[sequences_name] = timestamps_images[self.start_frame:-self.start_frame]

                # Left events only
                event = event_dir / 'events.h5'
                rec_map = event_dir / 'rectify_map.h5'

                h5f_location = h5py.File(event, 'r')
                self.h5f = h5f_location
                self.event_slicer[sequences_name] = EventSlicer(h5f_location)
                with h5py.File(rec_map, 'r') as h5_rect:
                    self.rectify_ev_map[sequences_name] = h5_rect['rectify_map'][()]

                self._finalizer = weakref.finalize(self, self.close_callback, self.h5f)

    def load_data(self, data_root, img_suffix, split):
        """Load annotation from directory.
        Args:
            data_path (str): Path to image directory
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None
        Returns:
            list[dict]: All image info of dataset.
        """

        data_path = Path(data_root)

        img_infos = []
        if split == 'train':
            main_path = data_path / 'train'
            sequences_namelist = self.train_sequences_namelist
        elif split == 'val':
            main_path = data_path / 'test'
            sequences_namelist = self.val_sequences_namelist
        
        assert main_path.is_dir(), str(main_path)
    
        for child in main_path.iterdir():
            if any(k in str(child) for k in sequences_namelist):
                img_dir = child / 'images' / 'left' / 'ev_inf'
                ann_dir = child / '11classes'
                sequences_name = osp.split(child)[1]

                num_img = len(list(mmcv.scandir(img_dir, img_suffix, recursive=True)))

                for idx, img in enumerate(sorted(mmcv.scandir(img_dir, img_suffix, recursive=True))):
                    if idx > (self.start_frame - 1) and idx < (num_img - self.start_frame):
                        img_info = dict(filename=img, img_dir=img_dir, filepath=img_dir / img, seq_index=idx-self.start_frame,
                                        seq_name=sequences_name, remove_rows='DSEC')
                        if ann_dir is not None:
                            seg_map = ann_dir / img
                            img_info['ann'] = dict(ann_dir=ann_dir, seg_map=seg_map)
                        img_infos.append(img_info)
            else:
                continue

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos

    def get_aux_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        return self.img_infos[idx]['ann']

    def prepare_event(self, seq_name, seq_index):
        # First entry corresponds to all events BEFORE the flow map
        # Second entry corresponds to all events AFTER the flow map (corresponding to the actual fwd flow)
        names = ['event_volume_old', 'event_volume_new']
        ts_start = [self.timestamps_flow[seq_name][seq_index] - self.delta_t_us, self.timestamps_flow[seq_name][seq_index]]
        ts_end = [self.timestamps_flow[seq_name][seq_index], self.timestamps_flow[seq_name][seq_index] + self.delta_t_us]

        event_info = {
            'sequence_index': seq_index,
            'timestamp': self.timestamps_flow[seq_name][seq_index]
        }

        if self.fixed_interval == True:
            for i in range(len(names)):
                event_data = self.event_slicer[seq_name].get_events(ts_start[i], ts_end[i])

                p = event_data['p']
                t = event_data['t']
                x = event_data['x']
                y = event_data['y']

                xy_rect = self.rectify_events(seq_name, x, y)
                x_rect = xy_rect[:, 0]
                y_rect = xy_rect[:, 1]

                event_representation = self.events_to_voxel_grid(p, t, x_rect, y_rect)
                event_info[names[i]] = event_representation[:, :-40, :]
        else:
            for i in range(len(names)):
                event_data = self.event_slicer[seq_name].get_events_fixed_num(ts_end[i], self.events)

                if self.events >= event_data['t'].size:
                    start_index = 0
                else:
                    start_index = -self.events

                p = event_data['p'][start_index:]
                t = event_data['t'][start_index:]
                x = event_data['x'][start_index:]
                y = event_data['y'][start_index:]

                xy_rect = self.rectify_events(seq_name, x, y)
                x_rect = xy_rect[:, 0]
                y_rect = xy_rect[:, 1]

                event_representation = self.events_to_voxel_grid(p, t, x_rect, y_rect)
                event_info[names[i]] = event_representation[:, :-40, :]

        return event_info

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_aux_info(idx)
        event_info = self.prepare_event(img_info['seq_name'], img_info['seq_index'])

        results = dict(img_info=img_info, ann_info=ann_info, event_info=event_info)
        results['seg_fields'] = []
        results['img_prefix'] = img_info['img_dir']
        results['seg_prefix'] = img_info['ann']['ann_dir']

        self.pipeline_load(results)
        return self.pipeline_process(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys intorduced by
                piepline.
        """
        img_info = self.img_infos[idx]
        ann_info = self.get_aux_info(idx)
        event_info = self.prepare_event(img_info['seq_name'], img_info['seq_index'])
        results = dict(img_info=img_info, ann_info=ann_info, event_info=event_info)

        self.pipeline_load(results)
        return self.pipeline_process(results)

    def format_results(self, results, **kwargs):
        """Place holder to format result to dataset specific output."""
        pass

    def get_gt_seg_maps(self, efficient_test=False):
        """Get ground truth segmentation maps for evaluation."""
        gt_seg_maps = []
        for img_info in self.img_infos:
            seg_map = img_info['ann']['seg_map']
            if efficient_test:
                gt_seg_map = seg_map
            else:
                gt_seg_map = mmcv.imread(
                    seg_map, flag='unchanged', backend='pillow')
            gt_seg_maps.append(gt_seg_map)
        return gt_seg_maps

    def get_classes_and_palette(self, classes=None, palette=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
            palette (Sequence[Sequence[int]]] | np.ndarray | None):
                The palette of segmentation map. If None is given, random
                palette will be generated. Default: None
        """
        if classes is None:
            self.custom_classes = False
            return self.CLASSES, self.PALETTE

        self.custom_classes = True
        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        if self.CLASSES:
            if not set(classes).issubset(self.CLASSES):
                raise ValueError('classes is not a subset of CLASSES.')

            # dictionary, its keys are the old label ids and its values
            # are the new label ids.
            # used for changing pixel labels in load_annotations.
            self.label_map = {}
            for i, c in enumerate(self.CLASSES):
                if c not in class_names:
                    self.label_map[i] = -1
                else:
                    self.label_map[i] = classes.index(c)

        palette = self.get_palette_for_custom_classes(class_names, palette)

        return class_names, palette

    def get_palette_for_custom_classes(self, class_names, palette=None):

        if self.label_map is not None:
            # return subset of palette
            palette = []
            for old_id, new_id in sorted(
                    self.label_map.items(), key=lambda x: x[1]):
                if new_id != -1:
                    palette.append(self.PALETTE[old_id])
            palette = type(self.PALETTE)(palette)

        elif palette is None:
            if self.PALETTE is None:
                palette = np.random.randint(0, 255, size=(len(class_names), 3))
            else:
                palette = self.PALETTE

        return palette

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 efficient_test=False,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU' and
                'mDice' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """

        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))
        eval_results = {}
        gt_seg_maps = self.get_gt_seg_maps(efficient_test)
        if self.CLASSES is None:
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)
        ret_metrics = eval_metrics(
            results,
            gt_seg_maps,
            num_classes,
            self.ignore_index,
            metric,
            label_map=self.label_map,
            reduce_zero_label=self.reduce_zero_label)
        class_table_data = [['Class'] + [m[1:] for m in metric] + ['Acc']]
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES
        ret_metrics_round = [
            np.round(ret_metric * 100, 2) for ret_metric in ret_metrics
        ]
        for i in range(num_classes):
            class_table_data.append([class_names[i]] +
                                    [m[i] for m in ret_metrics_round[2:]] +
                                    [ret_metrics_round[1][i]])
        summary_table_data = [['Scope'] +
                              ['m' + head
                               for head in class_table_data[0][1:]] + ['aAcc']]
        ret_metrics_mean = [
            np.round(np.nanmean(ret_metric) * 100, 2)
            for ret_metric in ret_metrics
        ]
        summary_table_data.append(['global'] + ret_metrics_mean[2:] +
                                  [ret_metrics_mean[1]] +
                                  [ret_metrics_mean[0]])
        print_log('per class results:', logger)
        table = AsciiTable(class_table_data)
        print_log('\n' + table.table, logger=logger)
        print_log('Summary:', logger)
        table = AsciiTable(summary_table_data)
        print_log('\n' + table.table, logger=logger)

        for i in range(1, len(summary_table_data[0])):
            eval_results[summary_table_data[0]
                         [i]] = summary_table_data[1][i] / 100.0
        if mmcv.is_list_of(results, str):
            for file_name in results:
                os.remove(file_name)
        return eval_results
