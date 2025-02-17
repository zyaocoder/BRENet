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
from mmseg.utils import get_root_logger, generate_input_representation, normalize_voxel_grid
from .builder import DATASETS
from .pipelines import Compose

@DATASETS.register_module()
class EventDDD17Dataset(Dataset):
    """Custom dataset for semantic segmentation. An example of file structure
    is as followed.

    .. code-block:: none

        ├── data
        │   ├── my_dataset
        │   │   ├── img_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   │   ├── val
        │   │   ├── ann_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{seg_map_suffix}
        │   │   │   │   ├── yyy{seg_map_suffix}
        │   │   │   │   ├── zzz{seg_map_suffix}
        │   │   │   ├── val

    The img/gt_semantic_seg pair of CustomDataset should be of the same
    except suffix. A valid img/gt_semantic_seg filename pair should be like
    ``xxx{img_suffix}`` and ``xxx{seg_map_suffix}`` (extension is also included
    in the suffix). If split is given, then ``xxx`` is specified in txt file.
    Otherwise, all files in ``img_dir/``and ``ann_dir`` will be loaded.
    Please refer to ``docs/tutorials/new_dataset.md`` for more details.


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
                 height = 260,
                 width = 364,
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

        self.pipeline = Compose(pipeline)
        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.ann_dir = ann_dir
        self.seg_map_suffix = seg_map_suffix
        self.height = height
        self.width = width
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes, palette)
        self.train_sequences_namelist = ['dir0', 'dir3', 'dir4',
                                        'dir6', 'dir7']
        self.val_sequences_namelist = ['dir1']
        # shape: [200, 346]
        self.fixed_interval = False
        self.interval = 50

        self.nr_events_per_data = 120000

        self.nr_events_data = 5
        self.nr_temporal_bins = num_bins        
        self.nr_events = self.nr_events_data * self.nr_events_per_data
        # self.nr_events = self.nr_temporal_bins * self.nr_events_per_data
        
        self.shape = [260, 346]
        self.separate_pol = False
        self.normalize_event = True        
        
        # join paths if data_root is specified
        assert str(self.data_root)

        self.img_timestamp_event_idx = {}
        self.event_data = {}

        # load annotations
        self.img_infos = self.load_data(self.data_root, self.img_suffix, self.split)
        self.load_event(self.data_root, self.split)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)

    def extract_events_from_memmap(self, t_events, xyp_events, img_index, img_timestamp_event_idx, nr_events):
        if self.fixed_interval:
            _, event_idx, event_idx_before = img_timestamp_event_idx[img_index]
            _, event_idx_after, _ = img_timestamp_event_idx[img_index + 2]
            event_idx_before = max([event_idx_before, 0])
            event_idx_after = max([event_idx_after, 0])
        else:
            _, event_idx, _ = img_timestamp_event_idx[img_index]
            event_idx_before = max([event_idx - nr_events, 0])
            event_idx_after = max([event_idx + nr_events, 0])

        # forward
        events_between_imgs_fwd = np.concatenate([
            np.array(t_events[event_idx_before:event_idx], dtype="int64"),
            np.array(xyp_events[event_idx_before:event_idx], dtype="int64")
        ], -1)
        events_between_imgs_fwd = events_between_imgs_fwd[:, [1, 2, 0, 3]]  # events have format xytp, and p is in [0,1]

        # backward
        events_between_imgs_back = np.concatenate([
            np.array(t_events[event_idx:event_idx_after], dtype="int64"),
            np.array(xyp_events[event_idx:event_idx_after], dtype="int64")
        ], -1)
        events_between_imgs_back = events_between_imgs_back[:, [1, 2, 0, 3]]  # events have format xytp, and p is in [0,1]


        return events_between_imgs_fwd, events_between_imgs_back

    def load_event(self, dataroot, split):
        data_path = Path(dataroot)

        if split == 'train':
            main_path = data_path / 'train'
        elif split == 'val':
            main_path = data_path / 'test'

        for child in main_path.iterdir():
            sequence_dir = child
            sequences_name = osp.split(child)[1]

            if self.interval == 10:
                img_timestamp_event_idx = np.load(os.path.join(sequence_dir, "index/index_10ms.npy"))
            elif self.interval == 50:
                img_timestamp_event_idx = np.load(os.path.join(sequence_dir, "index/index_50ms.npy"))
            elif self.interval == 250:
                img_timestamp_event_idx = np.load(os.path.join(sequence_dir, "index/index_250ms.npy"))
            else:
                raise ValueError('No index file under given interval!')
            
            events_t_file = os.path.join(sequence_dir, "events.dat.t")
            events_xyp_file = os.path.join(sequence_dir, "events.dat.xyp")


            num_events = int(os.path.getsize(events_t_file) / 8)
            t_events = np.memmap(events_t_file, dtype="int64", mode="r", shape=(num_events, 1))
            xyp_events = np.memmap(events_xyp_file, dtype="int16", mode="r", shape=(num_events, 3))
            self.img_timestamp_event_idx[sequences_name] = img_timestamp_event_idx
            self.event_data[sequences_name] = [t_events, xyp_events]

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
                img_dir = child / 'image'
                ann_dir = child / 'segmentation_masks'
                sequences_name = osp.split(child)[1]

                for _, img in enumerate(sorted(mmcv.scandir(img_dir, img_suffix, recursive=True))):
                    seq_index = int(img.split(".")[0]) - 1
                    img_info = dict(filename=img, img_dir=img_dir, filepath=img_dir / img, seq_index=seq_index,
                                    seq_name=sequences_name, remove_rows='DDD17')
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
        img_timestamp_event_idx = self.img_timestamp_event_idx[seq_name]
        t_events, xyp_events = self.event_data[seq_name]
        names = ['event_volume_old', 'event_volume_new']
        event_info = {
            'sequence_index': seq_index,
            'timestamp': img_timestamp_event_idx[seq_index][0]
        }

        events_fwd, events_back = self.extract_events_from_memmap(t_events, xyp_events, seq_index, img_timestamp_event_idx, self.nr_events)
        
        for i in range(len(names)):
            if i == 0:
                events = events_fwd
            else:
                events = events_back

            nr_events_loaded = events.shape[0]
            nr_events_temp = nr_events_loaded // self.nr_events_data
            
            id_end = 0
            event_tensor = None
            bin_per_slice = int(self.nr_temporal_bins/self.nr_events_data)
            for _ in range(self.nr_events_data):
                id_start = id_end
                id_end += nr_events_temp

                if id_end > nr_events_loaded:
                    id_end = nr_events_loaded

                event_representation = generate_input_representation(events[id_start:id_end],
                                                                            self.shape,
                                                                            nr_temporal_bins=bin_per_slice,
                                                                            separate_pol=self.separate_pol)

                event_representation = torch.from_numpy(event_representation)

                if self.normalize_event:
                    event_representation = normalize_voxel_grid(event_representation)

                if event_tensor is None:
                    event_tensor = event_representation
                else:
                    event_tensor = torch.cat([event_tensor, event_representation], dim=0)

            event_info[names[i]] = event_tensor[:, :-60, :]
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
