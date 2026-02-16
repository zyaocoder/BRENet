# Preparing Dataset
Our dataset preparation follows [ESS](https://github.com/uzh-rpg/ess) that contains: <br/>
(1) [DDD17](https://docs.google.com/document/u/1/d/1HM0CSmjO8nOpUeTvmPjopcBcVCk7KXvLUuiZFS6TWSg/pub) <br/>
(2) [DSEC](https://dsec.ifi.uzh.ch/) <br/>
The datasets should be put in ```BRENet/data/```. Or you can use Symlink:
```
mkdir -p data
ln -s /dataset_path/DSEC ./data
ln -s /dataset_path/DDD17 ./data
```

### DDD17
The original DDD17 dataset without semantic segmentation labels can be downloaded [here](https://docs.google.com/document/u/1/d/1HM0CSmjO8nOpUeTvmPjopcBcVCk7KXvLUuiZFS6TWSg/pub). Additional semantic labels can be downloaded [here](https://github.com/Shathe/Ev-SegNet). Please do not forget to cite DDD17 and Ev-SegNet if you are using the DDD17 with semantic labels. The dataset should have the following format:

    ./BRENet                                # current (project) directory
    └── data                                # various datasets
        └── DDD17
            ├── train
            │   ├── dir 0                   # various sequences
            │   │   ├── image               # images
            │   │   |   ├── 00000000.png
            │   │   |   └── ...
            │   │   ├── index               # event timestamp data
            │   │   |   ├── index_10ms.npy
            │   │   |   └── ...
            │   │   ├── segmentation_masks  # segmentation labels
            │   │   |   ├── 00000000.png
            │   │   |   └── ...
            │   │   ├── events.dat.t        # event data
            │   │   └── events.dat.xyp      # event data
            │   ├── dir 3
            │   ├── dir 4
            │   ├── dir 6
            │   └── dir 7
            └── test
                └── dir 1

### DSEC
The DSEC dataset can be downloaded [here](https://dsec.ifi.uzh.ch/dsec-datasets/download/). The dataset should have the following format:

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

### Processed dataset
We also provide the processed and easy-to-use versions of the two datasets which you can use use after directly unzipping the files: [DDD17 and DSEC](https://huggingface.co/datasets/Chrisathy/BRENet-dataset/tree/main).
