# Training Documentation

This documentation only shows the way to re-produce our [paper](https://peterl1n.github.io/RobustVideoMatting/). If you would like to remove or add a dataset to the training, you are responsible for adapting the training code yourself.

## Datasets

The following datasets are used during our training.

**IMPORTANT: If you choose to download our preprocessed versions. Please avoid repeated downloads and cache the data locally. All traffics cost our expense. Please be responsible. We may only provide the preprocessed version of a limited time.**

### Matting Datasets
* [VideoMatte240K](https://grail.cs.washington.edu/projects/background-matting-v2/#/datasets)
    * Download JPEG SD version (6G) for stage 1 and 2.
    * Download JPEG HD version (60G) for stage 3 and 4.
    * Manually move clips `0000`, `0100`, `0200`, `0300` from the training set to a validation set.
* ImageMatte
    * ImageMatte consists of [Distinctions-646](https://wukaoliu.github.io/HAttMatting/) and [Adobe Image Matting](https://sites.google.com/view/deepimagematting) datasets.
    * Only needed for stage 4.
    * You need to contact their authors to acquire.
    * After downloading both datasets, merge their samples together to form ImageMatte dataset.
    * Only keep samples of humans.
    * Full list of images we used in ImageMatte for training:
        * [imagematte_train.txt](/documentation/misc/imagematte_train.txt)
        * [imagematte_valid.txt](/documentation/misc/imagematte_valid.txt)
    * Full list of images we used for evaluation.
        * [aim_test.txt](/documentation/misc/aim_test.txt)
        * [d646_test.txt](/documentation/misc/d646_test.txt)
### Background Datasets
* Video Backgrounds
    * We process from [DVM Background Set](https://github.com/nowsyn/DVM) by selecting clips without humans and extract only the first 100 frames as JPEG sequence.
    * Full list of clips we used:
        * [dvm_background_train_clips.txt](/documentation/misc/dvm_background_train_clips.txt)
        * [dvm_background_test_clips.txt](/documentation/misc/dvm_background_test_clips.txt)
    * You can download our preprocessed versions:
        * [Train set (14.6G)](https://robustvideomatting.blob.core.windows.net/data/BackgroundVideosTrain.tar) (Manually move some clips to validation set)
        * [Test set (936M)](https://robustvideomatting.blob.core.windows.net/data/BackgroundVideosTest.tar) (Not needed for training. Only used for making synthetic test samples for evaluation)
* Image Backgrounds
    * Train set:
        * We crawled 8000 suitable images from Google and Flicker.
        * We will not publish these images.
    * [Test set](https://grail.cs.washington.edu/projects/background-matting-v2/#/datasets)
        * We use the validation background set from [BGMv2](https://grail.cs.washington.edu/projects/background-matting-v2/) project.
        * It contains about 200 images.
        * It is not used in our training. Only used for making synthetic test samples for evaluation.
        * But if you just want to quickly tryout training, you may use this as a temporary subsitute for the train set.

### Segmentation Datasets

* [COCO](https://cocodataset.org/#download)
    * Download [train2017.zip (18G)](http://images.cocodataset.org/zips/train2017.zip)
    * Download [panoptic_annotations_trainval2017.zip (821M)](http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip)
    * Note that our train script expects the panopitc version.
* [YouTubeVIS 2021](https://youtube-vos.org/dataset/vis/)
    * Download the train set. No preprocessing needed.
* [Supervisely Person Dataset](https://supervise.ly/explore/projects/supervisely-person-dataset-23304/datasets)
    * We used the supervisedly library to convert their encoding to bitmaps masks before using our script. We also resized down some of the large images to avoid disk loading bottleneck.
    * You can refer to [spd_preprocess.py](/documentation/misc/spd_preprocess.py)
    * Or, you can download our [preprocessed version (800M)](https://robustvideomatting.blob.core.windows.net/data/SuperviselyPersonDataset.tar)

## Training

For reference, our training was done on data center machines with 48 CPU cores, 300G CPU memory, and 4 Nvidia V100 32G GPUs.

During our official training, the code contains custom logics for our infrastructure. For release, the script has been cleaned up. There may be bugs existing in this version of the code but not in our official training. If you find problems, please file an issue.

After you have downloaded the datasets. Please configure `train_config.py` to provide paths to your datasets.

The training consists of 4 stages. For detail, please refer to the [paper](https://peterl1n.github.io/RobustVideoMatting/).

### Stage 1
```sh
python train.py \
    --model-variant mobilenetv3 \
    --dataset videomatte \
    --resolution-lr 512 \
    --seq-length-lr 15 \
    --learning-rate-backbone 0.0001 \
    --learning-rate-aspp 0.0002 \
    --learning-rate-decoder 0.0002 \
    --learning-rate-refiner 0 \
    --checkpoint-dir checkpoint/stage1 \
    --log-dir log/stage1 \
    --epoch-start 0 \
    --epoch-end 20
```

### Stage 2
```sh
python train.py \
    --model-variant mobilenetv3 \
    --dataset videomatte \
    --resolution-lr 512 \
    --seq-length-lr 50 \
    --learning-rate-backbone 0.00005 \
    --learning-rate-aspp 0.0001 \
    --learning-rate-decoder 0.0001 \
    --learning-rate-refiner 0 \
    --checkpoint checkpoint/stage1/epoch-19.pth \
    --checkpoint-dir checkpoint/stage2 \
    --log-dir log/stage2 \
    --epoch-start 20 \
    --epoch-end 22
```

### Stage 3
```sh
python train.py \
    --model-variant mobilenetv3 \
    --dataset videomatte \
    --train-hr \
    --resolution-lr 512 \
    --resolution-hr 2048 \
    --seq-length-lr 40 \
    --seq-length-hr 6 \
    --learning-rate-backbone 0.00001 \
    --learning-rate-aspp 0.00001 \
    --learning-rate-decoder 0.00001 \
    --learning-rate-refiner 0.0002 \
    --checkpoint checkpoint/stage2/epoch-21.pth \
    --checkpoint-dir checkpoint/stage3 \
    --log-dir log/stage3 \
    --epoch-start 22 \
    --epoch-end 23
```

### Stage 4
```sh
python train.py \
    --model-variant mobilenetv3 \
    --dataset imagematte \
    --train-hr \
    --resolution-lr 512 \
    --resolution-hr 2048 \
    --seq-length-lr 40 \
    --seq-length-hr 6 \
    --learning-rate-backbone 0.00001 \
    --learning-rate-aspp 0.00001 \
    --learning-rate-decoder 0.00005 \
    --learning-rate-refiner 0.0002 \
    --checkpoint checkpoint/stage3/epoch-22.pth \
    --checkpoint-dir checkpoint/stage4 \
    --log-dir log/stage4 \
    --epoch-start 23 \
    --epoch-end 28
```

<br><br><br>

## Evaluation

We synthetically composite test samples to both image and video backgrounds. Image samples (from D646, AIM) are augmented with synthetic motion.

We only provide the composited VideoMatte240K test set. They are used in our paper evaluation. For D646 and AIM, you need to acquire the data from their authors and composite them yourself. The composition scripts we used are saved in `/evaluation` folder as reference backup. You need to modify them based on your setup.

* [videomatte_512x512.tar (PNG 1.8G)](https://robustvideomatting.blob.core.windows.net/eval/videomatte_512x288.tar)
* [videomatte_1920x1080.tar (JPG 2.2G)](https://robustvideomatting.blob.core.windows.net/eval/videomatte_1920x1080.tar)

Evaluation scripts are provided in `/evaluation` folder.