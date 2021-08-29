# No frame left behind: Full Video Action Recognition [[arXiv]](https://arxiv.org/abs/2103.15395)

## Abstract

Not all video frames are equally informative for recognizing an action. It is computationally infeasible to train deep networks on all video frames when actions develop over hundreds of frames. A common heuristic is uniformly sampling a small number of video frames and using these to recognize the action. Instead, here we propose full video action recognition and consider all video frames. To make this computational tractable, we first cluster all frame activations along the temporal dimension based on their similarity with respect to the classification task, and then temporally aggregate the frames in the clusters into a smaller number of representations. Our method is end-to-end trainable and computationally efficient as it relies on temporally localized clustering in combination with fast Hamming distances in feature space. We evaluate on UCF101, HMDB51, Breakfast, and Something-Something V1 and V2, where we compare favorably to existing heuristic frame sampling methods.

This repo contains the toy dataset Move4MNIST example images and the PyTorch code of [No frame left behind: Full Video Action Recognition](https://arxiv.org/abs/2103.15395). Its implementation is based on [TSM](https://github.com/mit-han-lab/temporal-shift-module).

![framework](https://github.com/L-KID/Full-Video-Action-Recognition/blob/fix_branch/images/fig1_cvpr21.pdf_tex.pdf)
<object data="https://github.com/L-KID/Full-Video-Action-Recognition/blob/fix_branch/images/fig1_cvpr21.pdf_tex.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="https://github.com/L-KID/Full-Video-Action-Recognition/blob/fix_branch/images/fig1_cvpr21.pdf_tex.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="https://github.com/L-KID/Full-Video-Action-Recognition/blob/fix_branch/images/fig1_cvpr21.pdf_tex.pdf">Download PDF</a>.</p>
    </embed>
</object>

## Prerequisites

The code is built with [PyTorch](https://pytorch.org/) 1.7.0. You may need [ffmpeg](https://www.ffmpeg.org/) for video data pre-processing.

## Datasets and Preparation

We evaluate our model on [UCF101](http://crcv.ucf.edu/data/UCF101.php), [HMDB51](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/), [Something-Something-V1](https://20bn.com/datasets/something-something/v1) and [V2](https://20bn.com/datasets/something-something/v2), [Breakfast](https://serre-lab.clps.brown.edu/resource/breakfast-actions-dataset/). We also provide some examples of the fully controlled [Move4MNIST]() dataset we created in this repo.

We input frames that are extracted from videos. The [TSN](https://github.com/yjxiong/temporal-segment-networks) repo provides detailed guide of video data pre-processing.

## Code

The code is implemented based on [TSM](https://github.com/mit-han-lab/temporal-shift-module). We plug in our method and finetune on pretrained backbones since our method has no cost of extra parameters.

## Main Results

### Breakfast

Here backbones are pre-trained on ImageNet.

|     Model     |Backbone |#Frames|#Clusters|Top-1|
|---------------|---------|-------|---------|-----|
|ResNet-152     |ResNet152|64    | -       |41.1%|
|ActionVLAD     |ResNet152| 64   | -       |55.5%|
|VideoGraph     |ResNet152| 64   | -       |59.1%|
|TSM (our impl.)|ResNet50 |16    |-        |72.1%|
|Ours-slope     |ResNet50 |all   |16       |74.9%|
|Ours-cumulative|ResNet50 |all   |16       |**76.6%**|

### Something-Something V1 & V2

All methods in this table use ResNet50 backbone, which is pre-trained on ImageNet.

|Model|#Frames|#Clusters|Top-1 V1|Top-1 V2|
|-----|-------|---------|--------|--------|
| TSN |8      | -       |19.7%   |30.0%   |
| TRN-Multiscale|8| -   |38.9%   |48.8%   |
| TSM |8      | -       |45.6%   |59.1%   |
| TSM |16     |-        |47.2%   |63.4%   |
| STM |8      |-        |49.2%   |62.3%   |
| STM |16     |-        |50.7%   |64.2%   |
|Ours-slope|all|8       |46.7%   |60.2%   |
|Ours-cumulative|all|8  |49.5%   |62.7%   |
|Ours-cumulative|all|16 |**51.4%**|**65.1%**|

### UCF-101 & HMDB51

|Model|Backbone|Pre-train|#Frames|#Clusters|Top-1 UCF-101|Top-1 HMDB51|    
|-----|--------|---------|-------|---------|-------------|------------|
|TSM (our impl.)|ResNet50|Kinetics|1|-     |91.2%        |65.1%       | 
|TSN  |ResNet50|Kinetics |8      |-        |91.7%        |64.7%       |
|SI+DI+OF+DOF|ResNeXt50|ImageNet|dynamic images|-  |95.0%|71.5%       |
|TSM  |ResNet50|Kinetics |8      |-        |95.9%        |73.5%       |
|STM  |ResNet50|ImageNet+Kinetics|16|-     |96.2%        |72.2%       |
|Ours-slope|TSM-ResNet50|Kinetics|all|8    |96.2%        |73.3%       |
|Ours-cumulative|TSM-ResNet50|Kinetics|all|8|**96.4%**   |**73.4%**   |

## Training

We provide an example to train our method on the toy dataset Move4MNIST with pre-trained ResNet18:

	python3 main.py movingmnist RGB \
	     --arch resnet18 --num_segments 16 \
	     --gd 20 --lr 0.001 --epochs 40 \
	     --batch-size 16 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
	     --shift --shift_div=8 --shift_place=blockres --npb

## Testing

To test our method on the toy dataset Move4MNIST:

	python3 main.py --evaluate movingmnist RGB \
	     --arch resnet18 --num_segments 16 \
	     --batch-size 16 -j 16 --consensus_type=avg \
	     --resume=path_to_the_best_model.pth

## Citation

If you find this repository useful for your work, please cite as follows: 

```
@InProceedings{Liu_2021_CVPR,
    author    = {Liu, Xin and Pintea, Silvia L. and Nejadasl, Fatemeh Karimi and Booij, Olaf and van Gemert, Jan C.},
    title     = {No Frame Left Behind: Full Video Action Recognition},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {14892-14901}
}
```
