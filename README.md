# Learned Image Compression with Hierarchical Progressive Context Modeling [ICCV 2025]

[![arXiv](https://img.shields.io/badge/arXiv-2507.19125-b31b1b.svg)](https://arxiv.org/abs/2507.19125)  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## *News*

🎉Our new work based on HPCM, *"Scaling Learned Image Compression Models up to 1 Billion"*, has been released on arXiv! [[link]](https://arxiv.org/abs/2508.09075)

## Introduction

This repository is the official PyTorch implementation of the paper *"Learned Image Compression with Hierarchical Progressive Context Modeling"*.

**Abstract:** Context modeling is essential in learned image compression for accurately estimating the distribution of latents. While recent advanced methods have expanded context modeling capacity, they still struggle to efficiently exploit long-range dependency and diverse context information across different coding steps. In this paper, we introduce a novel Hierarchical Progressive Context Model (HPCM) for more efficient context information acquisition. Specifically, HPCM employs a hierarchical coding schedule to sequentially model the contextual dependencies among latents at multiple scales, which enables more efficient long-range context modeling. Furthermore, we propose a progressive context fusion mechanism that incorporates contextual information from previous coding steps into the current step, effectively exploiting diverse contextual information. Experimental results demonstrate that our method achieves state-of-the-art rate-distortion performance and strikes a better balance between compression performance and computational complexity.

## Highlights

Our **H**ierarchical **P**rogressive **C**ontext **M**odeling (**HPCM**) entropy model significantly advances the performance of Learned Image Compression (LIC) models with the key insights:

- Efficient global-local context modeling with Hierarchical Coding Schedule.
- Exploring effective rich context information through Progressive Context Fusion.

Our method achieves **state-of-the-art** compression performance gains on Kodak (-19.19%), CLIC (-18.37%), and Tecnick (-22.20%) over VTM-22.0.

## Performance

<img src="./assets/table1.png" alt="table1" style="zoom:25%;" />

## Installation

This implementation requires Python 3.8 and PyTorch 1.12.

1. Install the dependencies

   ```
   pip install -r requirements.txt
   ```
   
2. Compile the arithmetic coder

   If you need real bitstream writing, please compile the arithmetic coder using the following commands. The compiled files are located in the directories `src/entropy_models/entropy_coders/unbounded_rans/build`.

   ```
   cd src/entropy_models/entropy_coders/unbounded_rans
   sh setup.sh
   ```
   
   We provide the arithmetic coder for Linux, Python 3.8, specifically `_CXX.cpython-38-x86_64-linux-gnu.so` and `unbounded_ans.cpython-38-x86_64-linux-gnu.so`.

## Usage

### Dataset

Download [Flickr2W](https://github.com/liujiaheng/CompressionData) for training; [Kodak](https://r0k.us/graphics/kodak/), [CLIC](https://www.compression.cc/) and [TESTIMAGES](https://testimages.org/) for evaluation.

### Training

```
python train.py --model_name [HPCM_Base/HPCM_Large] --train_dataset [path-to-train-dataset] --test_dataset [path-to-test-dataset] --lambda [lambda] -e [epoch-num] -lr [learning-rate] -bs [batch-size]
```

### Testing

```
python test.py --model_name [HPCM_Base/HPCM_Large] --dataset [path-to-test-dataset] --checkpoint [path-to-checkpoint]
```

## Pretrained Model

HPCM-Base models:

| Lambda | Metric | Link                                                         | Lambda | Metric  | Link                                                         |
| ------ | ------ | ------------------------------------------------------------ | ------ | ------- | ------------------------------------------------------------ |
| 0.0018 | MSE    | [Link](https://drive.google.com/file/d/1nIoANbXzBNE0S_VoLo9ZDHU50lPMmeBP/view?usp=drive_link) | 2.4    | MS-SSIM | [Link](https://drive.google.com/file/d/1AZ9dY2J9Rn17YSQe_NYIOID-st1C-68O/view?usp=drive_link) |
| 0.0035 | MSE    | [Link](https://drive.google.com/file/d/15J_nl33_5R_qyTIzLAaT60ICn9BMGHlB/view?usp=drive_link) | 4.58   | MS-SSIM | [Link](https://drive.google.com/file/d/1Y8gEL4MRNB-TBbOMDUKeMTO_z1QhbwqL/view?usp=drive_link) |
| 0.0067 | MSE    | [Link](https://drive.google.com/file/d/1HIzsEqAPztaMh0Frqec4TtRwoc7uxO97/view?usp=drive_link) | 8.73   | MS-SSIM | [Link](https://drive.google.com/file/d/1hXK-X6GsjjiULy6FvU80Smob_2UOFeFJ/view?usp=drive_link) |
| 0.013  | MSE    | [Link](https://drive.google.com/file/d/1Snq7vkWQdApzCe-gK_V-WuRyMHQRL443/view?usp=drive_link) | 16.64  | MS-SSIM | [Link](https://drive.google.com/file/d/1antXt3M0ecOVejbpxL1U7CVx4TS_XPMQ/view?usp=drive_link) |
| 0.025  | MSE    | [Link](https://drive.google.com/file/d/1NFZD87BkfU28YnDqpzfphG0xDZDZpUA5/view?usp=drive_link) | 31.73  | MS-SSIM | [Link](https://drive.google.com/file/d/1X_Q0hHwAW0GOsHWLoq84YKYqXrduFe6b/view?usp=drive_link) |
| 0.0483 | MSE    | [Link](https://drive.google.com/file/d/1G5wm4KENBY2qSAQBxNw3Rz4JcMxH8HXu/view?usp=drive_link) | 60.5   | MS-SSIM | [Link](https://drive.google.com/file/d/1mX885h4eVwLvpeHpBHBoM1p4Z2VLV2y-/view?usp=drive_link) |

HPCM-Large models:

| Lambda | Metric | Link                                                         | Lambda | Metric  | Link                                                         |
| ------ | ------ | ------------------------------------------------------------ | ------ | ------- | ------------------------------------------------------------ |
| 0.0018 | MSE    | [Link](https://drive.google.com/file/d/1E1DUaPsIrfNPwfk4qD-630hhxx5n_BJ4/view?usp=drive_link) | 2.4    | MS-SSIM | [Link](https://drive.google.com/file/d/1RUM2a1wdI8Yj9-tvzO_MnHGZWZRp2-W6/view?usp=drive_link) |
| 0.0035 | MSE    | [Link](https://drive.google.com/file/d/15yDUVvEBn-7dMA9SBIQ2w28LJXBGntQo/view?usp=drive_link) | 4.58   | MS-SSIM | [Link](https://drive.google.com/file/d/1TL_QDlfzHvmerN1p0rn5mJbSNwn3LXXx/view?usp=drive_link) |
| 0.0067 | MSE    | [Link](https://drive.google.com/file/d/1yzZKji6RpsyQPD6KFr_weavVrlmn-V4R/view?usp=drive_link) | 8.73   | MS-SSIM | [Link](https://drive.google.com/file/d/1nIEJY9ecr9uA9XidtiQRXQ2rzm1DWKM0/view?usp=drive_link) |
| 0.013  | MSE    | [Link](https://drive.google.com/file/d/1L19zjwOpbbFPw0FxnyVLcHATxCaorjUV/view?usp=drive_link) | 16.64  | MS-SSIM | [Link](https://drive.google.com/file/d/1sKnWry4LIZPawwv08TH3l_41giUuElCx/view?usp=drive_link) |
| 0.025  | MSE    | [Link](https://drive.google.com/file/d/1oh8OwCLc8PEVMW1fc9LoC7G4385kHU5D/view?usp=drive_link) | 31.73  | MS-SSIM | [Link](https://drive.google.com/file/d/1rR0vFbQ2fOT7EgJbYg5f0OdiIT5jbPPu/view?usp=drive_link) |
| 0.0483 | MSE    | [Link](https://drive.google.com/file/d/1VWLPQeDzBZgb1D2mZ9jLzLppXL8gUanH/view?usp=drive_link) | 60.5   | MS-SSIM | [Link](https://drive.google.com/file/d/1ITR5JEzLjmdHLp20GYzIdwE8eEK2d7ns/view?usp=drive_link) |

## R-D Data

R-D data on CLIC Pro Valid and Tecnick datasets is in `R-D_Data.md`.

#### HPCM-Base, Kodak, PSNR

```
bpp = [0.1211, 0.1757, 0.2729, 0.4125, 0.5898, 0.8209]
psnr = [29.4022, 30.7547, 32.4012, 34.2063, 36.0145, 37.7525]
```

#### HPCM-Base, Kodak, MS-SSIM

```
bpp = [0.0974, 0.1447, 0.2124, 0.2958, 0.4082, 0.5724]
db_msssim = [13.2883, 14.8748, 16.5800, 18.1826, 19.7679, 21.4763]
```

#### HPCM-Large, Kodak, PSNR

```
bpp = [0.0951, 0.1537, 0.2438, 0.3778, 0.5516, 0.7843]
psnr = [28.9135, 30.4490, 32.1219, 33.9923, 35.8511, 37.7480]
```

#### HPCM-Large, Kodak, MS-SSIM

```
bpp = [0.0943, 0.1429, 0.2090, 0.2935, 0.4016, 0.5577]
db_msssim = [13.2098, 14.7740, 16.6119, 18.2272, 19.7947, 21.4393]
```

## Acknowledgement

Part of our code is implemented based on [CompressAI](https://github.com/InterDigitalInc/CompressAI) and [DCVC-DC](https://github.com/microsoft/DCVC/tree/main/DCVC-family/DCVC-DC). Thank for the excellent jobs!

## Citation

```
@article{li2025hpcm,
  title={Learned Image Compression with Hierarchical Progressive Context Modeling},
  author={Li, Yuqi and Zhang, Haotian and Li, Li and Liu, Dong},
  journal={arXiv preprint arXiv:2507.19125},
  year={2025}
}
```

## Contact

If you have any questions, please feel free to contact lyq010303@mail.ustc.edu.cn.