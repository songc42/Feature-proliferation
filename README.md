# Feature Proliferation --- the ``Cancer'' in StyleGAN and its Treatments

![](https://github.com/songc42/Feature-proliferation/blob/main/Impact_feature_proliferation.png)



This repository contains code for our `ICCV 2023` paper "Feature Proliferation --- the ``Cancer'' in StyleGAN and its Treatments" <br>

Abstract: Despite the success of StyleGAN in image synthesis, the images it synthesizes are not always perfect and the well-known truncation trick has become a standard post-processing technique for StyleGAN to synthesize high-quality images. Although effective, it has long been noted that the truncation trick tends to reduce the diversity of synthesized images and unnecessarily sacrifices many distinct image features. To address this issue, in this paper, we first delve into the StyleGAN image synthesis mechanism and discover an important phenomenon, namely Feature Proliferation, which demonstrates how specific features reproduce with forward propagation. Then, we show how the occurrence of Feature Proliferation results in StyleGAN image artifacts. As an analogy, we refer to it as the "cancer" in StyleGAN from its proliferating and malignant nature. Finally, we propose a novel feature rescaling method that identifies and modulates risky features to mitigate feature proliferation. Thanks to our discovery of Feature Proliferation, the proposed feature rescaling method is less destructive and retains more useful image features than the truncation trick, as it is more fine-grained and works in a lower-level feature space rather than a high-level latent space. Experimental results justify the validity of our claims and the effectiveness of the proposed feature rescaling method. <br>

[[Paper]](https://openaccess.thecvf.com/content/ICCV2023/html/Song_Feature_Proliferation_--_the_Cancer_in_StyleGAN_and_its_Treatments_ICCV_2023_paper.html) | 
[[Archiv]](https://openaccess.thecvf.com/content/ICCV2023/html/Song_Feature_Proliferation_--_the_Cancer_in_StyleGAN_and_its_Treatments_ICCV_2023_paper.html) | 
[[Colab]](https://openaccess.thecvf.com/content/ICCV2023/html/Song_Feature_Proliferation_--_the_Cancer_in_StyleGAN_and_its_Treatments_ICCV_2023_paper.html) | 
[[Video]](https://openaccess.thecvf.com/content/ICCV2023/html/Song_Feature_Proliferation_--_the_Cancer_in_StyleGAN_and_its_Treatments_ICCV_2023_paper.html) |
[![StyleGAN-XL + CLIP](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CasualGANPapers/unconditional-StyleGANXL-CLIP/blob/main/StyleganXL%2BCLIP.ipynb)

If you find our code or paper useful, please cite 
```bibtex
@inproceedings{song2023feature,  
  title={Feature Proliferation--the" Cancer" in StyleGAN and its Treatments},  
  author={Song, Shuang and Liang, Yuanbang and Wu, Jing and Lai, Yu-Kun and Qin, Yipeng},  
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},  
  pages={2360--2370},  
  year={2023}     
}
```
                





Requirements
==

*  Ubuntu 22.04.1，Python 3.7 and PyTorch 1.10.2 (or later). See [https://pytorch.org](https://pytorch.org) for PyTorch install instructions.
*  CUDA toolkit 11.3 or later
*  We recommand that you can simply create the enviroment using Anaconda:
   * `conda env create -f environment.yml`
   * `conda activate Feature_Proliferation`



Getting Started
----


