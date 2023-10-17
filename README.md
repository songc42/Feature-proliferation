# Feature Proliferation --- the ``Cancer'' in StyleGAN and its Treatments (ICCV 2023)





> Shuang Song, Yuanbang Liang, Jing Wu, Yu-Kun Lai, Yipeng Qin 
>
>Abstract: Despite the success of StyleGAN in image synthesis, the images it synthesizes are not always perfect and the well-known truncation trick has become a standard post-processing technique for StyleGAN to synthesize high-quality images. Although effective, it has long been noted that the truncation trick tends to reduce the diversity of synthesized images and unnecessarily sacrifices many distinct image features. To address this issue, in this paper, we first delve into the StyleGAN image synthesis mechanism and discover an important phenomenon, namely Feature Proliferation, which demonstrates how specific features reproduce with forward propagation. Then, we show how the occurrence of Feature Proliferation results in StyleGAN image artifacts. As an analogy, we refer to it as the "cancer" in StyleGAN from its proliferating and malignant nature. Finally, we propose a novel feature rescaling method that identifies and modulates risky features to mitigate feature proliferation. Thanks to our discovery of Feature Proliferation, the proposed feature rescaling method is less destructive and retains more useful image features than the truncation trick, as it is more fine-grained and works in a lower-level feature space rather than a high-level latent space. Experimental results justify the validity of our claims and the effectiveness of the proposed feature rescaling method. <br>

![](https://github.com/songc42/Feature-proliferation/blob/main/Impact_feature_proliferation.png)

[[Paper]](https://openaccess.thecvf.com/content/ICCV2023/html/Song_Feature_Proliferation_--_the_Cancer_in_StyleGAN_and_its_Treatments_ICCV_2023_paper.html) | 
[[Archiv]](https://arxiv.org/abs/2310.08921) | 
[[Video]](https://openaccess.thecvf.com/content/ICCV2023/html/Song_Feature_Proliferation_--_the_Cancer_in_StyleGAN_and_its_Treatments_ICCV_2023_paper.html) |
[![StyleGAN-XL + CLIP](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JT0xkkn-pyNb-Nt13Zj8AXZueOjs1_3P?usp=sharing)

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

Pre-trained models
==

| Dataset | Description
| :--- | :----------
|[FFHQ](https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT/view?usp=sharing) | StyleGAN2 model trained on FFHQ with 1024x1024 output resolution.
|[AFHQ](https://drive.google.com/file/d/17OU6C76FIol3ggdGXGUBwjhF3hGSF9V4/view?usp=drive_link) | StyleGAN2 model trained on AFHQ with 512x512 output resolution.
|[Metface](https://drive.google.com/file/d/1z6IVVaCJuFTksKwp1CM3emWOVHbrBip-/view?usp=sharing) | StyleGAN2 model trained on Metface with 1024x1024 output resolution.

The estimated mean and standard deviation of feature maps in `FFHQ, AFHQ, Metface` pretrained models are stored in folders `FFHQ_m_var, AFHQ_m_var, and Metface_m_var`.


Getting Started
----

For a quick start, you can simply use the following command:
```
python SG2_modify.py --outdir=out \
  --seed 79058 79325 78482 77963 \
  --dataset FFHQ 
```
and

```
python SG2_modify.py --outdir=out \
  --seed 78948 79565 78434 76977 \
  --dataset AFHQ 
```
