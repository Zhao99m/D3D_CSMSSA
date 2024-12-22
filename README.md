# PyTorch implementation 

<p align="center"><img src="./MNAD_files/overview.png" alt="no_image" width="40%" height="40%" /><img src="./MNAD_files/teaser.png" alt="no_image" width="60%" height="60%" /></p>
This is the implementation of the paper "Learning Memory-guided Normality for Anomaly Detection (CVPR 2020)".



## Dependencies
* Python 3.6
* PyTorch 1.1.0
* Numpy
* Sklearn
* timm==0.3.2
* mmcv==1.3.8


## Datasets
* USCD Ped2 [[dataset](https://github.com/StevenLiuWen/ano_pred_cvpr2018)]
* CUHK Avenue [[dataset](https://github.com/StevenLiuWen/ano_pred_cvpr2018)]
* ShanghaiTech [[dataset](https://github.com/StevenLiuWen/ano_pred_cvpr2018)]

These datasets are from an official github of "Future Frame Prediction for Anomaly Detection - A New Baseline (CVPR 2018)".

Download the datasets into ``dataset`` folder, like ``./dataset/ped2/``

## Update
* 02/04/21: We uploaded the codes based on reconstruction method, and pretrained wieghts for Ped2 reconstruction, Avenue prediction and Avenue reconstruction.


## Training

```bash

python Train.py # for training
```
* You can freely define parameters with your own settings like
```bash
python Train.py --gpus 1 --dataset_path 'your_dataset_directory' --dataset_type avenue --exp_dir 'your_log_directory'
```


## Evaluation


```bash
python Evaluate.py --dataset_type ped2 --model_dir your_model.pth --m_items_dir your_m_items.pt
```
* For the reconstruction task, you need to set the parameters as
```bash
python Evaluate.py --method recon --t_length 1 --alpha 0.7 --th 0.015 --dataset_type ped2 --model_dir your_model.pth --m_items_dir your_m_items.pt
```
* Test the model with our pre-trained model and memory items
```bash
python Evaluate.py --dataset_type ped2 --model_dir pretrained_model.pth --m_items_dir m_items.pt
```


  author={Park, Hyunjong and Noh, Jongyoun and Ham, Bumsub},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={14372--14381},
  year={2020}
}
```
