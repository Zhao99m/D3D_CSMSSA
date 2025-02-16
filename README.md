# PyTorch implementation 



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

```
