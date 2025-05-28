# NNDL Midterm Task 1

This repository contains the code for training and evaluating an image classification model based on ResNet-18 for the Caltech-101 dataset.

---

## 📁 Data Preparation

Before training the model, make sure to prepare the dataset and place it in the appropriate directory as specified in the `utils/dataloader.py` file.

---

## 🚀 Model Training

You can adjust the hyperparameters (such as learning rates, batch sizes, number of epochs, and whether to use pretrained weights) by editing the `train.sh` script.  
To start training, run the following commands:

```bash
cd resnet-18/pretrain
sh train.sh
````

---

## 📊 Model Evaluation

First, obtain the trained model checkpoint:

* Either use the checkpoint generated after training.
* Or download it directly from our [Google Drive repository](https://drive.google.com/drive/folders/1nDUGEP0sHxuvfOszKUVJeMYnZLN5b3Es?usp=drive_link).

Then, evaluate the model using the following commands:

```bash
cd resnet-18/pretrain
python eval.py --ckpt_path <path_to_your_checkpoint>
```

Replace `<path_to_your_checkpoint>` with the actual path to your checkpoint file.




