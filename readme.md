# IDD Segmentation using Attention U-Net

This project focuses on semantic segmentation of urban road scenes using the [IDD (Indian Driving Dataset)](https://idd.insaan.iiit.ac.in/) with an implementation of the Attention U-Net architecture. The solution is implemented in a way to work on Indian road environments, where challenges like unstructured roads and varied object categories require more robust models.

The model is implemented using TensorFlow and Keras, and the training pipeline is designed to be modular and reproducible using Python 3.11.9.

---

## Overview

The pipeline includes the following components:
- Dataset preparation and preprocessing (including custom mask generation).
- Training an Attention U-Net model.
- Evaluation on the validation set.
- Visualization of predictions.

The model is trained only if a pre-trained version is not found. Once trained, the model can be evaluated and used on custom input images.

>**Important:** *The model to be used is already available as in `attention_unet_model.h5` file in the main project directory.*

---

## Dataset Details

The model is trained on two subsets of the **IDD Segmentation** dataset:
1. `IDD_Segmentation`
2. `idd20kII`

Each subset includes:
- `leftImg8bit/`: contains RGB images for training and validation.
- `gtFine/`: contains ground truth segmentation masks in `labelLevel3Ids` format.

The masks used for training were generated using preprocessing scripts adapted from the public GitHub repository:

**GitHub Source for Mask Generation:**  
https://github.com/AutoNUE/public-code

Please ensure that the directory structure is preserved:

```
data/
└── IDD_Segmentation/
    ├── IDD_Segmentation/
    │   ├── leftImg8bit/
    │   └── gtFine/
    └── idd20kII/
        ├── leftImg8bit/
        └── gtFine/
```

The following commands are expected to run after cloning the public-code repository for mask generation.

```bash
.env/Scripts/python.exe public-code/preperation/createLabels.py \
  --datadir data/IDD_Segmentation/IDD_Segmentation \
  --id-type level3Id \
  --num-workers 4

```

and

```bash
.env/Scripts/python.exe public-code/preperation/createLabels.py \
  --datadir data/IDD_Segmentation/idd20kII \
  --id-type level3Id \
  --num-workers 4

```
---

## Environment Setup

Ensure the following dependencies are installed:

- Python 3.11.9
- TensorFlow >= 2.11
- NumPy
- OpenCV
- scikit-learn
- tqdm
- matplotlib

You can install the dependencies using:

```bash
pip install -r requirements.txt
```

Create a virtual environment (optional but recommended):

```bash
python -m venv .env
source .env/bin/activate  # For Unix-based systems
.env\Scripts\activate     # For Windows
```

---

## Running the Pipeline

The training script will check for an existing saved model (`attention_unet_model.h5`). If not found, it will initiate training using the provided datasets.

To run the training and evaluation:

```bash
jupyter lab
# Open the notebook and execute cells in order
```

If the model already exists, training will be skipped and the saved model will be loaded for evaluation or prediction.

---

## Output

- Trained model file: `attention_unet_model.h5`
- Evaluation metrics: Validation loss and accuracy
- Visualization: Optional plots for predicted masks on validation images

---

## Notes

- The project is structured modularly to support further experimentation with different architectures and datasets.
- For reproducibility, random seeds and consistent data splits are used.
- File paths in the notebook are system-specific by default; consider verifying and updating them for your own environment.
- The masks for the dataset that is shown or stored in this repository has already been created using the public-code repository.
