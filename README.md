# VAE from Scratch

A complete implementation of a Variational Autoencoder (VAE) built from scratch using PyTorch, trained on the CelebA faces dataset.

## Features

- Custom VAE implementation with encoder and decoder
- Training on CelebA dataset
- Reconstruction visualization
- Model checkpointing

## Dataset

This project uses the CelebA dataset from Kaggle: [CelebA Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)

### Setup Instructions

1. Download the dataset from Kaggle
2. Extract the `img_align_celeba.zip` file
3. Place the `img_align_celeba` folder in `data/celeba/img_align_celeba/`
4. The final structure should be: `data/celeba/img_align_celeba/img_align_celeba/*.jpg`

**Note**: The actual image files are not included in this repository due to size constraints. You must download them separately from Kaggle.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/vae-from-scratch.git
cd vae-from-scratch
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Run the training notebook:
```bash
jupyter notebook train.ipynb
```

The notebook includes:
- Data loading and preprocessing
- Model training (10 epochs)
- Loss plotting
- Model checkpoint saving

### Generating Reconstructions

After training, run the reconstruction cell in `train.ipynb` to generate and save reconstruction examples.

## Model Architecture

- **Latent Dimension**: 128
- **Input Size**: 64x64 RGB images
- **Encoder**: Convolutional layers with batch normalization
- **Decoder**: Transpose convolutional layers
- **Loss**: MSE reconstruction loss + KL divergence

## Training Details

- **Optimizer**: Adam (lr=5e-4)
- **Batch Size**: 64
- **Epochs**: 10
- **Device**: CUDA if available, otherwise CPU

## Results

The trained VAE can generate realistic face reconstructions. Check the `outputs/reconstructions/` folder for example reconstructions after running the notebook.

## Project Structure

```
vae-from-scratch/
├── models/
│   └── vae.py              # VAE model implementation
├── utils/
│   ├── dataset.py          # CelebA dataset loader
│   └── visualize.py        # Visualization utilities
├── data/
│   └── celeba/             # Dataset directory (download separately)
├── outputs/
│   ├── checkpoints/        # Model checkpoints
│   └── reconstructions/    # Generated reconstructions
├── train.ipynb             # Training notebook
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore rules
└── README.md              # This file
```

## Requirements

- Python 3.7+
- PyTorch 2.0+
- torchvision
- PIL
- matplotlib
- numpy
- pandas

## License

This project is for educational purposes. Please respect the CelebA dataset license and usage terms.

## Acknowledgments

- CelebA dataset by Liu et al.
- PyTorch community for the excellent deep learning framework