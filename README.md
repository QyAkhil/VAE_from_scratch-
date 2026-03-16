# VAE from Scratch

A complete implementation of a Variational Autoencoder (VAE) built from scratch using PyTorch, trained on the CelebA faces dataset.

## Features

- Custom VAE implementation with encoder and decoder
- Training on the CelebA dataset
- Reconstruction visualization
- Pretrained model checkpoint included (for quick inference)

## Dataset

This project uses the CelebA dataset from Kaggle: [CelebA Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)

### Setup Instructions

1. Download the dataset from Kaggle.
2. Extract the `img_align_celeba.zip` file.
3. Place the `img_align_celeba` folder in `data/celeba/img_align_celeba/`.
4. The final structure should be: `data/celeba/img_align_celeba/img_align_celeba/*.jpg`

> **Note:** The actual image files are not included in this repository due to size constraints. You must download them separately from Kaggle.

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
- Model training
- Loss plotting
- Saving model weights to `vae_celeba_best.pth`

### Generating Reconstructions

You can generate reconstructions either by running cells in `train.ipynb` or by using `generate.ipynb`.

- `generate.ipynb` contains inference code that loads `vae_celeba_best.pth` and creates reconstruction and sample outputs.

## Model Architecture

- **Latent Dimension**: 128
- **Input Size**: 64×64 RGB images
- **Encoder**: Convolutional layers with batch normalization
- **Decoder**: Transpose convolutional layers
- **Loss**: MSE reconstruction loss + KL divergence

## Training Details

- **Optimizer**: Adam (lr=5e-4)
- **Batch Size**: 64
- **Epochs**: Configurable in the notebook
- **Device**: CUDA if available, otherwise CPU

## Outputs

Generated outputs are saved under:

- `outputs/reconstructions/` — reconstruction examples
- `outputs/samples/` — random samples from the latent space

A pretrained model checkpoint is included as:

- `vae_celeba_best.pth`

## Project Structure

```
VAE_from_scratch/
├── config.py              # Training/configuration settings
├── generate.ipynb         # Inference / sample generation notebook
├── models/
│   └── vae.py             # VAE model implementation
├── utils/
│   └── dataset.py         # CelebA dataset loader
├── data/
│   └── celeba/            # Dataset directory (download separately)
├── outputs/
│   ├── reconstructions/   # Generated reconstructions
│   └── samples/           # Random samples from the VAE
├── train.ipynb            # Training notebook
├── requirements.txt       # Python dependencies
├── vae_celeba_best.pth    # Pretrained model weights
└── README.md              # This file
```

## Requirements

- Python 3.7+
- PyTorch 2.0+
- torchvision
- Pillow
- matplotlib
- numpy

## License

This project is for educational purposes. Please respect the CelebA dataset license and usage terms.

## Acknowledgments

- CelebA dataset by Liu et al.
- PyTorch community for the excellent deep learning framework
