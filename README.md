[![Python tests](https://github.com/Giuseppe499/PNPD_unrolling/actions/workflows/python_test.yml/badge.svg)](https://github.com/Giuseppe499/PNPD_unrolling/actions/workflows/python_test.yml)
[![Python lint](https://github.com/Giuseppe499/PNPD_unrolling/actions/workflows/python_lint.yml/badge.svg)](https://github.com/Giuseppe499/PNPD_unrolling/actions/workflows/python_lint.yml)
[![Python format](https://github.com/Giuseppe499/PNPD_unrolling/actions/workflows/python_format.yml/badge.svg)](https://github.com/Giuseppe499/PNPD_unrolling/actions/workflows/python_format.yml)

# PNPD Unrolling

Parameters estimation via unrolling of the Preconditioned Nested Primal Dual (PNPD) algorithm applied to image deblurring.

## Requirements

### git-lfs

This repository uses [git-lfs](https://git-lfs.com/) to store the pre-trained model weights.
Before cloning the repository, make sure to install git-lfs and to run

```
git lfs install
```

### Clone the repository

Clone the repository using git:

```bash
git clone git@github.com:Giuseppe499/PNPD_unrolling.git
```

Then, move into the project directory:

```bash
cd PNPD_unrolling
```

### BSDS500 dataset

The BSDS500 dataset is used for training and testing the model.
It can be downloaded from [here](https://github.com/BIDS/BSDS500/tree/master/BSDS500/data/images).
The default location is `./data/BSDS500/` and it should contain three subfolders: `test`, `train` and `val`.

### Python dependencies

Install the requirements that are listed in the `pyproject.toml` file.
You can use [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`. We also suggest using a [virtual environment](https://docs.python.org/3/library/venv.html) to manage dependencies.

run

```bash
uv sync
```

or

```bash
pip install .
```

To run the example scripts, [activate the virtual environment](https://docs.python.org/3/tutorial/venv.html#creating-virtual-environments) (if used) and execute the scripts using Python:

```bash
python test.py
```

## Usage

There are two main scripts: `train.py` and `test.py`.
The first one is used to train the model, while the second one is used to test it and to generate the plots shown in the paper.
To exactly reproduce the results shown in the paper, you can run `test.py` without training the model:

```bash
python test.py
```

This will use the pre-trained model provided in `results/PNPD_unrolling_SSIM.pth`
