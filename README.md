[![Python tests](https://github.com/Giuseppe499/learned_pseudoinverse_filters_prototipo_deblur/actions/workflows/python_test.yml/badge.svg)](https://github.com/Giuseppe499/PNPD_unrolling/actions/workflows/python_test.yml)
[![Python lint](https://github.com/Giuseppe499/PNPD_unrolling/actions/workflows/python_lint.yml/badge.svg)](https://github.com/Giuseppe499/PNPD_unrolling/actions/workflows/python_lint.yml)
[![Python format](https://github.com/Giuseppe499/PNPD_unrolling/actions/workflows/python_format.yml/badge.svg)](https://github.com/Giuseppe499/PNPD_unrolling/actions/workflows/python_format.yml)

# PNPD Unrolling

Parameters estimation via unrolling of the Preconditioned Nested Primal Dual (PNPD) algorithm applied to image deblurring.

## Requirements

### BSDS500 dataset

It can be downloaded from [here](https://github.com/BIDS/BSDS500/tree/master/BSDS500/data/images).
The default location is `./data/BSDS500/` and it should contain three subfolders: `test`, `train` and `val`.
