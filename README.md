# DeepCNV

> DeepCNV method is a novel cancer type classifier based on deep learning and copy number variants which has been implemented with Python.

## Demo
<p align="center">
  <img src="/images/demo.png" width="250" height="350" title="demo">
</p>

## Architecture
<p align="center">
  <img src="/images/architecture.png" title="architecture">
</p>

## Installation
First install `Python 3` then use the package manager [pip](https://pip.pypa.io/en/stable/) to install the following packages:

```bash
pygubu
matplotlib
numpy
pandas
keras
sklearn
```


## About Dataset
The DeepCNV method used from [cBioPortal](http://cbio.mskcc.org/cancergenomics/pancan_tcga/) dataset. The CNV data dataset contains 11 different types of cancer. In this study 6 cancer types that had more than 400 patient samples were selected.

> The dataset format should be as follows:

> |Gene_1 | Gene_2 |   ...  | Gene_n| Class |
> | ----- | ------ | ------ | ----- | -----:|
> |   +2  |   -1   |   ...  |   -2  |   C1  |
> |   -1  |   +2   |   ...  |   -1  |   C2  |
> |   --  |   --   |   ---  |   --  |   --  |
> |    0  |   -2   |   ...  |    0  |   C1  |
> |   +1  |    0   |   ...  |   +2  |   C3  |