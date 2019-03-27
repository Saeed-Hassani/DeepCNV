# DeepCNV
DeepCNV method is a novel cancer type classifier based on deep learning and copy number variants which has been implemented with Python.
## Demo

<p align="center">
  <img src="/images/demo.png" width="250" height="350" title="Demo">
</p>
## Architecture

<p align="center">
  <img src="/images/demo.png" width="250" height="350" title="Demo">
</p>

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the following packages:

```bash
**pygubu**
**matplotlib**
**numpy**
**pandas**
**keras**
**sklearn**
```

فرمت ورودی به برنامه:

|Gene_1 | Gene_2 |   ...  | Gene_n| Class |
| ----- | ------ | ------ | ----- | -----:|
|   +2  |   -1   |   ...  |   -2  |   C1  |
|   -1  |   +2   |   ...  |   -1  |   C2  |
|    .  |    .   |    .   |    .  |   .   |
|    0  |   -2   |   ...  |    0  |   C1  |
|   +1  |    0   |   ...  |   +2  |   C3  |

