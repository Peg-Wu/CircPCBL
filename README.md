# CircPCBL for Plant Circular RNA Prediction. 

## Required Packages 

* Python 3.9.0 (or a compatible version) 
* Pytorch 1.11.0 (or a compatible version) 
* NumPy 1.23.3 (or a compatible version) 
* Pandas 1.2.4 (or a compatible version) 
* Scikit-learn 1.1.1 (or a compatible version) 

## General Instructions for Conducting the CircPCBL Tool 

1. Prepare datasets (fasta file) 
2. Invoke CircPCBL with CMD 
3. Done! 

## Usage 
``` 
$ cd Model/ 
$ python xxx.py 
--usage xxx.py --input=seq.fasta --output=result.csv --batch_size=16 
``` 

* `xxx.py`: `Plant_CPU.py`; `Plant_GPU.py`; `Animal_CPU.py`; `Animal_GPU.py` 
1. [`Plant_CPU.py`](./Model/Plant_CPU.py): Used for plant circRNAs identification, running on CPU.  
2. [`Plant_GPU.py`](./Model/Plant_GPU.py): Used for plant circRNAs identification, running on GPU. 
3. [`Animal_CPU.py`](./Model/Animal_CPU.py): Used for animal circRNAs identification, running on CPU. 
4. [`Animal_GPU.py`](./Model/Animal_GPU.py): Used for animal circRNAs identification, running on GPU. 
* `--input`: File path of the input. (Must be in `fasta` format) 
* `--output`: Path for storing the result file. (Must be in `csv` format) 
* `--batch_size`: The default value is 16, it is recommended not to set it too large.

## Server
* CircPCBL is available for free on the server: [`www.circpcbl.cn`](http://circpcbl.cn/#/)

## Concat
* Please feel free to contact me if you have any questions. 
**Email**: `peg2_wu@163.com`