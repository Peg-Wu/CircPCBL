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
    [`Plant_CPU.py`](./Model/Plant_CPU.py): Used for plant circRNAs identification, running on CPU.  
    [`Plant_GPU.py`](./Model/Plant_GPU.py): Used for plant circRNAs identification, running on GPU. 
    [`Animal_CPU.py`](./Model/Animal_CPU.py): Used for animal circRNAs identification, running on CPU. 
    [`Animal_GPU.py`](./Model/Animal_GPU.py): Used for animal circRNAs identification, running on GPU. 
* `--input`: File path of the input. (Must be in **`fasta`** format) 
* `--output`: Path for storing the result file. (Must be in **`csv`** format) 