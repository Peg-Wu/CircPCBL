# CircPCBL for Plant Circular RNA Prediction. 
This repository provides the implementation of our paper: "[CircPCBL: Identificaiton of Plant CircRNAs with a CNN-BiGRU-GLT Model](https://doi.org/10.3390/plants12081652)," Pengpeng Wu, Zhenjun Nie, Zhiqiang Huang, Xiaodan Zhang.

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
--usage python xxx.py --input=seq.fasta --output=result.csv --batch_size=16 
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

## Contact
* Please feel free to contact me if you have any questions: `peg2_wu@163.com`

## Guide!
* Please review the following tools before utilizing CircPCBL:  
1. `OrfPredictor`: [`http://proteomics.ysu.edu/tools/OrfPredictor.html`](http://proteomics.ysu.edu/tools/OrfPredictor.html) 
2. `NCBI ORF Finder`: [`https://www.ncbi.nlm.nih.gov/orffinder/`](https://www.ncbi.nlm.nih.gov/orffinder/)   
3. `LGC for long non-coding RNAs`: [`https://ngdc.cncb.ac.cn/lgc/`](https://ngdc.cncb.ac.cn/lgc/) 
4. `tRNA-scan`: [`http://lowelab.ucsc.edu/tRNAscan-SE/`](http://lowelab.ucsc.edu/tRNAscan-SE/) 

* Check the coding capacity of the sequences, please use `1-3` 
* Confirm whether the sequences are tRNAs or not, please use `4` 

**Once you have verified that your test sequences are lncRNAs but not mRNAs or sncRNAs along with the above tools, our model will be of assistance in further determining whether they are circRNAs.**  

#### CircPCBL Video Tutorial
* `Chinese version`: [`Click here to view`](https://www.bilibili.com/video/BV1gh411774a/?spm_id_from=333.999.0.0&vd_source=71ef231609c4f6a95ebfadad530deed2)