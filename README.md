# CircPCBL
 A deep learning based method for identifying plants' circRNAs and other lncRNAs.

# 1.How to use CircPCBL?
## Method 1:
You can easily use CircPCBL via www.circpcbl.cn

## Method 2: 
### step 1:
You can download two files in our repositories:
(1) CircPCBL_CPU.py or CircPCBL_GPU.py
(2) model.pkl
### step 2:
Place the pkl file in a directory above the py file.
### step 3:
Change the fasta file path at **line 15** of the py file.

# 2.How can we speed up CircPCBL?
We recommend downloading CircPCBL_GPU.py or increasing the batch_size.

# 3.What can we do if the memory is insufficient?
Try reducing the batch_size.


# The flowchart for developing CircPCBL
![image](Graphic Abstract/Graphic Abstract.png)

## Picture Discription
(1) Part A describes all the datasets used in our work.
(2) Parts B and C demonstrate the one-hot and k-mer encoding processes, respectively.
(3) Parts D and E represent the architecture of CNN-BiGRU model and GLT model.
(4) Part F shows the course of outputting results.