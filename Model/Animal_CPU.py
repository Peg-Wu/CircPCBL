import re
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from itertools import product
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str)
parser.add_argument('--output', type=str, default='./result.csv')
parser.add_argument('--batch_size', type=int, default=16)
opt = parser.parse_known_args()[0]


# fasta to csv (extract the sequences)
pattern_title = re.compile(r'^>.*', re.M)
pattern_n = re.compile(r'\n')

with open(opt.input, 'r', encoding='utf-8') as f:  # upload your fasta file
    text = f.read()

data = re.split(pattern_title, text)
data = [re.sub(pattern_n, '', i) for i in data]
data = pd.DataFrame(data[1:])
data.columns = ['Sequence']

# compute k-mers features (k=1~4)
def Kmers_funct(seq, x):
    X = [None] * len(seq)
    for i in range(len(seq)):
        a = seq[i]
        t = 0
        l = []
        for index in range(len(a) - x + 1):
            t = a[index:index + x]
            if (len(t)) == x:
                l.append(t)
        X[i] = l
    return X

def nucleotide_type(k):
    z = []
    for i in product('ACGT', repeat=k):
        z.append(''.join(i))
    return z

def Kmers_frequency(seq, x):
    X = []
    char = nucleotide_type(x)
    for i in range(len(seq)):
        s = seq[i]
        frequence = []
        for a in char:
            number = s.count(a)
            char_frequence = number / (len(s) - x + 1)
            frequence.append(char_frequence)
        X.append(frequence)
    return X

feature_1mer = Kmers_funct(data.Sequence, 1)  # 1-mer
feature_2mer = Kmers_funct(data.Sequence, 2)  # 2-mer
feature_3mer = Kmers_funct(data.Sequence, 3)  # 3-mer
feature_4mer = Kmers_funct(data.Sequence, 4)  # 4-mer

feature_1mer_frequency = Kmers_frequency(feature_1mer, 1)  # 1-mer
feature_2mer_frequency = Kmers_frequency(feature_2mer, 2)  # 2-mer
feature_3mer_frequency = Kmers_frequency(feature_3mer, 3)  # 3-mer
feature_4mer_frequency = Kmers_frequency(feature_4mer, 4)  # 4-mer

feature = pd.concat([pd.DataFrame(feature_1mer_frequency),
                     pd.DataFrame(feature_2mer_frequency),
                     pd.DataFrame(feature_3mer_frequency),
                     pd.DataFrame(feature_4mer_frequency)], axis=1)

column_name = []

for k in range(1, 5):
    for each in nucleotide_type(k):
        column_name.append(each)

feature.columns = column_name
data = pd.concat([data, feature], axis=1)

# process the sequences
pat = re.compile('[A-Za-z]')

def pre_process(text):
    text = pat.findall(text)
    text = [each.lower() for each in text]
    return text

x = data.Sequence.apply(pre_process)

word_list = ['a', 'g', 'c', 't']
word_index = {'a': 0, 'g': 1, 'c': 2, 't': 3}

text = x.apply(lambda x: [word_index.get(word, 4) for word in x])

# fix the sequences' length
text_len = 1500
pad_text = [l + (text_len - len(l)) * [4] if len(l) < text_len else l[:text_len] for l in text]
pad_text = np.array(pad_text)
pad_text = np.concatenate([pad_text, data.iloc[:, 1:].values], axis=1)
X = torch.tensor(pad_text, dtype=torch.float32)

# define the network
one_hot_dim = len(word_list)
hidden_size = 30

class Net(nn.Module):
    def __init__(self, word_list, one_hot_dim, hidden_size, num_layers=1):
        super().__init__()

        self.conv_7 = nn.Conv1d(in_channels=one_hot_dim,
                                out_channels=32,
                                kernel_size=7,
                                stride=1,
                                padding=3)
        self.conv_3 = nn.Conv1d(in_channels=one_hot_dim,
                                out_channels=32,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.conv_5 = nn.Conv1d(in_channels=one_hot_dim,
                                out_channels=32,
                                kernel_size=5,
                                stride=1,
                                padding=2)
        self.max_pool_1 = nn.MaxPool1d(kernel_size=3,
                                       stride=1,
                                       padding=1)
        self.max_pool_2 = nn.MaxPool1d(kernel_size=3,
                                       stride=1,
                                       padding=1)
        self.max_pool_3 = nn.MaxPool1d(kernel_size=3,
                                       stride=1,
                                       padding=1)

        self.bigru = nn.GRU(32 * 3, hidden_size, num_layers, bidirectional=True)
        self.fc1 = nn.Linear(2 * hidden_size, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(64, 2)

        # GLT layer

        self.GLT1 = nn.Linear(340, 256)

        self.GLT2_1 = nn.Linear(128, 64)
        self.GLT2_2 = nn.Linear(128, 64)

        self.GLT3_1 = nn.Linear(32, 8)
        self.GLT3_2 = nn.Linear(32, 8)
        self.GLT3_3 = nn.Linear(32, 8)
        self.GLT3_4 = nn.Linear(32, 8)

        self.norm_8 = nn.LayerNorm([8])
        self.norm_32 = nn.LayerNorm([32])
        self.norm_64 = nn.LayerNorm([64])
        self.norm_128 = nn.LayerNorm([128])
        self.norm_256 = nn.LayerNorm([256])

    def forward(self, x):
        (x, x_features) = torch.split(x, text_len, dim=-1)
        x = x.long()
        x = F.one_hot(x, num_classes=len(word_list) + 1)

        x = x[:, :, :-1]
        x = x.permute(0, 2, 1)
        x = x.float()

        x1 = self.max_pool_1(F.relu(self.conv_7(x)))
        x2 = self.max_pool_2(F.relu(self.conv_3(x)))
        x3 = self.max_pool_3(F.relu(self.conv_5(x)))
        x = torch.cat([x1, x2, x3], dim=1)
        x = x.permute(2, 0, 1)
        x, _ = self.bigru(x)
        x = torch.sum(x, dim=0)
        x = F.relu(self.norm_128(self.fc1(x)))
        x = F.relu(self.norm_32(self.fc2(x)))

        # GLT layers
        x_features = F.relu(self.norm_256(self.GLT1(x_features)))
        (x1, x2) = torch.chunk(x_features, 2, dim=-1)
        x1 = F.relu(self.norm_64(self.GLT2_1(x1)))
        x2 = F.relu(self.norm_64(self.GLT2_2(x2)))
        x_features = torch.cat([x1, x2], dim=-1)
        idx = torch.randperm(x_features.shape[-1])
        x_features = x_features[:, idx].view(x_features.shape)
        (x1, x2, x3, x4) = torch.chunk(x_features, 4, dim=-1)
        x1 = F.relu(self.norm_8(self.GLT3_1(x1)))
        x2 = F.relu(self.norm_8(self.GLT3_2(x2)))
        x3 = F.relu(self.norm_8(self.GLT3_3(x3)))
        x4 = F.relu(self.norm_8(self.GLT3_4(x4)))
        x_features = torch.cat([x1, x2, x3, x4], dim=-1)

        x = torch.cat([x, x_features], dim=-1)
        x = self.fc3(x)

        return x

model = Net(word_list, one_hot_dim, hidden_size)
model.load_state_dict(torch.load('../Param/Animal.pkl', map_location=torch.device('cpu')))
# loss = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Dataset and DataLoader
ds_X = TensorDataset(X)
dl_X = DataLoader(ds_X, batch_size=opt.batch_size, shuffle=False)

# output results
result = torch.tensor([])
for (x,) in dl_X:
    y_pred = torch.argmax(model(x), dim=1)
    result = torch.cat([result, y_pred])

# to Dataframe
result = pd.concat([pd.DataFrame(range(len(result))), pd.DataFrame(result)], axis=1)
result.columns = ['Index', 'Predict_result']
result['Predict_result'] = np.where(result['Predict_result'] == 1, 'circRNA', 'lncRNA')
result.to_csv(opt.output, index=False)

print('finish!')
