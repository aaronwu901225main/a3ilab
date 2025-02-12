# -*- coding: utf-8 -*-
from __future__ import print_function
from torch.utils import data
from torchvision import datasets, transforms
import math
import numpy as np
import pandas as pd
import json
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import Counter
from pytorch_metric_learning import samplers
import sys
# sys.path.append('/root/notebooks/Phison/defect_classification-maim/clustimage_phison/clustimage/')
# sys.path.append('/root/notebooks/deepGD3_plus/defect_classification/PCB_dataset/clustimage_phison/clustimage')
# from clustimage import Clustimage
labelencoder = LabelEncoder()

# data_dir = "/root/notebooks/datasets_for_ma/"
# json_path = "/root/notebooks/Phison/dataset.json"
# noisy_label_path = "/root/notebooks/Phison/toRemove.txt"
# data_dir = "/home/a3ilab01/.datasets_for_ma/"
# json_path = "/home/a3ilab01/Downloads/dataset.json"
# noisy_label_path = "/home/a3ilab01/Downloads/toRemove.txt"
data_dir = "/root/notebooks/GPCE/datasets_for_ma/"
json_path = "/root/notebooks/GPCE/Phison/dataset.json"
noisy_label_path = "/root/notebooks/GPCE/Phison/toRemove.txt"


# data_dir = "/root/notebooks/Phison/.datasets_for_ma/"
# json_path = "/root/notebooks/Phison/dataset.json"
# noisy_label_path = "/root/notebooks/Phison/toRemove.txt"
def split_stratified_into_train_val_test(df_input, stratify_colname='y',
                                     frac_train=0.8, frac_val=0.1, frac_test=0.1,
                                     random_state=None):
    if frac_train + frac_val + frac_test != 1.0:
        raise ValueError('fractions %f, %f, %f do not add up to 1.0' % \
                         (frac_train, frac_val, frac_test))
    if stratify_colname not in df_input.columns:
        raise ValueError('%s is not a column in the dataframe' % (stratify_colname))

    X = df_input # Contains all columns.
    y = df_input[[stratify_colname]] # Dataframe of just the column on which to stratify.

    # Split original dataframe into train and temp dataframes.
    df_train, df_temp, y_train, y_temp = train_test_split(X,
                                                          y,
                                                          stratify=y,
                                                          test_size=(1.0 - frac_train),
                                                          random_state=random_state)

    # Split the temp dataframe into val and test dataframes.
    relative_frac_test = frac_test / (frac_val + frac_test)
    df_val, df_test, y_val, y_test = train_test_split(df_temp,
                                                      y_temp,
                                                      stratify=y_temp,
                                                      test_size=relative_frac_test,
                                                      random_state=random_state)

    assert len(df_input) == len(df_train) + len(df_val) + len(df_test)

    return df_train, df_val, df_test

def split_stratified_into_train_val(df_input, stratify_colname='y',
                                     frac_train=0.8, frac_val=0.1,
                                     random_state=None):
    if frac_train + frac_val != 1.0:
        raise ValueError('fractions %f, %f, %f do not add up to 1.0' % \
                         (frac_train, frac_val))
    if stratify_colname not in df_input.columns:
        raise ValueError('%s is not a column in the dataframe' % (stratify_colname))

    X = df_input # Contains all columns.
    y = df_input[[stratify_colname]] # Dataframe of just the column on which to stratify.

    # Split original dataframe into train and temp dataframes.
    df_train, df_val, y_train, y_val = train_test_split(X,
                                                          y,
                                                          stratify=y,
                                                          test_size=(1.0 - frac_train),
                                                          random_state=random_state)

    # Split the temp dataframe into val and test dataframes.
#     relative_frac_test = frac_test / (frac_val + frac_test)
#     df_val, df_test, y_val, y_test = train_test_split(df_temp,
#                                                       y_temp,
#                                                       stratify=y_temp,
#                                                       test_size=relative_frac_test,
#                                                       random_state=random_state)

    assert len(df_input) == len(df_train) + len(df_val)

    return df_train, df_val


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transform, mode='train'):
        self.dataframe = dataframe
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image = self.transform(Image.open((row["file_path"])).convert('RGB'))
#         image = self.transform(Image.open((row["file_path"])))
        label = np.asarray(row["class"])
        component_name = row["component_name"]
        file_path = row["file_path"]
        return (image, label, file_path, component_name)
def CreateDataset(seed,add_test, testing=None):
    random.seed(seed)
    print('==> Preparing data..')
    dataset_info = json.load(open(json_path, "r"))
    df = pd.DataFrame.from_dict(dataset_info, orient="index")
    df['file_path'] = df.index
    df["file_path"] = data_dir + df["file_path"].astype(str)
    
    # 分成6個class
    df.loc[df['class'] == "good", 'class'] = 0
    df.loc[df['class'] == "missing", 'class'] = 1
    df.loc[df['class'] == "shift", 'class'] = 2
    df.loc[df['class'] == "stand", 'class'] = 3
    df.loc[df['class'] == "broke", 'class'] = 4
    df.loc[df['class'] == "short", 'class'] = 5    
    # 移除資料集中的Label Noise   
    unwantedData = pd.read_csv(noisy_label_path, sep=",", header=None)[0].tolist()
    df = df[~df.file_path.isin(unwantedData)]    
    
    df['component_name'] = labelencoder.fit_transform(df['component_name'])
    component_name_counter = Counter(labelencoder.inverse_transform(df['component_name']))
    component_name_list = [key for key, _ in component_name_counter.most_common()]
    component_label_list = df['component_name'].value_counts().index.tolist()
    component_dict = {component_label_list[i]: component_name_list[i] for i in range(len(component_label_list))}
    # 將Test set從Training set中移除並重新切割資料集
    trainComponent = df['component_name'].value_counts().index.tolist()
    trainComponent.remove(1) # 元件A (樣本最多的)
    valComponent = random.sample(trainComponent, 6)
    for i in valComponent:
        trainComponent.remove(i)
    testComponent = random.sample(trainComponent, 6)
    for i in testComponent:
        trainComponent.remove(i)
    trainComponent.append(1)
    
    trainDatasetMask = df['component_name'].isin(trainComponent)
    train_df = df[trainDatasetMask].copy()
    
    print("Train component label: ")
    train_component_label = train_df['component_name'].value_counts().index.tolist()
    print(train_component_label)
    train_component_name=[]
    print("Train component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in train_component_label:  
            train_component_name.append(v)
    print(train_component_name)

#     import pdb;pdb.set_trace()
    # Set missing, stand samples as independent components
    if testing is None:
        train_df.loc[train_df['class'] == 1, ['component_name']] = 35 # missing
        train_df.loc[train_df['class'] == 3, ['component_name']] = 36 # stand
    train_df.loc[train_df['class'] == 0, 'class'] = 0
    train_df.loc[train_df['class'] == 1, 'class'] = 1
    train_df.loc[train_df['class'] == 2, 'class'] = 1
    train_df.loc[train_df['class'] == 3, 'class'] = 1
    train_df.loc[train_df['class'] == 4, 'class'] = 1
    train_df.loc[train_df['class'] == 5, 'class'] = 1
    
    # 將一部分的In-distribution old component分出來給val set和test set (ind_val, ind_test)
    train_df, ind_val, ind_test = split_stratified_into_train_val_test(train_df, stratify_colname='component_name', frac_train=0.8, frac_val=0.1, frac_test=0.1, random_state=seed)

    valDatasetMask = df['component_name'].isin(valComponent)
    val_df = df[valDatasetMask].copy()
    print("Val component label: ")
    val_component_label = val_df['component_name'].value_counts().index.tolist()
    print(val_component_label)
    val_component_name=[]
    print("Val component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in val_component_label:  
            val_component_name.append(v)
    print(val_component_name)
    
    if testing is None:
        # Set missing, stand samples as independent components
        val_df.loc[val_df['class'] == 1, ['component_name']] = 35
        val_df.loc[val_df['class'] == 3, ['component_name']] = 36
    # 分成2個class (Good and Bad)
    val_df.loc[val_df['class'] == 0, 'class'] = 0
    val_df.loc[val_df['class'] == 1, 'class'] = 1
    val_df.loc[val_df['class'] == 2, 'class'] = 1
    val_df.loc[val_df['class'] == 3, 'class'] = 1
    val_df.loc[val_df['class'] == 4, 'class'] = 1
    val_df.loc[val_df['class'] == 5, 'class'] = 1
    val_df = pd.concat([val_df, ind_val])
    testDatasetMask = df['component_name'].isin(testComponent)
    test_df = df[testDatasetMask].copy()
    print("Test component label: ")
    test_component_label = test_df['component_name'].value_counts().index.tolist()
    print(test_component_label)
    test_component_name=[]
    print("Test component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in test_component_label:  
            test_component_name.append(v)
    print(test_component_name)
    
    if testing is None:
        test_df.loc[test_df['class'] == 1, ['component_name']] = 35, #'solder_missing'
        test_df.loc[test_df['class'] == 3, ['component_name']] = 36, #'solder_stand'
    test_df.loc[test_df['class'] == 0, 'class'] = 0
    test_df.loc[test_df['class'] == 1, 'class'] = 1
    test_df.loc[test_df['class'] == 2, 'class'] = 1
    test_df.loc[test_df['class'] == 3, 'class'] = 1
    test_df.loc[test_df['class'] == 4, 'class'] = 1
    test_df.loc[test_df['class'] == 5, 'class'] = 1
    test_df = pd.concat([test_df, ind_test])
    
    with open(f"split_{seed}_component_name_label_mapping.txt", 'w') as f:
        f.write('Train: \n' + str(train_component_name) + '\n' + str(train_component_label) + '\n' +
                'Val: \n' + str(val_component_name) + '\n' + str(val_component_label) + '\n' + 
                'Test: \n' + str(test_component_name) +'\n' + str(test_component_label)
               )
    # 用來產生overkill和leakage數值的dataframe    
    test_df_mapping2_label = test_df.copy()
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 0, 'class'] = 0
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 1, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 2, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 3, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 4, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 5, 'class'] = 1

    name_of_each_component = test_df_mapping2_label['component_name'].value_counts().index.tolist()
    num_of_image_in_each_component = test_df_mapping2_label['component_name'].value_counts().values
    test_component_name_df = pd.DataFrame(list(zip(name_of_each_component, num_of_image_in_each_component)), columns =['component_name', 'total'])

    for name in set(test_df_mapping2_label['component_name'].values):
        temp_data = test_df_mapping2_label.loc[(test_df_mapping2_label["component_name"] == name)]
        for k, v in zip(temp_data['class'].value_counts().keys(), temp_data['class'].value_counts()):
            if k == 0:
                test_component_name_df.loc[test_component_name_df['component_name'] == name, 'good'] = temp_data['class'].value_counts().sort_index().values[0]
            elif k ==1:
                try:
                    test_component_name_df.loc[test_component_name_df['component_name'] == name, 'bad'] = temp_data['class'].value_counts().sort_index().values[1]
                except:
                    print(f"{name} only contains bad label.")
                    test_component_name_df.loc[test_component_name_df['component_name'] == name, 'bad'] = temp_data['class'].value_counts().sort_index().values[0]
    test_component_name_df['good'] = test_component_name_df['good'].fillna(0).astype(int)
    test_component_name_df['bad'] = test_component_name_df['bad'].fillna(0).astype(int)
    test_component_name_df = test_component_name_df[['component_name', 'total', 'good', 'bad']]    
    col = {'overkill': 0, 'leakage': 0}
    test_component_name_df = test_component_name_df.assign(**col)

    test_set_class = sorted(test_df['class'].value_counts().keys().tolist())   #由於每個component的label都不一樣，透過這個方式取得該component下的所有label
    print('add_test:',add_test)
    if add_test == True:
        # 取得new component的good sample給component classifier訓練
        for name in valComponent:
            good_new_component = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] == 0)].sample(frac=0.5, random_state=123)
            val_df = val_df.drop(good_new_component.index)
            bad_new_component_sample = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] != 0)]
            val_df = val_df.drop(bad_new_component_sample.index)
            train_df = pd.concat([train_df, good_new_component])
        for name in testComponent:
            good_new_component = test_df.loc[(test_df['component_name'] == name) & (test_df['class'] == 0)].sample(frac=0.5, random_state=123)
            test_df = test_df.drop(good_new_component.index)
            train_df = pd.concat([train_df, good_new_component])
            
    
    train_good_df = train_df.copy()
    train_good_df = train_good_df.loc[train_good_df['class']==0]
    a = Counter(train_good_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_good_df.loc[train_good_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_good_df[train_good_df['component_name']==i].index
            train_good_df=train_good_df.drop(df_idx)
            train_good_df = pd.concat([train_good_df, component])
            
    train_bad_df = train_df.copy()
    train_bad_df = train_bad_df.loc[train_bad_df['class']==1]
    a = Counter(train_bad_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_bad_df.loc[train_bad_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_bad_df[train_bad_df['component_name']==i].index
            train_bad_df=train_bad_df.drop(df_idx)
            train_bad_df = pd.concat([train_bad_df, component])
            
    train_df = pd.concat([train_good_df, train_bad_df])
    
    
    train_com_df = train_df.copy()
    good_samples = train_com_df.loc[train_df['class']==0]
    missing_samples = train_com_df.loc[(train_com_df['component_name']==35)]
    stand_samples = train_com_df.loc[(train_com_df['component_name']==36)]
    train_com_df = pd.concat([good_samples, missing_samples, stand_samples])
    
    val_good_df = val_df.copy()
    val_good_df = val_good_df.loc[val_good_df['class']==0]
    a = Counter(val_good_df['component_name'])
    for i in range(max(a)):
        if a[i] >5000:
            component = val_good_df.loc[val_good_df['component_name']==i]
            component = component.sample(n=5000,random_state=123,axis=0)
            df_idx = val_good_df[val_good_df['component_name']==i].index
            val_good_df=val_good_df.drop(df_idx)
            val_good_df = pd.concat([val_good_df, component])
            
    val_bad_df = val_df.copy()
    val_bad_df = val_bad_df.loc[val_bad_df['class']==1]
    a = Counter(val_bad_df['component_name'])
    for i in range(max(a)):
        if a[i] >5000:
            component = val_bad_df.loc[val_bad_df['component_name']==i]
            component = component.sample(n=5000,random_state=123,axis=0)
            df_idx = val_bad_df[val_bad_df['component_name']==i].index
            val_bad_df=val_bad_df.drop(df_idx)
            val_bad_df = pd.concat([val_bad_df, component])
            
    val_df = pd.concat([val_good_df, val_bad_df])

    
    print("Class distribution in Component Training set:")
    print(train_df['class'].value_counts())
    print("\nClass distribution in Val set:")
    print(val_df['class'].value_counts())
    print("\nClass distribution in Testing set:")
    print(test_df['class'].value_counts())
    print("Num of Images in Component Training set: ", sum(train_df['class'].value_counts().tolist()))
    print("Num of Images in Validation set: ", sum(val_df['class'].value_counts().tolist()))
    print("Num of Images in Testing set: ", sum(test_df['class'].value_counts().tolist()))
    return train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df 

def get_PHISON_ori(root, seed):
    input_size = 224
    num_classes = 37
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    train_transform = transforms.Compose(
        [
            transforms.Resize([input_size, input_size]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    
    train_cls_transform = transforms.Compose(
        [
            transforms.Resize([input_size, input_size]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )
    
    
    test_transform = transforms.Compose([
        transforms.Resize([input_size, input_size]),
        transforms.ToTensor(),  
        normalize])
    
    add_test = True
    
    train_cls_df, val_df, test_df, _, _, _, train_com_df= CreateDataset(seed ,add_test)

#     train_cls_df = train_cls_df.sample(frac=0.01, random_state=123)
#     train_com_df = train_com_df.sample(frac=0.01, random_state=123)
#     val_df = val_df.sample(frac=0.01, random_state=123)

    train_cls_dataset = CustomDataset(train_cls_df, transform=train_cls_transform)    
    train_com_dataset = CustomDataset(train_com_df, transform=train_transform)    
    
    
    test_dataset = CustomDataset(val_df, transform=test_transform)
    
    per_component_num = 128 // len(train_com_dataset.dataframe['component_name'].value_counts().index)
    per_class_num = 128 // len(train_cls_dataset.dataframe['class'].value_counts().index)
    train_com_sampler = samplers.MPerClassSampler(train_com_dataset.dataframe['component_name'], per_component_num, length_before_new_iter=len(train_com_dataset))
    train_cls_sampler = samplers.MPerClassSampler(train_cls_dataset.dataframe['class'], per_class_num, length_before_new_iter=len(train_cls_dataset))

    train_com_loader = torch.utils.data.DataLoader(
            train_com_dataset, batch_size=128, shuffle=(train_com_sampler is None),
            num_workers=8, pin_memory=True,drop_last=True, sampler=train_com_sampler)

    train_cls_loader = torch.utils.data.DataLoader(
            train_cls_dataset, batch_size=128, shuffle=(train_cls_sampler is None),
            num_workers=8, pin_memory=True,drop_last=True, sampler=train_cls_sampler)
    

    return input_size, num_classes, train_com_dataset, test_dataset, train_cls_loader, train_com_loader 



def CreateDataset_relabel_fourcls(seed , add_test, testing=None):
    # 1:A, 2:B, 3:C, 4:D, 7:F, 8:E
    random.seed(seed)
    print('==> Preparing data..')
    df = pd.read_csv("~/GPCE/Phison/dataset_relabel_ywl.csv")
    
    # 分成6個class
    df.loc[df['class'] == "good", 'class'] = 0
    df.loc[df['class'] == "missing", 'class'] = 1
    df.loc[df['class'] == "shift", 'class'] = 2
    df.loc[df['class'] == "stand", 'class'] = 3
    df.loc[df['class'] == "broke", 'class'] = 4
    df.loc[df['class'] == "short", 'class'] = 5    
    # 移除資料集中的Label Noise   
    unwantedData = pd.read_csv(noisy_label_path, sep=",", header=None)[0].tolist()
    df = df[~df.file_path.isin(unwantedData)]    
    
    df['component_name'] = labelencoder.fit_transform(df['component_name'])
    component_name_counter = Counter(labelencoder.inverse_transform(df['component_name']))
    component_name_list = [key for key, _ in component_name_counter.most_common()]
    component_label_list = df['component_name'].value_counts().index.tolist()
    component_dict = {component_label_list[i]: component_name_list[i] for i in range(len(component_label_list))}
#     import pdb;pdb.set_trace()
    # 將Test set從Training set中移除並重新切割資料集
    trainComponent = df['component_name'].value_counts().index.tolist()

    # trainComponent = [1,3,5,6,10,11,16,17]
    trainComponent = [1,2,3,5,6,10,11,14,16,17]
    trainDefect = [0,2,4,5]
    
#     trainComponent = [1,6,10,11]
#     trainDefect = [0,2,4,5]
    
    trainComponentDatasetMask = df['component_name'].isin(trainComponent)
    df = df[trainComponentDatasetMask].copy()

    trainDefectDatasetMask = df['class'].isin(trainDefect)
    df = df[trainDefectDatasetMask].copy()
    
#     newComponent = [4,8,9,12,13,14,20,2,3,5,10,11,17]

#     df.loc[df['class'] == 0, 'class'] = 0
#     df.loc[df['class'] == 1, 'class'] = 0
#     df.loc[df['class'] == 2, 'class'] = 0
#     df.loc[df['class'] == 3, 'class'] = 0
#     df.loc[df['class'] == 4, 'class'] = 1
#     df.loc[df['class'] == 5, 'class'] = 0
    
    trainComponent.remove(1)
    trainComponent.remove(6)
    
    valComponent = random.sample(trainComponent, 2)
    for i in valComponent:
        trainComponent.remove(i)
    testComponent = random.sample(trainComponent, 2)
    for i in testComponent:
        trainComponent.remove(i)
    trainComponent.append(1)
    testComponent.append(6)
    
    
    trainDatasetMask = df['component_name'].isin(trainComponent)
    train_df = df[trainDatasetMask].copy()
    
    print("Train component label: ")
    train_component_label = train_df['component_name'].value_counts().index.tolist()
    print(train_component_label)
    train_component_name=[]
    print("Train component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in train_component_label:  
            train_component_name.append(v)
    print(train_component_name)
    
    # 將一部分的In-distribution old component分出來給val set和test set (ind_val, ind_test)
    train_df, ind_val, ind_test = split_stratified_into_train_val_test(train_df, stratify_colname='component_name', frac_train=0.8, frac_val=0.1, frac_test=0.1, random_state=seed)

    valDatasetMask = df['component_name'].isin(valComponent)
    val_df = df[valDatasetMask].copy()
    print("Val component label: ")
    val_component_label = val_df['component_name'].value_counts().index.tolist()
    print(val_component_label)
    val_component_name=[]
    print("Val component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in val_component_label:  
            val_component_name.append(v)
    print(val_component_name)
    
    val_df = pd.concat([val_df, ind_val])
    testDatasetMask = df['component_name'].isin(testComponent)
    test_df = df[testDatasetMask].copy()
    print("Test component label: ")
    test_component_label = test_df['component_name'].value_counts().index.tolist()
    print(test_component_label)
    test_component_name=[]
    print("Test component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in test_component_label:  
            test_component_name.append(v)
    print(test_component_name)
    
        
    test_df = pd.concat([test_df, ind_test])
    
    with open(f"split_{seed}_component_name_label_mapping.txt", 'w') as f:
        f.write('Train: \n' + str(train_component_name) + '\n' + str(train_component_label) + '\n' +
                'Val: \n' + str(val_component_name) + '\n' + str(val_component_label) + '\n' + 
                'Test: \n' + str(test_component_name) +'\n' + str(test_component_label)
               )

    test_set_class = sorted(test_df['class'].value_counts().keys().tolist())   #由於每個component的label都不一樣，透過這個方式取得該component下的所有label
    print('add_test:',add_test)
    if add_test == True:
        # 取得new component的good sample給component classifier訓練
        for name in valComponent:
            good_new_component = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] == 0)].sample(frac=0.5, random_state=123)
            val_df = val_df.drop(good_new_component.index)
            bad_new_component_sample = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] != 0)]
            val_df = val_df.drop(bad_new_component_sample.index)
            train_df = pd.concat([train_df, good_new_component])
        for name in testComponent:
            good_new_component = test_df.loc[(test_df['component_name'] == name) & (test_df['class'] == 0)].sample(frac=0.5, random_state=123)
            test_df = test_df.drop(good_new_component.index)
            train_df = pd.concat([train_df, good_new_component])
    
    val_component_label = val_df['component_name'].value_counts().index.tolist()
    test_component_label = test_df['component_name'].value_counts().index.tolist()

    train_com_df = train_df.copy()
    good_samples = train_com_df.loc[train_df['class']==0]
    train_com_df = good_samples.copy()
    a = Counter(train_com_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_com_df.loc[train_com_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_com_df[train_com_df['component_name']==i].index
            train_com_df=train_com_df.drop(df_idx)
            train_com_df = pd.concat([train_com_df, component])
    
    ### train_defect_df
    train_good_df = train_df.copy()
    train_good_df = train_good_df.loc[train_good_df['class']==0]
    a = Counter(train_good_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_good_df.loc[train_good_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_good_df[train_good_df['component_name']==i].index
            train_good_df=train_good_df.drop(df_idx)
            train_good_df = pd.concat([train_good_df, component])
            
    train_bad_df = train_df.copy()
    train_bad_df = train_bad_df.loc[train_bad_df['class']!=0]
    a = Counter(train_bad_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_bad_df.loc[train_bad_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_bad_df[train_bad_df['component_name']==i].index
            train_bad_df=train_bad_df.drop(df_idx)
            train_bad_df = pd.concat([train_bad_df, component])
            
    train_df = pd.concat([train_good_df, train_bad_df])
    
    ### val_df
    val_good_df = val_df.copy()
    val_good_df = val_good_df.loc[val_good_df['class']==0]
    a = Counter(val_good_df['component_name'])
    for i in range(max(a)):
        if a[i] >5000:
            component = val_good_df.loc[val_good_df['component_name']==i]
            component = component.sample(n=5000,random_state=123,axis=0)
            df_idx = val_good_df[val_good_df['component_name']==i].index
            val_good_df=val_good_df.drop(df_idx)
            val_good_df = pd.concat([val_good_df, component])
            
    val_com_df = val_good_df.copy()
            
    val_bad_df = val_df.copy()
    val_bad_df = val_bad_df.loc[val_bad_df['class']!=0]
    a = Counter(val_bad_df['component_name'])
    for i in range(max(a)):
        if a[i] >5000:
            component = val_bad_df.loc[val_bad_df['component_name']==i]
            component = component.sample(n=5000,random_state=123,axis=0)
            df_idx = val_bad_df[val_bad_df['component_name']==i].index
            val_bad_df=val_bad_df.drop(df_idx)
            val_bad_df = pd.concat([val_bad_df, component])
            
    val_df = pd.concat([val_good_df, val_bad_df])
    
    print("Class distribution in Component Training set:")
    print(train_df['class'].value_counts())
    print("\nClass distribution in Val set:")
    print(val_df['class'].value_counts())
    print("\nClass distribution in Testing set:")
    print(test_df['class'].value_counts())
    print("Num of Images in Component Training set: ", sum(train_df['class'].value_counts().tolist()))
    print("Num of Images in Validation set: ", sum(val_df['class'].value_counts().tolist()))
    print("Num of Images in Testing set: ", sum(test_df['class'].value_counts().tolist()))
    return train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df, val_com_df

def CreateDataset_relabel_threecls(seed , add_test, testing=None):
    # 1:A, 2:B, 3:C, 4:D, 7:F, 8:E
    random.seed(seed)
    print('==> Preparing data..')
    df = pd.read_csv("~/GPCE/Phison/dataset_relabel_ywl.csv")
    
    # 分成6個class
    df.loc[df['class'] == "good", 'class'] = 0
    df.loc[df['class'] == "missing", 'class'] = 1
    df.loc[df['class'] == "shift", 'class'] = 2
    df.loc[df['class'] == "stand", 'class'] = 3
    df.loc[df['class'] == "broke", 'class'] = 4
    df.loc[df['class'] == "short", 'class'] = 5    
    # 移除資料集中的Label Noise   
    unwantedData = pd.read_csv(noisy_label_path, sep=",", header=None)[0].tolist()
    df = df[~df.file_path.isin(unwantedData)]    
    
    df['component_name'] = labelencoder.fit_transform(df['component_name'])
    component_name_counter = Counter(labelencoder.inverse_transform(df['component_name']))
    component_name_list = [key for key, _ in component_name_counter.most_common()]
    component_label_list = df['component_name'].value_counts().index.tolist()
    component_dict = {component_label_list[i]: component_name_list[i] for i in range(len(component_label_list))}
#     import pdb;pdb.set_trace()
    # 將Test set從Training set中移除並重新切割資料集
    trainComponent = df['component_name'].value_counts().index.tolist()

    trainComponent = [1,2,3,5,6,10,11,14,16,17]
    trainDefect = [2,4,5]
    
    trainComponentDatasetMask = df['component_name'].isin(trainComponent)
    df = df[trainComponentDatasetMask].copy()

    trainDefectDatasetMask = df['class'].isin(trainDefect)
    df = df[trainDefectDatasetMask].copy()
    
    trainComponent.remove(1)
    trainComponent.remove(6)
    
    valComponent = random.sample(trainComponent, 2)
    for i in valComponent:
        trainComponent.remove(i)
    testComponent = random.sample(trainComponent, 2)
    for i in testComponent:
        trainComponent.remove(i)
    trainComponent.append(1)
    testComponent.append(6)
    
    
    trainDatasetMask = df['component_name'].isin(trainComponent)
    train_df = df[trainDatasetMask].copy()
    
    print("Train component label: ")
    train_component_label = train_df['component_name'].value_counts().index.tolist()
    print(train_component_label)
    train_component_name=[]
    print("Train component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in train_component_label:  
            train_component_name.append(v)
    print(train_component_name)
    
    # 將一部分的In-distribution old component分出來給val set和test set (ind_val, ind_test)
    train_df, ind_val, ind_test = split_stratified_into_train_val_test(train_df, stratify_colname='component_name', frac_train=0.8, frac_val=0.1, frac_test=0.1, random_state=seed)

    valDatasetMask = df['component_name'].isin(valComponent)
    val_df = df[valDatasetMask].copy()
    print("Val component label: ")
    val_component_label = val_df['component_name'].value_counts().index.tolist()
    print(val_component_label)
    val_component_name=[]
    print("Val component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in val_component_label:  
            val_component_name.append(v)
    print(val_component_name)
    
    val_df = pd.concat([val_df, ind_val])
    testDatasetMask = df['component_name'].isin(testComponent)
    test_df = df[testDatasetMask].copy()
    print("Test component label: ")
    test_component_label = test_df['component_name'].value_counts().index.tolist()
    print(test_component_label)
    test_component_name=[]
    print("Test component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in test_component_label:  
            test_component_name.append(v)
    print(test_component_name)
    
        
    test_df = pd.concat([test_df, ind_test])
    
    with open(f"split_{seed}_component_name_label_mapping.txt", 'w') as f:
        f.write('Train: \n' + str(train_component_name) + '\n' + str(train_component_label) + '\n' +
                'Val: \n' + str(val_component_name) + '\n' + str(val_component_label) + '\n' + 
                'Test: \n' + str(test_component_name) +'\n' + str(test_component_label)
               )

    test_set_class = sorted(test_df['class'].value_counts().keys().tolist())   #由於每個component的label都不一樣，透過這個方式取得該component下的所有label
    print('add_test:',add_test)
    if add_test == True:
        # 取得new component的good sample給component classifier訓練
        for name in valComponent:
            good_new_component = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] == 0)].sample(frac=0.5, random_state=123)
            val_df = val_df.drop(good_new_component.index)
            bad_new_component_sample = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] != 0)]
            val_df = val_df.drop(bad_new_component_sample.index)
            train_df = pd.concat([train_df, good_new_component])
        for name in testComponent:
            good_new_component = test_df.loc[(test_df['component_name'] == name) & (test_df['class'] == 0)].sample(frac=0.5, random_state=123)
            test_df = test_df.drop(good_new_component.index)
            train_df = pd.concat([train_df, good_new_component])
    
    val_component_label = val_df['component_name'].value_counts().index.tolist()
    test_component_label = test_df['component_name'].value_counts().index.tolist()

    train_com_df = train_df.copy()
    good_samples = train_com_df.loc[train_df['class']==2]
    train_com_df = good_samples.copy()
    a = Counter(train_com_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_com_df.loc[train_com_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_com_df[train_com_df['component_name']==i].index
            train_com_df=train_com_df.drop(df_idx)
            train_com_df = pd.concat([train_com_df, component])
    
    ### train_defect_df
    train_good_df = train_df.copy()
    train_good_df = train_good_df.loc[train_good_df['class']==2]
    a = Counter(train_good_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_good_df.loc[train_good_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_good_df[train_good_df['component_name']==i].index
            train_good_df=train_good_df.drop(df_idx)
            train_good_df = pd.concat([train_good_df, component])
            
    train_bad_df = train_df.copy()
    train_bad_df = train_bad_df.loc[train_bad_df['class']!=0]
    a = Counter(train_bad_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_bad_df.loc[train_bad_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_bad_df[train_bad_df['component_name']==i].index
            train_bad_df=train_bad_df.drop(df_idx)
            train_bad_df = pd.concat([train_bad_df, component])
            
    train_df = pd.concat([train_good_df, train_bad_df])
    
    ### val_df
    val_good_df = val_df.copy()
    val_good_df = val_good_df.loc[val_good_df['class']==2]
    a = Counter(val_good_df['component_name'])
    for i in range(max(a)):
        if a[i] >5000:
            component = val_good_df.loc[val_good_df['component_name']==i]
            component = component.sample(n=5000,random_state=123,axis=0)
            df_idx = val_good_df[val_good_df['component_name']==i].index
            val_good_df=val_good_df.drop(df_idx)
            val_good_df = pd.concat([val_good_df, component])
            
    val_com_df = val_good_df.copy()
            
    val_bad_df = val_df.copy()
    val_bad_df = val_bad_df.loc[val_bad_df['class']!=0]
    a = Counter(val_bad_df['component_name'])
    for i in range(max(a)):
        if a[i] >5000:
            component = val_bad_df.loc[val_bad_df['component_name']==i]
            component = component.sample(n=5000,random_state=123,axis=0)
            df_idx = val_bad_df[val_bad_df['component_name']==i].index
            val_bad_df=val_bad_df.drop(df_idx)
            val_bad_df = pd.concat([val_bad_df, component])
            
    val_df = pd.concat([val_good_df, val_bad_df])
    
    print("Class distribution in Component Training set:")
    print(train_df['class'].value_counts())
    print("\nClass distribution in Val set:")
    print(val_df['class'].value_counts())
    print("\nClass distribution in Testing set:")
    print(test_df['class'].value_counts())
    print("Num of Images in Component Training set: ", sum(train_df['class'].value_counts().tolist()))
    print("Num of Images in Validation set: ", sum(val_df['class'].value_counts().tolist()))
    print("Num of Images in Testing set: ", sum(test_df['class'].value_counts().tolist()))
    return train_df, val_bad_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df, val_com_df

def CreateDataset_relabel_good(seed , add_test, testing=None):
    # 1:A, 2:B, 3:C, 4:D, 7:F, 8:E
    random.seed(seed)
    print('==> Preparing data..')
    df = pd.read_csv("~/GPCE/Phison/dataset_relabel_ywl.csv")
    
    # 分成6個class
    df.loc[df['class'] == "good", 'class'] = 0
    df.loc[df['class'] == "missing", 'class'] = 1
    df.loc[df['class'] == "shift", 'class'] = 2
    df.loc[df['class'] == "stand", 'class'] = 3
    df.loc[df['class'] == "broke", 'class'] = 4
    df.loc[df['class'] == "short", 'class'] = 5    
    # 移除資料集中的Label Noise   
    unwantedData = pd.read_csv(noisy_label_path, sep=",", header=None)[0].tolist()
    df = df[~df.file_path.isin(unwantedData)]    
    
    df['component_name'] = labelencoder.fit_transform(df['component_name'])
    component_name_counter = Counter(labelencoder.inverse_transform(df['component_name']))
    component_name_list = [key for key, _ in component_name_counter.most_common()]
    component_label_list = df['component_name'].value_counts().index.tolist()
    component_dict = {component_label_list[i]: component_name_list[i] for i in range(len(component_label_list))}
#     import pdb;pdb.set_trace()
    # 將Test set從Training set中移除並重新切割資料集
    trainComponent = df['component_name'].value_counts().index.tolist()

    trainComponent = [1,2,3,5,6,10,11,14,16,17]
    trainDefect = [0,2,4,5]
    
#     trainComponent = [1,6,10,11]
#     trainDefect = [0,2,4,5]
    
    trainComponentDatasetMask = df['component_name'].isin(trainComponent)
    df = df[trainComponentDatasetMask].copy()

    trainDefectDatasetMask = df['class'].isin(trainDefect)
    df = df[trainDefectDatasetMask].copy()
    
#     newComponent = [4,8,9,12,13,14,20,2,3,5,10,11,17]

    df.loc[df['class'] == 0, 'class'] = 0
    df.loc[df['class'] == 1, 'class'] = 1
    df.loc[df['class'] == 2, 'class'] = 1
    df.loc[df['class'] == 3, 'class'] = 1
    df.loc[df['class'] == 4, 'class'] = 1
    df.loc[df['class'] == 5, 'class'] = 1
    
    trainComponent.remove(1)
    trainComponent.remove(6)
    
    valComponent = random.sample(trainComponent, 2)
    for i in valComponent:
        trainComponent.remove(i)
    testComponent = random.sample(trainComponent, 2)
    for i in testComponent:
        trainComponent.remove(i)
    trainComponent.append(1)
    testComponent.append(6)

    trainDatasetMask = df['component_name'].isin(trainComponent)
    train_df = df[trainDatasetMask].copy()
    
    print("Train component label: ")
    train_component_label = train_df['component_name'].value_counts().index.tolist()
    print(train_component_label)
    train_component_name=[]
    print("Train component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in train_component_label:  
            train_component_name.append(v)
    print(train_component_name)
    
    # 將一部分的In-distribution old component分出來給val set和test set (ind_val, ind_test)
    train_df, ind_val, ind_test = split_stratified_into_train_val_test(train_df, stratify_colname='component_name', frac_train=0.8, frac_val=0.1, frac_test=0.1, random_state=seed)

    valDatasetMask = df['component_name'].isin(valComponent)
    val_df = df[valDatasetMask].copy()
    print("Val component label: ")
    val_component_label = val_df['component_name'].value_counts().index.tolist()
    print(val_component_label)
    val_component_name=[]
    print("Val component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in val_component_label:  
            val_component_name.append(v)
    print(val_component_name)
    
    val_df = pd.concat([val_df, ind_val])
    testDatasetMask = df['component_name'].isin(testComponent)
    test_df = df[testDatasetMask].copy()
    print("Test component label: ")
    test_component_label = test_df['component_name'].value_counts().index.tolist()
    print(test_component_label)
    test_component_name=[]
    print("Test component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in test_component_label:  
            test_component_name.append(v)
    print(test_component_name)
    
        
    test_df = pd.concat([test_df, ind_test])
    
    with open(f"split_{seed}_component_name_label_mapping.txt", 'w') as f:
        f.write('Train: \n' + str(train_component_name) + '\n' + str(train_component_label) + '\n' +
                'Val: \n' + str(val_component_name) + '\n' + str(val_component_label) + '\n' + 
                'Test: \n' + str(test_component_name) +'\n' + str(test_component_label)
               )

    test_set_class = sorted(test_df['class'].value_counts().keys().tolist())   #由於每個component的label都不一樣，透過這個方式取得該component下的所有label
    print('add_test:',add_test)
    if add_test == True:
        # 取得new component的good sample給component classifier訓練
        for name in valComponent:
            good_new_component = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] == 0)].sample(frac=0.5, random_state=123)
            val_df = val_df.drop(good_new_component.index)
            bad_new_component_sample = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] != 0)]
            val_df = val_df.drop(bad_new_component_sample.index)
            train_df = pd.concat([train_df, good_new_component])
        for name in testComponent:
            good_new_component = test_df.loc[(test_df['component_name'] == name) & (test_df['class'] == 0)].sample(frac=0.5, random_state=123)
            test_df = test_df.drop(good_new_component.index)
            train_df = pd.concat([train_df, good_new_component])
    
    val_component_label = val_df['component_name'].value_counts().index.tolist()
    test_component_label = test_df['component_name'].value_counts().index.tolist()

    train_com_df = train_df.copy()
    good_samples = train_com_df.loc[train_df['class']==0]
    train_com_df = good_samples.copy()
    a = Counter(train_com_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_com_df.loc[train_com_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_com_df[train_com_df['component_name']==i].index
            train_com_df=train_com_df.drop(df_idx)
            train_com_df = pd.concat([train_com_df, component])
    
    ### train_defect_df
    train_good_df = train_df.copy()
    train_good_df = train_good_df.loc[train_good_df['class']==0]
    a = Counter(train_good_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_good_df.loc[train_good_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_good_df[train_good_df['component_name']==i].index
            train_good_df=train_good_df.drop(df_idx)
            train_good_df = pd.concat([train_good_df, component])
            
    train_bad_df = train_df.copy()
    train_bad_df = train_bad_df.loc[train_bad_df['class']!=0]
    a = Counter(train_bad_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_bad_df.loc[train_bad_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_bad_df[train_bad_df['component_name']==i].index
            train_bad_df=train_bad_df.drop(df_idx)
            train_bad_df = pd.concat([train_bad_df, component])
            
    train_df = pd.concat([train_good_df, train_bad_df])
    
    ### val_df
    val_good_df = val_df.copy()
    val_good_df = val_good_df.loc[val_good_df['class']==0]
    a = Counter(val_good_df['component_name'])
    for i in range(max(a)):
        if a[i] >5000:
            component = val_good_df.loc[val_good_df['component_name']==i]
            component = component.sample(n=5000,random_state=123,axis=0)
            df_idx = val_good_df[val_good_df['component_name']==i].index
            val_good_df=val_good_df.drop(df_idx)
            val_good_df = pd.concat([val_good_df, component])
            
    val_com_df = val_good_df.copy()
            
    val_bad_df = val_df.copy()
    val_bad_df = val_bad_df.loc[val_bad_df['class']!=0]
    a = Counter(val_bad_df['component_name'])
    for i in range(max(a)):
        if a[i] >5000:
            component = val_bad_df.loc[val_bad_df['component_name']==i]
            component = component.sample(n=5000,random_state=123,axis=0)
            df_idx = val_bad_df[val_bad_df['component_name']==i].index
            val_bad_df=val_bad_df.drop(df_idx)
            val_bad_df = pd.concat([val_bad_df, component])
            
    val_df = pd.concat([val_good_df, val_bad_df])
    
    print("Class distribution in Component Training set:")
    print(train_df['class'].value_counts())
    print("\nClass distribution in Val set:")
    print(val_df['class'].value_counts())
    print("\nClass distribution in Testing set:")
    print(test_df['class'].value_counts())
    print("Num of Images in Component Training set: ", sum(train_df['class'].value_counts().tolist()))
    print("Num of Images in Validation set: ", sum(val_df['class'].value_counts().tolist()))
    print("Num of Images in Testing set: ", sum(test_df['class'].value_counts().tolist()))
    return train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df, val_com_df

def CreateDataset_relabel_shift(seed , add_test, testing=None):
    # 1:A, 2:B, 3:C, 4:D, 7:F, 8:E
    random.seed(seed)
    print('==> Preparing data..')
    df = pd.read_csv("~/GPCE/Phison/dataset_relabel_ywl.csv")
    
    # 分成6個class
    df.loc[df['class'] == "good", 'class'] = 0
    df.loc[df['class'] == "missing", 'class'] = 1
    df.loc[df['class'] == "shift", 'class'] = 2
    df.loc[df['class'] == "stand", 'class'] = 3
    df.loc[df['class'] == "broke", 'class'] = 4
    df.loc[df['class'] == "short", 'class'] = 5    
    # 移除資料集中的Label Noise   
    unwantedData = pd.read_csv(noisy_label_path, sep=",", header=None)[0].tolist()
    df = df[~df.file_path.isin(unwantedData)]    
    
    df['component_name'] = labelencoder.fit_transform(df['component_name'])
    component_name_counter = Counter(labelencoder.inverse_transform(df['component_name']))
    component_name_list = [key for key, _ in component_name_counter.most_common()]
    component_label_list = df['component_name'].value_counts().index.tolist()
    component_dict = {component_label_list[i]: component_name_list[i] for i in range(len(component_label_list))}
#     import pdb;pdb.set_trace()
    # 將Test set從Training set中移除並重新切割資料集
    trainComponent = df['component_name'].value_counts().index.tolist()

    trainComponent = [1,2,3,5,6,10,11,14,16,17]
    trainDefect = [0,2,4,5]
    
#     trainComponent = [1,6,10,11]
#     trainDefect = [0,2,4,5]
    
    trainComponentDatasetMask = df['component_name'].isin(trainComponent)
    df = df[trainComponentDatasetMask].copy()

    trainDefectDatasetMask = df['class'].isin(trainDefect)
    df = df[trainDefectDatasetMask].copy()
    
#     newComponent = [4,8,9,12,13,14,20,2,3,5,10,11,17]

    df.loc[df['class'] == 0, 'class'] = 0
    df.loc[df['class'] == 1, 'class'] = 0
    df.loc[df['class'] == 2, 'class'] = 1
    df.loc[df['class'] == 3, 'class'] = 0
    df.loc[df['class'] == 4, 'class'] = 0
    df.loc[df['class'] == 5, 'class'] = 0
    
    trainComponent.remove(1)
    trainComponent.remove(6)
    
    valComponent = random.sample(trainComponent, 2)
    for i in valComponent:
        trainComponent.remove(i)
    testComponent = random.sample(trainComponent, 2)
    for i in testComponent:
        trainComponent.remove(i)
    trainComponent.append(1)
    testComponent.append(6)

    trainDatasetMask = df['component_name'].isin(trainComponent)
    train_df = df[trainDatasetMask].copy()
    
    print("Train component label: ")
    train_component_label = train_df['component_name'].value_counts().index.tolist()
    print(train_component_label)
    train_component_name=[]
    print("Train component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in train_component_label:  
            train_component_name.append(v)
    print(train_component_name)
    
    # 將一部分的In-distribution old component分出來給val set和test set (ind_val, ind_test)
    train_df, ind_val, ind_test = split_stratified_into_train_val_test(train_df, stratify_colname='component_name', frac_train=0.8, frac_val=0.1, frac_test=0.1, random_state=seed)

    valDatasetMask = df['component_name'].isin(valComponent)
    val_df = df[valDatasetMask].copy()
    print("Val component label: ")
    val_component_label = val_df['component_name'].value_counts().index.tolist()
    print(val_component_label)
    val_component_name=[]
    print("Val component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in val_component_label:  
            val_component_name.append(v)
    print(val_component_name)
    
    val_df = pd.concat([val_df, ind_val])
    testDatasetMask = df['component_name'].isin(testComponent)
    test_df = df[testDatasetMask].copy()
    print("Test component label: ")
    test_component_label = test_df['component_name'].value_counts().index.tolist()
    print(test_component_label)
    test_component_name=[]
    print("Test component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in test_component_label:  
            test_component_name.append(v)
    print(test_component_name)
    
        
    test_df = pd.concat([test_df, ind_test])
    
    with open(f"split_{seed}_component_name_label_mapping.txt", 'w') as f:
        f.write('Train: \n' + str(train_component_name) + '\n' + str(train_component_label) + '\n' +
                'Val: \n' + str(val_component_name) + '\n' + str(val_component_label) + '\n' + 
                'Test: \n' + str(test_component_name) +'\n' + str(test_component_label)
               )

    test_set_class = sorted(test_df['class'].value_counts().keys().tolist())   #由於每個component的label都不一樣，透過這個方式取得該component下的所有label
    print('add_test:',add_test)
    if add_test == True:
        # 取得new component的good sample給component classifier訓練
        for name in valComponent:
            good_new_component = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] == 0)].sample(frac=0.5, random_state=123)
            val_df = val_df.drop(good_new_component.index)
            bad_new_component_sample = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] != 0)]
            val_df = val_df.drop(bad_new_component_sample.index)
            train_df = pd.concat([train_df, good_new_component])
        for name in testComponent:
            good_new_component = test_df.loc[(test_df['component_name'] == name) & (test_df['class'] == 0)].sample(frac=0.5, random_state=123)
            test_df = test_df.drop(good_new_component.index)
            train_df = pd.concat([train_df, good_new_component])
    
    val_component_label = val_df['component_name'].value_counts().index.tolist()
    test_component_label = test_df['component_name'].value_counts().index.tolist()

    train_com_df = train_df.copy()
    good_samples = train_com_df.loc[train_df['class']==0]
    train_com_df = good_samples.copy()
    a = Counter(train_com_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_com_df.loc[train_com_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_com_df[train_com_df['component_name']==i].index
            train_com_df=train_com_df.drop(df_idx)
            train_com_df = pd.concat([train_com_df, component])
    
    ### train_defect_df
    train_good_df = train_df.copy()
    train_good_df = train_good_df.loc[train_good_df['class']==0]
    a = Counter(train_good_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_good_df.loc[train_good_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_good_df[train_good_df['component_name']==i].index
            train_good_df=train_good_df.drop(df_idx)
            train_good_df = pd.concat([train_good_df, component])
            
    train_bad_df = train_df.copy()
    train_bad_df = train_bad_df.loc[train_bad_df['class']!=0]
    a = Counter(train_bad_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_bad_df.loc[train_bad_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_bad_df[train_bad_df['component_name']==i].index
            train_bad_df=train_bad_df.drop(df_idx)
            train_bad_df = pd.concat([train_bad_df, component])
            
    train_df = pd.concat([train_good_df, train_bad_df])
    
    ### val_df
    val_good_df = val_df.copy()
    val_good_df = val_good_df.loc[val_good_df['class']==0]
    a = Counter(val_good_df['component_name'])
    for i in range(max(a)):
        if a[i] >5000:
            component = val_good_df.loc[val_good_df['component_name']==i]
            component = component.sample(n=5000,random_state=123,axis=0)
            df_idx = val_good_df[val_good_df['component_name']==i].index
            val_good_df=val_good_df.drop(df_idx)
            val_good_df = pd.concat([val_good_df, component])
            
    val_com_df = val_good_df.copy()
            
    val_bad_df = val_df.copy()
    val_bad_df = val_bad_df.loc[val_bad_df['class']!=0]
    a = Counter(val_bad_df['component_name'])
    for i in range(max(a)):
        if a[i] >5000:
            component = val_bad_df.loc[val_bad_df['component_name']==i]
            component = component.sample(n=5000,random_state=123,axis=0)
            df_idx = val_bad_df[val_bad_df['component_name']==i].index
            val_bad_df=val_bad_df.drop(df_idx)
            val_bad_df = pd.concat([val_bad_df, component])
            
    val_df = pd.concat([val_good_df, val_bad_df])
    
    print("Class distribution in Component Training set:")
    print(train_df['class'].value_counts())
    print("\nClass distribution in Val set:")
    print(val_df['class'].value_counts())
    print("\nClass distribution in Testing set:")
    print(test_df['class'].value_counts())
    print("Num of Images in Component Training set: ", sum(train_df['class'].value_counts().tolist()))
    print("Num of Images in Validation set: ", sum(val_df['class'].value_counts().tolist()))
    print("Num of Images in Testing set: ", sum(test_df['class'].value_counts().tolist()))
    return train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df, val_com_df

def CreateDataset_relabel_broke(seed , add_test, testing=None):
    # 1:A, 2:B, 3:C, 4:D, 7:F, 8:E
    random.seed(seed)
    print('==> Preparing data..')
    df = pd.read_csv("~/GPCE/Phison/dataset_relabel_ywl.csv")
    
    # 分成6個class
    df.loc[df['class'] == "good", 'class'] = 0
    df.loc[df['class'] == "missing", 'class'] = 1
    df.loc[df['class'] == "shift", 'class'] = 2
    df.loc[df['class'] == "stand", 'class'] = 3
    df.loc[df['class'] == "broke", 'class'] = 4
    df.loc[df['class'] == "short", 'class'] = 5    
    # 移除資料集中的Label Noise   
    unwantedData = pd.read_csv(noisy_label_path, sep=",", header=None)[0].tolist()
    df = df[~df.file_path.isin(unwantedData)]    
    
    df['component_name'] = labelencoder.fit_transform(df['component_name'])
    component_name_counter = Counter(labelencoder.inverse_transform(df['component_name']))
    component_name_list = [key for key, _ in component_name_counter.most_common()]
    component_label_list = df['component_name'].value_counts().index.tolist()
    component_dict = {component_label_list[i]: component_name_list[i] for i in range(len(component_label_list))}
#     import pdb;pdb.set_trace()
    # 將Test set從Training set中移除並重新切割資料集
    trainComponent = df['component_name'].value_counts().index.tolist()

    trainComponent = [1,2,3,5,6,10,11,14,16,17]
    trainDefect = [0,2,4,5]
    
#     trainComponent = [1,6,10,11]
#     trainDefect = [0,2,4,5]
    
    trainComponentDatasetMask = df['component_name'].isin(trainComponent)
    df = df[trainComponentDatasetMask].copy()

    trainDefectDatasetMask = df['class'].isin(trainDefect)
    df = df[trainDefectDatasetMask].copy()

    
#     newComponent = [4,8,9,12,13,14,20,2,3,5,10,11,17]

    df.loc[df['class'] == 0, 'class'] = 0
    df.loc[df['class'] == 1, 'class'] = 0
    df.loc[df['class'] == 2, 'class'] = 0
    df.loc[df['class'] == 3, 'class'] = 0
    df.loc[df['class'] == 4, 'class'] = 1
    df.loc[df['class'] == 5, 'class'] = 0
    
    trainComponent.remove(1)
    trainComponent.remove(6)
    
    valComponent = random.sample(trainComponent, 2)
    for i in valComponent:
        trainComponent.remove(i)
    testComponent = random.sample(trainComponent, 2)
    for i in testComponent:
        trainComponent.remove(i)
    trainComponent.append(1)
    testComponent.append(6)
    
    
    trainDatasetMask = df['component_name'].isin(trainComponent)
    train_df = df[trainDatasetMask].copy()
    
    print("Train component label: ")
    train_component_label = train_df['component_name'].value_counts().index.tolist()
    print(train_component_label)
    train_component_name=[]
    print("Train component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in train_component_label:  
            train_component_name.append(v)
    print(train_component_name)
    
    # 將一部分的In-distribution old component分出來給val set和test set (ind_val, ind_test)
    train_df, ind_val, ind_test = split_stratified_into_train_val_test(train_df, stratify_colname='component_name', frac_train=0.8, frac_val=0.1, frac_test=0.1, random_state=seed)

    valDatasetMask = df['component_name'].isin(valComponent)
    val_df = df[valDatasetMask].copy()
    print("Val component label: ")
    val_component_label = val_df['component_name'].value_counts().index.tolist()
    print(val_component_label)
    val_component_name=[]
    print("Val component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in val_component_label:  
            val_component_name.append(v)
    print(val_component_name)
    
    val_df = pd.concat([val_df, ind_val])
    testDatasetMask = df['component_name'].isin(testComponent)
    test_df = df[testDatasetMask].copy()
    print("Test component label: ")
    test_component_label = test_df['component_name'].value_counts().index.tolist()
    print(test_component_label)
    test_component_name=[]
    print("Test component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in test_component_label:  
            test_component_name.append(v)
    print(test_component_name)
    
        
    test_df = pd.concat([test_df, ind_test])
    
    with open(f"split_{seed}_component_name_label_mapping.txt", 'w') as f:
        f.write('Train: \n' + str(train_component_name) + '\n' + str(train_component_label) + '\n' +
                'Val: \n' + str(val_component_name) + '\n' + str(val_component_label) + '\n' + 
                'Test: \n' + str(test_component_name) +'\n' + str(test_component_label)
               )

    test_set_class = sorted(test_df['class'].value_counts().keys().tolist())   #由於每個component的label都不一樣，透過這個方式取得該component下的所有label
    print('add_test:',add_test)
    if add_test == True:
        # 取得new component的good sample給component classifier訓練
        for name in valComponent:
            good_new_component = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] == 0)].sample(frac=0.5, random_state=123)
            val_df = val_df.drop(good_new_component.index)
            bad_new_component_sample = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] != 0)]
            val_df = val_df.drop(bad_new_component_sample.index)
            train_df = pd.concat([train_df, good_new_component])
        for name in testComponent:
            good_new_component = test_df.loc[(test_df['component_name'] == name) & (test_df['class'] == 0)].sample(frac=0.5, random_state=123)
            test_df = test_df.drop(good_new_component.index)
            train_df = pd.concat([train_df, good_new_component])
    
    val_component_label = val_df['component_name'].value_counts().index.tolist()
    test_component_label = test_df['component_name'].value_counts().index.tolist()

    train_com_df = train_df.copy()
    good_samples = train_com_df.loc[train_df['class']==0]
    train_com_df = good_samples.copy()
    a = Counter(train_com_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_com_df.loc[train_com_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_com_df[train_com_df['component_name']==i].index
            train_com_df=train_com_df.drop(df_idx)
            train_com_df = pd.concat([train_com_df, component])
    
    ### train_defect_df
    train_good_df = train_df.copy()
    train_good_df = train_good_df.loc[train_good_df['class']==0]
    a = Counter(train_good_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_good_df.loc[train_good_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_good_df[train_good_df['component_name']==i].index
            train_good_df=train_good_df.drop(df_idx)
            train_good_df = pd.concat([train_good_df, component])
            
    train_bad_df = train_df.copy()
    train_bad_df = train_bad_df.loc[train_bad_df['class']!=0]
    a = Counter(train_bad_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_bad_df.loc[train_bad_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_bad_df[train_bad_df['component_name']==i].index
            train_bad_df=train_bad_df.drop(df_idx)
            train_bad_df = pd.concat([train_bad_df, component])
            
    train_df = pd.concat([train_good_df, train_bad_df])
    
    ### val_df
    val_good_df = val_df.copy()
    val_good_df = val_good_df.loc[val_good_df['class']==0]
    a = Counter(val_good_df['component_name'])
    for i in range(max(a)):
        if a[i] >5000:
            component = val_good_df.loc[val_good_df['component_name']==i]
            component = component.sample(n=5000,random_state=123,axis=0)
            df_idx = val_good_df[val_good_df['component_name']==i].index
            val_good_df=val_good_df.drop(df_idx)
            val_good_df = pd.concat([val_good_df, component])
            
    val_com_df = val_good_df.copy()
            
    val_bad_df = val_df.copy()
    val_bad_df = val_bad_df.loc[val_bad_df['class']!=0]
    a = Counter(val_bad_df['component_name'])
    for i in range(max(a)):
        if a[i] >5000:
            component = val_bad_df.loc[val_bad_df['component_name']==i]
            component = component.sample(n=5000,random_state=123,axis=0)
            df_idx = val_bad_df[val_bad_df['component_name']==i].index
            val_bad_df=val_bad_df.drop(df_idx)
            val_bad_df = pd.concat([val_bad_df, component])
            
    val_df = pd.concat([val_good_df, val_bad_df])
    
    print("Class distribution in Component Training set:")
    print(train_df['class'].value_counts())
    print("\nClass distribution in Val set:")
    print(val_df['class'].value_counts())
    print("\nClass distribution in Testing set:")
    print(test_df['class'].value_counts())
    print("Num of Images in Component Training set: ", sum(train_df['class'].value_counts().tolist()))
    print("Num of Images in Validation set: ", sum(val_df['class'].value_counts().tolist()))
    print("Num of Images in Testing set: ", sum(test_df['class'].value_counts().tolist()))
    return train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df, val_com_df

def CreateDataset_relabel_short(seed , add_test, testing=None):
    # 1:A, 2:B, 3:C, 4:D, 7:F, 8:E
    random.seed(seed)
    print('==> Preparing data..')
    df = pd.read_csv("~/Phison/dataset_relabel_ywl.csv")
    
    # 分成6個class
    df.loc[df['class'] == "good", 'class'] = 0
    df.loc[df['class'] == "missing", 'class'] = 1
    df.loc[df['class'] == "shift", 'class'] = 2
    df.loc[df['class'] == "stand", 'class'] = 3
    df.loc[df['class'] == "broke", 'class'] = 4
    df.loc[df['class'] == "short", 'class'] = 5    
    # 移除資料集中的Label Noise   
    unwantedData = pd.read_csv(noisy_label_path, sep=",", header=None)[0].tolist()
    df = df[~df.file_path.isin(unwantedData)]    
    
    df['component_name'] = labelencoder.fit_transform(df['component_name'])
    component_name_counter = Counter(labelencoder.inverse_transform(df['component_name']))
    component_name_list = [key for key, _ in component_name_counter.most_common()]
    component_label_list = df['component_name'].value_counts().index.tolist()
    component_dict = {component_label_list[i]: component_name_list[i] for i in range(len(component_label_list))}
#     import pdb;pdb.set_trace()
    # 將Test set從Training set中移除並重新切割資料集
    trainComponent = df['component_name'].value_counts().index.tolist()

    trainComponent = [1,2,3,5,6,10,11,14,16,17]
    trainDefect = [0,2,4,5]
    
#     trainComponent = [1,6,10,11]
#     trainDefect = [0,2,4,5]
    
    trainComponentDatasetMask = df['component_name'].isin(trainComponent)
    df = df[trainComponentDatasetMask].copy()

    trainDefectDatasetMask = df['class'].isin(trainDefect)
    df = df[trainDefectDatasetMask].copy()
    
#     newComponent = [4,8,9,12,13,14,20,2,3,5,10,11,17]

    df.loc[df['class'] == 0, 'class'] = 0
    df.loc[df['class'] == 1, 'class'] = 0
    df.loc[df['class'] == 2, 'class'] = 0
    df.loc[df['class'] == 3, 'class'] = 0
    df.loc[df['class'] == 4, 'class'] = 0
    df.loc[df['class'] == 5, 'class'] = 1
    
    trainComponent.remove(1)
    trainComponent.remove(6)
    
    valComponent = random.sample(trainComponent, 2)
    for i in valComponent:
        trainComponent.remove(i)
    testComponent = random.sample(trainComponent, 2)
    for i in testComponent:
        trainComponent.remove(i)
    trainComponent.append(1)
    testComponent.append(6)
    
    trainDatasetMask = df['component_name'].isin(trainComponent)
    train_df = df[trainDatasetMask].copy()
    
    print("Train component label: ")
    train_component_label = train_df['component_name'].value_counts().index.tolist()
    print(train_component_label)
    train_component_name=[]
    print("Train component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in train_component_label:  
            train_component_name.append(v)
    print(train_component_name)
    
    # 將一部分的In-distribution old component分出來給val set和test set (ind_val, ind_test)
    train_df, ind_val, ind_test = split_stratified_into_train_val_test(train_df, stratify_colname='component_name', frac_train=0.8, frac_val=0.1, frac_test=0.1, random_state=seed)

    valDatasetMask = df['component_name'].isin(valComponent)
    val_df = df[valDatasetMask].copy()
    print("Val component label: ")
    val_component_label = val_df['component_name'].value_counts().index.tolist()
    print(val_component_label)
    val_component_name=[]
    print("Val component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in val_component_label:  
            val_component_name.append(v)
    print(val_component_name)
    
    val_df = pd.concat([val_df, ind_val])
    testDatasetMask = df['component_name'].isin(testComponent)
    test_df = df[testDatasetMask].copy()
    print("Test component label: ")
    test_component_label = test_df['component_name'].value_counts().index.tolist()
    print(test_component_label)
    test_component_name=[]
    print("Test component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in test_component_label:  
            test_component_name.append(v)
    print(test_component_name)
    
        
    test_df = pd.concat([test_df, ind_test])

    
    
    with open(f"split_{seed}_component_name_label_mapping.txt", 'w') as f:
        f.write('Train: \n' + str(train_component_name) + '\n' + str(train_component_label) + '\n' +
                'Val: \n' + str(val_component_name) + '\n' + str(val_component_label) + '\n' + 
                'Test: \n' + str(test_component_name) +'\n' + str(test_component_label)
               )

    test_set_class = sorted(test_df['class'].value_counts().keys().tolist())   #由於每個component的label都不一樣，透過這個方式取得該component下的所有label
    print('add_test:',add_test)
    if add_test == True:
        # 取得new component的good sample給component classifier訓練
        for name in valComponent:
            good_new_component = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] == 0)].sample(frac=0.5, random_state=123)
            val_df = val_df.drop(good_new_component.index)
            bad_new_component_sample = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] != 0)]
            val_df = val_df.drop(bad_new_component_sample.index)
            train_df = pd.concat([train_df, good_new_component])
        for name in testComponent:
            good_new_component = test_df.loc[(test_df['component_name'] == name) & (test_df['class'] == 0)].sample(frac=0.5, random_state=123)
            test_df = test_df.drop(good_new_component.index)
            train_df = pd.concat([train_df, good_new_component])
    
    val_component_label = val_df['component_name'].value_counts().index.tolist()
    test_component_label = test_df['component_name'].value_counts().index.tolist()

    train_com_df = train_df.copy()
    good_samples = train_com_df.loc[train_df['class']==0]
    train_com_df = good_samples.copy()
    a = Counter(train_com_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_com_df.loc[train_com_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_com_df[train_com_df['component_name']==i].index
            train_com_df=train_com_df.drop(df_idx)
            train_com_df = pd.concat([train_com_df, component])
    
    ### train_defect_df
    train_good_df = train_df.copy()
    train_good_df = train_good_df.loc[train_good_df['class']==0]
    a = Counter(train_good_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_good_df.loc[train_good_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_good_df[train_good_df['component_name']==i].index
            train_good_df=train_good_df.drop(df_idx)
            train_good_df = pd.concat([train_good_df, component])
            
    train_bad_df = train_df.copy()
    train_bad_df = train_bad_df.loc[train_bad_df['class']!=0]
    a = Counter(train_bad_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_bad_df.loc[train_bad_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_bad_df[train_bad_df['component_name']==i].index
            train_bad_df=train_bad_df.drop(df_idx)
            train_bad_df = pd.concat([train_bad_df, component])
            
    train_df = pd.concat([train_good_df, train_bad_df])
    
    ### val_df
    val_good_df = val_df.copy()
    val_good_df = val_good_df.loc[val_good_df['class']==0]
    a = Counter(val_good_df['component_name'])
    for i in range(max(a)):
        if a[i] >5000:
            component = val_good_df.loc[val_good_df['component_name']==i]
            component = component.sample(n=5000,random_state=123,axis=0)
            df_idx = val_good_df[val_good_df['component_name']==i].index
            val_good_df=val_good_df.drop(df_idx)
            val_good_df = pd.concat([val_good_df, component])
            
    val_com_df = val_good_df.copy()
            
    val_bad_df = val_df.copy()
    val_bad_df = val_bad_df.loc[val_bad_df['class']!=0]
    a = Counter(val_bad_df['component_name'])
    for i in range(max(a)):
        if a[i] >5000:
            component = val_bad_df.loc[val_bad_df['component_name']==i]
            component = component.sample(n=5000,random_state=123,axis=0)
            df_idx = val_bad_df[val_bad_df['component_name']==i].index
            val_bad_df=val_bad_df.drop(df_idx)
            val_bad_df = pd.concat([val_bad_df, component])
            
    val_df = pd.concat([val_good_df, val_bad_df])
    
    print("Class distribution in Component Training set:")
    print(train_df['class'].value_counts())
    print("\nClass distribution in Val set:")
    print(val_df['class'].value_counts())
    print("\nClass distribution in Testing set:")
    print(test_df['class'].value_counts())
    print("Num of Images in Component Training set: ", sum(train_df['class'].value_counts().tolist()))
    print("Num of Images in Validation set: ", sum(val_df['class'].value_counts().tolist()))
    print("Num of Images in Testing set: ", sum(test_df['class'].value_counts().tolist()))
    return train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df, val_com_df



def get_SVHN(root):
    input_size = 32
    num_classes = 10

    # NOTE: these are not correct mean and std for SVHN, but are commonly used
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    train_dataset = datasets.SVHN(
        root + "/SVHN", split="train", transform=transform, download=True
    )
    test_dataset = datasets.SVHN(
        root + "/SVHN", split="test", transform=transform, download=True
    )
    return input_size, num_classes, train_dataset, test_dataset


def get_CIFAR10(root):
    # https://github.com/talreiss/Mean-Shifted-Anomaly-Detection/blob/main/utils.py#L110-L132
    input_size = 32
    num_classes = 10

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    # Alternative
    # normalize = transforms.Normalize(
    #     (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    # )

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_dataset = datasets.CIFAR10(
        root + "/CIFAR10", train=True, transform=train_transform, download=True
    )

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    test_dataset = datasets.CIFAR10(
        root + "/CIFAR10", train=False, transform=test_transform, download=False
    )

    return input_size, num_classes, train_dataset, test_dataset


def get_CIFAR100(root):
    input_size = 32
    num_classes = 100
    normalize = transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762))

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    train_dataset = datasets.CIFAR100(
        root + "/CIFAR100", train=True, transform=train_transform, download=True
    )

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    test_dataset = datasets.CIFAR100(
        root + "/CIFAR100", train=False, transform=test_transform, download=False
    )

    return input_size, num_classes, train_dataset, test_dataset

def get_PHISON_df(root, seed):
    input_size = 224
    num_classes = 23
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    train_transform = transforms.Compose(
        [
            transforms.Resize([input_size, input_size]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    
    train_cls_transform = transforms.Compose(
        [
            transforms.Resize([input_size, input_size]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )
    
    
    test_transform = transforms.Compose([
        transforms.Resize([input_size, input_size]),
        transforms.ToTensor(),  
        normalize])
    
    add_test = True
    
    train_com_df, train_regroup_df, df = CreateDataset_regroup_new(seed ,add_test)

#     train_cls_df = train_cls_df.sample(frac=0.01, random_state=123)
#     train_com_df = train_com_df.sample(frac=0.01, random_state=123)
#     val_df = val_df.sample(frac=0.01, random_state=123)


#     train_com_df = train_com_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.5, random_state=123))

    train_com_dataset = CustomDataset(train_com_df, transform=train_transform)  
    train_regroup_dataset = CustomDataset(train_regroup_df, transform=train_transform)  
    
    per_component_num = 128 // len(train_com_dataset.dataframe['component_name'].value_counts().index)
    train_com_sampler = samplers.MPerClassSampler(train_com_dataset.dataframe['component_name'], per_component_num, length_before_new_iter=len(train_com_dataset))


    train_com_loader = torch.utils.data.DataLoader(
            train_com_dataset, batch_size=128, shuffle=(train_com_sampler is None),
            num_workers=8, pin_memory=True,drop_last=True, sampler=train_com_sampler)
    
    train_regroup_loader = torch.utils.data.DataLoader(
            train_regroup_dataset, batch_size=128, shuffle=False,
            num_workers=8, pin_memory=True,drop_last=True)

    return train_com_loader, train_regroup_df ,train_regroup_loader ,df

def get_label(seed,clust ,new_group , new_group_component_name):
        
    # Load model
#     cl = Clustimage(method='pca')
#     cl.load(f'clust/{seed}_pretrain_all_clustimage_model')
    label_0=[]
    
    cn = new_group_component_name
#     cn = train_regroup_df['component_name'].tolist()
    Counter_cn = Counter(cn)
    
    result = np.where(clust.results['labels'] == new_group)
    result = result[0].tolist()
    label0=[]
    for i in result:
        label0.append(cn[i])
    counter_label = Counter(label0)
    
    for i in range(len(set(Counter_cn))):
        if counter_label[i]>= (Counter_cn[i]/2):
            label_0.append(i)
        
    return label_0

def get_PHISON_regroup(root, seed):
    input_size = 224
#     num_classes = 23
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    train_transform = transforms.Compose(
        [
            transforms.Resize([input_size, input_size]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    
    train_cls_transform = transforms.Compose(
        [
            transforms.Resize([input_size, input_size]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )
    
    
    test_transform = transforms.Compose([
        transforms.Resize([input_size, input_size]),
        transforms.ToTensor(),  
        normalize])
    
    add_test = True
    
    train_cls_regroup_df, val_regroup_df, test_regroup_df, train_component_label, val_component_label, test_component_label, train_com_regroup_df = CreateDataset_regroup_due(seed ,add_test)
    
    num_classes = len(set(train_com_regroup_df['component_name']))

#     train_cls_regroup_df = train_cls_regroup_df.sample(frac=0.01, random_state=123)
#     train_com_regroup_df = train_com_regroup_df.sample(frac=0.01, random_state=123)
#     val_regroup_df = val_regroup_df.sample(frac=0.01, random_state=123)
#     train_com_df = train_com_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.5, random_state=123))

    train_cls_dataset = CustomDataset(train_cls_regroup_df, transform=train_cls_transform)    
    train_com_dataset = CustomDataset(train_com_regroup_df, transform=train_transform)  
    
    test_dataset = CustomDataset(val_regroup_df, transform=test_transform)
    
    per_component_num = 128 // len(train_com_dataset.dataframe['component_name'].value_counts().index)
    per_class_num = 128 // len(train_cls_dataset.dataframe['class'].value_counts().index)
    train_com_sampler = samplers.MPerClassSampler(train_com_dataset.dataframe['component_name'], per_component_num, length_before_new_iter=len(train_com_dataset))
    train_cls_sampler = samplers.MPerClassSampler(train_cls_dataset.dataframe['class'], per_class_num, length_before_new_iter=len(train_cls_dataset))

    train_com_loader = torch.utils.data.DataLoader(
            train_com_dataset, batch_size=128, shuffle=(train_com_sampler is None),
            num_workers=8, pin_memory=True,drop_last=True, sampler=train_com_sampler)

    train_cls_loader = torch.utils.data.DataLoader(
            train_cls_dataset, batch_size=128, shuffle=(train_cls_sampler is None),
            num_workers=8, pin_memory=True,drop_last=True, sampler=train_cls_sampler)



    return input_size ,num_classes ,train_com_loader, train_cls_loader, test_dataset ,train_cls_dataset,train_com_dataset
def get_PHISON_regroup_2(root, seed):
    input_size = 224
#     num_classes = 23
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    train_transform = transforms.Compose(
        [
            transforms.Resize([input_size, input_size]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    
    train_cls_transform = transforms.Compose(
        [
            transforms.Resize([input_size, input_size]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )
    
    
    test_transform = transforms.Compose([
        transforms.Resize([input_size, input_size]),
        transforms.ToTensor(),  
        normalize])
    
    add_test = True
    
    train_cls_regroup_df, val_regroup_df, test_regroup_df, train_component_label, val_component_label, test_component_label, train_com_regroup_df = CreateDataset_regroup_due_2(seed ,add_test)
    
    num_classes = len(set(train_com_regroup_df['component_name']))

#     train_cls_regroup_df = train_cls_regroup_df.sample(frac=0.01, random_state=123)
#     train_com_regroup_df = train_com_regroup_df.sample(frac=0.01, random_state=123)
#     val_regroup_df = val_regroup_df.sample(frac=0.01, random_state=123)
#     train_com_df = train_com_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.5, random_state=123))

    train_cls_dataset = CustomDataset(train_cls_regroup_df, transform=train_cls_transform)    
    train_com_dataset = CustomDataset(train_com_regroup_df, transform=train_transform)  
    
    test_dataset = CustomDataset(val_regroup_df, transform=test_transform)
#     test_com_dataset = CustomDataset(val_com_df, transform=test_transform)
    
    per_component_num = 128 // len(train_com_dataset.dataframe['component_name'].value_counts().index)
    per_class_num = 128 // len(train_cls_dataset.dataframe['class'].value_counts().index)
    train_com_sampler = samplers.MPerClassSampler(train_com_dataset.dataframe['component_name'], per_component_num, length_before_new_iter=len(train_com_dataset))
    train_cls_sampler = samplers.MPerClassSampler(train_cls_dataset.dataframe['class'], per_class_num, length_before_new_iter=len(train_cls_dataset))

    train_com_loader = torch.utils.data.DataLoader(
            train_com_dataset, batch_size=128, shuffle=(train_com_sampler is None),
            num_workers=8, pin_memory=True,drop_last=True, sampler=train_com_sampler)

    train_cls_loader = torch.utils.data.DataLoader(
            train_cls_dataset, batch_size=128, shuffle=(train_cls_sampler is None),
            num_workers=8, pin_memory=True,drop_last=True, sampler=train_cls_sampler)



    return input_size ,num_classes ,train_com_loader, train_cls_loader, test_dataset ,train_cls_dataset,train_com_dataset 
def get_PHISON_regroup_3(root, seed):
    input_size = 224
#     num_classes = 23
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    train_transform = transforms.Compose(
        [
            transforms.Resize([input_size, input_size]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    
    train_cls_transform = transforms.Compose(
        [
            transforms.Resize([input_size, input_size]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )
    
    
    test_transform = transforms.Compose([
        transforms.Resize([input_size, input_size]),
        transforms.ToTensor(),  
        normalize])
    
    add_test = True
    
    train_cls_regroup_df, val_regroup_df, test_regroup_df, train_component_label, val_component_label, test_component_label, train_com_regroup_df, val_com_df = CreateDataset_regroup_due_2_seed1212(seed ,add_test)
    
#     train_cls_regroup_df, val_regroup_df, test_regroup_df, train_component_label, val_component_label, test_component_label, train_com_regroup_df, val_com_df = CreateDataset_regroup_due_2_sixcls(seed ,add_test)
    
    num_classes = len(set(train_com_regroup_df['component_name']))

#     train_cls_regroup_df = train_cls_regroup_df.sample(frac=0.01, random_state=123)
#     train_com_regroup_df = train_com_regroup_df.sample(frac=0.01, random_state=123)
#     val_regroup_df = val_regroup_df.sample(frac=0.01, random_state=123)
#     train_com_df = train_com_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.5, random_state=123))

    train_cls_dataset = CustomDataset(train_cls_regroup_df, transform=train_cls_transform)    
    train_com_dataset = CustomDataset(train_com_regroup_df, transform=train_transform)  
    
    test_dataset = CustomDataset(val_regroup_df, transform=test_transform)
    test_com_dataset = CustomDataset(val_com_df, transform=test_transform)
    
    per_component_num = 64 // len(train_com_dataset.dataframe['component_name'].value_counts().index)
    per_class_num = 64 // len(train_cls_dataset.dataframe['class'].value_counts().index)
    train_com_sampler = samplers.MPerClassSampler(train_com_dataset.dataframe['component_name'], per_component_num, length_before_new_iter=len(train_com_dataset))
    train_cls_sampler = samplers.MPerClassSampler(train_cls_dataset.dataframe['class'], per_class_num, length_before_new_iter=len(train_cls_dataset))

    train_com_loader = torch.utils.data.DataLoader(
            train_com_dataset, batch_size=64, shuffle=(train_com_sampler is None),
            num_workers=4, pin_memory=True,drop_last=True, sampler=train_com_sampler)

    train_cls_loader = torch.utils.data.DataLoader(
            train_cls_dataset, batch_size=64, shuffle=(train_cls_sampler is None),
            num_workers=4, pin_memory=True,drop_last=True, sampler=train_cls_sampler)



    return input_size ,num_classes ,train_com_loader, train_cls_loader, test_dataset ,train_cls_dataset,train_com_dataset, test_com_dataset
def get_PHISON_regroup_4(root, seed):
    input_size = 224
#     num_classes = 23
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    train_transform = transforms.Compose(
        [
            transforms.Resize([input_size, input_size]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    
    train_cls_transform = transforms.Compose(
        [
            transforms.Resize([input_size, input_size]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )
    
    
    test_transform = transforms.Compose([
        transforms.Resize([input_size, input_size]),
        transforms.ToTensor(),  
        normalize])
    
    add_test = True
    
    train_cls_regroup_df, val_regroup_df, test_regroup_df, train_component_label, val_component_label, test_component_label, train_com_regroup_df = CreateDataset_regroup_due_2_seed1(seed ,add_test)
    
    num_classes = len(set(train_com_regroup_df['component_name']))

#     train_cls_regroup_df = train_cls_regroup_df.sample(frac=0.01, random_state=123)
#     train_com_regroup_df = train_com_regroup_df.sample(frac=0.01, random_state=123)
#     val_regroup_df = val_regroup_df.sample(frac=0.01, random_state=123)
#     train_com_df = train_com_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.5, random_state=123))

    train_cls_dataset = CustomDataset(train_cls_regroup_df, transform=train_cls_transform)    
    train_com_dataset = CustomDataset(train_com_regroup_df, transform=train_transform)  
    
    test_dataset = CustomDataset(val_regroup_df, transform=test_transform)
#     test_com_dataset = CustomDataset(val_com_df, transform=test_transform)
    
    per_component_num = 128 // len(train_com_dataset.dataframe['component_name'].value_counts().index)
    per_class_num = 128 // len(train_cls_dataset.dataframe['class'].value_counts().index)
    train_com_sampler = samplers.MPerClassSampler(train_com_dataset.dataframe['component_name'], per_component_num, length_before_new_iter=len(train_com_dataset))
    train_cls_sampler = samplers.MPerClassSampler(train_cls_dataset.dataframe['class'], per_class_num, length_before_new_iter=len(train_cls_dataset))

    train_com_loader = torch.utils.data.DataLoader(
            train_com_dataset, batch_size=128, shuffle=(train_com_sampler is None),
            num_workers=8, pin_memory=True,drop_last=True, sampler=train_com_sampler)

    train_cls_loader = torch.utils.data.DataLoader(
            train_cls_dataset, batch_size=128, shuffle=(train_cls_sampler is None),
            num_workers=8, pin_memory=True,drop_last=True, sampler=train_cls_sampler)



    return input_size ,num_classes ,train_com_loader, train_cls_loader, test_dataset ,train_cls_dataset,train_com_dataset

def get_PHISON_regroup_2_sixcls(root, seed):
    input_size = 224
#     num_classes = 23
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    train_transform = transforms.Compose(
        [
            transforms.Resize([input_size, input_size]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    
    train_cls_transform = transforms.Compose(
        [
            transforms.Resize([input_size, input_size]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )
    
    
    test_transform = transforms.Compose([
        transforms.Resize([input_size, input_size]),
        transforms.ToTensor(),  
        normalize])
    
    add_test = True
    
    train_cls_regroup_df, val_regroup_df, test_regroup_df, train_component_label, val_component_label, test_component_label, train_com_regroup_df, val_com_df = CreateDataset_regroup_due_2_sixcls(seed ,add_test)
#     import pdb;pdb.set_trace()
    num_classes = len(set(train_com_regroup_df['component_name']))

#     train_cls_regroup_df = train_cls_regroup_df.sample(frac=0.01, random_state=123)
#     train_com_regroup_df = train_com_regroup_df.sample(frac=0.01, random_state=123)
#     val_regroup_df = val_regroup_df.sample(frac=0.01, random_state=123)
#     train_com_df = train_com_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.5, random_state=123))

    train_cls_dataset = CustomDataset(train_cls_regroup_df, transform=train_cls_transform)    
    train_com_dataset = CustomDataset(train_com_regroup_df, transform=train_transform)  
    
    test_dataset = CustomDataset(val_regroup_df, transform=test_transform)
    test_com_dataset = CustomDataset(val_com_df, transform=test_transform)
    
    per_component_num = 128 // len(train_com_dataset.dataframe['component_name'].value_counts().index)
    per_class_num = 128 // len(train_cls_dataset.dataframe['class'].value_counts().index)
    train_com_sampler = samplers.MPerClassSampler(train_com_dataset.dataframe['component_name'], per_component_num, length_before_new_iter=len(train_com_dataset))
    train_cls_sampler = samplers.MPerClassSampler(train_cls_dataset.dataframe['class'], per_class_num, length_before_new_iter=len(train_cls_dataset))

    train_com_loader = torch.utils.data.DataLoader(
            train_com_dataset, batch_size=128, shuffle=(train_com_sampler is None),
            num_workers=8, pin_memory=True,drop_last=True, sampler=train_com_sampler)

    train_cls_loader = torch.utils.data.DataLoader(
            train_cls_dataset, batch_size=128, shuffle=(train_cls_sampler is None),
            num_workers=8, pin_memory=True,drop_last=True, sampler=train_cls_sampler)



    return input_size ,num_classes ,train_com_loader, train_cls_loader, test_dataset ,train_cls_dataset,train_com_dataset, test_com_dataset 



def get_PHISON_regroup_six(root, seed):
    input_size = 224
#     num_classes = 23
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    train_transform = transforms.Compose(
        [
            transforms.Resize([input_size, input_size]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    
    train_cls_transform = transforms.Compose(
        [
            transforms.Resize([input_size, input_size]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )
    
    
    test_transform = transforms.Compose([
        transforms.Resize([input_size, input_size]),
        transforms.ToTensor(),  
        normalize])
    
    add_test = True
    
    train_cls_regroup_df, val_regroup_df, test_regroup_df, train_component_label, val_component_label, test_component_label, train_com_regroup_df = CreateDataset_regroup_due_six(seed ,add_test)
    
    num_classes = len(set(train_com_regroup_df['component_name']))

#     train_cls_regroup_df = train_cls_regroup_df.sample(frac=0.01, random_state=123)
#     train_com_regroup_df = train_com_regroup_df.sample(frac=0.01, random_state=123)
#     val_regroup_df = val_regroup_df.sample(frac=0.01, random_state=123)
#     train_com_df = train_com_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.5, random_state=123))

    train_cls_dataset = CustomDataset(train_cls_regroup_df, transform=train_cls_transform)    
    train_com_dataset = CustomDataset(train_com_regroup_df, transform=train_transform)  
    
    test_dataset = CustomDataset(val_regroup_df, transform=test_transform)
    
    per_component_num = 128 // len(train_com_dataset.dataframe['component_name'].value_counts().index)
    per_class_num = 128 // len(train_cls_dataset.dataframe['class'].value_counts().index)
    train_com_sampler = samplers.MPerClassSampler(train_com_dataset.dataframe['component_name'], per_component_num, length_before_new_iter=len(train_com_dataset))
    train_cls_sampler = samplers.MPerClassSampler(train_cls_dataset.dataframe['class'], per_class_num, length_before_new_iter=len(train_cls_dataset))

    train_com_loader = torch.utils.data.DataLoader(
            train_com_dataset, batch_size=128, shuffle=(train_com_sampler is None),
            num_workers=8, pin_memory=True,drop_last=True, sampler=train_com_sampler)

    train_cls_loader = torch.utils.data.DataLoader(
            train_cls_dataset, batch_size=128, shuffle=(train_cls_sampler is None),
            num_workers=8, pin_memory=True,drop_last=True, sampler=train_cls_sampler)



    return input_size ,num_classes ,train_com_loader, train_cls_loader, test_dataset ,train_cls_dataset,train_com_dataset


def get_PHISON(root, seed):
    input_size = 224
    num_classes = 23
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    train_transform = transforms.Compose(
        [
            transforms.Resize([input_size, input_size]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    
    train_cls_transform = transforms.Compose(
        [
            transforms.Resize([input_size, input_size]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )
    
    
    test_transform = transforms.Compose([
        transforms.Resize([input_size, input_size]),
        transforms.ToTensor(),  
        normalize])
    
    add_test = True
    
    # train_cls_df, val_df, test_df, _, _, _, train_com_df, val_com_df  = CreateDataset_relabel(seed ,add_test)
    # train_cls_df, val_df, test_df, _, _, _, train_com_df, val_com_df = CreateDataset_relabel_sixcls(seed ,add_test)
    train_cls_df, val_df, test_df, _, _, _, train_com_df, val_com_df = CreateDataset_relabel_sixcls_randomsplit(seed ,add_test)

#     train_cls_df = train_cls_df.sample(frac=0.01, random_state=123)
#     train_com_df = train_com_df.sample(frac=0.01, random_state=123)
#     val_df = val_df.sample(frac=0.01, random_state=123)

    train_cls_dataset = CustomDataset(train_cls_df, transform=train_cls_transform)    
    train_com_dataset = CustomDataset(train_com_df, transform=train_transform)    
    
    
    test_dataset = CustomDataset(val_df, transform=test_transform)
    test_com_dataset = CustomDataset(val_com_df, transform=test_transform)
    
    per_component_num = 128 // len(train_com_dataset.dataframe['component_name'].value_counts().index)
    per_class_num = 128 // len(train_cls_dataset.dataframe['class'].value_counts().index)
    train_com_sampler = samplers.MPerClassSampler(train_com_dataset.dataframe['component_name'], per_component_num, length_before_new_iter=len(train_com_dataset))
    train_cls_sampler = samplers.MPerClassSampler(train_cls_dataset.dataframe['class'], per_class_num, length_before_new_iter=len(train_cls_dataset))

    train_com_loader = torch.utils.data.DataLoader(
            train_com_dataset, batch_size=128, shuffle=(train_com_sampler is None),
            num_workers=8, pin_memory=True,drop_last=True, sampler=train_com_sampler)

    train_cls_loader = torch.utils.data.DataLoader(
            train_cls_dataset, batch_size=128, shuffle=(train_cls_sampler is None),
            num_workers=8, pin_memory=True,drop_last=True, sampler=train_cls_sampler)

#     test_com_loader = torch.utils.data.DataLoader(
#             test_dataset, batch_size=128, shuffle=(train_com_sampler is None),
#             num_workers=8, pin_memory=True,drop_last=True, sampler=train_com_sampler)
#     kwargs = {"num_workers": 8, "pin_memory": True}

#     train_loader = torch.utils.data.DataLoader(
#         train_cls_dataset,
#         batch_size=32,
#         shuffle=True,
#         drop_last=True, 
#         **kwargs
#     )

#     train_com_loader = torch.utils.data.DataLoader(
#         train_com_dataset,
#         batch_size=32,
#         shuffle=True,
#         drop_last=True, 
#         **kwargs
#     )

    return input_size ,num_classes ,train_com_loader, train_cls_loader, test_dataset ,train_cls_dataset,train_com_dataset, test_com_dataset

def get_PHISON_good(root, seed):
    input_size = 224
    num_classes = 23
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    train_transform = transforms.Compose(
        [
            transforms.Resize([input_size, input_size]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    
    train_cls_transform = transforms.Compose(
        [
            transforms.Resize([input_size, input_size]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )
    
    
    test_transform = transforms.Compose([
        transforms.Resize([input_size, input_size]),
        transforms.ToTensor(),  
        normalize])
    
    add_test = True
    
#     train_cls_df, val_df, test_df, _, _, _, train_com_df,val_com_df = CreateDataset_relabel_sixcls(seed ,add_test)
    train_cls_df, val_df, test_df, _, _, _, train_com_df,val_com_df = CreateDataset_relabel_good(seed ,add_test)

#     train_cls_df = train_cls_df.sample(frac=0.01, random_state=123)
#     train_com_df = train_com_df.sample(frac=0.01, random_state=123)
#     val_df = val_df.sample(frac=0.01, random_state=123)

    train_cls_dataset = CustomDataset(train_cls_df, transform=train_cls_transform)    
    train_com_dataset = CustomDataset(train_com_df, transform=train_transform)    
    
    
    test_dataset = CustomDataset(val_df, transform=test_transform)
    test_com_dataset = CustomDataset(val_com_df, transform=test_transform)
    
    per_component_num = 128 // len(train_com_dataset.dataframe['component_name'].value_counts().index)
    per_class_num = 128 // len(train_cls_dataset.dataframe['class'].value_counts().index)
    train_com_sampler = samplers.MPerClassSampler(train_com_dataset.dataframe['component_name'], per_component_num, length_before_new_iter=len(train_com_dataset))
    train_cls_sampler = samplers.MPerClassSampler(train_cls_dataset.dataframe['class'], per_class_num, length_before_new_iter=len(train_cls_dataset))

    train_com_loader = torch.utils.data.DataLoader(
            train_com_dataset, batch_size=128, shuffle=(train_com_sampler is None),
            num_workers=8, pin_memory=True,drop_last=True, sampler=train_com_sampler)

    train_cls_loader = torch.utils.data.DataLoader(
            train_cls_dataset, batch_size=128, shuffle=(train_cls_sampler is None),
            num_workers=8, pin_memory=True,drop_last=True, sampler=train_cls_sampler)



    return input_size ,num_classes ,train_com_loader, train_cls_loader, test_dataset ,train_cls_dataset,train_com_dataset, test_com_dataset

def get_PHISON_shift(root, seed):
    input_size = 224
    num_classes = 23
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    train_transform = transforms.Compose(
        [
            transforms.Resize([input_size, input_size]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    
    train_cls_transform = transforms.Compose(
        [
            transforms.Resize([input_size, input_size]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )
    
    
    test_transform = transforms.Compose([
        transforms.Resize([input_size, input_size]),
        transforms.ToTensor(),  
        normalize])
    
    add_test = True
    
#     train_cls_df, val_df, test_df, _, _, _, train_com_df,val_com_df = CreateDataset_relabel_sixcls(seed ,add_test)
    train_cls_df, val_df, test_df, _, _, _, train_com_df,val_com_df = CreateDataset_relabel_shift(seed ,add_test)

#     train_cls_df = train_cls_df.sample(frac=0.01, random_state=123)
#     train_com_df = train_com_df.sample(frac=0.01, random_state=123)
#     val_df = val_df.sample(frac=0.01, random_state=123)

    train_cls_dataset = CustomDataset(train_cls_df, transform=train_cls_transform)    
    train_com_dataset = CustomDataset(train_com_df, transform=train_transform)    
    
    
    test_dataset = CustomDataset(val_df, transform=test_transform)
    test_com_dataset = CustomDataset(val_com_df, transform=test_transform)
    
    per_component_num = 128 // len(train_com_dataset.dataframe['component_name'].value_counts().index)
    per_class_num = 128 // len(train_cls_dataset.dataframe['class'].value_counts().index)
    train_com_sampler = samplers.MPerClassSampler(train_com_dataset.dataframe['component_name'], per_component_num, length_before_new_iter=len(train_com_dataset))
    train_cls_sampler = samplers.MPerClassSampler(train_cls_dataset.dataframe['class'], per_class_num, length_before_new_iter=len(train_cls_dataset))

    train_com_loader = torch.utils.data.DataLoader(
            train_com_dataset, batch_size=128, shuffle=(train_com_sampler is None),
            num_workers=8, pin_memory=True,drop_last=True, sampler=train_com_sampler)

    train_cls_loader = torch.utils.data.DataLoader(
            train_cls_dataset, batch_size=128, shuffle=(train_cls_sampler is None),
            num_workers=8, pin_memory=True,drop_last=True, sampler=train_cls_sampler)



    return input_size ,num_classes ,train_com_loader, train_cls_loader, test_dataset ,train_cls_dataset,train_com_dataset, test_com_dataset

def get_PHISON_broke(root, seed):
    input_size = 224
    num_classes = 23
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    train_transform = transforms.Compose(
        [
            transforms.Resize([input_size, input_size]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    
    train_cls_transform = transforms.Compose(
        [
            transforms.Resize([input_size, input_size]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )
    
    
    test_transform = transforms.Compose([
        transforms.Resize([input_size, input_size]),
        transforms.ToTensor(),  
        normalize])
    
    add_test = True
    
#     train_cls_df, val_df, test_df, _, _, _, train_com_df,val_com_df = CreateDataset_relabel_sixcls(seed ,add_test)
    train_cls_df, val_df, test_df, _, _, _, train_com_df,val_com_df = CreateDataset_relabel_broke(seed ,add_test)

#     train_cls_df = train_cls_df.sample(frac=0.01, random_state=123)
#     train_com_df = train_com_df.sample(frac=0.01, random_state=123)
#     val_df = val_df.sample(frac=0.01, random_state=123)

    train_cls_dataset = CustomDataset(train_cls_df, transform=train_cls_transform)    
    train_com_dataset = CustomDataset(train_com_df, transform=train_transform)    
    
    
    test_dataset = CustomDataset(val_df, transform=test_transform)
    test_com_dataset = CustomDataset(val_com_df, transform=test_transform)
    
    per_component_num = 128 // len(train_com_dataset.dataframe['component_name'].value_counts().index)
    per_class_num = 128 // len(train_cls_dataset.dataframe['class'].value_counts().index)
    train_com_sampler = samplers.MPerClassSampler(train_com_dataset.dataframe['component_name'], per_component_num, length_before_new_iter=len(train_com_dataset))
    train_cls_sampler = samplers.MPerClassSampler(train_cls_dataset.dataframe['class'], per_class_num, length_before_new_iter=len(train_cls_dataset))

    train_com_loader = torch.utils.data.DataLoader(
            train_com_dataset, batch_size=128, shuffle=(train_com_sampler is None),
            num_workers=8, pin_memory=True,drop_last=True, sampler=train_com_sampler)

    train_cls_loader = torch.utils.data.DataLoader(
            train_cls_dataset, batch_size=128, shuffle=(train_cls_sampler is None),
            num_workers=8, pin_memory=True,drop_last=True, sampler=train_cls_sampler)


    return input_size ,num_classes ,train_com_loader, train_cls_loader, test_dataset ,train_cls_dataset,train_com_dataset, test_com_dataset

def get_PHISON_short(root, seed):
    input_size = 224
    num_classes = 23
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    train_transform = transforms.Compose(
        [
            transforms.Resize([input_size, input_size]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    
    train_cls_transform = transforms.Compose(
        [
            transforms.Resize([input_size, input_size]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )
    
    
    test_transform = transforms.Compose([
        transforms.Resize([input_size, input_size]),
        transforms.ToTensor(),  
        normalize])
    
    add_test = True
    
#     train_cls_df, val_df, test_df, _, _, _, train_com_df,val_com_df = CreateDataset_relabel_sixcls(seed ,add_test)
    # train_cls_df, val_df, test_df, _, _, _, train_com_df,val_com_df = CreateDataset_relabel_short_randomsplit(seed ,add_test)
    train_cls_df, val_df, test_df, _, _, _, train_com_df,val_com_df = CreateDataset_relabel_short(seed ,add_test)

#     train_cls_df = train_cls_df.sample(frac=0.01, random_state=123)
#     train_com_df = train_com_df.sample(frac=0.01, random_state=123)
#     val_df = val_df.sample(frac=0.01, random_state=123)

    train_cls_dataset = CustomDataset(train_cls_df, transform=train_cls_transform)    
    train_com_dataset = CustomDataset(train_com_df, transform=train_transform)    
    
    
    test_dataset = CustomDataset(val_df, transform=test_transform)
    test_com_dataset = CustomDataset(val_com_df, transform=test_transform)
    
    per_component_num = 128 // len(train_com_dataset.dataframe['component_name'].value_counts().index)
    per_class_num = 128 // len(train_cls_dataset.dataframe['class'].value_counts().index)
    train_com_sampler = samplers.MPerClassSampler(train_com_dataset.dataframe['component_name'], per_component_num, length_before_new_iter=len(train_com_dataset))
    train_cls_sampler = samplers.MPerClassSampler(train_cls_dataset.dataframe['class'], per_class_num, length_before_new_iter=len(train_cls_dataset))

    train_com_loader = torch.utils.data.DataLoader(
            train_com_dataset, batch_size=128, shuffle=(train_com_sampler is None),
            num_workers=8, pin_memory=True,drop_last=True, sampler=train_com_sampler)

    train_cls_loader = torch.utils.data.DataLoader(
            train_cls_dataset, batch_size=128, shuffle=(train_cls_sampler is None),
            num_workers=8, pin_memory=True,drop_last=True, sampler=train_cls_sampler)


    return input_size ,num_classes ,train_com_loader, train_cls_loader, test_dataset ,train_cls_dataset,train_com_dataset, test_com_dataset

def get_PHISON_fourcls(root, seed):
    input_size = 224
    num_classes = 23
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    train_transform = transforms.Compose(
        [
            transforms.Resize([input_size, input_size]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    
    train_cls_transform = transforms.Compose(
        [
            transforms.Resize([input_size, input_size]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )
    
    
    test_transform = transforms.Compose([
        transforms.Resize([input_size, input_size]),
        transforms.ToTensor(),  
        normalize])
    
    add_test = True
    
#     train_cls_df, val_df, test_df, _, _, _, train_com_df,val_com_df = CreateDataset_relabel_sixcls(seed ,add_test)
    train_cls_df, val_df, test_df, _, _, _, train_com_df,val_com_df = CreateDataset_relabel_fourcls(seed ,add_test)


    train_cls_dataset = CustomDataset(train_cls_df, transform=train_cls_transform)    
    train_com_dataset = CustomDataset(train_com_df, transform=train_transform)    
    
    
    test_dataset = CustomDataset(val_df, transform=test_transform)
    test_com_dataset = CustomDataset(val_com_df, transform=test_transform)
    
    per_component_num = 128 // len(train_com_dataset.dataframe['component_name'].value_counts().index)
    per_class_num = 128 // len(train_cls_dataset.dataframe['class'].value_counts().index)
    train_com_sampler = samplers.MPerClassSampler(train_com_dataset.dataframe['component_name'], per_component_num, length_before_new_iter=len(train_com_dataset))
    train_cls_sampler = samplers.MPerClassSampler(train_cls_dataset.dataframe['class'], per_class_num, length_before_new_iter=len(train_cls_dataset))

    train_com_loader = torch.utils.data.DataLoader(
            train_com_dataset, batch_size=128, shuffle=(train_com_sampler is None),
            num_workers=8, pin_memory=True,drop_last=True, sampler=train_com_sampler)

    train_cls_loader = torch.utils.data.DataLoader(
            train_cls_dataset, batch_size=128, shuffle=(train_cls_sampler is None),
            num_workers=8, pin_memory=True,drop_last=True, sampler=train_cls_sampler)



    return input_size ,num_classes ,train_com_loader, train_cls_loader, test_dataset ,train_cls_dataset,train_com_dataset, test_com_dataset

def get_PHISON_threecls(root, seed):
    input_size = 224
    num_classes = 23
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    train_transform = transforms.Compose(
        [
            transforms.Resize([input_size, input_size]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    
    train_cls_transform = transforms.Compose(
        [
            transforms.Resize([input_size, input_size]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )
    
    
    test_transform = transforms.Compose([
        transforms.Resize([input_size, input_size]),
        transforms.ToTensor(),  
        normalize])
    
    add_test = True
    
#     train_cls_df, val_df, test_df, _, _, _, train_com_df,val_com_df = CreateDataset_relabel_sixcls(seed ,add_test)
    train_cls_df, val_df, test_df, _, _, _, train_com_df,val_com_df = CreateDataset_relabel_threecls(seed ,add_test)


    train_cls_dataset = CustomDataset(train_cls_df, transform=train_cls_transform)    
    train_com_dataset = CustomDataset(train_com_df, transform=train_transform)    
    
    
    test_dataset = CustomDataset(val_df, transform=test_transform)
    test_com_dataset = CustomDataset(val_com_df, transform=test_transform)
    
    per_component_num = 128 // len(train_com_dataset.dataframe['component_name'].value_counts().index)
    per_class_num = 128 // len(train_cls_dataset.dataframe['class'].value_counts().index)
    train_com_sampler = samplers.MPerClassSampler(train_com_dataset.dataframe['component_name'], per_component_num, length_before_new_iter=len(train_com_dataset))
    train_cls_sampler = samplers.MPerClassSampler(train_cls_dataset.dataframe['class'], per_class_num, length_before_new_iter=len(train_cls_dataset))

    train_com_loader = torch.utils.data.DataLoader(
            train_com_dataset, batch_size=128, shuffle=(train_com_sampler is None),
            num_workers=8, pin_memory=True,drop_last=True, sampler=train_com_sampler)

    train_cls_loader = torch.utils.data.DataLoader(
            train_cls_dataset, batch_size=128, shuffle=(train_cls_sampler is None),
            num_workers=8, pin_memory=True,drop_last=True, sampler=train_cls_sampler)



    return input_size ,num_classes ,train_com_loader, train_cls_loader, test_dataset ,train_cls_dataset,train_com_dataset, test_com_dataset





    

all_datasets = {
#     "SVHN": get_SVHN,
#     "CIFAR10": get_CIFAR10,
#     "CIFAR100": get_CIFAR100,
    "PHISON": get_PHISON,
    "PHISON_fourcls": get_PHISON_fourcls,
    "PHISON_threecls": get_PHISON_threecls,
    "PHISON_good": get_PHISON_good,
    "PHISON_shift": get_PHISON_shift,
    "PHISON_broke": get_PHISON_broke,
    "PHISON_short": get_PHISON_short,
#     "PHISON_ori":get_PHISON_ori,
#     "PHISON_df":get_PHISON_df,
#     "PHISON_regroup":get_PHISON_regroup,
#     "PHISON_regroup2":get_PHISON_regroup_2,
#     "PHISON_regroup3":get_PHISON_regroup_3,
#     "PHISON_regroup4":get_PHISON_regroup_4,
#     "PHISON_regroup5":get_PHISON_regroup_5,
#     "PHISON_regroup2_fewshot":get_PHISON_regroup_2_fewshot,
#     "PHISON_regroup_six":get_PHISON_regroup_2_sixcls,
#     'fruit2':get_fruit2,
#     'MVtecAD':get_mvtecad,
#     'MVtecAD_texture':get_mvtecad_tex,
#     'fruit_8':get_fruit_8,
#     'fruit_8_MSSH':get_fruit_8_MSSH
}


def get_dataset(dataset, seed, root="./"):
    return all_datasets[dataset](root ,seed)
def get_dataset_bs(dataset, seed, bs, root="./"):
    return all_datasets[dataset](root ,seed, bs)

def get_fewshot_dataset(dataset, seed, nshot, root="./"):
    return all_datasets[dataset](root ,seed, nshot)

def get_dataloaders(dataset, train_batch_size=128, root="./"):
    ds = all_datasets[dataset](root)
    input_size, num_classes, train_dataset, test_dataset = ds

    kwargs = {"num_workers": 8, "pin_memory": True}

    train_loader = data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True, **kwargs
    )

    test_loader = data.DataLoader(
        test_dataset, batch_size=1000, shuffle=False, **kwargs
    )

    return train_loader, test_loader, input_size, num_classes
