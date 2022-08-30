import os
import sys
import re
import six
import copy
import time
import math
#import lmdb
import torch

from natsort import natsorted
import itertools
from PIL import Image
from copy import deepcopy
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, ConcatDataset, Subset
from torch._utils import _accumulate
import torchvision.transforms as transforms
#from train import target_select

global source
global target

class Batch_Balanced_Dataset(object):

    def __init__(self, opt):
        """
        Modulate the data ratio in the batch.
        For example, when select_data is "MJ-ST" and batch_ratio is "0.5-0.5",
        the 50% of the batch is filled with MJ and the other 50% of the batch is filled with ST.

        opt.batch_ratio: 一个batch中包含的不同数据集的比例
        opt.total_data_usage_ratio: 对于每一个数据及，使用这个数据集的百分之多少，默认是1（100%）
        """
        self.opt = opt
        _,source = target_select(opt.target)
        #log = open(f'./saved_models/{opt.exp_name}/log_dataset.txt', 'a')
        dashed_line = '-' * 80
        print(dashed_line)
        #log.write(dashed_line + '\n')
        #print(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}')
        #log.write(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}\n')
        #assert len(opt.select_data) == len(opt.batch_ratio)

        # 为每个dataloader应用collate函数，直接输出一整个batch，
        #_AlignCollate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        self.data_loader_list = []
        self.dataloader_iter_list = []
        batch_size_list = []
        Total_batch_size = 0
        for selected_d, batch_ratio_d in zip(source, opt.batch_ratio):  #opt.select_data
            _batch_size = max(round(opt.batch_size * float(batch_ratio_d)), 1)
            print("_batch_size: ",_batch_size)  # 48
            print("opt.batch_ratio: ",opt.batch_ratio)   #0.5
            print(dashed_line,"domain: ",selected_d)
            #log.write(dashed_line + '\n')
            _dataset= hierarchical_dataset(selected_d)  #source domain dataset
            total_number_dataset = len(_dataset) # 현재 데이터세트에 포함된 이미지 수
            print("hierarchical_dataset output:")
            #print("_dataset: ",_dataset)
            print("dataset len(현재 데이터세트에 포함된 이미지 수): ",total_number_dataset)
            #log.write(_dataset_log)

            """
            The total number of data can be modified with opt.total_data_usage_ratio.
            ex) opt.total_data_usage_ratio = 1 indicates 100% usage, and 0.2 indicates 20% usage.
            See 4.2 section in our paper.
            """
            print(opt.total_data_usage_ratio)  # 1 
            number_dataset = int(total_number_dataset * float(opt.total_data_usage_ratio) ) # 사용비율(
            print("사용 비율: ",number_dataset)
            #print(opt)
            print("A","="*80)
           # print(opt.fix_dataset_num)  fix_dataset_num 사용 X
           # if opt.fix_dataset_num != -1: number_dataset = opt.fix_dataset_num
            dataset_split = [number_dataset, total_number_dataset - number_dataset] # List[int] e.g. [50, 50]
            indices = range(total_number_dataset)
            
            # accumulate函数： _accumulate([1,2,3,4,5]) --> 1 3 6 10 15
            # Subset就是根据indices取一个数据集的子集，indice根据opt.total_data_usage_ratio来取值
            _dataset, _ = [Subset(_dataset, indices[offset - length:offset])
                           for offset, length in zip(_accumulate(dataset_split), dataset_split)]
            #selected_d_log = f'num total samples of {selected_d}: {total_number_dataset} x {opt.total_data_usage_ratio} (total_data_usage_ratio) = {len(_dataset)}\n'
            #selected_d_log += f'num samples of {selected_d} per batch: {opt.batch_size} x {float(batch_ratio_d)} (batch_ratio) = {_batch_size}'
            #print(selected_d_log)
            #log.write(selected_d_log + '\n')
            batch_size_list.append(str(_batch_size))
            Total_batch_size += _batch_size

            _data_loader = torch.utils.data.DataLoader(
                _dataset, batch_size=_batch_size,
                shuffle=True,
                num_workers=int(opt.workers),
                pin_memory=False, drop_last=True) #collate_fn=_AlignCollate, 
            self.data_loader_list.append(_data_loader)
            self.dataloader_iter_list.append(iter(_data_loader))
            
        print("_data_loader:",_data_loader)
        test = _data_loader.__iter__().__next__()
        print(len(test))
        print(test[0])  #sensor 
        print(test[1])  #label
        # 아마 for문 마지막 user일 듯
        print("B","="*80)
        print("self.data_loader_list:",self.data_loader_list)  #2
        print("self.dataloader_iter_list:",self.dataloader_iter_list)  #2


        Total_batch_size_log = f'{dashed_line}\n'
        batch_size_sum = '+'.join(batch_size_list)
        Total_batch_size_log += f'Total_batch_size: {batch_size_sum} = {Total_batch_size}\n'
        Total_batch_size_log += f'{dashed_line}'
        opt.batch_size = Total_batch_size

        #print(Total_batch_size_log)
        #log.write(Total_batch_size_log + '\n')
        #log.close()

    def get_batch(self, meta_target_index=-1, no_pseudo=False): # 如果指定了meta_target_index，则忽略第meta_target_index个数据集
        balanced_batch_images = []
        balanced_batch_texts = []
        
        print("값이 잘 들어왔는지 확인(traing dataset)")
        print(self.dataloader_iter_list)
        for i, data_loader_iter in enumerate(self.dataloader_iter_list):     # user02, user03         
            print(i)
            print(data_loader_iter)
            
            if i == meta_target_index: continue
            # 유사 레이블 데이터세트를 샘플링할 필요가 없고 현재 의사 레이블 데이터세트를 포함하는 경우 건너뜁니다.
            if i == len(self.dataloader_iter_list) - 1 and no_pseudo and self.has_pseudo_label_dataset(): continue 
            try:
                image, text = data_loader_iter.next()
                balanced_batch_images.append(image)
                balanced_batch_texts += [text]
            except StopIteration: # 데이터 세트의 이미지 수가 충분하지 않은 경우 훈련을 위해 반복자를 다시 작성하십시오.
                self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                image, text = self.dataloader_iter_list[i].next()
                balanced_batch_images.append(image)
                balanced_batch_texts += [text]
            except ValueError:
                pass

        balanced_batch_images = torch.cat(balanced_batch_images, 0)
        balanced_batch_texts = torch.cat(balanced_batch_texts, 0)  # 새로 추가
       # print("get_batch output_")    # train_datatset.get_batch 함수에 들어가는 값 살펴보기
       # print("balanced_batch_images: ",balanced_batch_images)
       # print("balanced_batch_texts: ",balanced_batch_texts)

        return balanced_batch_images, balanced_batch_texts
    
    def get_meta_test_batch(self, meta_target_index=-1): # 如果指定了meta_target_index，则忽略第meta_target_index个数据集
        
        if meta_target_index == self.opt.source_num:
            assert len(self.data_loader_list) == self.opt.source_num + 1, 'There is no target dataset'
        balanced_batch_images = []
        balanced_batch_texts = []

        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            if i == meta_target_index:
                try:
                    image, text = data_loader_iter.next()
                    balanced_batch_images.append(image)
                    balanced_batch_texts += text
                except StopIteration: # 如果一个数据集图片数量不够了，则重新构建迭代器进行训练
                    self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                    image, text = self.dataloader_iter_list[i].next()
                    balanced_batch_images.append(image)
                    balanced_batch_texts += text
                except ValueError:
                    pass
        # print(balanced_batch_images[0].shape)
        balanced_batch_images = torch.cat(balanced_batch_images, 0)

        return balanced_batch_images, balanced_batch_texts

    def add_target_domain_dataset(self, dataset, opt):
        _AlignCollate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        avg_batch_size = opt.batch_size // opt.source_num
        batch_size = len(dataset) if len(dataset) <= avg_batch_size else avg_batch_size
        self_training_loader = torch.utils.data.DataLoader(
                        dataset, batch_size=batch_size,
                        shuffle=True,  # 'True' to check training progress with validation function.
                        num_workers=int(opt.workers), pin_memory=False, collate_fn=_AlignCollate, drop_last=True)
        if self.has_pseudo_label_dataset():
            self.data_loader_list[opt.source_num] = self_training_loader
            self.dataloader_iter_list[opt.source_num] = (iter(self_training_loader))
        else:
            self.data_loader_list.append(self_training_loader)
            self.dataloader_iter_list.append(iter(self_training_loader))

    def add_pseudo_label_dataset(self, dataset, opt):
        avg_batch_size = opt.batch_size // opt.source_num
        batch_size = len(dataset) if len(dataset) <= avg_batch_size else avg_batch_size
        self_training_loader = torch.utils.data.DataLoader(
                        dataset, batch_size=batch_size,
                        shuffle=True,  # 'True' to check training progress with validation function.
                        num_workers=int(opt.workers), pin_memory=False, collate_fn=self_training_collate)
        if self.has_pseudo_label_dataset():
            self.data_loader_list[opt.source_num] = self_training_loader
            self.dataloader_iter_list[opt.source_num] = (iter(self_training_loader))
        else:
            self.data_loader_list.append(self_training_loader)
            self.dataloader_iter_list.append(iter(self_training_loader))

    def add_residual_pseudo_label_dataset(self, dataset, opt):
        avg_batch_size = opt.batch_size // opt.source_num
        batch_size = len(dataset) if len(dataset) <= avg_batch_size else avg_batch_size
        self_training_loader = torch.utils.data.DataLoader(
                        dataset, batch_size=batch_size,
                        shuffle=True,  # 'True' to check training progress with validation function.
                        num_workers=int(opt.workers), pin_memory=False, collate_fn=self_training_collate)
        if self.has_residual_pseudo_label_dataset():
            self.data_loader_list[opt.source_num + 1] = self_training_loader
            self.dataloader_iter_list[opt.source_num + 1] = (iter(self_training_loader))
        else:
            self.data_loader_list.append(self_training_loader)
            self.dataloader_iter_list.append(iter(self_training_loader))

    def has_pseudo_label_dataset(self):
        return True if len(self.data_loader_list) > self.opt.source_num else False

    def has_residual_pseudo_label_dataset(self):
        return True if len(self.data_loader_list) > self.opt.source_num + 1 else False
        
class Batch_Balanced_Sampler(object):
    def __init__(self, dataset_len, batch_size):
        dataset_len.insert(0,0)
        self.dataset_len = dataset_len
        self.start_index = list(itertools.accumulate(self.dataset_len))[:-1]
        self.batch_size = batch_size # 每个子数据集的batchsize
        self.counter = 0

    def __len__(self):
        return self.dataset_len

    def __iter__(self):
        data_index = []
        while True:
            for i in range(len(self.start_index)):
                data_index.extend([self.start_index[i] + (self.counter * self.batch_size + j) % self.dataset_len[i + 1] for j in range(self.batch_size)])
            yield data_index
            data_index = []
            self.counter += 1
        

class Batch_Balanced_Dataset0(object):

    def __init__(self, opt):
        """
        Modulate the data ratio in the batch.
        For example, when select_data is "MJ-ST" and batch_ratio is "0.5-0.5",
        the 50% of the batch is filled with MJ and the other 50% of the batch is filled with ST.

        opt.batch_ratio: 一个batch中包含的不同数据集的比例z
        opt.total_data_usage_ratio: 对于每一个数据及，使用这个数据集的百分之多少，默认是1（100%）
        """
        self.opt = opt
        #log = open(f'./saved_models/{opt.exp_name}/log_dataset.txt', 'a')
        #dashed_line = '-' * 80
        #print(dashed_line)
        #log.write(dashed_line + '\n')
        #print(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}')
        #log.write(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}\n')
        #assert len(opt.select_data) == len(opt.batch_ratio)

        # 为每个dataloader应用collate函数，直接输出一整个batch，
        #_AlignCollate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        self.data_loader_list = []
        self.dataloader_iter_list = []
        self.batch_size_list = []
        Total_batch_size = 0

        self.dataset_list = []
        self.dataset_len_list = []

        self.pseudo_dataloader = None
        self.pseudo_batch_size = -1

        for selected_d, batch_ratio_d in zip(source, opt.batch_ratio):  #opt.select_data  (User02-USer03),(0.5-0.5)
            _batch_size = max(round(opt.batch_size * float(batch_ratio_d)), 1)
            print(dashed_line)
            #log.write(dashed_line + '\n')
            _dataset= hierarchical_dataset(selected_d)  #source domain dataset
            total_number_dataset = len(_dataset) #현재 데이터세트에 포함된 이미지 수
            
            
            #log.write(_dataset_log)

            """
            The total number of data can be modified with opt.total_data_usage_ratio.
            ex) opt.total_data_usage_ratio = 1 indicates 100% usage, and 0.2 indicates 20% usage.
            See 4.2 section in our paper.
            """
            number_dataset = int(total_number_dataset * float(opt.total_data_usage_ratio)) # 사용 비율
            #if opt.fix_dataset_num != -1: number_dataset = opt.fix_dataset_num
            dataset_split = [number_dataset, total_number_dataset - number_dataset] # List[int] e.g. [50, 50]
            indices = range(total_number_dataset)
            
            # accumulate函数： _accumulate([1,2,3,4,5]) --> 1 3 6 10 15
            # Subset就是根据indices取一个数据集的子集，indice根据opt.total_data_usage_ratio来取值
            _dataset, _ = [Subset(_dataset, indices[offset - length:offset])
                           for offset, length in zip(_accumulate(dataset_split), dataset_split)]
            #selected_d_log = f'num total samples of {selected_d}: {total_number_dataset} x {opt.total_data_usage_ratio} (total_data_usage_ratio) = {len(_dataset)}\n'
            #selected_d_log += f'num samples of {selected_d} per batch: {opt.batch_size} x {float(batch_ratio_d)} (batch_ratio) = {_batch_size}'
            #print(selected_d_log)
            #log.write(selected_d_log + '\n')
            self.batch_size_list.append(str(_batch_size))
            Total_batch_size += _batch_size

            self.dataset_list.append(_dataset)
            self.dataset_len_list.append(number_dataset)



        concatenated_dataset = ConcatDataset(self.dataset_list)
       # assert len(concatenated_dataset) == sum(self.dataset_len_list)

        batch_sampler = Batch_Balanced_Sampler(self.dataset_len_list, _batch_size)
        self.data_loader = iter(torch.utils.data.DataLoader(
            concatenated_dataset,
            batch_sampler=batch_sampler,
            num_workers=int(opt.workers),
            pin_memory=False)) #collate_fn=_AlignCollate,

        #Total_batch_size_log = f'{dashed_line}\n'
        batch_size_sum = '+'.join(self.batch_size_list)
        self.batch_size_list = list(map(int, self.batch_size_list))
        #Total_batch_size_log += f'Total_batch_size: {batch_size_sum} = {Total_batch_size}\n'
        #Total_batch_size_log += f'{dashed_line}'
        opt.batch_size = Total_batch_size

       # print(Total_batch_size_log)
        #log.write(Total_batch_size_log + '\n')
        #log.close()

    def get_batch(self, meta_target_index=-1, no_pseudo=False): # meta_target_index가 지정되면 meta_target_index번째 데이터 세트는 무시됩니다.
        
        imgs, texts = next(self.data_loader)
        # 如果未指定或指定为伪标签数据集，则直接返回所有
        if meta_target_index == -1 or meta_target_index >= len(self.batch_size_list): return imgs, texts
        start_index_list = list(itertools.accumulate(self.batch_size_list))
        start_index_list.insert(0, 0)

        ret_imgs, ret_texts = [], []
        for i in range(len(self.batch_size_list)): 
            if i == meta_target_index: continue
            ret_imgs.extend(imgs[start_index_list[i] : start_index_list[i] + self.batch_size_list[i]])
            ret_texts.extend(texts[start_index_list[i] : start_index_list[i] + self.batch_size_list[i]])
        ret_imgs = torch.stack(ret_imgs, 0)
        
        # assert self.has_pseudo_label_dataset() == True, '의사 레이블 데이터 세트는 비어 있을 수 없습니다.'
        if self.has_pseudo_label_dataset():
            try:
                psuedo_imgs, pseudo_texts = next(self.pseudo_dataloader_iter)
            except StopIteration:
                self.pseudo_dataloader_iter = iter(self.pseudo_dataloader)
                psuedo_imgs, pseudo_texts = next(self.pseudo_dataloader_iter)
            ret_imgs = torch.cat([ret_imgs, psuedo_imgs], 0)
            ret_texts += pseudo_texts

        return ret_imgs, ret_texts

    def get_meta_test_batch(self, meta_target_index=-1): # 如果指定了meta_target_index，则忽略第meta_target_index个数据集
        
        #assert meta_target_index != -1, 'Meta target index should be specified'
        if meta_target_index >= len(self.batch_size_list) and self.has_pseudo_label_dataset(): 
            try:
                img, text = next(self.pseudo_dataloader_iter)
            except StopIteration:
                self.pseudo_dataloader_iter = iter(self.pseudo_dataloader)
                img, text = next(self.pseudo_dataloader_iter)

            return img, text
        
        imgs, texts = next(self.data_loader)
        start_index_list = list(itertools.accumulate(self.batch_size_list))
        start_index_list.insert(0, 0)
        ret_img, ret_text = None, None
        for i in range(len(self.batch_size_list)): 
            if i == meta_target_index:
                ret_img = imgs[start_index_list[i]:start_index_list[i] + self.batch_size_list[i]]
                ret_text = texts[start_index_list[i]:start_index_list[i] + self.batch_size_list[i]]

        return ret_img, ret_text

    def add_pseudo_label_dataset(self, dataset, opt):
        avg_batch_size = opt.batch_size // opt.source_num
        batch_size = len(dataset) if len(dataset) <= avg_batch_size else avg_batch_size
        self.pseudo_batch_size = batch_size
        self.pseudo_dataloader = torch.utils.data.DataLoader(
                        dataset, batch_size=batch_size,
                        shuffle=True,  # 'True' to check training progress with validation function.
                        num_workers=int(opt.workers), pin_memory=False, collate_fn=self_training_collate)
        self.pseudo_dataloader_iter = iter(self.pseudo_dataloader)


    def has_pseudo_label_dataset(self):
        return True if self.pseudo_dataloader else False

#USER LIST
USER_TRAIN= [
    "user01",
    "user02",
    "user03",
    "user04",
    "user05",
    "user08",
    "user09",
    "user10",
    "user11",

]

def target_select(opt):
    print("input:", opt)
    
    #target domain 설정
    target = str(opt)
    print("target= ",target)
    
    source = []
    for j in range(len(USER_TRAIN)):
        if target != USER_TRAIN[j]:
            source.append(USER_TRAIN[j])
    print("source= ",source)
        
    return target,source

def hierarchical_dataset(user):
    # dataset 경로 지정
    path="./2020_e4/"
    
    dataset_list = []
  
    #target data load
    data = path + str(user) + '_e4.npy'
    label = path + str(user) + '_label.npy'
    
    #데이터 로드
    valid_dataset = np.load(data)
    valid_label = np.load(label)
    
    #print(len(train_dataset))
    idx_cnt = 0
    len(valid_dataset)
    for idx in range(len(valid_dataset)):
        one_x = valid_dataset[idx][:-5, 0]
        one_y = valid_dataset[idx][:-5, 1]
        one_z = valid_dataset[idx][:-5, 2]
        new_x = np.reshape(one_x, (5, 15))  # reshape to 5 by 15 matrix
        new_y = np.reshape(one_y, (5, 15))
        new_z = np.reshape(one_z, (5, 15))
        fin_d = np.concatenate((new_x, new_y, new_z), axis=0)  # concatenate 2D arrays
        fin_d = np.reshape(fin_d, (1, 15, 15))
        one_l = valid_label[idx, 0]
    
        dataset = (fin_d,one_l)  #fin_d[0] : 3차원
        dataset_list.append(dataset)
        
    #concat_dataset = ConcatDataset(dataset_list)
    print("ex)",dataset_list[0])
    print("sensor shape:",dataset_list[0][0].shape)
        
    print("데이터 셋 갯수:",len(dataset_list))
    
    return dataset_list

class RawDataset(Dataset):

    def __init__(self, root, opt):
        self.opt = opt
        self.image_path_list = []
        for dirpath, dirnames, filenames in os.walk(root):
            for name in filenames:
                _, ext = os.path.splitext(name)
                ext = ext.lower()
                if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                    self.image_path_list.append(os.path.join(dirpath, name))

        self.image_path_list = natsorted(self.image_path_list)
        self.nSamples = len(self.image_path_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        try:
            if self.opt.rgb:
                img = Image.open(self.image_path_list[index]).convert('RGB')  # for color image
            else:
                img = Image.open(self.image_path_list[index]).convert('L')

        except IOError:
            
            (f'Corrupted image for {index}')
            # make dummy image and dummy label for corrupted image.
            if self.opt.rgb:
                img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
            else:
                img = Image.new('L', (self.opt.imgW, self.opt.imgH))

        return (img, self.image_path_list[index])


class ResizeNormalize(object):

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class NormalizePAD(object):

    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img


class AlignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)

        if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
            resized_max_w = self.imgW
            input_channel = 3 if images[0].mode == 'RGB' else 1
            transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

            resized_images = []
            for image in images:
                w, h = image.size
                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)

                resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_images.append(transform(resized_image))
                # resized_image.save('./image_test/%d_test.jpg' % w)

            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

        else:
            transform = ResizeNormalize((self.imgW, self.imgH))
            image_tensors = [transform(image) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors, labels


def self_training_collate(batch):
    imgs, labels = [], []
    for img, label in batch:
        imgs.append(img)
        labels.append(label)
    
    return torch.stack(imgs), labels

class SelfTrainingDataset(Dataset):
    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = labels
    
    def __getitem__(self, index):
        return self.imgs[index], self.labels[index]

    def __len__(self):
        #assert len(self.imgs) == len(self.labels)
        return len(self.imgs)



def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)
