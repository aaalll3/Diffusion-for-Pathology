import os
import csv
import argparse
import re
import random
import matplotlib.pyplot as plt
import skimage
from skimage.io import imread,imshow,imsave
# from skimage.color import 


# For test TODO
test_path = "E:/X/11/PRAT/code/Diffusion-for-Pathology/__portable/"
test_name1 = "TCGA-BH-A1FC-DX1_xmin52216_ymin33355_MPP-0.2500.png"
test_name2 = "TCGA-S3-AA15-DX1_xmin55486_ymin28926_MPP-0.2500.png"
roi_name = "roiBounds.csv"
decode_name = "gtruth_codes.tsv"

class Info:
    def __init__(self,decode=None,roi=None):
        self.names = []
        self.n_iidx = {}
        self.ids = []
        self.i_iidx = {}
        self.bounds = {}
        self.decode = {}
        my_pattern = r'^TCGA-(.{2})-(.{4})-(.*)_xmin(\d*)_ymin(\d*)_MPP-(\d*\.\d*)\.(png|jpg|jpeg|bmp)'
        self.pattern = re.compile(my_pattern)
        self.ext_pattern = re.compile(r'\.(.*)')
        self.debug = False
        if decode:
            line1=True
            with open(decode,'r') as tsv_file:
                lines = tsv_file.readlines()
                for row in lines:
                    if line1:
                        line1 = False
                        continue
                    row = row.strip().split('\t')
                    region_type = row[0]
                    code = int(row[1])
                    self.decode[code] = region_type
                    
        if roi:
            line1=True
            with open(roi,'r') as csv_file:
                reader = csv.reader(csv_file)
                for row in reader:
                    if line1:
                        line1 = False
                        continue
                    name = row[0]
                    xmin = int(row[1])
                    ymin = int(row[2])
                    xmax = int(row[3])
                    ymax = int(row[4])
                    self.bounds[name]=[xmin,ymin,xmax,ymax]

    def add(self,names):
        if type(names) is list:
            if self.debug:
                print('adding a list of file')
            for name in names:
                if self.pattern.search(name):
                    self.n_iidx[name]=len(self.names)
                    self.i_iidx[name]=len(self.names)
                    self.names.append(name)
                    self.ids.append(name.split('_')[0])
                else:
                    print(f'miss match on {name}')
        elif type(names) is str:
            if self.debug:
                print('adding one file')
            if self.pattern.search(names):
                self.n_iidx[names]=len(self.names)
                self.i_iidx[names]=len(self.names)
                self.names.append(names)
                self.ids.append(names.split('_')[0])
            else:
                print(f'miss match on {names}')
        else:
            print(f'Not adding {names}')

    def get_id(self,name):
        if type(name) is str:
            return self.ids[self.n_iidx[name]]
        elif type(name) is int:
            return self.ids[name]
        else:
            print(f'Info.get_id:miss type >{name}')
    
    def get_idx(self,name):
        if type(name) is int:
            if self.ext_pattern.search(name):
                return self.n_iidx[name]
            else:
                return self.i_iidx[name]
        else:
            print(f'Info.get_idx:miss type >{name}')
        
    def get_name(self,idx):
        if type(idx) is str:
            return self.names[self.i_iidx[name]]
        elif type(idx) is int:
            return self.names[idx]
        else:
            print(f'Info.get_name:miss type >{idx}')
    
    def get_boundary(self,name):
        if type(name) is str:
            if self.ext_pattern.search(name):
                return self.bounds[self.get_id(name)]
            else:
                return self.bounds[name]
        elif type(name) is int:
            return self.bounds[self.get_id(name)]
        else:
            print(f'Info.get_name:miss type >{name}')
    
    def parse(self):
        matches = self.pattern.search(self.name)
        self.dye = matches.group(0)
        self.sid = matches.group(1)
        self.MPP = matches.group(5)
        
class Prepare:
    def __init__(self,root_dir = None,dst_dir = None):
        roi_file = "roiBounds.csv"
        decode_file = "gtruth_codes.tsv"
        self.root_dir = root_dir
        self.dst_dir = dst_dir
        self.image_dir = os.path.join(self.root_dir,'images')
        self.mask_dir = os.path.join(self.root_dir,'masks')
        self.meta_dir = os.path.join(self.root_dir,'meta')
        self.sample_names = os.listdir(self.image_dir)
        self.infos = Info(roi=os.path.join(self.meta_dir,roi_file),decode=os.path.join(self.meta_dir,decode_file))
        self.infos.add(self.sample_names)
        self.set = set()
        self.debug = False
    
    def config(self):
        pass
    
    def get_id(self,name):
        return self.infos.get_id(name)
    
    def get_idx(self,name):
        return self.infos.get_idx(name)
    
    def get_name(self,idx):
        return self.infos.get_name(idx)
    
    def get_boundary(self,name):
        return self.infos.get_boundary(name)
    
    def cut_one(self,_id,shape=(128,128),save=False,ext='png'):
        dst_dir = self.dst_dir
        if dst_dir is None:
            dst_dir = os.path.curdir
        im_path = os.path.join(self.image_dir,self.get_name(_id))
        boundary = self.get_boundary(_id)
        im = imread(im_path)
        im_shape = im.shape
        xmax = im_shape[0]
        ymax = im_shape[1]
        while(True):
            xrand = random.randint(0,xmax-shape[0])
            yrand = random.randint(0,ymax-shape[1])
            if self.set is not None and not (xrand,yrand) in self.set:
                self.set.add((xrand,yrand))
                print('add')
                break
        dst_path = os.path.join(dst_dir,f'{shape[0]}x{shape[1]}',f'{self.get_id(_id)}_({xrand}_{yrand}).{ext}')
        # print(im.shape)
        if len(im_shape) == 2:
            im = im[xrand:xrand+shape[0],yrand:yrand+shape[1]]
        elif len(im_shape) == 3:
            im = im[xrand:xrand+shape[0],yrand:yrand+shape[1],:]
        # print(im.shape)
        if save:
            if self.debug:
                print(f'save to {dst_path}')
            imsave(dst_path,im)
        return im
    
    def cut_m(self,_id,shape=(128,128),number=1,save=False,ext='png'):
        self.set.clear()
        cropped=[]
        for ii in number:
            cropped.append(self.cut_one(_id,shape,save,ext))
        return cropped
    
    def cut_all(self,shape=(128,128),number=1,save=False,ext='png'):
        for _id in self.infos.ids:
            self.cut_m(_id,shape,number,save,ext)

# TODO

debugging = False

if __name__ == "__main__" and debugging:
    im1 = os.path.join(test_path,'images',test_name1)
    mk1 = os.path.join(test_path,'masks',test_name1)
    im2 = os.path.join(test_path,'images',test_name2)
    mk2 = os.path.join(test_path,'masks',test_name2)
    roi = os.path.join(test_path,'meta',roi_name)
    dc = os.path.join(test_path,'meta',decode_name)
    infos = Info(dc,roi)
    print('initialized info')
    infos.add([test_name1,test_name2])
    # TODO
    for name in infos.names:
        print(name)
        im_path = os.path.join(test_path,'images',name)
        im = imread(im_path)
        # imshow(im)
        # plt.show()
        print(infos.bounds[infos.get_id(name)])

    pc = Prepare(root_dir=test_path,dst_dir=os.path.join(test_path,'playground'))
    print(pc.sample_names)
    print(pc.infos.names)
    print(pc.infos.get_boundary(0))
    imshow(pc.cut_one(0,save = True))
    plt.show()

if __name__=="__main__":
    data_dir = "/users/Etu2/21213002/CrowdsourcingDataset-Amgadetal2019"
    dst_dir = "/users/Etu2/21213002/CrowdsourcingDataset-Amgadetal2019/crop"
    pc = Prepare(root_dir=data_dir,dst_dir=dst_dir)
    pc.cut_all(shape=(128,128),number=20,save=True)