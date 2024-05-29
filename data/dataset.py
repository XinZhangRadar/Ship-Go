import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np

from .util.mask import (bbox2mask, bboxes2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)
import xml.etree.ElementTree as ET
import xml.dom.minidom
import cv2 

from pycocotools.coco import COCO 
import  os, cv2, shutil
from lxml import etree, objectify
from tqdm import tqdm
from PIL import Image
import numpy as np
import time
import json

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    # import pdb;pdb.set_trace()
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images

def pil_loader(path):
    return Image.open(path).convert('RGB')


# For SSDD
class I2IDataset(data.Dataset):
    CLASSES = ('ship',)

    ENVS = ('offshore','inshore')
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader,output_anns=None, out_anns_path=None):
        import pdb;pdb.set_trace()
        imgs = make_dataset(data_root)     
        self.ENV_image_list = {env: make_dataset(os.path.join(data_root.split('flist')[0], 'flist',env+'.txt')) for env in self.ENVS} 
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size
        self.env2label = {env: i for i, env in enumerate(self.ENVS)}
        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        self.output_anns = output_anns
        self.out_anns_path = out_anns_path
        if self.output_anns:
            if not os.path.exists(self.out_anns_path ): 
                os.mkdir(self.out_anns_path)
    def __getitem__(self, index):
        ret = {}
        ann = {}
        path = self.imgs[index]

        img = self.loader(path)
        img_size = np.array(img.size) #(w,h)
        img = self.tfs(img)
        img_tf_size = img.shape[1:] #(h,w)
        w_scale = img_tf_size[1]/img_size[0]
        h_scale = img_tf_size[0]/img_size[1]
        ann['height'] = img_tf_size[0]
        ann['weight'] = img_tf_size[1]
        ann['depth'] = 1 # for SAR images
        ann['filename'] ='Out_'+path.split('/')[-1]
        if self.mask_mode == 'det':
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],dtype=np.float32)
            ann_path = path.replace('image','Annotations').replace('jpg','xml')
            # ann = self.get_ann_info(ann_path)
            # bboxes = self.resize_bboxes(ann['bboxes'],scale_factor)
            # mask = self.get_mask(bboxes)
            mask, objects = self.get_mask_from_det(ann_path, scale_factor)
            if self.output_anns:
                ann['objects'] = objects
                self.genxml(self.out_anns_path, ann)
        elif self.mask_mode == 'seg':
            scale_factor = np.array([w_scale, h_scale],dtype=np.float32)
            ann_path = path.replace('image','Annotations_seg').replace('jpg','xml')
            mask = self.get_mask_from_seg(ann_path, scale_factor)
            if self.output_anns:
                det_ann_path = path.replace('image','Annotations').replace('jpg','xml')
                objects = self.resize_det_anns(det_ann_path, np.array([w_scale, h_scale, w_scale, h_scale],dtype=np.float32))
                ann['objects'] = objects
                self.genxml(self.out_anns_path, ann)
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        #ENVS:

        for env in self.ENV_image_list.keys():
            if path.split('/')[-1] in self.ENV_image_list[env]:
                image_env = self.env2label[env]
                break
        # mask_img = (mask_img + 1)*127
        # cv2.imwrite('mask_img.jpg',mask_img.permute(1,2,0).numpy())
        # image_env = 1
        ret['gt_image'] = img[0].unsqueeze(0)
        ret['cond_image'] = cond_image[0].unsqueeze(0)
        ret['mask_image'] = mask_img[0].unsqueeze(0)
        ret['mask'] = mask[0].unsqueeze(0)
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        ret['image_env'] = image_env
        return ret

    def __len__(self):
        return len(self.imgs)

    def resize_det_anns(self,xml_path,scale_factor):
        objects = self.get_ann_info(xml_path)
        bboxes = self.resize_bboxes(objects['bboxes'],scale_factor)
        objects['bboxes'] = bboxes
        return objects


    def get_mask_from_det(self, xml_path, scale_factor):
        objects = self.resize_det_anns(xml_path, scale_factor)
        #import pdb;pdb.set_trace()
        # ann = self.get_ann_info(xml_path)
        # bboxes = self.resize_bboxes(ann['bboxes'],scale_factor)
        mask = bboxes2mask(self.image_size, ann['bboxes'])
        return torch.from_numpy(mask).permute(2,0,1), objects


    def get_mask_from_seg(self, xml_path, scale_factor):

        DomTree = xml.dom.minidom.parse(xml_path)
        annotation = DomTree.documentElement
        filenamelist = annotation.getElementsByTagName('filename')
        filename = filenamelist[0].childNodes[0].data
        objectlist = annotation.getElementsByTagName('object')
        height, width = self.image_size[:2]
        mask = np.ones((height, width, 1), dtype=np.uint8)
        for objects in objectlist:
            namelist = objects.getElementsByTagName('name')
            objectname = namelist[0].childNodes[0].data
            segm = objects.getElementsByTagName('segm')                    
            one_segm_points_list = []
            for one_segm in segm:                
                points = one_segm.getElementsByTagName('point')               
                for point in points:
                    
                    x = point.childNodes[0].data.split(',')[0]
                    y = point.childNodes[0].data.split(',')[1]
                    
                    one_segm_points_list.append([x, y])
         
            pts = np.array(one_segm_points_list, np.int32)
            pts = pts * scale_factor
            pts = np.fix(pts).astype(np.int) 

            cv2.fillPoly(mask, [pts], 0)

        return torch.from_numpy(mask).permute(2,0,1)     


    def resize_bboxes(self, bboxes, scale_factor):
        """Resize bounding boxes with ``results['scale_factor']``."""
        bboxes = bboxes * scale_factor
        return np.fix(bboxes).astype(np.int) 



    def get_ann_info(self, xml_path):
        """Get annotation from XML file by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.CLASSES:
                continue
            label = self.cat2label[name]
            bnd_box = obj.find('bndbox')
            # TODO: check whether it is necessary to use int
            # Coordinates may be float type
            bbox = [
                int(float(bnd_box.find('xmin').text)),
                int(float(bnd_box.find('ymin').text)),
                int(float(bnd_box.find('xmax').text)),
                int(float(bnd_box.find('ymax').text))
            ]
            ignore = False
 
            bboxes.append(bbox)
            labels.append(label)
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)

        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64))
        return ann

    def genxml(self,ann_savepath='VOC2007/Annotations/',ann=None):
        #import pdb;pdb.set_trace()
        filename = ann['filename']
        objs = []
        bboxes = ann['objects']['bboxes']
        labels = ann['objects']['labels']
        for i, bbox in enumerate(bboxes):
            name = self.CLASSES[labels[i]]

            xmin = (int)(bbox[0])
            ymin = (int)(bbox[1])
            xmax = (int)(bbox[2])
            ymax = (int)(bbox[3])
            obj = [name, xmin, ymin, xmax, ymax]
            if not(xmin-xmax==0 or ymin-ymax==0):
                objs.append(obj)  
        annopath = os.path.join(ann_savepath,filename[:-3] + "xml") 
       

        E = objectify.ElementMaker(annotate=False)
        anno_tree = E.annotation(
            E.folder('VOC'),
            E.filename(filename),
            E.source(
                E.database('ShipGo'),
                E.annotation('VOC'),
                E.image('ShipGo')
            ),
            E.size(
                E.width(ann['weight']),
                E.height(ann['height']),
                E.depth(ann['depth'])
            ),
            E.segmented(0)
        )

        for obj in objs:
            E2 = objectify.ElementMaker(annotate=False)
            anno_tree2 = E2.object(
                E.name(obj[0]),
                E.pose(),
                E.truncated("0"),
                E.difficult(0),
                E.bndbox(
                    E.xmin(obj[1]),
                    E.ymin(obj[2]),
                    E.xmax(obj[3]),
                    E.ymax(obj[4])
                )
            )
            anno_tree.append(anno_tree2)
        etree.ElementTree(anno_tree).write(annopath, pretty_print=True)

#For HRSID
class I2IHRSIDDataset(data.Dataset):
    CLASSES = ('ship',)
    ENVS = ('offshore','inshore_1','inshore_2')
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader,output_anns=None, out_anns_path=None):
        # import pdb;pdb.set_trace()
        imgs = make_dataset(data_root)
        self.ENV_image_list = {env: make_dataset(os.path.join(data_root.split('flist')[0], 'flist',env+'.txt')) for env in self.ENVS} 
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size
        self.env2label = {env: i for i, env in enumerate(self.ENVS)}
        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        self.output_anns = output_anns
        self.out_anns_path = out_anns_path
        if self.output_anns:
            if not os.path.exists(self.out_anns_path ): 
                os.mkdir(self.out_anns_path)

    def __getitem__(self, index):
        ret = {}
        ann = {}
        path = self.imgs[index]

        img = self.loader(path)
        img_size = np.array(img.size) #(w,h)
        img = self.tfs(img)
        img_tf_size = img.shape[1:] #(h,w)
        w_scale = img_tf_size[1]/img_size[0]
        h_scale = img_tf_size[0]/img_size[1]
        ann['height'] = img_tf_size[0]
        ann['weight'] = img_tf_size[1]
        ann['depth'] = 1 # for SAR images
        ann['filename'] ='Out_'+path.split('/')[-1]
        #import pdb;pdb.set_trace()


        if self.mask_mode == 'det':
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],dtype=np.float32)
            ann_path = path.replace('JPEGImages','Annotations').replace('jpg','xml')
            mask, objects = self.get_mask_from_det(ann_path, scale_factor)
            if self.output_anns:
                ann['objects'] = objects
                self.genxml(self.out_anns_path, ann)


        elif self.mask_mode == 'seg':
            scale_factor = np.array([w_scale, h_scale],dtype=np.float32)
            ann_path = path.replace('JPEGImages','Annotations_seg').replace('jpg','xml')
            mask = self.get_mask_from_seg(ann_path, scale_factor)
            if self.output_anns:
                det_ann_path = path.replace('JPEGImages','Annotations').replace('jpg','xml')
                objects = self.resize_det_anns(det_ann_path, np.array([w_scale, h_scale, w_scale, h_scale],dtype=np.float32))
                ann['objects'] = objects
                self.genxml(self.out_anns_path, ann)



        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        #ENVS:
        for env in self.ENV_image_list.keys():
            if path.split('/')[-1] in self.ENV_image_list[env]:
                image_env = self.env2label[env]
                break

        # mask_img = (mask_img + 1)*127
        # cv2.imwrite('mask_img.jpg',mask_img.permute(1,2,0).numpy())
        # image_env = 2
        ret['gt_image'] = img[0].unsqueeze(0)
        ret['cond_image'] = cond_image[0].unsqueeze(0)
        ret['mask_image'] = mask_img[0].unsqueeze(0)
        ret['mask'] = mask[0].unsqueeze(0)
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        ret['image_env'] = image_env

        return ret

    def __len__(self):
        return len(self.imgs)

    def resize_det_anns(self,xml_path,scale_factor):
        objects = self.get_ann_info(xml_path)
        #import pdb;pdb.set_trace()
        bboxes = self.resize_bboxes(objects['bboxes'],scale_factor)
        objects['bboxes'] = bboxes
        return objects


    def get_mask_from_det(self, xml_path, scale_factor):
        objects = self.resize_det_anns(xml_path, scale_factor)
        #import pdb;pdb.set_trace()
        # ann = self.get_ann_info(xml_path)
        # bboxes = self.resize_bboxes(ann['bboxes'],scale_factor)
        mask = bboxes2mask(self.image_size, ann['bboxes'])
        return torch.from_numpy(mask).permute(2,0,1), objects


    def get_mask_from_seg(self, xml_path, scale_factor):

        DomTree = xml.dom.minidom.parse(xml_path)
        annotation = DomTree.documentElement
        filenamelist = annotation.getElementsByTagName('filename')
        filename = filenamelist[0].childNodes[0].data
        objectlist = annotation.getElementsByTagName('object')
        # size = annotation.getElementsByTagName('size')
        # height = int(size[0].getElementsByTagName('height')[0].childNodes[0].data)
        # width = int(size[0].getElementsByTagName('height')[0].childNodes[0].data)
        height, width = self.image_size[:2]

        mask = np.ones((height, width, 1), dtype=np.uint8)

        #im = np.zeros([im.shape[0], im.shape[1], 3], )
        
        for objects in objectlist:
            namelist = objects.getElementsByTagName('name')
            objectname = namelist[0].childNodes[0].data
            segm = objects.getElementsByTagName('segm')                    
            one_segm_points_list = []
            
            for one_segm in segm:
                
                points = one_segm.getElementsByTagName('point')
                
                for point in points:
                    
                    x = point.childNodes[0].data.split(',')[0]
                    y = point.childNodes[0].data.split(',')[1]
                    
                    one_segm_points_list.append([x, y])
    
            
            
            pts = np.array(one_segm_points_list, np.int32)
            pts = pts * scale_factor
            pts = np.fix(pts).astype(np.int) 

            cv2.fillPoly(mask, [pts], 0)

        return torch.from_numpy(mask).permute(2,0,1)     


    def resize_bboxes(self, bboxes, scale_factor):
        """Resize bounding boxes with ``results['scale_factor']``."""
        bboxes = bboxes * scale_factor
        return np.fix(bboxes).astype(np.int) 



    def get_ann_info(self, xml_path):
        """Get annotation from XML file by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.CLASSES:
                continue
            label = self.cat2label[name]
            bnd_box = obj.find('bndbox')
            # TODO: check whether it is necessary to use int
            # Coordinates may be float type
            bbox = [
                int(float(bnd_box.find('xmin').text)),
                int(float(bnd_box.find('ymin').text)),
                int(float(bnd_box.find('xmax').text)),
                int(float(bnd_box.find('ymax').text))
            ]
            ignore = False
 
            bboxes.append(bbox)
            labels.append(label)
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)

        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64))
        return ann

    # def get_mask(self,bboxs):
    #     mask = bboxes2mask(self.image_size, bboxs)
    #     return torch.from_numpy(mask).permute(2,0,1)



    def genxml(self,ann_savepath='VOC2007/Annotations/',ann=None):
        #import pdb;pdb.set_trace()
        filename = ann['filename']
        objs = []
        bboxes = ann['objects']['bboxes']
        labels = ann['objects']['labels']
        for i, bbox in enumerate(bboxes):
            name = self.CLASSES[labels[i]]

            xmin = (int)(bbox[0])
            ymin = (int)(bbox[1])
            xmax = (int)(bbox[2])
            ymax = (int)(bbox[3])
            obj = [name, xmin, ymin, xmax, ymax]
            if not(xmin-xmax==0 or ymin-ymax==0):
                objs.append(obj)  
        annopath = os.path.join(ann_savepath,filename[:-3] + "xml") 
       

        E = objectify.ElementMaker(annotate=False)
        anno_tree = E.annotation(
            E.folder('VOC'),
            E.filename(filename),
            E.source(
                E.database('ShipGo'),
                E.annotation('VOC'),
                E.image('ShipGo')
            ),
            E.size(
                E.width(ann['weight']),
                E.height(ann['height']),
                E.depth(ann['depth'])
            ),
            E.segmented(0)
        )

        for obj in objs:
            E2 = objectify.ElementMaker(annotate=False)
            anno_tree2 = E2.object(
                E.name(obj[0]),
                E.pose(),
                E.truncated("0"),
                E.difficult(0),
                E.bndbox(
                    E.xmin(obj[1]),
                    E.ymin(obj[2]),
                    E.xmax(obj[3]),
                    E.ymax(obj[4])
                )
            )
            anno_tree.append(anno_tree2)
        etree.ElementTree(anno_tree).write(annopath, pretty_print=True)


#For MSTAR
class I2IMSTARDataset(data.Dataset):
    CLASSES = ('1', '7', '8', '5', '4', '0', '3', '2', '6', '9')
    ENVS = ('grass',)
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader,output_anns=None, out_anns_path=None):
        imgs = make_dataset(data_root)
        # import pdb;pdb.set_trace()
        self.ENV_image_list = {env: make_dataset(os.path.join(data_root.split('flist')[0], 'flist',env+'.txt')) for env in self.ENVS} 
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size
        self.env2label = {env: i for i, env in enumerate(self.ENVS)}
        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        self.output_anns = output_anns
        self.out_anns_path = out_anns_path
        if self.output_anns:
            if not os.path.exists(self.out_anns_path ): 
                os.mkdir(self.out_anns_path)
    def __getitem__(self, index):
        ret = {}
        ann = {}
        path = self.imgs[index]

        img = self.loader(path)
        img_size = np.array(img.size) #(w,h)
        img = self.tfs(img)
        img_tf_size = img.shape[1:] #(h,w)
        w_scale = img_tf_size[1]/img_size[0]
        h_scale = img_tf_size[0]/img_size[1]
        ann['height'] = img_tf_size[0]
        ann['weight'] = img_tf_size[1]
        ann['depth'] = 1 # for SAR images
        ann['filename'] ='Out_'+path.split('/')[-1]
        #import pdb;pdb.set_trace()

        if self.mask_mode == 'det':
            # import pdb;pdb.set_trace()
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],dtype=np.float32)

            ann_path = path.replace('JPEGImages','Annotations').replace('jpg','xml')
            mask, objects = self.get_mask_from_det(ann_path, scale_factor)
            if self.output_anns:
                ann['objects'] = objects
                self.genxml(self.out_anns_path, ann)
        elif self.mask_mode == 'seg':
            scale_factor = np.array([w_scale, h_scale],dtype=np.float32)
            ann_path = path.replace('image','Annotations_seg').replace('jpg','xml')
            mask = self.get_mask_from_seg(ann_path, scale_factor)
            if self.output_anns:
                det_ann_path = path.replace('image','Annotations').replace('jpg','xml')
                objects = self.resize_det_anns(det_ann_path, np.array([w_scale, h_scale, w_scale, h_scale],dtype=np.float32))
                ann['objects'] = objects
                self.genxml(self.out_anns_path, ann)



        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        #ENVS:
        for env in self.ENV_image_list.keys():
            if path.split('/')[-1] in self.ENV_image_list[env]:
                image_env = self.env2label[env]
                break

        # mask_img = (mask_img + 1)*127
        # cv2.imwrite('mask_img.jpg',mask_img.permute(1,2,0).numpy())
        # image_env = 1



        ret['gt_image'] = img[0].unsqueeze(0)
        ret['cond_image'] = cond_image[0].unsqueeze(0)
        ret['mask_image'] = mask_img[0].unsqueeze(0)
        ret['mask'] = mask[0].unsqueeze(0)
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        ret['image_env'] = image_env
        return ret

    def __len__(self):
        return len(self.imgs)

    def resize_det_anns(self,xml_path,scale_factor):
        # import pdb;pdb.set_trace()

        objects = self.get_ann_info(xml_path)
        #import pdb;pdb.set_trace()
        bboxes = self.resize_bboxes(objects['bboxes'],scale_factor)
        objects['bboxes'] = bboxes
        return objects


    def get_mask_from_det(self, xml_path, scale_factor):
        objects = self.resize_det_anns(xml_path, scale_factor)
        # import pdb;pdb.set_trace()
        # ann = self.get_ann_info(xml_path)
        # bboxes = self.resize_bboxes(ann['bboxes'],scale_factor)
        mask = bboxes2mask(self.image_size, objects['bboxes'])
        return torch.from_numpy(mask).permute(2,0,1), objects


    def get_mask_from_seg(self, xml_path, scale_factor):

        DomTree = xml.dom.minidom.parse(xml_path)
        annotation = DomTree.documentElement
        filenamelist = annotation.getElementsByTagName('filename')
        filename = filenamelist[0].childNodes[0].data
        objectlist = annotation.getElementsByTagName('object')
        # size = annotation.getElementsByTagName('size')
        # height = int(size[0].getElementsByTagName('height')[0].childNodes[0].data)
        # width = int(size[0].getElementsByTagName('height')[0].childNodes[0].data)
        height, width = self.image_size[:2]

        mask = np.ones((height, width, 1), dtype=np.uint8)

        #im = np.zeros([im.shape[0], im.shape[1], 3], )
        
        for objects in objectlist:
            namelist = objects.getElementsByTagName('name')
            objectname = namelist[0].childNodes[0].data
            segm = objects.getElementsByTagName('segm')                    
            one_segm_points_list = []
            
            for one_segm in segm:
                
                points = one_segm.getElementsByTagName('point')
                
                for point in points:
                    
                    x = point.childNodes[0].data.split(',')[0]
                    y = point.childNodes[0].data.split(',')[1]
                    
                    one_segm_points_list.append([x, y])
    
            
            
            pts = np.array(one_segm_points_list, np.int32)
            pts = pts * scale_factor
            pts = np.fix(pts).astype(np.int) 

            cv2.fillPoly(mask, [pts], 0)

        return torch.from_numpy(mask).permute(2,0,1)     


    def resize_bboxes(self, bboxes, scale_factor):
        """Resize bounding boxes with ``results['scale_factor']``."""
        bboxes = bboxes * scale_factor
        return np.fix(bboxes).astype(np.int) 



    def get_ann_info(self, xml_path):
        """Get annotation from XML file by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        # import pdb;pdb.set_trace()
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.CLASSES:
                continue
            label = self.cat2label[name]
            bnd_box = obj.find('bndbox')
            # TODO: check whether it is necessary to use int
            # Coordinates may be float type
            bbox = [
                int(float(bnd_box.find('xmin').text)),
                int(float(bnd_box.find('ymin').text)),
                int(float(bnd_box.find('xmax').text)),
                int(float(bnd_box.find('ymax').text))
            ]
            ignore = False
 
            bboxes.append(bbox)
            labels.append(label)
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)

        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64))
        return ann

    # def get_mask(self,bboxs):
    #     mask = bboxes2mask(self.image_size, bboxs)
    #     return torch.from_numpy(mask).permute(2,0,1)



    def genxml(self,ann_savepath='VOC2007/Annotations/',ann=None):
        #import pdb;pdb.set_trace()
        filename = ann['filename']
        objs = []
        bboxes = ann['objects']['bboxes']
        labels = ann['objects']['labels']
        for i, bbox in enumerate(bboxes):
            name = self.CLASSES[labels[i]]

            xmin = (int)(bbox[0])
            ymin = (int)(bbox[1])
            xmax = (int)(bbox[2])
            ymax = (int)(bbox[3])
            obj = [name, xmin, ymin, xmax, ymax]
            if not(xmin-xmax==0 or ymin-ymax==0):
                objs.append(obj)  
        annopath = os.path.join(ann_savepath,filename[:-3] + "xml") 
       

        E = objectify.ElementMaker(annotate=False)
        anno_tree = E.annotation(
            E.folder('VOC'),
            E.filename(filename),
            E.source(
                E.database('ShipGo'),
                E.annotation('VOC'),
                E.image('ShipGo')
            ),
            E.size(
                E.width(ann['weight']),
                E.height(ann['height']),
                E.depth(ann['depth'])
            ),
            E.segmented(0)
        )

        for obj in objs:
            E2 = objectify.ElementMaker(annotate=False)
            anno_tree2 = E2.object(
                E.name(obj[0]),
                E.pose(),
                E.truncated("0"),
                E.difficult(0),
                E.bndbox(
                    E.xmin(obj[1]),
                    E.ymin(obj[2]),
                    E.xmax(obj[3]),
                    E.ymax(obj[4])
                )
            )
            anno_tree.append(anno_tree2)
        etree.ElementTree(anno_tree).write(annopath, pretty_print=True)
