from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models
import warnings
import math
from PIL import Image
from torchvision.transforms import functional as TF # Added for functional transforms
import random # Added for random decisions

warnings.filterwarnings("ignore")



                 #dir_gainDPM="gain/DPM/", 
                 #dir_gainDPMcars="gain/carsDPM/", 
                 #dir_gainIRT2="gain/IRT2/", 
                 #dir_gainIRT2cars="gain/carsIRT2/", 
                 #dir_buildings="png/", 
                 #dir_antenna= , 
                    

class RadioUNet_c(Dataset):
    """RadioMapSeer Loader for accurate buildings and no measurements (RadioUNet_c)"""
    def __init__(self,maps_inds=np.zeros(1), phase="train",
                 ind1=0,ind2=0, 
                 dir_dataset="dataset/",
                 numTx=80,                   
                 thresh=0.2,
                 simulation="DPM",
                 carsSimul="no",
                 carsInput="no",
                 IRT2maxW=1,
                 cityMap="complete",
                 missing=1,
                 transform= transforms.ToTensor()):
        """
        Args:
            maps_inds: optional shuffled sequence of the maps. Leave it as maps_inds=0 (default) for the standart split.
            phase:"train", "val", "test", "custom". If "train", "val" or "test", uses a standard split.
                  "custom" means that the loader will read maps ind1 to ind2 from the list maps_inds.
            ind1,ind2: First and last indices from maps_inds to define the maps of the loader, in case phase="custom". 
            dir_dataset: directory of the RadioMapSeer dataset.
            numTx: Number of transmitters per map. Default and maximal value of numTx = 80.                 
            thresh: Pathlos threshold between 0 and 1. Defaoult is the noise floor 0.2.
            simulation:"DPM", "IRT2", "rand". Default= "DPM"
            carsSimul:"no", "yes". Use simulation with or without cars. Default="no".
            carsInput:"no", "yes". Take inputs with or without cars channel. Default="no".
            IRT2maxW: in case of "rand" simulation, the maximal weight IRT2 can take. Default=1.
            cityMap: "complete", "missing", "rand". Use the full city, or input map with missing buildings "rand" means that there is 
                      a random number of missing buildings.
            missing: 1 to 4. in case of input map with missing buildings, and not "rand", the number of missing buildings. Default=1.
            transform: Transform to apply on the images of the loader.  Default= transforms.ToTensor())
                 
        Output:
            inputs: The RadioUNet inputs.  
            image_gain
            
        """
        

        
        #self.phase=phase
                
        if maps_inds.size==1:
            self.maps_inds=np.arange(0,700,1,dtype=np.int16)
            #Determenistic "random" shuffle of the maps:
            np.random.seed(42)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds=maps_inds
            
        if phase=="train":
            self.ind1=0
            self.ind2=500
        elif phase=="val":
            self.ind1=501
            self.ind2=600
        elif phase=="test":
            self.ind1=601
            self.ind2=699
        else: # custom range
            self.ind1=ind1
            self.ind2=ind2
            
        self.dir_dataset = dir_dataset
        self.numTx =  numTx                
        self.thresh=thresh
        
        self.simulation=simulation
        self.carsSimul=carsSimul
        self.carsInput=carsInput
        if simulation=="DPM" :
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/DPM/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsDPM/"
        elif simulation=="IRT2":
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsIRT2/"
        elif  simulation=="rand":
            if carsSimul=="no":
                self.dir_gainDPM=self.dir_dataset+"gain/DPM/"
                self.dir_gainIRT2=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gainDPM=self.dir_dataset+"gain/carsDPM/"
                self.dir_gainIRT2=self.dir_dataset+"gain/carsIRT2/"
        
        self.IRT2maxW=IRT2maxW
        
        self.cityMap=cityMap
        self.missing=missing
        if cityMap=="complete":
            self.dir_buildings=self.dir_dataset+"png/buildings_complete/"
            print(self.dir_buildings)
        else:
            self.dir_buildings = self.dir_dataset+"png/buildings_missing" # a random index will be concatenated in the code
        #else:  #missing==number
        #    self.dir_buildings = self.dir_dataset+ "png/buildings_missing"+str(missing)+"/"
        #print(self.dir_buildings)
              
        self.transform= transform
        
        self.dir_Tx = self.dir_dataset+ "png/antennas/" 
        #later check if reading the JSON file and creating antenna images on the fly is faster
        if carsInput!="no":
            self.dir_cars = self.dir_dataset+ "png/cars/" 
        
        self.height = 256
        self.width = 256

        
    def __len__(self):
        return (self.ind2-self.ind1+1)*self.numTx
    
    def __getitem__(self, idx):
        
        idxr=np.floor(idx/self.numTx).astype(int)
        idxc=idx-idxr*self.numTx 
        dataset_map_ind=self.maps_inds[idxr+self.ind1]+1
        #names of files that depend only on the map:
        name1 = str(dataset_map_ind) + ".png"
        #names of files that depend on the map and the Tx:
        name2 = str(dataset_map_ind) + "_" + str(idxc) + ".png"
        
        #Load buildings:
        if self.cityMap == "complete":
            img_name_buildings = os.path.join(self.dir_buildings, name1)
        else:
            if self.cityMap == "rand":
                self.missing=np.random.randint(low=1, high=5)
            version=np.random.randint(low=1, high=7)
            img_name_buildings = os.path.join(self.dir_buildings+str(self.missing)+"/"+str(version)+"/", name1)
            
            str(self.missing)
        image_buildings = np.asarray(io.imread(img_name_buildings))   
        
        #Load Tx (transmitter):
        img_name_Tx = os.path.join(self.dir_Tx, name2)
        image_Tx = np.asarray(io.imread(img_name_Tx))
        
        #Load radio map:
        if self.simulation!="rand":
            img_name_gain = os.path.join(self.dir_gain, name2)  
            image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain)),axis=2)/255
        else: #random weighted average of DPM and IRT2
            img_name_gainDPM = os.path.join(self.dir_gainDPM, name2) 
            img_name_gainIRT2 = os.path.join(self.dir_gainIRT2, name2) 
            #image_gainDPM = np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/255
            #image_gainIRT2 = np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/255
            w=np.random.uniform(0,self.IRT2maxW) # IRT2 weight of random average
            image_gain= w*np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/256  \
                        + (1-w)*np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/256
        
        #pathloss threshold transform
        if self.thresh>0:
            mask = image_gain < self.thresh
            image_gain[mask]=self.thresh
            image_gain=image_gain-self.thresh*np.ones(np.shape(image_gain))
            image_gain=image_gain/(1-self.thresh)
                 
        
        #inputs to radioUNet
        if self.carsInput=="no":
            inputs=np.stack([image_buildings, image_Tx], axis=2)        
            #The fact that the buildings and antenna are normalized  256 and not 1 promotes convergence, 
            #so we can use the same learning rate as RadioUNets
        else: #cars
            #Normalization, so all settings can have the same learning rate
            image_buildings=image_buildings/256
            image_Tx=image_Tx/256
            img_name_cars = os.path.join(self.dir_cars, name1)
            image_cars = np.asarray(io.imread(img_name_cars))/256
            inputs=np.stack([image_buildings, image_Tx, image_cars], axis=2)
            #note that ToTensor moves the channel from the last asix to the first!

        
        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            image_gain = self.transform(image_gain).type(torch.float32)
            #note that ToTensor moves the channel from the last asix to the first!


        return (inputs, image_gain, name1)
    
    
    
    

class RadioUNet_c_sprseIRT4(Dataset):
    """RadioMapSeer Loader for accurate buildings and no measurements (RadioUNet_c)"""
    def __init__(self,maps_inds=np.zeros(1), phase="train",
                 ind1=0,ind2=0, 
                 dir_dataset="RadioMapSeer/",
                 numTx=2,                  
                 thresh=0.2,
                 simulation="IRT4",
                 carsSimul="no",
                 carsInput="no",
                 cityMap="complete",
                 missing=1,
                 num_samples=300,
                 transform= transforms.ToTensor()):
        """
        Args:
            maps_inds: optional shuffled sequence of the maps. Leave it as maps_inds=0 (default) for the standart split.
            phase:"train", "val", "test", "custom". If "train", "val" or "test", uses a standard split.
                  "custom" means that the loader will read maps ind1 to ind2 from the list maps_inds.
            ind1,ind2: First and last indices from maps_inds to define the maps of the loader, in case phase="custom". 
            dir_dataset: directory of the RadioMapSeer dataset.
            numTx: Number of transmitters per map. Default = 2. Note that IRT4 works only with numTx<=2.                
            thresh: Pathlos threshold between 0 and 1. Defaoult is the noise floor 0.2.
            simulation: default="IRT4", with an option to "DPM", "IRT2".
            carsSimul:"no", "yes". Use simulation with or without cars. Default="no".
            carsInput:"no", "yes". Take inputs with or without cars channel. Default="no".
            cityMap: "complete", "missing", "rand". Use the full city, or input map with missing buildings "rand" means that there is 
                      a random number of missing buildings.
            missing: 1 to 4. in case of input map with missing buildings, and not "rand", the number of missing buildings. Default=1.
            num_samples: number of samples in the sparse IRT4 radio map. Default=300.
            transform: Transform to apply on the images of the loader.  Default= transforms.ToTensor())
            
        Output:
            
        """
        if maps_inds.size==1:
            self.maps_inds=np.arange(0,700,1,dtype=np.int16)
            #Determenistic "random" shuffle of the maps:
            np.random.seed(42)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds=maps_inds
            
        if phase=="train":
            self.ind1=0
            self.ind2=500
        elif phase=="val":
            self.ind1=501
            self.ind2=600
        elif phase=="test":
            self.ind1=601
            self.ind2=699
        else: # custom range
            self.ind1=ind1
            self.ind2=ind2
            
        self.dir_dataset = dir_dataset
        self.numTx=  numTx                
        self.thresh=thresh
        
        self.simulation=simulation
        self.carsSimul=carsSimul
        self.carsInput=carsInput
        if simulation=="IRT4":
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/IRT4/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsIRT4/"
        
        elif simulation=="DPM" :
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/DPM/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsDPM/"
        elif simulation=="IRT2":
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsIRT2/"  
        
        
        self.cityMap=cityMap
        self.missing=missing
        if cityMap=="complete":
            self.dir_buildings=self.dir_dataset+"png/buildings_complete/"
        else:
            self.dir_buildings = self.dir_dataset+"png/buildings_missing" # a random index will be concatenated in the code
        #else:  #missing==number
        #    self.dir_buildings = self.dir_dataset+ "png/buildings_missing"+str(missing)+"/"
            
              
        self.transform= transform
        
        self.num_samples=num_samples
        
        self.dir_Tx = self.dir_dataset+ "png/antennas/" 
        #later check if reading the JSON file and creating antenna images on the fly is faster
        if carsInput!="no":
            self.dir_cars = self.dir_dataset+ "png/cars/" 
        
        self.height = 256
        self.width = 256

        
        
        
        
    def __len__(self):
        return (self.ind2-self.ind1+1)*self.numTx
    
    def __getitem__(self, idx):
        
        idxr=np.floor(idx/self.numTx).astype(int)
        idxc=idx-idxr*self.numTx 
        dataset_map_ind=self.maps_inds[idxr+self.ind1]+1
        #names of files that depend only on the map:
        name1 = str(dataset_map_ind) + ".png"
        #names of files that depend on the map and the Tx:
        name2 = str(dataset_map_ind) + "_" + str(idxc) + ".png"
        
        #Load buildings:
        if self.cityMap == "complete":
            img_name_buildings = os.path.join(self.dir_buildings, name1)
        else:
            if self.cityMap == "rand":
                self.missing=np.random.randint(low=1, high=5)
            version=np.random.randint(low=1, high=7)
            img_name_buildings = os.path.join(self.dir_buildings+str(self.missing)+"/"+str(version)+"/", name1)
            str(self.missing)
        image_buildings = np.asarray(io.imread(img_name_buildings))   
        
        #Load Tx (transmitter):
        img_name_Tx = os.path.join(self.dir_Tx, name2)
        image_Tx = np.asarray(io.imread(img_name_Tx))
        
        #Load radio map:
        if self.simulation!="rand":
            img_name_gain = os.path.join(self.dir_gain, name2)  
            image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain)),axis=2)/256
        else: #random weighted average of DPM and IRT2
            img_name_gainDPM = os.path.join(self.dir_gainDPM, name2) 
            img_name_gainIRT2 = os.path.join(self.dir_gainIRT2, name2) 
            #image_gainDPM = np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/255
            #image_gainIRT2 = np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/255
            w=np.random.uniform(0,self.IRT2maxW) # IRT2 weight of random average
            image_gain= w*np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/256  \
                        + (1-w)*np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/256
        
        #pathloss threshold transform
        if self.thresh>0:
            mask = image_gain < self.thresh
            image_gain[mask]=self.thresh
            image_gain=image_gain-self.thresh*np.ones(np.shape(image_gain))
            image_gain=image_gain/(1-self.thresh)
        
        #Saprse IRT4 samples, determenistic and fixed samples per map
        image_samples = np.zeros((self.width,self.height))
        seed_map=np.sum(image_buildings) # Each map has its fixed samples, independent of the transmitter location.
        np.random.seed(seed_map)       
        x_samples=np.random.randint(0, 255, size=self.num_samples)
        y_samples=np.random.randint(0, 255, size=self.num_samples)
        image_samples[x_samples,y_samples]= 1
        
        #inputs to radioUNet
        if self.carsInput=="no":
            inputs=np.stack([image_buildings, image_Tx], axis=2)        
            #The fact that the buildings and antenna are normalized  256 and not 1 promotes convergence, 
            #so we can use the same learning rate as RadioUNets
        else: #cars
            #Normalization, so all settings can have the same learning rate
            image_buildings=image_buildings/256
            image_Tx=image_Tx/256
            img_name_cars = os.path.join(self.dir_cars, name1)
            image_cars = np.asarray(io.imread(img_name_cars))/256
            inputs=np.stack([image_buildings, image_Tx, image_cars], axis=2)
            #note that ToTensor moves the channel from the last asix to the first!
        
        

        
        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            image_gain = self.transform(image_gain).type(torch.float32)
            image_samples = self.transform(image_samples).type(torch.float32)


        return [inputs, image_gain, image_samples]
    
    
    
    
    
    
    
class RadioUNet_s(Dataset):
    """RadioMapSeer Loader for accurate buildings and no measurements (RadioUNet_c)"""
    def __init__(self,maps_inds=np.zeros(1), phase="train",
                 ind1=0,ind2=0, 
                 dir_dataset="RadioUNet/RadioMapSeer/",
                 numTx=80,                  
                 thresh=0.2,
                 simulation="DPM",
                 carsSimul="no",
                 carsInput="no",
                 IRT2maxW=1,
                 cityMap="complete",
                 missing=1,
                 fix_samples=0,
                 num_samples_low= 10, 
                 num_samples_high= 300,
                 transform= transforms.ToTensor()):
        """
        Args:
            maps_inds: optional shuffled sequence of the maps. Leave it as maps_inds=0 (default) for the standart split.
            phase:"train", "val", "test", "custom". If "train", "val" or "test", uses a standard split.
                  "custom" means that the loader will read maps ind1 to ind2 from the list maps_inds.
            ind1,ind2: First and last indices from maps_inds to define the maps of the loader, in case phase="custom". 
            dir_dataset: directory of the RadioMapSeer dataset.
            numTx: Number of transmitters per map. Default and maximal value of numTx = 80.                 
            thresh: Pathlos threshold between 0 and 1. Defaoult is the noise floor 0.2.
            simulation:"DPM", "IRT2", "rand". Default= "DPM"
            carsSimul:"no", "yes". Use simulation with or without cars. Default="no".
            carsInput:"no", "yes". Take inputs with or without cars channel. Default="no".
            IRT2maxW: in case of "rand" simulation, the maximal weight IRT2 can take. Default=1.
            cityMap: "complete", "missing", "rand". Use the full city, or input map with missing buildings "rand" means that there is 
                      a random number of missing buildings.
            missing: 1 to 4. in case of input map with missing buildings, and not "rand", the number of missing buildings. Default=1.
            fix_samples: fixed or a random number of samples. If zero, fixed, else, fix_samples is the number of samples. Default = 0.
            num_samples_low: if random number of samples, this is the minimum number of samples. Default = 10. 
            num_samples_high: if random number of samples, this is the maximal number of samples. Default = 300.
            transform: Transform to apply on the images of the loader.  Default= transforms.ToTensor())
                 
        Output:
            inputs: The RadioUNet inputs.  
            image_gain
            
        """
        

        
        #self.phase=phase
                
        if maps_inds.size==1:
            self.maps_inds=np.arange(0,700,1,dtype=np.int16)
            #Determenistic "random" shuffle of the maps:
            np.random.seed(42)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds=maps_inds
            
        if phase=="train":
            self.ind1=0
            self.ind2=500
        elif phase=="val":
            self.ind1=501
            self.ind2=600
        elif phase=="test":
            self.ind1=601
            self.ind2=699
        else: # custom range
            self.ind1=ind1
            self.ind2=ind2
            
        self.dir_dataset = dir_dataset
        self.numTx=  numTx                
        self.thresh=thresh
        
        self.simulation=simulation
        self.carsSimul=carsSimul
        self.carsInput=carsInput
        if simulation=="DPM" :
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/DPM/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsDPM/"
        elif simulation=="IRT2":
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsIRT2/"
        elif  simulation=="rand":
            if carsSimul=="no":
                self.dir_gainDPM=self.dir_dataset+"gain/DPM/"
                self.dir_gainIRT2=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gainDPM=self.dir_dataset+"gain/carsDPM/"
                self.dir_gainIRT2=self.dir_dataset+"gain/carsIRT2/"
        
        self.IRT2maxW=IRT2maxW
        
        self.cityMap=cityMap
        self.missing=missing
        if cityMap=="complete":
            self.dir_buildings=self.dir_dataset+"png/buildings_complete/"
        else:
            self.dir_buildings = self.dir_dataset+"png/buildings_missing" # a random index will be concatenated in the code
        #else:  #missing==number
        #    self.dir_buildings = self.dir_dataset+ "png/buildings_missing"+str(missing)+"/"
            
         
        self.fix_samples= fix_samples
        self.num_samples_low= num_samples_low 
        self.num_samples_high= num_samples_high
                
        self.transform= transform
        
        self.dir_Tx = self.dir_dataset+ "png/antennas/" 
        #later check if reading the JSON file and creating antenna images on the fly is faster
        if carsInput!="no":
            self.dir_cars = self.dir_dataset+ "png/cars/" 
        
        self.height = 256
        self.width = 256

        
    def __len__(self):
        return (self.ind2-self.ind1+1)*self.numTx
    
    def __getitem__(self, idx):
        
        idxr=np.floor(idx/self.numTx).astype(int)
        idxc=idx-idxr*self.numTx 
        dataset_map_ind=self.maps_inds[idxr+self.ind1]+1
        #names of files that depend only on the map:
        name1 = str(dataset_map_ind) + ".png"
        #names of files that depend on the map and the Tx:
        name2 = str(dataset_map_ind) + "_" + str(idxc) + ".png"
        
        #Load buildings:
        if self.cityMap == "complete":
            img_name_buildings = os.path.join(self.dir_buildings, name1)
        else:
            if self.cityMap == "rand":
                self.missing=np.random.randint(low=1, high=5)
            version=np.random.randint(low=1, high=7)
            img_name_buildings = os.path.join(self.dir_buildings+str(self.missing)+"/"+str(version)+"/", name1)
            str(self.missing)
        image_buildings = np.asarray(io.imread(img_name_buildings))/256  
        
        #Load Tx (transmitter):
        img_name_Tx = os.path.join(self.dir_Tx, name2)
        image_Tx = np.asarray(io.imread(img_name_Tx))/256
        
        #Load radio map:
        if self.simulation!="rand":
            img_name_gain = os.path.join(self.dir_gain, name2)  
            image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain)),axis=2)/256
        else: #random weighted average of DPM and IRT2
            img_name_gainDPM = os.path.join(self.dir_gainDPM, name2) 
            img_name_gainIRT2 = os.path.join(self.dir_gainIRT2, name2) 
            #image_gainDPM = np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/255
            #image_gainIRT2 = np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/255
            w=np.random.uniform(0,self.IRT2maxW) # IRT2 weight of random average
            image_gain= w*np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/256  \
                        + (1-w)*np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/256
        
        #pathloss threshold transform
        if self.thresh>0:
            mask = image_gain < self.thresh
            image_gain[mask]=self.thresh
            image_gain=image_gain-self.thresh*np.ones(np.shape(image_gain))
            image_gain=image_gain/(1-self.thresh)
            
        #image_gain=image_gain*256 # we use this normalization so all RadioUNet methods can have the same learning rate.
                                  # Namely, the loss of RadioUNet_s is 256 the loss of RadioUNet_c
                                  # Important: when evaluating the accuracy, remember to devide the errors by 256!
                 
        #input measurements
        image_samples = np.zeros((256,256))
        if self.fix_samples==0:
            num_samples=np.random.randint(self.num_samples_low, self.num_samples_high, size=1)
        else:
            num_samples=np.floor(self.fix_samples).astype(int)               
        x_samples=np.random.randint(0, 255, size=num_samples)
        y_samples=np.random.randint(0, 255, size=num_samples)
        image_samples[x_samples,y_samples]= image_gain[x_samples,y_samples,0]
        
        #inputs to radioUNet
        if self.carsInput=="no":
            inputs=np.stack([image_buildings, image_Tx, image_samples], axis=2)        
            #The fact that the buildings and antenna are normalized  256 and not 1 promotes convergence, 
            #so we can use the same learning rate as RadioUNets
        else: #cars
            #Normalization, so all settings can have the same learning rate
            img_name_cars = os.path.join(self.dir_cars, name1)
            image_cars = np.asarray(io.imread(img_name_cars))/256
            inputs=np.stack([image_buildings, image_Tx, image_samples, image_cars], axis=2)
            #note that ToTensor moves the channel from the last asix to the first!

        
        
        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            image_gain = self.transform(image_gain).type(torch.float32)
            #note that ToTensor moves the channel from the last asix to the first!


        return (inputs, image_gain, num_samples)
    
    
    
    

class RadioUNet_s_sprseIRT4(Dataset):
    """RadioMapSeer Loader for accurate buildings and no measurements (RadioUNet_c)"""
    def __init__(self,maps_inds=np.zeros(1), phase="train",
                 ind1=0,ind2=0, 
                 dir_dataset="RadioMapSeer/",
                 numTx=2,                  
                 thresh=0.2,
                 simulation="IRT4",
                 carsSimul="no",
                 carsInput="no",
                 cityMap="complete",
                 missing=1,
                 data_samples=300,
                 fix_samples=0,
                 num_samples_low= 10, 
                 num_samples_high= 299,
                 transform= transforms.ToTensor()):
        """
        Args:
            maps_inds: optional shuffled sequence of the maps. Leave it as maps_inds=0 (default) for the standart split.
            phase:"train", "val", "test", "custom". If "train", "val" or "test", uses a standard split.
                  "custom" means that the loader will read maps ind1 to ind2 from the list maps_inds.
            ind1,ind2: First and last indices from maps_inds to define the maps of the loader, in case phase="custom". 
            dir_dataset: directory of the RadioMapSeer dataset.
            numTx: Number of transmitters per map. Default = 2. Note that IRT4 works only with numTx<=2.                
            thresh: Pathlos threshold between 0 and 1. Defaoult is the noise floor 0.2.
            simulation: default="IRT4", with an option to "DPM", "IRT2".
            carsSimul:"no", "yes". Use simulation with or without cars. Default="no".
            carsInput:"no", "yes". Take inputs with or without cars channel. Default="no".
            cityMap: "complete", "missing", "rand". Use the full city, or input map with missing buildings "rand" means that there is 
                      a random number of missing buildings.
            missing: 1 to 4. in case of input map with missing buildings, and not "rand", the number of missing buildings. Default=1.
            data_samples: number of samples in the sparse IRT4 radio map. Default=300. All input samples are taken from the data_samples
            fix_samples: fixed or a random number of samples. If zero, fixed, else, fix_samples is the number of samples. Default = 0.
            num_samples_low: if random number of samples, this is the minimum number of samples. Default = 10. 
            num_samples_high: if random number of samples, this is the maximal number of samples. Default = 300.
            transform: Transform to apply on the images of the loader.  Default= transforms.ToTensor())
            
        Output:
            
        """
        if maps_inds.size==1:
            self.maps_inds=np.arange(0,700,1,dtype=np.int16)
            #Determenistic "random" shuffle of the maps:
            np.random.seed(42)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds=maps_inds
            
        if phase=="train":
            self.ind1=0
            self.ind2=500
        elif phase=="val":
            self.ind1=501
            self.ind2=600
        elif phase=="test":
            self.ind1=601
            self.ind2=699
        else: # custom range
            self.ind1=ind1
            self.ind2=ind2
            
        self.dir_dataset = dir_dataset
        self.numTx=  numTx                
        self.thresh=thresh
        
        self.simulation=simulation
        self.carsSimul=carsSimul
        self.carsInput=carsInput
        if simulation=="IRT4":
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/IRT4/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsIRT4/"
        
        elif simulation=="DPM" :
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/DPM/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsDPM/"
        elif simulation=="IRT2":
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsIRT2/"  
        
        
        self.cityMap=cityMap
        self.missing=missing
        if cityMap=="complete":
            self.dir_buildings=self.dir_dataset+"png/buildings_complete/"
        else:
            self.dir_buildings = self.dir_dataset+"png/buildings_missing" # a random index will be concatenated in the code
        #else:  #missing==number
        #    self.dir_buildings = self.dir_dataset+ "png/buildings_missing"+str(missing)+"/"
            
         
        self.data_samples=data_samples
        self.fix_samples= fix_samples
        self.num_samples_low= num_samples_low 
        self.num_samples_high= num_samples_high
        
        self.transform= transform
        
        
        self.dir_Tx = self.dir_dataset+ "png/antennas/" 
        #later check if reading the JSON file and creating antenna images on the fly is faster
        if carsInput!="no":
            self.dir_cars = self.dir_dataset+ "png/cars/" 
        
        self.height = 256
        self.width = 256

        
        
        
        
    def __len__(self):
        return (self.ind2-self.ind1+1)*self.numTx
    
    def __getitem__(self, idx):
        
        idxr=np.floor(idx/self.numTx).astype(int)
        idxc=idx-idxr*self.numTx 
        dataset_map_ind=self.maps_inds[idxr+self.ind1]+1
        #names of files that depend only on the map:
        name1 = str(dataset_map_ind) + ".png"
        #names of files that depend on the map and the Tx:
        name2 = str(dataset_map_ind) + "_" + str(idxc) + ".png"
        
        #Load buildings:
        if self.cityMap == "complete":
            img_name_buildings = os.path.join(self.dir_buildings, name1)
        else:
            if self.cityMap == "rand":
                self.missing=np.random.randint(low=1, high=5)
            version=np.random.randint(low=1, high=7)
            img_name_buildings = os.path.join(self.dir_buildings+str(self.missing)+"/"+str(version)+"/", name1)
            str(self.missing)
        image_buildings = np.asarray(io.imread(img_name_buildings))  #Will be normalized later, after random seed is computed from it
        
        #Load Tx (transmitter):
        img_name_Tx = os.path.join(self.dir_Tx, name2)
        image_Tx = np.asarray(io.imread(img_name_Tx))/256 
        
        #Load radio map:
        if self.simulation!="rand":
            img_name_gain = os.path.join(self.dir_gain, name2)  
            image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain)),axis=2)/256
        else: #random weighted average of DPM and IRT2
            img_name_gainDPM = os.path.join(self.dir_gainDPM, name2) 
            img_name_gainIRT2 = os.path.join(self.dir_gainIRT2, name2) 
            #image_gainDPM = np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/255
            #image_gainIRT2 = np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/255
            w=np.random.uniform(0,self.IRT2maxW) # IRT2 weight of random average
            image_gain= w*np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/256  \
                        + (1-w)*np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/256
        
        #pathloss threshold transform
        if self.thresh>0:
            mask = image_gain < self.thresh
            image_gain[mask]=self.thresh
            image_gain=image_gain-self.thresh*np.ones(np.shape(image_gain))
            image_gain=image_gain/(1-self.thresh)
        
        image_gain=image_gain*256 # we use this normalization so all RadioUNet methods can have the same learning rate.
                                  # Namely, the loss of RadioUNet_s is 256 the loss of RadioUNet_c
                                  # Important: when evaluating the accuracy, remember to devide the errors by 256!
                    
        #Saprse IRT4 samples, determenistic and fixed samples per map
        sparse_samples = np.zeros((self.width,self.height))
        seed_map=np.sum(image_buildings) # Each map has its fixed samples, independent of the transmitter location.
        np.random.seed(seed_map)       
        x_samples=np.random.randint(0, 255, size=self.data_samples)
        y_samples=np.random.randint(0, 255, size=self.data_samples)
        sparse_samples[x_samples,y_samples]= 1
        
        #input samples from the sparse gain samples
        input_samples = np.zeros((256,256))
        if self.fix_samples==0:
            num_in_samples=np.random.randint(self.num_samples_low, self.num_samples_high, size=1)
        else:
            num_in_samples=np.floor(self.fix_samples).astype(int)
            
        data_inds=range(self.data_samples)
        input_inds=np.random.permutation(data_inds)[0:num_in_samples[0]]      
        x_samples_in=x_samples[input_inds]
        y_samples_in=y_samples[input_inds]
        input_samples[x_samples_in,y_samples_in]= image_gain[x_samples_in,y_samples_in,0]
        
        #normalize image_buildings, after random seed computed from it as an int
        image_buildings=image_buildings/256
        
        #inputs to radioUNet
        if self.carsInput=="no":
            inputs=np.stack([image_buildings, image_Tx, input_samples], axis=2)        
            #The fact that the buildings and antenna are normalized  256 and not 1 promotes convergence, 
            #so we can use the same learning rate as RadioUNets
        else: #cars
            #Normalization, so all settings can have the same learning rate
            img_name_cars = os.path.join(self.dir_cars, name1)
            image_cars = np.asarray(io.imread(img_name_cars))/256
            inputs=np.stack([image_buildings, image_Tx, input_samples, image_cars], axis=2)
            #note that ToTensor moves the channel from the last asix to the first!
        
        

        
        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            image_gain = self.transform(image_gain).type(torch.float32)
            sparse_samples = self.transform(sparse_samples).type(torch.float32)
            


        return [inputs, image_gain, sparse_samples]
    
    

    

    

#  --- Normalization Parameters ---
# IMPORTANT: These are placeholders. For best results, you should calculate
# these min/max statistics from your *training dataset only* and then use those
# fixed values for normalizing train, validation, and test sets.
HM_GLOBAL_MIN = 0.0
HM_GLOBAL_MAX = 237.0  # Based on your "max difference is 237m"
RM_GLOBAL_MIN = -200.0 # Based on your augmentation script clipping
RM_GLOBAL_MAX = -20.0


class LunarLoader(Dataset):
    """
    Dataset loader for augmented heightmap, radio map (gain), and TX location
    data stored as .png files.
    Determines files based on original map indices and number of augmentations per map.
    Handles splitting data into train/validation/test phases.
    """

    def __init__(self, root_dir,
                 phase="train",
                 num_original_maps_total=50,
                 augmentations_per_map=20,
                 train_split_ratio=0.7,
                 val_split_ratio=0.15, # Test split will be the remainder
                 random_seed=42,
                 transform=None): # Recommended: pass torchvision_transforms.ToTensor() or a Compose
        """
        Args:
            root_dir (string): Base directory where 'heightmaps', 'radiomaps',
                               'tx_locations' subfolders with .png files are located.
            phase (string): "train", "val", or "test" to select the data split.
            num_original_maps_total (int): Total number of unique original maps
                                           before augmentation.
            augmentations_per_map (int): Number of augmented samples created
                                         for each original map.
            train_split_ratio (float): Proportion of original maps for training.
            val_split_ratio (float): Proportion of original maps for validation.
            random_seed (int): Seed for shuffling original map indices.
            transform (callable, optional): Optional transform to be applied
                                            to each loaded PIL image.
                                            Typically torchvision.transforms.ToTensor() or a
                                            Compose object including it.
        """
        self.root_dir = root_dir
        self.augmentations_per_map = augmentations_per_map
        self.transform = transform
        self.phase = phase # Store phase for error messages

        self.heightmap_dir = os.path.join(root_dir, 'heightmaps')
        self.radiomap_dir = os.path.join(root_dir, 'radiomaps')
        self.tx_location_dir = os.path.join(root_dir, 'tx_locations')
            
        all_original_map_ids = np.arange(num_original_maps_total)
        
        np.random.seed(random_seed)
        np.random.shuffle(all_original_map_ids)

        n_train = int(math.floor(train_split_ratio * num_original_maps_total))
        n_val =int(math.floor(val_split_ratio * num_original_maps_total))
        
        if n_train + n_val > num_original_maps_total:
            n_val = num_original_maps_total - n_train
            if n_val < 0: n_val = 0
        
        if phase == "train":
            self.current_phase_original_map_ids = all_original_map_ids[:n_train]
        elif phase == "val":
            self.current_phase_original_map_ids = all_original_map_ids[n_train : n_train + n_val]
        elif phase == "test":
            self.current_phase_original_map_ids = all_original_map_ids[n_train + n_val:]
        else:
            raise ValueError(f"Invalid phase '{phase}'. Must be 'train', 'val', or 'test'.")

        if len(self.current_phase_original_map_ids) == 0 and num_original_maps_total > 0:
             print(f"Warning: Phase '{phase}' resulted in an empty set of original map IDs. "
                   f"Total original maps: {num_original_maps_total}, "
                   f"Train count used: {n_train}, Val count used: {n_val}. "
                   f"Consider adjusting split ratios or total map count.")
        
        print(f"Dataset phase: '{phase}', using {len(self.current_phase_original_map_ids)} original map IDs for this instance.")


    def __len__(self):
        return len(self.current_phase_original_map_ids) * self.augmentations_per_map

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if not (0 <= idx < self.__len__()):
            if self.__len__() == 0:
                 raise IndexError(f"Cannot get item for index {idx} from an empty dataset (phase: {self.phase}). Check dataset initialization and splits.")
            raise IndexError(f"Index {idx} out of bounds for dataset of length {self.__len__()}")

        original_map_group_idx = idx // self.augmentations_per_map # Integer division
        augmentation_idx = idx % self.augmentations_per_map
        
        actual_original_map_id = self.current_phase_original_map_ids[original_map_group_idx]
        base_name = f"orig{actual_original_map_id}_aug{augmentation_idx}"

        # Update to .png extension
        hm_path = os.path.join(self.heightmap_dir, f'hm_{base_name}.png')
        rm_path = os.path.join(self.radiomap_dir, f'rm_{base_name}.png')
        tx_path = os.path.join(self.tx_location_dir, f'tx_{base_name}.png')

        try:
            # Load PNG images using PIL and ensure grayscale ('L')
            heightmap_pil = Image.open(hm_path).convert('L')
            radiomap_pil = Image.open(rm_path).convert('L')
            tx_location_pil = Image.open(tx_path).convert('L')
        except FileNotFoundError:
            print(f"Error: PNG File not found for generated base_name '{base_name}'.")
            print(f"  Index: {idx}, Original Map Group Idx: {original_map_group_idx}, Actual Original Map ID: {actual_original_map_id}, Augmentation Idx: {augmentation_idx}")
            print(f"  Expected HM path: {hm_path}")
            raise
        except Exception as e:
            print(f"Error loading PNG files for base_name {base_name}: {e}")
            raise

        # --- Apply transform (e.g., ToTensor) or manual conversion ---
        if self.transform:
            heightmap_tensor = self.transform(heightmap_pil)
            radiomap_tensor = self.transform(radiomap_pil)
            tx_location_tensor = self.transform(tx_location_pil)
        else:
            # Manual conversion if no transform is provided
            # (Converts 0-255 PIL Image to 0-1 float tensor with shape [1, H, W])
            def pil_to_tensor(pil_img):
                img_np = np.array(pil_img, dtype=np.float32) / 255.0
                return torch.from_numpy(img_np).unsqueeze(0)

            heightmap_tensor = pil_to_tensor(heightmap_pil)
            radiomap_tensor = pil_to_tensor(radiomap_pil)
            tx_location_tensor = pil_to_tensor(tx_location_pil)
            
        # Ensure tensors are float32
        heightmap_tensor = heightmap_tensor.type(torch.float32)
        radiomap_tensor = radiomap_tensor.type(torch.float32)
        tx_location_tensor = tx_location_tensor.type(torch.float32)

        # Stack inputs: (Heightmap, TX Location Map)
        inputs_tensor = torch.cat((heightmap_tensor, tx_location_tensor), dim=0) # Shape: (2, H, W)
        gain_tensor = radiomap_tensor # Shape: (1, H, W)
            
        return (inputs_tensor, gain_tensor, base_name)
    
    
class LunarLoader2(Dataset):
    """
    Dataset loader for augmented heightmap, radio map (gain), and TX location
    data stored as .npy files.
    It determines which files to load based on an overall index, the number of
    original maps, and the number of augmentations per map. It also handles
    splitting data into train/validation/test phases based on original map indices.
    """

    def __init__(self, root_dir,
                 phase="train",
                 num_original_maps_total=50,
                 augmentations_per_map=20,
                 train_split_ratio=0.7,
                 val_split_ratio=0.15, # Test split will be the remainder
                 random_seed=42,
                 hm_norm_params=(HM_GLOBAL_MIN, HM_GLOBAL_MAX),
                 rm_norm_params=(RM_GLOBAL_MIN, RM_GLOBAL_MAX),
                 transform=None):
        """
        Args:
            root_dir (string): Base directory where 'heightmaps', 'radiomaps',
                               'tx_locations' subfolders are located.
            phase (string): "train", "val", or "test" to select the data split.
            num_original_maps_total (int): Total number of unique original maps
                                           before augmentation (e.g., files indexed 0 to 49).
            augmentations_per_map (int): Number of augmented samples created
                                         for each original map.
            train_split_ratio (float): Proportion of original maps to use for training.
            val_split_ratio (float): Proportion of original maps to use for validation.
            random_seed (int): Seed for shuffling original map indices for reproducible splits.
            hm_norm_params (tuple): (min_val, max_val) for heightmap normalization.
            rm_norm_params (tuple): (min_val, max_val) for radio map normalization.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.augmentations_per_map = augmentations_per_map
        self.transform = transform

        self.heightmap_dir = os.path.join(root_dir, 'heightmaps')
        self.radiomap_dir = os.path.join(root_dir, 'radiomaps')
        self.tx_location_dir = os.path.join(root_dir, 'tx_locations')

        # Normalization parameters
        self.hm_min, self.hm_max = hm_norm_params
        self.rm_min, self.rm_max = rm_norm_params
       
        self.hm_norm_range = (self.hm_max - self.hm_min)
        if self.hm_norm_range == 0:
            print("Warning: Heightmap normalization range is 0. Max and Min are equal. Setting range to 1 to avoid div by zero.")
            self.hm_norm_range = 1.0
       
        self.rm_norm_range = (self.rm_max - self.rm_min)
        if self.rm_norm_range == 0:
            print("Warning: Radio map normalization range is 0. Max and Min are equal. Setting range to 1 to avoid div by zero.")
            self.rm_norm_range = 1.0
           
        # Determine the original map indices for this phase (train/val/test)
        all_original_map_ids = np.arange(num_original_maps_total)
       
        np.random.seed(random_seed)
        np.random.shuffle(all_original_map_ids)

        n_train = int(np.floor(train_split_ratio * num_original_maps_total))
        n_val = int(np.floor(val_split_ratio * num_original_maps_total))
       
        # Ensure splits are sensible
        if n_train + n_val > num_original_maps_total:
            print(f"Warning: Sum of train ({train_split_ratio*100}%) and val ({val_split_ratio*100}%) ratios exceeds 100%. Adjusting validation count.")
            n_val = num_original_maps_total - n_train
            if n_val < 0 : n_val = 0 # Should not happen if n_train is also sensible

        if phase == "train":
            self.current_phase_original_map_ids = all_original_map_ids[:n_train]
        elif phase == "val":
            self.current_phase_original_map_ids = all_original_map_ids[n_train : n_train + n_val]
        elif phase == "test":
            self.current_phase_original_map_ids = all_original_map_ids[n_train + n_val:]
        else:
            raise ValueError(f"Invalid phase '{phase}'. Must be 'train', 'val', or 'test'.")

        if len(self.current_phase_original_map_ids) == 0 and num_original_maps_total > 0:
             print(f"Warning: Phase '{phase}' resulted in an empty set of original map IDs. "
                   f"Total original maps: {num_original_maps_total}, "
                   f"Train count: {n_train}, Val count: {n_val}. "
                   f"Consider adjusting split ratios or total map count.")
       
        print(f"Dataset phase: '{phase}', using {len(self.current_phase_original_map_ids)} original map IDs for this instance.")

    def __len__(self):
        # Total number of samples is the number of selected original maps for this phase * augmentations per map
        return len(self.current_phase_original_map_ids) * self.augmentations_per_map

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if not (0 <= idx < self.__len__()):
            # This check is useful, especially if __len__ might be zero.
            if self.__len__() == 0:
                 raise IndexError(f"Cannot get item for index {idx} from an empty dataset (phase: {self.phase if hasattr(self, 'phase') else 'unknown'}). Check dataset initialization and splits.")
            raise IndexError(f"Index {idx} out of bounds for dataset of length {self.__len__()}")

        # Determine which original map ID (from the shuffled list for this phase) and which augmentation this idx refers to
        # `original_map_group_idx` is an index into `self.current_phase_original_map_ids`
        original_map_group_idx = int(np.floor(idx / self.augmentations_per_map))
        augmentation_idx = idx % self.augmentations_per_map # This will be 0 to (augmentations_per_map - 1)
       
        # Get the actual original map ID (e.g., from 0 to 49 if num_original_maps_total is 50)
        actual_original_map_id = self.current_phase_original_map_ids[original_map_group_idx]

        # Construct base filename (e.g., "orig0_aug0")
        base_name = f"orig{actual_original_map_id}_aug{augmentation_idx}"

        hm_path = os.path.join(self.heightmap_dir, f'hm_{base_name}.npy')
        rm_path = os.path.join(self.radiomap_dir, f'rm_{base_name}.npy')
        tx_path = os.path.join(self.tx_location_dir, f'tx_{base_name}.npy')

        try:
            heightmap = np.load(hm_path)
            heightmap = heightmap - np.min(heightmap)
            radiomap = np.load(rm_path)
            # heightmap = nan
            radiomap = np.nan_to_num(radiomap, nan=RM_GLOBAL_MIN)
            tx_location = np.load(tx_path)
        except FileNotFoundError:
            # Provide more context for debugging missing files
            print(f"Error: File not found for generated base_name '{base_name}'.")
            print(f"  Index: {idx}, Original Map Group Idx: {original_map_group_idx}, Actual Original Map ID: {actual_original_map_id}, Augmentation Idx: {augmentation_idx}")
            print(f"  Expected HM path: {hm_path}")
            raise # Re-raise the exception
        except Exception as e:
            print(f"Error loading files for base_name {base_name}: {e}")
            raise

        # --- Normalization ---
        heightmap_normalized = (heightmap.astype(np.float32) - self.hm_min) / self.hm_norm_range
        radiomap_normalized = (radiomap.astype(np.float32) - self.rm_min) / self.rm_norm_range
        tx_location_processed = tx_location.astype(np.float32) # Already 0 or 1

        # --- Convert to PyTorch Tensors and add channel dimension ---
        heightmap_tensor = torch.from_numpy(heightmap_normalized).unsqueeze(0) # (1, H, W)
        tx_location_tensor = torch.from_numpy(tx_location_processed).unsqueeze(0) # (1, H, W)
        gain_tensor = torch.from_numpy(radiomap_normalized).unsqueeze(0) # (1, H, W)

        # Stack inputs: (Normalized Heightmap, TX Location Map)
        inputs_tensor = torch.cat((heightmap_tensor, tx_location_tensor), dim=0) # (2, H, W)


        if self.transform:
            inputs_tensor = self.transform(inputs_tensor).type(torch.float32)
            gain_tensor = self.transform(gain_tensor).type(torch.float32)
           
        return (inputs_tensor, gain_tensor, base_name)

class LunarLoader2_los(Dataset):
    """
    Dataset loader for augmented heightmap, radio map (gain), TX location,
    and optionally Line of Sight (LOS) map. Data stored as .npy files.
    It determines which files to load based on an overall index, the number of
    original maps, and the number of augmentations per map. It also handles
    splitting data into train/validation/test phases based on original map indices.
    """

    def __init__(self, root_dir,
                 phase="train",
                 num_original_maps_total=50,
                 augmentations_per_map=20,
                 train_split_ratio=0.7,
                 val_split_ratio=0.15, # Test split will be the remainder
                 random_seed=42,
                 hm_norm_params=(HM_GLOBAL_MIN, HM_GLOBAL_MAX),
                 rm_norm_params=(RM_GLOBAL_MIN, RM_GLOBAL_MAX),
                 use_los_input=False, # New parameter for LOS input
                 transform=None, 
                 heavy_aug = False,
                 non_global_norm= False):
        """
        Args:
            root_dir (string): Base directory where 'heightmaps', 'radiomaps',
                               'tx_locations', and optionally 'los_maps' subfolders are located.
            phase (string): "train", "val", or "test" to select the data split.
            num_original_maps_total (int): Total number of unique original maps
                                           before augmentation (e.g., files indexed 0 to 49).
            augmentations_per_map (int): Number of augmented samples created
                                           for each original map.
            train_split_ratio (float): Proportion of original maps to use for training.
            val_split_ratio (float): Proportion of original maps to use for validation.
            random_seed (int): Seed for shuffling original map indices for reproducible splits.
            hm_norm_params (tuple): (min_val, max_val) for heightmap normalization.
            rm_norm_params (tuple): (min_val, max_val) for radio map normalization.
            use_los_input (bool): If True, loads and includes the LOS map as an input channel.
                                   Defaults to False.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.phase = phase 
        self.augmentations_per_map = augmentations_per_map
        self.transform = transform
        self.use_los_input = use_los_input
        self.heavy_aug = heavy_aug
        self.non_global_norm = non_global_norm
        
        self.heightmap_dir = os.path.join(root_dir, 'heightmaps')
        self.radiomap_dir = os.path.join(root_dir, 'radiomaps')
        self.tx_location_dir = os.path.join(root_dir, 'tx_locations')
        if self.use_los_input:
            self.los_dir = os.path.join(root_dir, 'los_maps') 

        # Normalization parameters
        self.hm_min, self.hm_max = hm_norm_params
        self.rm_min, self.rm_max = rm_norm_params

        self.hm_norm_range = (self.hm_max - self.hm_min)
        if self.hm_norm_range == 0:
            print("Warning: Heightmap normalization range is 0. Max and Min are equal. Setting range to 1.0 to avoid div by zero.")
            self.hm_norm_range = 1.0

        self.rm_norm_range = (self.rm_max - self.rm_min)
        if self.rm_norm_range == 0:
            print("Warning: Radio map normalization range is 0. Max and Min are equal. Setting range to 1.0 to avoid div by zero.")
            self.rm_norm_range = 1.0

        # Determine the original map indices for this phase (train/val/test)
        all_original_map_ids = np.arange(num_original_maps_total)

        np.random.seed(random_seed)
        np.random.shuffle(all_original_map_ids)

        n_train = int(np.floor(train_split_ratio * num_original_maps_total))
        n_val = int(np.floor(val_split_ratio * num_original_maps_total))

        if n_train + n_val > num_original_maps_total:
            print(f"Warning: Sum of train ({train_split_ratio*100}%) and val ({val_split_ratio*100}%) ratios exceeds 100%. Adjusting validation count.")
            n_val = num_original_maps_total - n_train
            if n_val < 0 : n_val = 0

        if phase == "train":
            self.current_phase_original_map_ids = all_original_map_ids[:n_train]
        elif phase == "val":
            self.current_phase_original_map_ids = all_original_map_ids[n_train : n_train + n_val]
        elif phase == "test":
            self.current_phase_original_map_ids = all_original_map_ids[n_train + n_val:]
        else:
            raise ValueError(f"Invalid phase '{phase}'. Must be 'train', 'val', or 'test'.")

        if len(self.current_phase_original_map_ids) == 0 and num_original_maps_total > 0:
              print(f"Warning: Phase '{self.phase}' resulted in an empty set of original map IDs. "
                    f"Total original maps: {num_original_maps_total}, "
                    f"Train count: {n_train}, Val count: {n_val}. "
                    f"Consider adjusting split ratios or total map count.")

        print(f"Dataset phase: '{self.phase}', using {len(self.current_phase_original_map_ids)} original map IDs for this instance. LOS input: {self.use_los_input}")

    def __len__(self):
        # Total number of samples is the number of selected original maps for this phase * augmentations per map
        return len(self.current_phase_original_map_ids) * self.augmentations_per_map

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if not (0 <= idx < self.__len__()):
            if self.__len__() == 0:
                raise IndexError(f"Cannot get item for index {idx} from an empty dataset (phase: {self.phase}). Check dataset initialization and splits.")
            raise IndexError(f"Index {idx} out of bounds for dataset of length {self.__len__()}")

        original_map_group_idx = int(np.floor(idx / self.augmentations_per_map))
        augmentation_idx = idx % self.augmentations_per_map
        actual_original_map_id = self.current_phase_original_map_ids[original_map_group_idx]

        base_name = f"orig{actual_original_map_id}_aug{augmentation_idx}"

        hm_path = os.path.join(self.heightmap_dir, f'hm_{base_name}.npy')
        rm_path = os.path.join(self.radiomap_dir, f'rm_{base_name}.npy')
        tx_path = os.path.join(self.tx_location_dir, f'tx_{base_name}.npy')
        los_path = None
        if self.use_los_input:
            los_path = os.path.join(self.los_dir, f'los_{base_name}.npy')

        try:
            heightmap = np.load(hm_path)
            # Normalize heightmap locally first (e.g. to make min height 0 for that specific map)
            # This seems to be what the original line implies, before global normalization
            heightmap = heightmap - np.min(heightmap)
            if self.non_global_norm:
                self.hm_norm_range = np.max(heightmap)
            radiomap = np.load(rm_path)
            radiomap = np.nan_to_num(radiomap, nan=RM_GLOBAL_MIN) # Use the defined RM_GLOBAL_MIN
            tx_location = np.load(tx_path)

            los_map = None
            if self.use_los_input:
                los_map = np.load(los_path)

        except FileNotFoundError as e:
            missing_path = e.filename
            print(f"Error: File not found: {missing_path} (generated base_name '{base_name}').")
            print(f"  Index: {idx}, Original Map Group Idx: {original_map_group_idx}, Actual Original Map ID: {actual_original_map_id}, Augmentation Idx: {augmentation_idx}")
            if self.use_los_input and los_path and missing_path == los_path:
                 print(f"  Attempted to load LOS from: {los_path}")
            elif missing_path == hm_path:
                 print(f"  Attempted to load HM from: {hm_path}")
            elif missing_path == rm_path:
                 print(f"  Attempted to load RM from: {rm_path}")
            elif missing_path == tx_path:
                 print(f"  Attempted to load TX from: {tx_path}")
            raise
        except Exception as e:
            print(f"Error loading files for base_name {base_name}: {e}")
            raise

        # --- Normalization ---
        heightmap_normalized = (heightmap.astype(np.float32) - self.hm_min) / self.hm_norm_range
        radiomap_normalized = (radiomap.astype(np.float32) - self.rm_min) / self.rm_norm_range
        tx_location_processed = tx_location.astype(np.float32) # Assumed to be 0 or 1 already

        # --- Convert to PyTorch Tensors and add channel dimension ---
        heightmap_tensor = torch.from_numpy(heightmap_normalized).unsqueeze(0) # (1, H, W)
        tx_location_tensor = torch.from_numpy(tx_location_processed).unsqueeze(0) # (1, H, W)
        gain_tensor = torch.from_numpy(radiomap_normalized).unsqueeze(0) # (1, H, W)

     

        if self.use_los_input:
            if los_map is None: # Should have been caught by FileNotFoundError if path was bad
                raise ValueError(f"LOS map is None for base_name {base_name} even though use_los_input is True. This should not happen.")
            # Assuming LOS map is already in a suitable range (e.g., 0 or 1, or probabilities 0-1)
            # If it requires specific normalization, add it here.
            los_map_processed = los_map.astype(np.float32)
            los_tensor = torch.from_numpy(los_map_processed).unsqueeze(0) # (1, H, W)
            # input_channels.append(los_tensor)

        if self.heavy_aug:
            if random.random() < 0.5:
                heightmap_tensor = TF.hflip(heightmap_tensor)
                tx_location_tensor = TF.hflip(tx_location_tensor)
                gain_tensor = TF.hflip(gain_tensor)
                if self.use_los_input:
                    los_tensor = TF.hflip(los_tensor)

            # Random Vertical Flip
            if random.random() < 0.5:
                heightmap_tensor = TF.vflip(heightmap_tensor)
                tx_location_tensor = TF.vflip(tx_location_tensor)
                gain_tensor = TF.vflip(gain_tensor)
                if self.use_los_input:
                    los_tensor = TF.vflip(los_tensor)
                    
            k = random.randint(0, 3)
            if k > 0:
                heightmap_tensor = TF.rotate(heightmap_tensor, angle=k*90)
                tx_location_tensor = TF.rotate(tx_location_tensor, angle=k*90)
                gain_tensor = TF.rotate(gain_tensor, angle=k*90)
                if self.use_los_input:
                    los_tensor = TF.rotate(los_tensor, angle=k*90)
                    
            if random.random() < 0.5:
                heightmap_tensor = torch.transpose(heightmap_tensor, 1, 2)
                tx_location_tensor = torch.transpose(tx_location_tensor, 1, 2)
                gain_tensor = torch.transpose(gain_tensor, 1, 2)
                if self.use_los_input:
                    los_tensor = torch.transpose(los_tensor, 1, 2)
                    
            shape_aug = transforms.RandomResizedCrop(
                size=(256, 256),
                scale=(0.1, 1.0),
                ratio=(0.5, 2.0)
            )
            # i, j, h, w = shape_aug.get_params(img=heightmap_tensor, scale=shape_aug.scale, ratio=shape_aug.ratio)
            # heightmap_tensor = TF.resized_crop(heightmap_tensor, top=i, left=j, height=h, width=w, size=shape_aug.size)
            # tx_location_tensor = TF.resized_crop(tx_location_tensor, top=i, left=j, height=h, width=w, size=shape_aug.size)
            # gain_tensor = TF.resized_crop(gain_tensor, top=i, left=j, height=h, width=w, size=shape_aug.size)
            # if self.use_los_input:
            #     los_tensor = TF.resized_crop(los_tensor, top=i, left=j, height=h, width=w, size=shape_aug.size)
            shape_aug = transforms.RandomResizedCrop(
                size=(256, 256),
                scale=(0.1, 1.0),
                ratio=(0.5, 2.0)
            )

            i, j, h, w = shape_aug.get_params(img=heightmap_tensor, scale=shape_aug.scale, ratio=shape_aug.ratio)

            tx_coords = tx_location_tensor.nonzero(as_tuple=False)

            if tx_coords.numel() > 0:

                tx_y = tx_coords[0, 1].item()
                tx_x = tx_coords[0, 2].item()
                
                img_height, img_width = heightmap_tensor.shape[-2:]

                top_min = max(0, tx_y - h + 1)
                top_max = min(img_height - h, tx_y)

                left_min = max(0, tx_x - w + 1)
                left_max = min(img_width - w, tx_x)

                if top_max >= top_min:
                    i = torch.randint(top_min, top_max + 1, size=(1,)).item()
                    
                if left_max >= left_min:
                    j = torch.randint(left_min, left_max + 1, size=(1,)).item()

            heightmap_tensor = TF.resized_crop(heightmap_tensor, top=i, left=j, height=h, width=w, size=shape_aug.size)
            tx_location_tensor = TF.resized_crop(tx_location_tensor, top=i, left=j, height=h, width=w, size=shape_aug.size)
            gain_tensor = TF.resized_crop(gain_tensor, top=i, left=j, height=h, width=w, size=shape_aug.size)
            if self.use_los_input:
                los_tensor = TF.resized_crop(los_tensor, top=i, left=j, height=h, width=w, size=shape_aug.size)
    
        if self.use_los_input:
            inputs_tensor = torch.cat((heightmap_tensor, tx_location_tensor, los_tensor), dim=0)
        else:
            inputs_tensor = torch.cat((heightmap_tensor, tx_location_tensor), dim=0)
            
            
        if self.transform:
            # Note: Standard torchvision transforms often expect PIL Image or (C,H,W) tensor.
            # If your custom transform expects something else, ensure compatibility.
            # The current setup provides inputs_tensor as (C, H, W) and gain_tensor as (1, H, W).
            inputs_tensor = self.transform(inputs_tensor) #.type(torch.float32) # type cast can be part of transform
            gain_tensor = self.transform(gain_tensor) #.type(torch.float32)

        # Ensure correct type after potential transform
        inputs_tensor = inputs_tensor.type(torch.float32)
        gain_tensor = gain_tensor.type(torch.float32)

        return (inputs_tensor, gain_tensor, base_name)
    


class LunarLoader2_TF(Dataset):
    """
    Dataset loader for augmented heightmap, radio map (gain), and TX location
    data stored as .npy files.
    It determines which files to load based on an overall index, the number of
    original maps, and the number of augmentations per map. It also handles
    splitting data into train/validation/test phases based on original map indices.
    Includes random horizontal and vertical flipping.
    """

    def __init__(self, root_dir,
                 phase="train",
                 num_original_maps_total=50,
                 augmentations_per_map=20,
                 train_split_ratio=0.7,
                 val_split_ratio=0.15, # Test split will be the remainder
                 random_seed=42,
                 hm_norm_params=(HM_GLOBAL_MIN, HM_GLOBAL_MAX),
                 rm_norm_params=(RM_GLOBAL_MIN, RM_GLOBAL_MAX),
                 transform=None):
        """
        Args:
            root_dir (string): Base directory where 'heightmaps', 'radiomaps',
                               'tx_locations' subfolders are located.
            phase (string): "train", "val", or "test" to select the data split.
            num_original_maps_total (int): Total number of unique original maps
                                           before augmentation (e.g., files indexed 0 to 49).
            augmentations_per_map (int): Number of augmented samples created
                                           for each original map.
            train_split_ratio (float): Proportion of original maps to use for training.
            val_split_ratio (float): Proportion of original maps to use for validation.
            random_seed (int): Seed for shuffling original map indices for reproducible splits.
            hm_norm_params (tuple): (min_val, max_val) for heightmap normalization.
            rm_norm_params (tuple): (min_val, max_val) for radio map normalization.
            transform (callable, optional): Optional transform to be applied on a sample
                                            (typically after flipping and tensor creation).
        """
        self.root_dir = root_dir
        self.phase = phase # Store phase
        self.augmentations_per_map = augmentations_per_map
        self.transform = transform

        self.heightmap_dir = os.path.join(root_dir, 'heightmaps')
        self.radiomap_dir = os.path.join(root_dir, 'radiomaps')
        self.tx_location_dir = os.path.join(root_dir, 'tx_locations')

        # Normalization parameters
        self.hm_min, self.hm_max = hm_norm_params
        self.rm_min, self.rm_max = rm_norm_params
        
        self.hm_norm_range = (self.hm_max - self.hm_min)
        if self.hm_norm_range == 0:
            print("Warning: Heightmap normalization range is 0. Max and Min are equal. Setting range to 1 to avoid div by zero.")
            self.hm_norm_range = 1.0
        
        self.rm_norm_range = (self.rm_max - self.rm_min)
        if self.rm_norm_range == 0:
            print("Warning: Radio map normalization range is 0. Max and Min are equal. Setting range to 1 to avoid div by zero.")
            self.rm_norm_range = 1.0
            
        # Determine the original map indices for this phase (train/val/test)
        all_original_map_ids = np.arange(num_original_maps_total)
        
        np.random.seed(random_seed)
        np.random.shuffle(all_original_map_ids)

        n_train = int(np.floor(train_split_ratio * num_original_maps_total))
        n_val = int(np.floor(val_split_ratio * num_original_maps_total))
        
        if n_train + n_val > num_original_maps_total:
            print(f"Warning: Sum of train ({train_split_ratio*100}%) and val ({val_split_ratio*100}%) ratios exceeds 100%. Adjusting validation count.")
            n_val = num_original_maps_total - n_train
            if n_val < 0 : n_val = 0

        if phase == "train":
            self.current_phase_original_map_ids = all_original_map_ids[:n_train]
        elif phase == "val":
            self.current_phase_original_map_ids = all_original_map_ids[n_train : n_train + n_val]
        elif phase == "test":
            self.current_phase_original_map_ids = all_original_map_ids[n_train + n_val:]
        else:
            raise ValueError(f"Invalid phase '{phase}'. Must be 'train', 'val', or 'test'.")

        if len(self.current_phase_original_map_ids) == 0 and num_original_maps_total > 0:
            print(f"Warning: Phase '{self.phase}' resulted in an empty set of original map IDs. "
                  f"Total original maps: {num_original_maps_total}, "
                  f"Train count: {n_train}, Val count: {n_val}. "
                  f"Consider adjusting split ratios or total map count.")
        
        print(f"Dataset phase: '{self.phase}', using {len(self.current_phase_original_map_ids)} original map IDs for this instance.")

    def __len__(self):
        return len(self.current_phase_original_map_ids) * self.augmentations_per_map

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if not (0 <= idx < self.__len__()):
            if self.__len__() == 0:
                raise IndexError(f"Cannot get item for index {idx} from an empty dataset (phase: {self.phase}). Check dataset initialization and splits.")
            raise IndexError(f"Index {idx} out of bounds for dataset of length {self.__len__()}")

        original_map_group_idx = int(np.floor(idx / self.augmentations_per_map))
        augmentation_idx = idx % self.augmentations_per_map
        actual_original_map_id = self.current_phase_original_map_ids[original_map_group_idx]
        base_name = f"orig{actual_original_map_id}_aug{augmentation_idx}"

        hm_path = os.path.join(self.heightmap_dir, f'hm_{base_name}.npy')
        rm_path = os.path.join(self.radiomap_dir, f'rm_{base_name}.npy')
        tx_path = os.path.join(self.tx_location_dir, f'tx_{base_name}.npy')

        try:
            heightmap = np.load(hm_path)
            heightmap = heightmap - np.min(heightmap) # Instance-specific min subtraction
            radiomap = np.load(rm_path)
            radiomap = np.nan_to_num(radiomap, nan=RM_GLOBAL_MIN) # Use the defined global min for nan
            tx_location = np.load(tx_path)
        except FileNotFoundError:
            print(f"Error: File not found for generated base_name '{base_name}'.")
            print(f"  Index: {idx}, Original Map Group Idx: {original_map_group_idx}, Actual Original Map ID: {actual_original_map_id}, Augmentation Idx: {augmentation_idx}")
            print(f"  Expected HM path: {hm_path}")
            raise
        except Exception as e:
            print(f"Error loading files for base_name {base_name}: {e}")
            raise

        # --- Normalization ---
        heightmap_normalized = (heightmap.astype(np.float32) - self.hm_min) / self.hm_norm_range
        radiomap_normalized = (radiomap.astype(np.float32) - self.rm_min) / self.rm_norm_range
        tx_location_processed = tx_location.astype(np.float32)

        # --- Convert to PyTorch Tensors and add channel dimension ---
        heightmap_tensor = torch.from_numpy(heightmap_normalized).unsqueeze(0) # (1, H, W)
        tx_location_tensor = torch.from_numpy(tx_location_processed).unsqueeze(0) # (1, H, W)
        gain_tensor = torch.from_numpy(radiomap_normalized).unsqueeze(0) # (1, H, W)

        # --- Apply Random Flips Consistently (NEW) ---
        # These transforms expect tensors in (C, H, W) format.
        
        # Random Horizontal Flip
        if random.random() < 0.5:
            heightmap_tensor = TF.hflip(heightmap_tensor)
            tx_location_tensor = TF.hflip(tx_location_tensor)
            gain_tensor = TF.hflip(gain_tensor)

        # Random Vertical Flip
        if random.random() < 0.5:
            heightmap_tensor = TF.vflip(heightmap_tensor)
            tx_location_tensor = TF.vflip(tx_location_tensor)
            gain_tensor = TF.vflip(gain_tensor)
            
        # --- Stack inputs: (Normalized Heightmap, TX Location Map) ---
        inputs_tensor = torch.cat((heightmap_tensor, tx_location_tensor), dim=0) # (2, H, W)

        # --- Apply any other user-defined transforms ---
        # Note: If self.transform includes geometric transforms, it might conflict
        # or apply them inconsistently if not designed carefully.
        # This self.transform is usually for things like further normalization if needed,
        # or converting to a specific type if not already done by an earlier step.
        if self.transform:
            # Careful: if self.transform is a torchvision transform expecting PIL or a specific tensor format
            # ensure inputs_tensor and gain_tensor are compatible.
            # Also, applying the same `self.transform` instance to multiple tensors will lead to
            # independent random choices if `self.transform` contains random components.
            # For consistency with flips, those are handled explicitly above.
            inputs_tensor = self.transform(inputs_tensor) 
            gain_tensor = self.transform(gain_tensor)
        
        # Ensure correct type after all transformations
        inputs_tensor = inputs_tensor.type(torch.float32)
        gain_tensor = gain_tensor.type(torch.float32)
            
        return (inputs_tensor, gain_tensor, base_name)