 
#%% 
import skimage.color as sc
import numpy as np
import matplotlib.pyplot as plt
import PySimpleGUI as sg
import scipy.io as si
import cv2
import glob
import os
from skimage.filters import laplace

class OledSolarImage:
  
    def __init__(self, number_of_wires = 5, transpose = False, order = 0, average_filter_size = [15,15], preprocess = True):
        self._number_of_wire = number_of_wires
 
        self.transpose = transpose
        self.order = order
        self.average_filter_size =average_filter_size
        self.preprocess = preprocess
        
    @staticmethod
    def read(filepath, skiprows = 5):
        array = []
        with open(filepath) as f:
            lines = f.readlines()
            rows = skiprows
            for l in lines:
                if rows != 0:
                    rows -= 1
                    continue
                l = l.split(' ')
                l = [i for i in l if i!='']
                l[-1] = l[-1].strip('\n')
                array.append(l)
        return np.array(array, dtype=np.float)
    def _preProcess(self, npd = False):
        if npd:
            self.File = np.load(self.path_to_file)
        else:
            self.File = si.loadmat(self.path_to_file)['sA']
        # self.File = self.read(self.path_to_file)
        # Not considering edgy conditions of wire cooler than noise - only seen once!
        # File = File - np.mean(File)
        # File = File-np.amin(File)
        if self.transpose:
            self.File = self.File.T     
        if self.preprocess:   
            Median = np.median(self.File )
        # 2. process the image in a horizontal view
            if self.order >= 1:
                self.image = np.exp(self.File)**self.order > np.exp(Median)**self.order
            else:
                self.averageFilt()
    
        else:
            self.image = self.File
 
    
    def segmentation(self, path_to_file, area_to_check=[[0,102,204,306,408],[102,204,306,408,502]] ,threshold = 0.4):
        self.path_to_file = path_to_file
        self.ind = []
        self.ind_hotspot=[]
        self._preProcess(npd = True)
     
        # if self._number_of_wire == 5 and area_to_check ==[]:
        #     #! Hardcode  - last 10 rows cannot be wires
        #     area_to_check = [0,102,204,306,408,502]
            
        # else:
        #     if area_to_check == []:
        #         raise KeyError('Please specifiy approximate area of wires in a list, e.g. [0,100,200,300,400,512] for 5 wires setup.')
        if area_to_check:
            self.roi = area_to_check
        else:
            self.roi = self._ROI() 
        l = np.min([len(self.roi[0]), len(self.roi[1])])
        for i in range(0, l):
            # expand +10 points as margin
            # heuristic - width less than width of wire, skip
            width = abs(self.roi[0][i] - self.roi[1][i])
            if width < 45:
                pass
            #! Noted increased ROI to increase sensitivity of weak signal
            addition = int(0*width)
            if self.roi[0][i] -addition < 0:
                start = 0
            else:
                start = self.roi[0][i]-addition
            
            if self.roi[1][i] +addition > self.image.shape[0]:
                end = self.image.shape[0]
            else:
                end = self.roi[1][i] + addition
            # print(width,start,end)
            ROI = self.image[start :end]
            
            zscore =  self.zscoreRow(ROI)
                
            self.ROI = self.scale((ROI.sum(axis =1))/ROI.shape[1])
            ind_size = len(self.ind)
            
            self._slide(self.ROI, start, threshold)
           
            #! Hardcode rule - zscore strech larger than 5 means hot spot
            if np.amax(zscore) >  6 and (len(self.ind) - ind_size) > 0:
                self.ind_hotspot.append(self.ind[-2])
                self.ind_hotspot.append(self.ind[-1])

    
    def _ROI(self, filter_size = 40, stepSize = 2):
        x, y =filter_size,filter_size
        
        skipX =stepSize

        inp = self.image.copy()*0
        targ = self.image.copy()

        stepX =  inp.shape[0] // x
        stepY = inp.shape[1] // y

        for i in range(0,stepX*x, skipX):
            for j in range(0, stepY*x, y):
                data = targ[i:i+x, j:j+y]
        
                mass = np.where(data>data.mean())
                if  (mass[0][-1] - mass[0][0]) * (mass[1][-1] - mass[1][0]) / (x*y) <= 0.5:
        
                    inp[i:i+x, j:j+y] = (inp[i:i+x, j:j+y]+np.interp(data, [np.mean(data) , np.max(data)], [0,1])) /2


        # plt.imshow(inp)
        # plt.figure()
        temp = inp.sum(axis =1)/inp.shape[1]
        temp = np.interp(temp, [temp.mean(), temp.max()], [0,1])
        self.temp = self.moveAverage(temp,15)
        # plt.plot(temp)     
        return self._slideCoarse(temp)                                                                                                 
    
    def drawArea(self, save_path=None, extraSpace = 5, color = (0,255,0), show = True):
        i = 0
        temp = sc.gray2rgb(self.File.copy())
        temp = temp / np.amax(temp) *255
        temp = temp.astype(np.uint8)
         
        while i < len(self.ind)//2*2:
            if len(temp.shape) < 3:
                color = (np.amax(self.File))
            cv2.rectangle(temp, (0+extraSpace, self.ind[i]-extraSpace), (self.File.shape[1]-extraSpace, self.ind[i+1]+extraSpace), color,2);
            i+=2
            #! draw hot spot se
        if len(self.ind_hotspot) != 0:
            i=0
            while i < len(self.ind_hotspot)//2*2:
                color = (255,0,0)
                cv2.rectangle(temp, (0+extraSpace, self.ind[i]-extraSpace), (self.File.shape[1]-extraSpace, self.ind[i+1]+extraSpace), color,2);
                i+=2

       

        if self.transpose:
            self.temp = np.transpose(temp, (1,0,2))
        else: self.temp = temp
        plt.imshow(self.temp)
        plt.title(f"Detected {int(len(self.ind)/2)} Wire, {int(len(self.ind_hotspot)/2)} hot spot")

        plt.xlabel('Column index')
        plt.ylabel('Row index')
        if save_path != None:
            plt.savefig(save_path +'\\' + self.path_to_file.split('.')[0]+'.png', dpi = 150)
        if show:
            plt.show()
    
    def averageFilt(self):
    
        kernel = np.ones((self.average_filter_size[0],self.average_filter_size[1]),np.float32)/(self.average_filter_size[0] * self.average_filter_size[1])
        self.image = cv2.filter2D(self.File,-1,kernel,cv2.BORDER_ISOLATED)
    
    @staticmethod
    def _slideCoarse(img_sideview_array, threshold=0.4):
    # window size to be 17 for testing // to be less than width of a wire ~ above half width
        temp_front_edge = []
        temp_back_edge = []
        newarray = img_sideview_array-threshold
        flags = 'front'
        #! noted original setting range from 2
        for i in range(1,len(img_sideview_array)-1):
    
            if newarray[i-1] <= 0 and newarray[i+1] > 0 and newarray[i]< 0:
                temp_front_edge.append(i)
 
            if newarray[i-1] >= 0 and newarray[i+1]< 0 and newarray[i]<0:
                temp_back_edge.append(i)
       
        # for i in range(1,len(img_sideview_array)-1):
        #     if flags == 'front':
        #         if newarray[i-1] <= 0 and newarray[i+1] > 0 and newarray[i]< 0:
        #             temp_front_edge.append(i)
        #             flags = 'back'
        #     else:
        #         if newarray[i-1] >= 0 and newarray[i+1]< 0 and newarray[i]<0:
        #             temp_back_edge.append(i)
        #             flags = 'front'
                    
        return [temp_front_edge, temp_back_edge]

    def _slide(self,  img_sideview_array, offset, threshold):
        # window size to be 17 for testing // to be less than width of a wire ~ above half width
        edges = self._slideCoarse(img_sideview_array, threshold)
 
        if edges[0] != [] and edges[1] != []:
            temp_front_edge = edges[0]
            temp_back_edge = edges[1]
            
            if len(temp_front_edge) == len(temp_back_edge) and len(temp_front_edge) < 3 and temp_front_edge[0] < temp_back_edge[0]:
            
                self.ind.append(temp_front_edge[0] + offset)
                self.ind.append(temp_back_edge[-1]+  offset)
             

    @staticmethod
    def scale(arr1, Range=1):
        s = Range/ (np.amax(arr1) - np.amin(arr1))
        return (arr1-np.amin(arr1))*s
    @staticmethod
    def zscoreRow(imageArray):
        zscore = []
        for i in imageArray:
            zscore.append(np.max((i-i.mean())/np.std(i,ddof=0)))
        return zscore
    @staticmethod
    def moveAverage(image ,windowSize=5):
        imageArray = image.copy()
        steps = len(imageArray)-len(imageArray)%windowSize
        for i in range(steps):
            imageArray[i] = np.mean(imageArray[i:i+windowSize])
        return imageArray
#%% 
import time
if __name__ == "__main__":
 
    os.chdir('C:\\Data preparation\\Auto segmentation\\Train')
    Files = glob.glob("*.npy")
    p = r'C:\OledSolar\Data preparation\Auto segmentation\Train\r'
    processor = OledSolarImage(number_of_wires = 4, transpose = False, average_filter_size= [15,15])
    # a = time.time()
    for i in Files:
        ## Hardcoded rule - no boarder considered in the last 10 rows of data
        processor.segmentation(i,  area_to_check = [[0, 100, 200, 300,400], [ 100, 200, 300,400,500]], threshold = 0.35)
        processor.drawArea(save_path =p , show=False)
        print(f'Finished {i}')
    # b = time.time()
    # print(b-a)
