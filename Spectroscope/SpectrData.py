# -*- coding:utf-8 -*-
#!/usr/bin/env python
""" Классы данных спектра и визуализации
Created on 22.12.2016, @author: dmagin
"""
import numpy as NP
from numpy import linalg as LA

import SpectrPlot.Geometry as Geo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as tri
#from matplotlib.colors import LightSource

def dicMarkStyle():
    # Словарь стилей маркера
    # http://matplotlib.org/api/markers_api.html#module-matplotlib.markers
    #return {v:k for k, v in Markers.MarkerStyle().markers.items()}
    dictMS = {'o': 'circle', 'h': 'hexagon', 'D': 'diamond', '*': 'star', '.': 'point', ',': 'pixel', 'None': 'nothing'}
    return {v:k for k, v in dictMS.items()}

def mColorMap():
    # перечень доступных цветовых карт
    # http://matplotlib.org/examples/color/colormaps_reference.html
    # https://matplotlib.org/users/colormaps.html?highlight=colormap#id8
    cmaps = [#'== Perceptually Uniform Sequential ==',
                'viridis', 'inferno', 'plasma', 'magma',

             #'== Sequential =='
                'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu',   
                'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn', 

             #'== Sequential (2) =='
                'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
                'spring', 'summer', 'autumn', 'winter',    
                'cool', 'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper',
                
             #'== Diverging =='
                'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 
                'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',

             #'== Qualitative =='
                'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',  
                'Set1', 'Set2', 'Set3',
                #'tab10', 'tab20', 'tab20b', 'tab20c',

             #'== Miscellaneous =='
                'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
                'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
                'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar',
                ]
    
    return cmaps

class DataSpectr(object):
    '''Класс данных спектра'''
    
    def Spectr2Projections(self, sVal, mVector):
        '''Получаем вырожденные 2D-проекции спектра'''
        self.vIndex = [] # индексы вырожденных спектров
        self.vS = [] # собственные числа вырожденных спектров
        self.vDataX, self.vDataY = [], [] # собственные функции вырожденных спектров
        self.vDataZ = [] # собственные функции невырожденных спектров
         
        NumDims = len(sVal)
        self.numSp = NumDims - 1 # Всего уровней
        self.numSp2, self.numZ = 0, 0 # Вырожденных и невырожденных уровней
        bDegen = False # Признак вырожденного уровня 
        for i in range(0, NumDims):
            if bDegen:
                bDegen = False
                continue
            sV = sVal[i]
            if abs(sV) < self.error/10: continue # исключаем нулевой уровень Должен быть один
            if i == NumDims-1: # последний уровень
                self.vDataZ.append(mVector[i])
                self.numZ += 1
            elif abs(sV - sVal[i+1]) < self.error: # вырожденный уровень
                self.vIndex.append(i)
                self.vDataX.append(mVector[i])
                self.vDataY.append(mVector[i+1])
                self.vS.append(sV)
                self.numSp2 += 1
                bDegen = True
            else:
                self.vDataZ.append(mVector[i])
                self.numZ += 1

    def GetFunction(self):
        # Формирование функции от дистанции между узлами - добавление возмущения
        # F(R2) = w*Rd + 1/Rd, где w = dist^3/n^2, Rd = R2^degree
        size = len(self.mR2)
        weight = self.distortion**3/(self.Size**2)
        mRD = self.mR2 + NP.eye(size) # для исключения деления на ноль при отрицательных степенях (из-за нулевой диагонали)
        mTemp = mRD**self.degree
        mF = weight*mTemp + 1/mTemp
        #for i in range(size): mF[i][i] = 0
        return mF

    def SetSpectrData(self):
        # Создание спектров
        mG = Geo.mR2G(self.GetFunction())
        sG, vG = LA.eigh(mG)
        #print(sG)
        self.Spectr2Projections(sG, vG.T)

    def CalcNumOfSpectrum(self):
        # Теоретический расчет числа вырожденных спектров
        if self.Symm == 6:
            NumSp = 3*self.Size*(self.Size-1)
            NumSpDeg = NumSp/3
        elif self.Symm == 4:
            NumSp = self.Size*self.Size
            NumSpDeg = (NumSp-NumSp%2)/4
            NumSp -= 1     
        else:
            NumSp = Size*Size
            NumSpDeg = int(NumSp/Symm)
        
        self.numSp, self.numSp2 = int(NumSp), int(NumSpDeg) 
        self.numZ = self.numSp - 2*self.numSp2 # Количество невырожденных спектров

    def BaseGrids():
        return ['Polygon', 'Radial']

    def BaseSymms():
        return [i for i in range(3, 14)]
        
    def SetPoints(self):
        self.vSet = []
        if self.Border >= self.Size:
            self.Border = self.Size-1
         
        if self.SetGrid == 'Polygon':
            self.vSet = Geo.mPolygonGrid(order=self.Symm, size = [self.Border, self.Size])

        elif self.SetGrid == 'Radial':
            self.vSet = Geo.mRadialGrid(order=self.Symm, size = [self.Border, self.Size])


    def SetSpectr(self):
        self.SetPoints()
        self.mR2 = Geo.mSetToD2(self.vSet) # Получили матрицу квадратов расстояний
        #self.CalcNumOfSpectrum()
        self.SetSpectrData() 

    def __init__(self, SetGrid='Polygon', Symm=6, spSize=[0, 10], distortion=-2., degree=1, error=0.000001):
        self.SetGrid = SetGrid
        self.SetMode = 'all'
        self.Symm = Symm
        self.Border, self.Size = spSize[0], spSize[1]
        self.distortion = distortion
        self.degree = degree
        self.error = error
        self.SetSpectr()

class ShowSpectr(DataSpectr):
    '''Визуализация спектров'''
    def PlotTypes():
        return ['Points', 'Contour', 'Fill']

    def NumPlotsList():
        return [1, 4, 9, 16]
        
    def SetPlots(self):
        self.fig.clear()
        self.ax = []
        nr, nc = 1, 1
        if self.NumPlots == 4: nr, nc = 2, 2 
        elif self.NumPlots == 9: nr, nc = 3, 3
        elif self.NumPlots == 16: nr, nc = 4, 4
        for i in range(self.NumPlots):
            ax = self.fig.add_subplot(nr, nc, i+1, frame_on=False) # aspect='equal' 2, 2, i+1 nrows=nr, ncols=nc, plot_number=i+1, alpha=self.alpha
            ax.set_aspect('equal') #, 'datalim'
            #ax.axison = False
            ax.set_axis_off()
            self.ax.append(ax) 
        #self.ax.set_projection('lambert')
    
    def IniCanvas(self, ini=True):
        '''Инициализация холста'''
        if ini:
            self.fig = plt.figure(dpi=200, figsize=(8, 6)) # facecolor='white', figsize=(8, 6), linewidth=1 
            plt.subplots_adjust(left=0.0, right=1.0, bottom=0., top=0.97, wspace = 0.05, hspace = 0.05)
        self.SetPlots()
    
    def getImgColorMap(self):
        cmap = self.cmap
        if self.inverseColor: cmap = cmap + "_r"
        gradient = NP.linspace(0, 1, 256)
        gradient = NP.vstack((gradient, gradient))
        imgFig = plt.figure() 
        imgAx = imgFig.add_subplot(1, 1, 1, frame_on=False)
        imgFig.subplots_adjust(top=1, bottom=0., left=0, right=1)
        imgAx.imshow(gradient, aspect='auto', cmap=cmap)
        imgAx.set_axis_off()
        return imgFig         
        
    def getMask(self, iMask, fmask, bmask=False):
        # Mask to hide triangles
        maxlim = 100
        lim = Geo.Val2Val(maxlim - iMask, [0, maxlim], [min(fmask), max(fmask)])
        return NP.where(fmask > lim, not bmask, bmask)
    
    def ValidateSpIndex(self):
        if self.iSpectr >= self.numSp2 - self.NumPlots:
            self.iSpectr = self.numSp2 - self.NumPlots 
        if self.iSpectr < 0: self.iSpectr = 0

    def ValidateColorIndex(self):
        maxIndex = self.numZ-1
        if maxIndex < 0: return 
        if self.iColor < 0: self.iColor = 0
        elif self.iColor > maxIndex: self.iColor = maxIndex  
        
    def ValidateStyleIndex(self):
        maxIndex = self.numZ-1
        if maxIndex < 0: return 
        if self.iStyle < -1: self.iStyle = -1
        elif self.iStyle > maxIndex: self.iStyle = maxIndex
        
    def ValidateIndexes(self):
        self.ValidateSpIndex()  
        self.ValidateColorIndex()
        self.ValidateStyleIndex()

    def UpdateSpectr(self, ax, pIndex, zIndex, sIndex, title=''):
        # Вывод спектра
        ax.clear()
        ax.axison = False
        ax.use_sticky_edges = False
        ax.autoscale_view(tight=None, scalex=True, scaley=True)
        if title != '':  ax.set_title(title, fontsize=9, loc='left')

        vX, vY = Geo.RotateData(self.vDataX[pIndex], self.vDataY[pIndex])
        if len(self.vDataZ) == 0:
            vZ, vS = None, None
        else:
            vZ, vS = self.vDataZ[zIndex], self.vDataZ[sIndex]

        cmap = self.cmap
        if self.inverseColor: cmap = cmap + "_r"

        figSize = self.fig.get_size_inches()
        if self.PlotType == 'Points':
            vSize = self.pSize*(figSize[0]*figSize[0])/(2*self.NumPlots)
            if sIndex >= 0:
                maxVal = NP.max(vS)
                self.vSize = (vS*vS)/(maxVal*maxVal) + 0.01
                if self.inversepSize:
                    vSize /= (100*self.vSize)
                else:
                    vSize *= (self.vSize)

            #try: # Тут бывает какая-то ошибка с вектором цвета, надо бы разобраться
            ax.scatter(vX, vY, s=vSize, c=vZ, cmap=cmap, marker='o', alpha=self.alpha) #, norm=None, vmin=None, vmax=None, linewidths=None, verts=None, edgecolors=None

        else: # triangulation
            triang = tri.Triangulation(vX, vY)

            if sIndex >= 0:
                xm = Geo.Agregat(vX[triang.triangles], self.MaskFunction, 1)
                ym = Geo.Agregat(vY[triang.triangles], self.MaskFunction, 1)
                triang.set_mask(self.getMask(sIndex, xm*xm + ym*ym, self.varSIndex))

            if self.PlotType == 'Contour':
                density = int(self.pSize)
                ax.tricontour(triang, vZ, density, cmap=cmap, linewidths=figSize[0]/8, alpha=self.alpha) 
                
            else: #'Fill':
                shad = 'flat'
                if self.inversepSize: shad = 'gouraud' 
                ax.tripcolor(triang, vZ, cmap=cmap, edgecolors='none', alpha=self.alpha, shading=shad)
         

    def ShowSp(self):
        for i in range(self.NumPlots):
            # Проекция
            pIndex = self.iSpectr
            if self.varPIndex:  pIndex += i
            if pIndex >= len(self.vIndex): continue

            # Цвет
            zIndex = self.iColor
            if self.varZIndex:     zIndex += i
            if zIndex > self.numZ-1: zIndex = self.numZ-1  
            
            # Стиль
            sIndex = self.iStyle
            if self.varSIndex:  sIndex += i
            if sIndex > self.numZ-1: sIndex = self.numZ-1  

            title = ''
            if self.ShowTitle:
                spInd = self.vIndex[pIndex]
                if self.vS[pIndex] < 0: spInd += 1 # Поскольку нулевой уровень исключаем, то отрицательные индексы сдвигаются относительно положительных 
                title = str(spInd) + ':' + str(round(self.vS[pIndex], 3))
            
            self.UpdateSpectr(self.ax[i], pIndex, zIndex, sIndex, title) 

    def ChangeIndex(self, val=0):
        self.iSpectr = val
        self.ValidateSpIndex()

    def SetStyle(self, val=0):
        self.iStyle = int(val)
        self.ValidateStyleIndex()

    def ChangepSize(self, val=0):
        self.pSize = int(val)

    def SetColor(self, index=0):
        if index >= 0: self.iColor = index
        self.ValidateColorIndex()

    def SetAlpha(self, val=0):
        self.alpha = val
       
    def ChangeDistortion(self, val=0):
        self.distortion = val
        self.SetSpectrData()

    def ChangeDegree(self, val=0):
        self.degree = val
        self.SetSpectrData()

    def SetForm(self):
        self.SetSpectr()
        self.ValidateIndexes()
        self.SetColor(-1)
            
    def dictDefaultValues(self):
        return {'SetGrid': 'Polygon', 'SetMode': 'all', 'Symm': 6, 
                'Size': 10, 'SizeRange': [2, 30], 'Border': 0, 'numSp2': 0, 'numZ': 0,  
                'distortion': -2., 'distortionRange': [-2., 2.], 'degree': 1, 'degreeRange': [0.0, 2.0], 'error': 0.000002,
                'NumPlots': 1, 'PlotType': 'Points', 'ShowTitle': False,
                'iSpectr': 0, 'iColor': 0, 'iStyle': -1,
                'varPIndex': True, 'varZIndex': False, 'varSIndex': False,
                'cmap': 'magma', 'inverseColor': False, 'alpha': 0.7,
                'pSize': 8, 'pSizeRange': [1, 100], 'inversepSize': False,
                #'useMask': False, 'iMask': 0, 'inverseMask': False, 
                'MaskFunction': 'Mean',
                }

    def Spectr2Dict(self):
        dictSp = self.dictDefaultValues() 
        for key in dictSp.keys():
            try: dictSp[key] = getattr(self, key) #dictSp[key] = self.__getattr__(key)
            except: print('Spectr2Dict(): invalid attribute name: ' + key)
        return dictSp
        
    def LoadFromDict(self, Dict=None, ini=False):
        dictSp = self.dictDefaultValues()
        if Dict is not None:
            dictSp.update(Dict)

        for key, value in dictSp.items():
            try: self.__setattr__(key, value)
            except: print('LoadFromDict(): invalid attribute name: ' + key)
        
        DataSpectr.__init__(self, SetGrid=dictSp['SetGrid'], Symm=dictSp['Symm'], spSize=[dictSp['Border'], dictSp['Size']], distortion=dictSp['distortion'], degree=dictSp['degree'], error=dictSp['error'])

        self.IniCanvas(ini)
        self.SetForm()
            
    def __init__(self, dictAttr = None):
        self.LoadFromDict(dictAttr, ini=True)


#mask = tri.TriAnalyzer(triang).get_flat_tri_mask(0.01)
#triang.set_mask(mask)
#triang.set_mask(NP.hypot(vX[triang.triangles].mean(axis=1), vY[triang.triangles].mean(axis=1)) 0> 0.1)
#refiner = tri.UniformTriRefiner(triang)
#tri_refi, z_test_refi = refiner.refine_field(vColor, subdiv=3)
#ax.tricontour(triang, vColor**2, density, cmap=cmap, linewidths=line_w, alpha=self.alpha)

    #ax.triplot(triang, linestyle='-', alpha=self.alpha) 
    #ax.tripcolor(triang, vColor, cmap=cmap, edgecolors='none', alpha=self.alpha) #shading='flat' edgecolors='none'
    #ax.tricontourf(triang, vColor, density,  hatches=['-', '/', '\\', '//'], extend='both', cmap=cmap, alpha=self.alpha, antialiased=False) #cmap=cmap, antialiased: [ True | False ]
# Web
#    if self.inversepSize: # выводим узлы одновременно с сеткой
#        ax.triplot(triang, marker='', linestyle='-', alpha=self.alpha)
#        ax.scatter(vX, vY, s=vSize, c=vColor, marker='o', alpha=self.alpha) #, norm=None, vmin=None, vmax=None, linewidths=None, verts=None, edgecolors=None
#    else:
#        ax.triplot(triang, marker='', linestyle='-', alpha=self.alpha)
