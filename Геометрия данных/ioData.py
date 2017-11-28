# -*- coding:utf-8 -*-
'''
Created on 12.05.2011, modifyed 07.09.2013, 28.11.17
@author: dm
'''
import numpy as np

def File2Data(fnData):
    # Загрузка файла данных в список (вектор) данных.
    # Строка (запись) должна иметь формат:    Узел1    Узел2    Значение
    # Идентификация узлов - числовая
    infile = open(fnData, 'r')
    lData = []
    for line in infile:
        record = [n for n in line.split()]
        Val = [int(record[0]), int(record[1]), float(record[2])]
        lData.append(Val)
    infile.close()
    return lData

def Data2Array(lData, offset=1):
    # Загрузка данных в массив
    # Данные представляют собой список записей формата:  Узел1    Узел2    Значение
    vMax = np.max(lData, 0)
    NumNodes = int(max(vMax[0], vMax[1]))
    mData = np.zeros((NumNodes+offset, NumNodes+offset))
    for record in lData:
        ind1, ind2, val = record[0], record[1], record[2]
        mData[ind1, ind2] = val 
        mData[ind2, ind1] = val 
    return mData, NumNodes

def mMinor(mData, lNodes):
    #Выборка данных по заданному списку узлов
    size = len(lNodes)
    mMinor = np.zeros((size, size))
    for i in range(size):
        indI = lNodes[i]
        for j in range(i+1, size):
            indJ = lNodes[j]
            mMinor[i, j] = mData[indI, indJ]
            mMinor[j, i] = mMinor[i, j]  
    return mMinor

def vDistance(mData, lNodes, node):
    #Вектор дистанции для заданного узла и базиса
    size = len(lNodes)
    vDist = np.zeros(size)
    for i in range(size):
        indI = lNodes[i]
        vDist[i] = mData[indI, node]
    return vDist


def mNodesCoordinate(nRow, nCol, startNode = 1):
    # Создает двумерный массив числовых идентификаторов узлов
    # nRow - количество строк, nCol - количество колонок
    mNodes = np.zeros((nRow, nCol))
    node = int(startNode)
    for i in range(nRow):
        for j in range(nCol):
            mNodes[i][j] = int(node)
            node += 1
    return mNodes

def SetOfNodes(mNodes, rowSize, colSize, rowStart=0, colStart=0):
    #Выборка координат узлов из массива
    lSet = []
    for row in range(rowSize):
        for col in range(colSize):
            lSet.append(int(mNodes[rowStart + row, colStart + col]))
    return lSet


""" Архив кода
def M2File(mTable, fileName='1.alg', format='%3d'):
    outfile = open(fileName, 'w')
    for row in mTable:
        for column in row:
            outfile.write(format % column)
        outfile.write('\n')
    outfile.close()

def SliceOfResistance(mNC, mSet, mResDat, xSize, ySize, disp):
    ''' Вывод спектральных срезов 
    mSet - относительные координаты узлов конфигурации
    mRes - исходная (общая) матрица резистенсов
    BaseNode - текущий (базовый) узел, относительно которого считаются абсолютные координаты узлов конфигурации
    '''
    mR0 = GF.mUniR(mSet)
    mG0 = GF.mR2G(mR0)
    sG0, vG0 = LA.eigh(mG0)
    
    NumLayers = len(mSet)
    #GF.pprint(mR)
    
    for yBase in range(ySize - disp):
        print(yBase)
        for Number in reversed(range(NumLayers)):
            if sG0[Number]*sG0[Number] < 0.01:
                continue
            #print "%3.2f" %sG0[Number],
            for xBase in range(xSize - disp):
                mR = mRofSet(mNC, mSet, mResDat, (xBase, yBase))
                sC, mComp = GF.mR2Comp(mR)
                scNorm = sC[Number]/sG0[Number]
                #print "%3.2f" %scNorm, 
            print()

def TestUniR():
    Size = 6
    Depth = Size 
    dx, dy, dz = Size, Size, Depth
    
    mNodes = Geo.mSet3(dx, dy, dz)
    mR = GF.mUniR3(mNodes)
    mLap = GF.mR2Laplas(mR)
  
    mC = GF.mV2M(mNodes, NP.diag(mLap), dx, dy, dz)
    
    for i in range(Depth):
        GF.pprint(mC[i], "%3.3f")

def File2Umain(fnData='main.dat', fnNode='main.alg'):
    infile = open(fnData, 'r')
    uList = [float(w) for w in infile.read().split()]
    infile.close()
    return uList

    infile = open(fnNode, 'r')
    nodes_list = []
    for line in infile:
        nodes = [int(n) for n in line.split()]
        nodes_list.append(nodes)
    infile.close()
    
    lNodes = []
    for nodes in nodes_list:
        if not(nodes[0] in lNodes):
            lNodes.append(nodes[0]) 
        if not(nodes[1] in lNodes):
            lNodes.append(nodes[1]) 

    mSize = len(lNodes)
    mUmain = NP.zeros((mSize, mSize))

    for umn, nodes in zip(uList, nodes_list):
        i, j = lNodes.index(nodes[0]), lNodes.index(nodes[1])
        #print umn, nodes, i, j
        mUmain[i][j] = umn
        mUmain[j][i] = umn

    return mUmain, lNodes

def mUmain2Uadj(mUmain):
    mainSize = len(mUmain)
    adjSize = mainSize - 1
    mUadj = NP.zeros((adjSize, adjSize))
    for col in range(adjSize):
        mUadj[col][col] = mUmain[col+1][0] 

    for row in range(adjSize):
        for col in range(adjSize):
            if col == row: continue
            mUadj[col][row] = (mUmain[0][row+1] + mUmain[col+1][0] - mUmain[col+1][row+1])/2. 
    return mUadj

def mGetABMN(mCoordOfNode, mSet, BaseNode, method='main'):
    mABMN = []
    if method == 'main': 
        # измеряем главные (питающие) напряжения между всеми узлами AB = MN
        for disp1 in mSet:
            for disp2 in mSet:
                if disp2 <= disp1: continue
                idA = mCoordOfNode[BaseNode[1]+disp1[1]][BaseNode[0]+disp1[0]]
                idB = mCoordOfNode[BaseNode[1]+disp2[1]][BaseNode[0]+disp2[0]]
                idM, idN = idA, idB
                mABMN.append([idA, idB, idM, idN])

    elif method == 'adj':
        # измеряем все напряжения (главные и смежные) между базовым узлом и остальными
        # базовым считается первый узел в измерительной установке - mSet[0]
        idA = mCoordOfNode[BaseNode[1]+mSet[0][1]][BaseNode[0]+mSet[0][0]]
        idM = idA 
        for disp1 in mSet:
            for disp2 in mSet:
                if disp2 > disp1: continue
                z = math.hypot(disp1, disp2)
                
                if disp2 == mSet[0]: continue
                idN = mCoordOfNode[BaseNode[1]+disp2[1]][BaseNode[0]+disp2[0]]
                mABMN.append([idA, idB, idM, idN])
    return mABMN

#bNode = [0, 2]
#method = 'main'
#mABMN = MS.mGetABMN(CoordOfNodes, setLine5, [2, 2], method)
#MS.M2File(mABMN, method+'.alg')

#method = 'adj'
#mABMN = MS.mGetABMN(CoordOfNodes, setLine5, [2, 2], method)
#MS.M2File(mABMN, method+'.alg')

#uMain, lNodes = MS.File2Umain(fnData='1.dat', fnNode='main.alg')
#print NP.around(uMain, 3)
#print uMain
#print lNodes
#uAdj = MS.mUmain2Uadj(uMain)
#print uAdj

#mSet = setCrest5
#mSet = setGrid4
#mSet = setLine3
#NumLayers = len(mSet)
#disp = 3

#MS.SliceOfResistance(mNC, mSet, mResDat, xSize, ySize, disp)

#mR0 = GF.mUniR(mSet)
#GF.pprint(mR0)

#sG0, vG0 = GF.mR2Comp(mR0)

#print vG0 

#x0, y0, z0 = 3, 3, 1.5
#contrast = 10.0

#for yBase in range(10):
    #print yBase
#for Number in reversed(range(NumLayers)):
        #if sG0[Number] < 0.1:
            #continue
    #print "%3.2f" %sG0[Number],
        #for xBase in range(10):
            #mdR = GF.mPointR(mSet, xBase - x0, yBase - y0, z0, contrast)
            #sC, mComp = GF.mR2Comp(mR0 + mdR)
            #scNorm = sC[Number]/sG0[Number]
            #print "%3.2f" %scNorm,   
            #print "%3.2f" %sC[Number],   
            #print "%3.2f" %sG0[Number],   
    #print


for z in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]:
    mdR = GF.mPointR(mSet, 1, 0, z, 0.4)
    sC, mComp = GF.mR2Comp(mR0 + mdR)
    scNorm = sC[Number]/sG0[Number]
    print sC 
    #GF.pprint(mComp)

#sC, mComp = GF.mR2Comp(R9_Kushman)
#print sC 
#GF.pprint(mComp)
#mSet = setLine4
mR0 = GF.mUniR(mSet)
mG0 = GF.mR2G(mR0)
sG0, vG0 = LA.eigh(mG0)
#print sG 
#GF.pprint(vG)

sC, mComp = GF.mR2Comp(R5_Kushman)
print sC 
print
GF.pprint(mComp)


#mR = GF.mUniR(mSet)     # Расчет матрицы квадратов координат узлов

#mR = GF.mRtoD2(R91_Kushman, 10)

#mR0 = GF.mUniR(mSet)     # Расчет однородных резистенсов для заданного набора узлов
#mdR = GF.mPointR(mSet, 2, 2, 2., 1) # Расчет влияния однородного включения
#mR = mR0
#mR = GF.mRtoD2(mR + mdR, 1)

#mR = R4grid #R4line_inv
#mG = GF.mR2G(mR)        # Преобразование резистенсов в матрицу Грина

#sC, mComp = GF.mR2Comp(mR0 + mdR)

"""
