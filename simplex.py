# -*- coding:utf-8 -*-
""" Свойства симплексов и графов"""
import math
import numpy as np
from numpy import linalg as la

class mDistance(object):
    # Distance matrix. Different degree of euclidean distance is available
    # todo:	support different dimensions of coordinates
 	def __init__(self, lPoints, degree=2, coeff=1.):
	    numPoints = len(lPoints)
	    self.mD = np.zeros((numPoints, numPoints))
	    for i in range(numPoints):
	        for j in range(i+1, numPoints):
	            diff = np.array(lPoints[i]) - np.array(lPoints[j])
	            self.mD[i, j] = np.inner(diff, diff)
	            if degree != 2:
	            	self.mD[i, j] = math.sqrt(self.mD[i, j])**degree
	            self.mD[j, i] = self.mD[i, j]
	

class Point(object):
	"""Point is vector"""

	def VectorMult(v1, v2, metrica):
		return np.inner(v1, np.inner(metrica, v2))

	def MetricTensor(simplex=None, vType="dec", size=2):
		if simplex == None:		return np.eye(size)
		return simplex.MetricTensor(vType)

	def Distance(P1, P2, simplex=None, vType="dec"):
		metrica = Point.MetricTensor(simplex, vType, len(P1.dec))

		if vType == "di2":	dx = P1.di2 - P2.di2
		elif vType == "bG":	dx = P1.bG - P2.bG
		elif vType == "bL":	dx = P1.bL - P2.bL
		else: 				dx = P1.dec - P2.dec
		dist = Point.VectorMult(dx, dx, metrica)
		if vType == "res" and dist != 0: dist = 1/math.sqrt(dist)
		return dist

	def Norma(self, simplex=None, vType="dec"):
		metrica = Point.MetricTensor(simplex, vType, len(self.dec))
		if vType == "di2":		vector = self.di2
		elif vType == "bG":		vector = self.bG
		elif vType == "bL":		vector = self.bL
		else:					vector = self.dec
		return Point.VectorMult(vector, vector, metrica)

	def ScalarProd(P1, P2, simplex=None, vType="dec"):
		metrica = Point.MetricTensor(simplex, vType, len(P1.dec))
		if vType == "di2":	v1, v2 = P1.di2, P2.di2
		elif vType == "bG":	v1, v2 = P1.bG, P2.bG
		elif vType == "bL":	v1, v2 = P1.bL, P2.bL
		else: 				v1, v2 = P1.dec, P2.dec
		return Point.VectorMult(v1, v2, metrica)

	def BC2dec(simplex, bcCoord):
		numPoints = simplex.size
		numCoord = numPoints - 1
		vDec = np.zeros(numCoord)
		if simplex == None: return vDec
		for i in range(numPoints):
			for j in range(numCoord):
				vDec[j] += simplex.lPoints[i].dec[j]*bcCoord[i]
		return vDec


	def vDi(self, lPoints, vType="dec"):
		size = len(lPoints)
		vDi = np.zeros(size)
		for i in range(len(lPoints)):
			vDi[i] = Point.Distance(self, lPoints[i], None, vType)
		return vDi

	def Dec2Simplex(self, simplex):
		self.di = self.vDi(simplex.lPoints, "dec")
		self.di2 = -self.di/2
		self.bG = np.inner(simplex.mL, self.di2)
		self.bL = np.inner(simplex.mG, self.bG)
		self.bc = self.bG + simplex.bcCenter # barycentric

	def __init__(self, vCoord, simplex=None, vType="dec"):
		if type(vCoord) == list:
			vCoord = np.array(vCoord)
		if vType == "dec":
			self.dec = vCoord
		elif vType == "bc":
			if simplex != None:	self.dec = Point.BC2dec(simplex, vCoord)
			else:	self.bc = vCoord
		elif vType == "bG":
			if simplex != None:
				self.bc = vCoord + simplex.bcCenter # barycentric
				self.dec = Point.BC2dec(simplex, self.bc)
			else:	self.bG = vCoord
		else:
			return
		if simplex != None:	self.Dec2Simplex(simplex)


class Matrix(object):
	'''Matrix minor'''
	def Minor(matrix, row=0, col=0):
		mTemp = np.vstack((matrix[:row,:], matrix[row+1:,:]))
		return np.hstack((mTemp[:,:col], mTemp[:,col+1:]))

	def mtrDistance(matrix, edging=False):
		vProd = np.outer(np.ones(len(matrix)), np.diag(matrix))
		mTrans = vProd + np.transpose(vProd) - 2*matrix
		if edging:
			return Matrix.VectorEdging(mTrans, np.diag(matrix), 0)
		return mTrans

	def mtrLaplacian(matrix):
		return np.eye(len(matrix))*np.sum(matrix, 0) - matrix

	def mLaplacian2Distance(mLap):
		mFund = la.inv(Matrix.Minor(mLap))
		return Matrix.mtrDistance(mFund, True)

	def VectorEdging(matrix, vector=[], scalar=0):
		size = len(matrix)
		mResult = np.vstack((vector, matrix))
		cV = np.sum(mResult, 1)
		cV.shape = (size+1, 1)
		cV[0] = scalar
		for i in range(size):	cV[i+1] = vector[i]
		return np.hstack((cV, mResult))

	def List2Matrix(lRecords, symm=True):
		#lRecords - list of records: [el1, el2, value]
		lItems = []
		#First cycle. Define of elements set
		for record in lRecords:
			lItems.append(record[0])
			lItems.append(record[1])

		lItems = list(set(lItems)) #unique elements
		numPoints = len(lItems)
		matrix = np.zeros((numPoints, numPoints))
		#Second cycle. Define of matrix
		for record in lRecords:
			ind1 = lItems.index(record[0])
			ind2 = lItems.index(record[1])
			matrix[ind1,ind2] = record[2]
			if symm: matrix[ind2,ind1] = matrix[ind1,ind2]
		return (matrix, lItems)

	def __init__(self, mValues, symm=True):
		if type(mValues) == list:
			self.matrix, self.items = Matrix.List2Matrix(mValues, symm)
		else:
			self.matrix = mValues
			self.items = []

class Simplex(object):
	"""Simplex is set of N points with known distance matrix and laplacian"""
	def mGreen(self, vWeight=[]):
		# Green:
		w2 = np.inner(self.vWeight, self.vWeight)
		vGreen = self.vWeight / w2

		mOne = np.eye(self.size) - np.outer(vGreen, self.vWeight)
		mG = np.matmul(mOne, np.matmul(self.mD2, mOne))
		return mG

	def radius(): return self.rs
	def volume(): return self.volume

	def IniMetric(self, mData, dtype='Distance'):
		if dtype == 'Distance':
			self.mD = mData
		elif dtype == 'Laplacian':
			self.mL = mData
			self.mD = Matrix.mLaplacian2Distance(self.mL)
		else: #dtype == 'Connect':
			self.mL = Matrix.mtrLaplacian(mData)
			self.mD = Matrix.mLaplacian2Distance(self.mL)

		self.Dm = Matrix.VectorEdging(-self.mD/2, self.vWeight, 0)
		self.Lm = la.inv(self.Dm)
		if dtype == 'Distance': self.mL = self.Lm[1:, 1:]

		self.rs = self.Lm[0, 0] #sphere distance
		self.dCM = la.det(self.Dm) #Caley-Menger determinant
		self.uK = -1/self.dCM #laplacian potential
		self.volume = math.sqrt(-self.dCM)/math.factorial(self.size-1) #volume of simplex

		# barycentric coordinate of sphere center
		self.bcSphere = self.Lm[0, 1:]

	def __init__(self, mData, dtype='Distance', vWeight=[]):
		self.size = np.shape(mData)[0]

		if vWeight == []:	self.vWeight = np.ones(self.size)
		else:				self.vWeight = np.array(vWeight)

		self.IniMetric(mData, dtype)





def testSimplex(lPoints):
	sTest = Simplex(lPoints, [1, 1, 1])


def Grid(xSize, ySize=1, zSize=1):
	lPoints = []
	for i in range(xSize):
		for j in range(ySize):
			for k in range(zSize):
				lPoints.append([i, j, k])
	return lPoints

def ResistanceSimplex():
	size = 5
	lGrid = Grid(size, size, size)
	simRes = Simplex(lGrid, "res")
	#print(simRes.vbCenter)
	#for i in range(size):
	#	print(simRes.vbCenter[i*size: (i+1)*size])

	P = Point([-1, 0, 0], simRes, "res")
	Q = Point([-1, 1, 0], simRes, "res")
	print(Point.Distance(P, Q, simRes, "di2"))
	print(Point.Distance(P, Q, simRes, "bal"))

#for i in range(1):
#	testSimplex([[0, 0], [0, 3+i], [4+i, 0]])

#testSimplex([[0, 0], [0, 4], [4, 0]])
#testSimplex([[0, 0], [0, 5], [5, 0]])
#ResistanceSimplex()
