# -*- coding:utf-8 -*-
""" Геометрия сети """
import math
import numpy as np
from numpy import linalg as la

class Point(object):
	"""Point is not vector"""

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

	def Dec2PointBase(self, pBase):
		self.di = self.vDi(pBase.lPoints, "dec")
		self.di2 = -self.di/2
		self.bL = np.inner(pBase.mG, self.bG)
		self.bc = self.bG + pBase.bcCenter # barycentric

	def __init__(self, vCoord, pBase=None, vType="dec", id=''):
		self.id = id
		if type(vCoord) == list:
			vCoord = np.array(vCoord)
		if vType == "dec":
			self.dec = vCoord
			if pBase != None:	self.Dec2PointBase(pBase)
		elif vType == "bi":
			self.bi = vCoord
		elif vType == "di":
			self.di = vCoord
		else:
			return


class dbCoord(object):

	def projection(self): # projection of point to base
		vPrDistance = -2*self.di + self.norma*np.ones(len(self.di))
		return dbCoord(vPrDistance[1:], self.pBase)

	def mutNorma(self, db):
		return Vector.ScalarProd(self.di, db.bi) # mutual norma

	def distance(self, db, sign=1):
		# sign=1: points are on the same side; -1: on other sides
		return -2*(self.mutNorma(db) + sign*math.sqrt(self.norma*db.norma))

	def pdistance(self, db):
		# distance of proections
		return self.norma + db.norma - 2*self.mutNorma(db)

	def bi2di(bi, pBase):
		return np.inner(pBase.Dm, bi)

	def di2bi(di, pBase):
		return np.inner(pBase.Lm, di)

	def power(self): # power of point
		return -2*self.bi[0]

	def radius(self): # distance to base sphere center
		return self.power() + self.pBase.rs

	def centrality(self): # index of centrality
		return -self.power()/self.pBase.rs

	def __init__(self, vDistance, pBase):
		self.pBase = pBase
		self.di = np.hstack((1, -vDistance/2))
		self.bi = dbCoord.di2bi(self.di, pBase)
		self.norma = np.inner(self.di, self.bi) # negative distance to base


class Vector(object):
	'''Vector operations'''
	def ScalarProd(v1, v2):
		return np.inner(v1, v2)

	def vDistance(vPoint, lPoints, degree=2, coeff=1.):
		#Create distance vector on base of list of decart coordinates
		numComponents = len(lPoints)
		vD = np.zeros(numComponents)
		decP = np.array(vPoint)
		for i in range(numComponents):
			decI = np.array(lPoints[i])
			if len(decP) > len(decI):	decI.resize((1, len(decP)))
			elif len(decP) < len(decI):	decP.resize((1, len(decI)))
			diff = decP - decI
			dist = np.inner(diff, diff)
			if degree != 2:	dist = math.sqrt(dist)**degree
			vD[i] = dist*coeff
		return vD


class Matrix(object):
	def Bilinear(matrix, v1, v2=None):
		if v2 == None:
			return np.inner(v1, np.inner(matrix, v1))
		else:
			return np.inner(v1, np.inner(matrix, v2))

	def Minor(matrix, row=0, col=0):
		'''Matrix minor'''
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

	def mDistance(lPoints, degree=2, coeff=1.):
		#Create distance matrix on base of list of decart coordinates
		numPoints = len(lPoints)
		mD = np.zeros((numPoints, numPoints))
		for i in range(numPoints):
			vD = Vector.vDistance(lPoints[i], lPoints, degree, coeff)
			mD[i] = vD
		return mD

	def ScalarProduct(mD, lIndex):
		if len(lIndex) == 3: # vectors are: ki & kj
			i,j,k = lIndex[0], lIndex[1], lIndex[2]
			return (mD[i,k] + mD[j,k] - mD[i,j])/2
		else: # vectors are: ij & kl
			i,j,k,l = lIndex[0], lIndex[1], lIndex[2], lIndex[3]
			return (mD[i,l] + mD[j,k] - mD[i,k] - mD[j,l])/2

	def VectorEdging(matrix, vector=[], scalar=0):
		#Insert vector and scalar to matrix
		mResult = np.vstack((vector, matrix))
		cV = np.hstack((scalar, vector))
		cV.shape = (len(cV), 1)
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
			matrix[ind1,ind2] = 1
			if len(record) > 2:	matrix[ind1,ind2] = record[2]
			if symm: matrix[ind2,ind1] = matrix[ind1,ind2]
		return (matrix, lItems)

	def __init__(self, mValues, symm=True):
		if type(mValues) == list:
			self.matrix, self.items = Matrix.List2Matrix(mValues, symm)
		else:
			self.matrix = mValues
			self.items = []

class PointBase(object):
	"""PointBase is set of N points with known distance matrix and laplacian"""
	def mGreen(self, vWeight=[]): # Green matrix
		w2 = np.inner(self.vWeight, self.vWeight)
		vGreen = self.vWeight / w2

		mOne = np.eye(self.size) - np.outer(vGreen, self.vWeight)
		mG = np.matmul(mOne, np.matmul(self.mD2, mOne))
		return mG

	def radius(self): return self.rs
	def connectivity(self): return 1/self.rs
	def volume(self): #volume of set
		if (self.dCM) > 0: # wrong
			return -math.sqrt(self.dCM)/math.factorial(self.size-1)
		else:	# correct
			return math.sqrt(-self.dCM)/math.factorial(self.size-1)

	def vol2(self): #volume^2 of set
		return -self.dCM/math.factorial(self.size-1)**2

	def sq2(self): # sum of square^2 edges
		return np.trace(self.mL)/(self.uK*math.factorial(self.size-2)**2)

	def ra(self): # average half distance
		return np.sum(self.mD)/(self.size*self.size)/2

	def dcs(self): # distance from centroid to sphere center
		return self.rs - self.ra()

	def rh(self): # almost radius of in-sphere
		return 1/np.trace(self.mL) # sum of 1/h^2

	def symIndex(self): # index of symmetry
		return self.ra()/self.rs

	def hIndex(self): # index of height
		return self.rh()/self.rs

	def sqIndex(self): # index of height
		return self.sq2()/self.rs**(self.size-2)

	def vIndex(self): # index of volume
		return self.vol2()/self.rs**(self.size-1)

	def value(self, idParam):
		# value of base parameter
		if idParam == 'rs':
			return self.rs
		elif idParam == 'ra':
			return self.ra()
		elif idParam == 'rh':
			return self.rh()
		elif idParam == 'dcs':
			return self.dcs()
		elif idParam == 'vol':
			return self.volume()
		elif idParam == 'sq2':
			return self.sq2()
		elif idParam == 'sIndex':
			return self.symIndex()
		elif idParam == 'hIndex':
			return self.hIndex()
		elif idParam == 'sqIndex':
			return self.sqIndex()
		elif idParam == 'vIndex':
			return self.vIndex()
		return 0

	def IniMetric(self, mData, dtype='Distance'):
		if dtype == 'Distance':
			self.mD = mData
		elif dtype == 'Laplacian':
			self.mL = mData
			self.mD = Matrix.mLaplacian2Distance(self.mL)
		else: #dtype == 'Connect':
			self.mL = Matrix.mtrLaplacian(mData)
			self.mD = Matrix.mLaplacian2Distance(self.mL)

		self.Dm = Matrix.VectorEdging(-self.mD/2, self.vWeight, 0) 	#Distance metric tensor
		self.Lm = la.inv(self.Dm)									#Laplace metric tensor
		if dtype == 'Distance': self.mL = self.Lm[1:, 1:]

		self.rs = self.Lm[0, 0] #sphere distance
		self.dCM = la.det(self.Dm) #Caley-Menger determinant
		self.uK = -1/self.dCM #laplacian potential

		# barycentric coordinate of sphere center
		self.bcSphere = self.Lm[0, 1:]

	def __init__(self, mData, dtype='Distance', vWeight=[]):
		self.size = np.shape(mData)[0]

		if vWeight == []:	self.vWeight = np.ones(self.size)
		else:				self.vWeight = np.array(vWeight)

		self.IniMetric(mData, dtype)

class FullPointBase(PointBase):
	"""Full graph. All nodes are connected with each other"""
	def __init__(self, size=3, connect=1):
		mLap = -np.ones((size, size))*connect
		for i in range(size):
			mLap[i, i] = (size-1)*connect
		PointBase.__init__(self, mLap, 'Laplacian')
		
class ChainPointBase(PointBase):
	"""Chain. May be opened or closed"""
	def __init__(self, size=3, connect=1, closed=False):
		mAdj = np.zeros((size, size))
		for i in range(size):
			if i > 0:		mAdj[i, i-1] = connect
			if i+1 < size:	mAdj[i, i+1] = connect
		if closed:
			mAdj[0, size-1] = connect
			mAdj[size-1, 0] = connect
		PointBase.__init__(self, Matrix.mtrLaplacian(mAdj), 'Laplacian')

class StarPointBase(PointBase):
	"""Star graph. All nodes are connected with center"""
	def __init__(self, size=3, connect=1):
		mAdj = np.zeros((size, size))
		for i in range(1, size):
			mAdj[0, i] = connect
			mAdj[i, 0] = connect
		PointBase.__init__(self, Matrix.mtrLaplacian(mAdj), 'Laplacian')



def Grid(xSize, ySize=1, zSize=1):
	lPoints = []
	for i in range(xSize):
		for j in range(ySize):
			for k in range(zSize):
				lPoints.append([i, j, k])
	return lPoints

def ResistanceBase():
	size = 5
	lGrid = Grid(size, size, size)
	simRes = PointBase(lGrid, "res")
	#print(simRes.vbCenter)
	#for i in range(size):
	#	print(simRes.vbCenter[i*size: (i+1)*size])

	P = Point([-1, 0, 0], simRes, "res")
	Q = Point([-1, 1, 0], simRes, "res")
	print(Point.Distance(P, Q, simRes, "di2"))
	print(Point.Distance(P, Q, simRes, "bal"))

