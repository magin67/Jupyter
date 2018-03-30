# -*- coding:utf-8 -*-
""" Нелокальная геометрия, март 2018 """
import math
import numpy as np
from numpy import linalg as la

class Point(object):
	"""Point is not vector"""

	def QuadrForm(v1, v2, metrica):
		return np.inner(v1, np.inner(metrica, v2))

	def MetricTensor(Base=None, vType="dec", size=2):
		if Base == None:		return np.eye(size)
		return Base.MetricTensor(vType)

	def Distance(P1, P2, Base=None, vType="dec"):
		metrica = Point.MetricTensor(Base, vType, len(P1.dec))

		if vType == "di2":	dx = P1.di2 - P2.di2
		elif vType == "bG":	dx = P1.bG - P2.bG
		elif vType == "bL":	dx = P1.bL - P2.bL
		else: 				dx = P1.dec - P2.dec
		dist = Point.QuadrForm(dx, dx, metrica)
		if vType == "res" and dist != 0: dist = 1/math.sqrt(dist)
		return dist

	def Norma(self, Base=None, vType="dec"):
		metrica = Point.MetricTensor(Base, vType, len(self.dec))
		if vType == "di2":		vector = self.di2
		elif vType == "bG":		vector = self.bG
		elif vType == "bL":		vector = self.bL
		else:					vector = self.dec
		return Point.QuadrForm(vector, vector, metrica)

	def ScalarProd(P1, P2, Base=None, vType="dec"):
		metrica = Point.MetricTensor(Base, vType, len(P1.dec))
		if vType == "di2":	v1, v2 = P1.di2, P2.di2
		elif vType == "bG":	v1, v2 = P1.bG, P2.bG
		elif vType == "bL":	v1, v2 = P1.bL, P2.bL
		else: 				v1, v2 = P1.dec, P2.dec
		return Point.QuadrForm(v1, v2, metrica)

	def BC2dec(Base, bcCoord):
		numPoints = Base.size
		numCoord = numPoints - 1
		vDec = np.zeros(numCoord)
		if Base == None: return vDec
		for i in range(numPoints):
			for j in range(numCoord):
				vDec[j] += Base.lPoints[i].dec[j]*bcCoord[i]
		return vDec

	def vDi(self, lPoints, vType="dec"):
		size = len(lPoints)
		vDi = np.zeros(size)
		for i in range(len(lPoints)):
			vDi[i] = Point.Distance(self, lPoints[i], None, vType)
		return vDi

	def Dec2BaseSet(self, Base):
		self.di = self.vDi(Base.lPoints, "dec")
		self.di2 = -self.di/2
		self.bL = np.inner(Base.mG, self.bG)
		self.bc = self.bG + Base.bcCo # barycentric

	def __init__(self, vCoord, Base=None, vType="dec", id=''):
		self.id = id
		if type(vCoord) == list:
			vCoord = np.array(vCoord)
		if vType == "dec":
			self.dec = vCoord
			if Base != None:	self.Dec2PointBase(Base)
		elif vType == "bi":
			self.bi = vCoord
		elif vType == "di":
			self.di = vCoord
		else:
			return


class dbCoord(object):

	def projection(self): # projection of point to base
		vPrDistance = -2*self.di + self.norma*np.ones(len(self.di))
		return dbCoord(vPrDistance[1:], self.Base)

	def mutNorma(self, db):
		return Tuple.Convolution(self.di, db.bi) # mutual norma

	def distance(self, db, sign=1):
		# sign=1: points are on the same side; -1: on other sides
		return -2*(self.mutNorma(db) + sign*math.sqrt(self.norma*db.norma))

	def pdistance(self, db):
		# distance of proections
		return self.norma + db.norma - 2*self.mutNorma(db)

	def bi2di(bi, Base):
		return np.inner(Base.Gm, bi)

	def di2bi(di, Base):
		return np.inner(Base.Lm, di)

	def orbital(self): # power of point
		return -2*self.bi[0]

	def radius(self): # distance to base sphere center
		return self.orbital() + self.Base.ro

	def centrality(self): # index of centrality
		return -self.orbital()/self.Base.ro

	def __init__(self, tDistance, Base):
		self.Base = Base
		self.di = np.hstack((1, -tDistance/2))
		self.bi = dbCoord.di2bi(self.di, Base)
		self.norma = np.inner(self.di, self.bi) # negative distance to base


class Tuple(object):
	'''Tuple operations'''
	def Convolution(v1, v2):
		return np.inner(v1, v2)

	def Distance(tPoint, lPoints, degree=2, coeff=1.):
		#Create distance tuple on base of list of decart coordinates
		numComponents = len(lPoints)
		tD = np.zeros(numComponents)
		decP = np.array(tPoint)
		for i in range(numComponents):
			decI = np.array(lPoints[i])
			if len(decP) > len(decI):	decI.resize((1, len(decP)))
			elif len(decP) < len(decI):	decP.resize((1, len(decI)))
			diff = decP - decI
			dist = np.inner(diff, diff)
			if degree != 2:	dist = math.sqrt(dist)**degree
			tD[i] = dist*coeff
		return tD


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
			return Matrix.Edging(mTrans, np.diag(matrix), 0)
		return mTrans

	def mtrLaplacian(matrix):
		return np.eye(len(matrix))*np.sum(matrix, 0) - matrix

	def Laplacian2Gramian(mLap):
		mFund = la.inv(Matrix.Minor(mLap))
		return -Matrix.mtrDistance(mFund, True)/2

	def mDistance(lPoints, degree=2, coeff=1.):
		#Create distance matrix on base of list of decart coordinates
		numPoints = len(lPoints)
		mD = np.zeros((numPoints, numPoints))
		for i in range(numPoints):
			mD[i] = Tuple.Distance(lPoints[i], lPoints, degree, coeff)
		return mD

	def ScalarProduct(mD, lIndex):
		if len(lIndex) == 3: # vectors are: ki & kj
			i,j,k = lIndex[0], lIndex[1], lIndex[2]
			return (mD[i,k] + mD[j,k] - mD[i,j])/2
		else: # vectors are: ij & kl
			i,j,k,l = lIndex[0], lIndex[1], lIndex[2], lIndex[3]
			return (mD[i,l] + mD[j,k] - mD[i,k] - mD[j,l])/2

	def Edging(matrix, vector=[], scalar=0):
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

class BaseSet(object):
	"""Set of N (=self.size) elements with known scalar products or laplacian"""
	def mGreen(self, vWeight=[]): # Green matrix
		w2 = np.inner(self.vWeight, self.vWeight)
		vGreen = self.vWeight / w2

		mOne = np.eye(self.size) - np.outer(vGreen, self.vWeight)
		mG = np.matmul(mOne, np.matmul(self.mD2, mOne))
		return mG

	def radius(self): return self.ro
	def connectivity(self): return 1/self.ro
	def volume(self): #volume of set
		if (self.detG) > 0: # wrong
			return -math.sqrt(self.detG)/math.factorial(self.size-1)
		else:	# correct
			return math.sqrt(-self.detG)/math.factorial(self.size-1)

	def vol2(self): #volume^2 of set
		return -self.detG/math.factorial(self.size-1)**2

	def sq2(self): # sum of square^2 edges
		return np.trace(self.mL)/(self.cf*math.factorial(self.size-2)**2)

	def ra(self): # average scalar products
		return -np.sum(self.mG)/(self.size*self.size)

	def dcs(self): # distance from centroid to sphere center
		return self.ro - self.ra()

	def rh(self): # almost radius of in-sphere
		return 1/np.trace(self.mL) # sum of 1/h^2

	def symIndex(self): # index of symmetry
		return self.ra()/self.ro

	def hIndex(self): # index of height
		return self.rh()/self.ro

	def sqIndex(self): # index of height
		return self.sq2()/self.ro**(self.size-2)

	def vIndex(self): # index of volume
		return self.vol2()/self.ro**(self.size-1)

	def value(self, idParam):
		# value of base parameter
		if idParam == 'ro':
			return self.ro
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
			ones = np.ones(self.size)
			self.mG = np.outer(ones, self.norms) + np.outer(self.norms, ones) - mData/2
		elif dtype == 'Laplacian':
			self.mL = mData
			self.mG = Matrix.Laplacian2Gramian(self.mL)
		else: #dtype == 'Connect':
			self.mL = Matrix.mtrLaplacian(mData)
			self.mG = Matrix.Laplacian2Gramian(self.mL)

        # metric tensors:
		self.Gm = Matrix.Edging(self.mG, self.ch, 0) 	# Major gramian
		self.Lm = la.inv(self.Gm)						# Major laplacian 
		if dtype == 'Distance': self.mL = self.Lm[1:, 1:]

		self.ro = self.Lm[0, 0]     #orthogonal center norm - base sphere quadrance
		self.detG = la.det(self.Gm) #Caley-Menger determinant
		self.cf = -1/self.detG      #laplacian cofactor

		# barycentric coordinate of orthogonal center
		self.bcCo = self.Lm[0, 1:]

	def __init__(self, mData, dtype='Distance', norms=[], ch=[]):
		self.size = np.shape(mData)[0]
		if norms == []:	self.norms = np.zeros(self.size)
		else:			self.norms = np.array(norms)
		if ch == []:	self.ch = np.ones(self.size)
		else:			self.ch = np.array(ch)
		self.IniMetric(mData, dtype)

class FullGraph(BaseSet):
	"""Full graph. All nodes are connected with each other"""
	def __init__(self, size=3, connect=1):
		mLap = -np.ones((size, size))*connect
		for i in range(size):
			mLap[i, i] = (size-1)*connect
		BaseSet.__init__(self, mLap, 'Laplacian')

class ChainGraph(BaseSet):
	"""Chain. May be opened or closed"""
	def __init__(self, size=3, connect=1, closed=False):
		mAdj = np.zeros((size, size))
		for i in range(size):
			if i > 0:		mAdj[i, i-1] = connect
			if i+1 < size:	mAdj[i, i+1] = connect
		if closed:
			mAdj[0, size-1] = connect
			mAdj[size-1, 0] = connect
		BaseSet.__init__(self, Matrix.mtrLaplacian(mAdj), 'Laplacian')

class StarGraph(BaseSet):
	"""Star graph. All nodes are connected with center"""
	def __init__(self, size=3, connect=1):
		mAdj = np.zeros((size, size))
		for i in range(1, size):
			mAdj[0, i] = connect
			mAdj[i, 0] = connect
		BaseSet.__init__(self, Matrix.mtrLaplacian(mAdj), 'Laplacian')

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
	simRes = BaseSet(lGrid, "res")
	#print(simRes.vbCenter)
	#for i in range(size):
	#	print(simRes.vbCenter[i*size: (i+1)*size])

	P = Point([-1, 0, 0], simRes, "res")
	Q = Point([-1, 1, 0], simRes, "res")
	print(Point.Distance(P, Q, simRes, "di2"))
	print(Point.Distance(P, Q, simRes, "bal"))