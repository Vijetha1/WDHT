from scipy import io as sio
from scipy.misc import imresize
import numpy as np
import pdb
from scipy.spatial.distance import cdist

def getData(dataset='CIFAR10', channels_last=True):
	if dataset == 'CIFAR10':
		#This matrix is made by the MATLAB/MatConvNet/DPSH_IJCAI_ version 1.0_beta23 code. As per the code, the data should be in RGB format(verified visually) 
		data = sio.loadmat('./datasets/cifar-10.mat')
	trainData = data['train_data']
	trainLabels = data['train_L']
	queryData = data['test_data']
	queryLabels = data['test_L']
	galleryData = data['data_set']
	galleryLabels = data['dataset_L']
	if channels_last:
		trainData = np.transpose(trainData, (3, 0, 1, 2))
		queryData = np.transpose(queryData, (3, 0, 1, 2))
		galleryData = np.transpose(galleryData, (3, 0, 1, 2))
	else:
		raise NotImplementedError
	return trainData, trainLabels, queryData, queryLabels, galleryData, galleryLabels

def makeDataSet(imageIds, labels, labelType = 'oneHot', nImagesPerClassTrain=500, nImagesPerClassTest = 100):
	if labelType == 'oneHot':
		nClasses = labels.shape[1]
	nTrainImages = int(imageIds.shape[0]*0.7)
	trainImages = imageIds[0:nTrainImages]
	testImages = imageIds[nTrainImages:]
	trainLabels = labels[0:nTrainImages]
	testLabels = labels[nTrainImages:]
	trainSetImageIds = np.zeros((nClasses, nImagesPerClassTrain), dtype='uint32')
	trainSetLabels = np.zeros((nClasses, nImagesPerClassTrain, nClasses))
	for i in range(nClasses):
		consider = trainLabels[:, i] == 1
		curImageIds = trainImages[consider]
		curLabels = trainLabels[consider,:]
		curImageIds, curLabels = shuffleInUnison(curImageIds, curLabels)
		trainSetImageIds[i,:] = np.reshape(curImageIds[0:nImagesPerClassTrain], (nImagesPerClassTrain,))
		trainSetLabels[i, :, :] = curLabels[0:nImagesPerClassTrain]
	testSetImageIds = np.zeros((nClasses, nImagesPerClassTest), dtype='uint32')
	testSetLabels = np.zeros((nClasses, nImagesPerClassTest, nClasses))
	for i in range(nClasses):
		consider = testLabels[:, i] == 1
		curImageIds = testImages[consider]
		curLabels = testLabels[consider,:]
		curImageIds, curLabels = shuffleInUnison(curImageIds, curLabels)
		testSetImageIds[i,:] = np.reshape(curImageIds[0:nImagesPerClassTest], (nImagesPerClassTest,))
		testSetLabels[i, :, :] = curLabels[0:nImagesPerClassTest]
	trainSetImageIds = np.reshape(trainSetImageIds, (nClasses*nImagesPerClassTrain,))
	trainSetLabels = np.reshape(trainSetLabels, (nClasses*nImagesPerClassTrain, nClasses))
	testSetImageIds = np.reshape(testSetImageIds, (nClasses*nImagesPerClassTest,))
	testSetLabels = np.reshape(testSetLabels, (nClasses*nImagesPerClassTest,nClasses))
	return trainSetImageIds, trainSetLabels, testSetImageIds, testSetLabels


def resizeImages(images, resizeHeight=256, resizeWidth = 256):
	resizedImages = np.zeros((images.shape[0], 3, resizeHeight, resizeWidth))
	for i in range(resizedImages.shape[0]):
		resizedImages[i,:,:,:] = np.transpose(imresize(images[i], (resizeHeight, resizeWidth)), (2, 0, 1))
	return resizedImages


def cropImages(images, cropHeight=227, cropWidth=227):
	croppedImages = np.zeros((images.shape[0], 3, cropHeight, cropWidth))
	for i in range(croppedImages.shape[0]):
		randX = np.random.randint(images.shape[2]-cropHeight)
		randY = np.random.randint(images.shape[3]-cropWidth)
		croppedImages[i,:,:,:] = images[i,:,randX:randX+cropHeight,randY:randY+cropWidth]
	return croppedImages


def meanSubtract(images, sourceDataSet='IMAGENET', order='RGB'):
	if order == 'RGB':
	    images[:, 0, :, :] -= 123.68
	    images[:, 1, :, :] -= 116.779
	    images[:, 2, :, :] -= 103.939 # values copied from https://github.com/heuritech/convnets-keras/blob/master/convnetskeras/convnets.py
	elif order == 'BGR':
	    images[:, 0, :, :] -= 103.939
	    images[:, 1, :, :] -= 116.779
	    images[:, 2, :, :] -= 123.68 # values copied from https://github.com/heuritech/convnets-keras/blob/master/convnetskeras/convnets.py
	    #this is the order in which the keras.applications models are trained. 
	return images


def shuffleInUnison(images, labels):
	perm = np.random.permutation(images.shape[0])
	images = images[perm]
	labels = labels[perm]
	return images, labels


def generatePairs(images, labels, batch_size):
	n_classes = np.unique(labels).shape[0]
	images_classwise = np.zeros((n_classes, images.shape[0]/n_classes, images.shape[1], images.shape[2], images.shape[3]))
	for i in range(n_classes):
		curClass = labels == i
		images_classwise[i,:,:,:,:] = images[curClass,:,:,:]
	randomLabels = np.random.randint(10, size=batch_size)
	simLabels = randomLabels[0:batch_size/2]
	dissimLabels = randomLabels[batch_size/2:batch_size]
	imagePairs = []
	similarity = []
	queryLabs = []
	databaseLabs = []
	for i in range(len(simLabels)):
		randomImgNums = np.random.randint(images.shape[0]/n_classes, size=2)
		imagePairs.append([images_classwise[simLabels[i], randomImgNums[0], :, :, :], images_classwise[simLabels[i], randomImgNums[1], :, :, :]])
		similarity.append(1)
		queryLabs.append(simLabels[i])
		databaseLabs.append(simLabels[i])
	for i in range(len(dissimLabels)):
		randomImgNums = np.random.randint(images.shape[0]/n_classes, size=2)
		secondImageClass = dissimLabels[i]
		while(secondImageClass==dissimLabels[i]):
			secondImageClass = np.random.randint(n_classes)
		imagePairs.append([images_classwise[dissimLabels[i], randomImgNums[0], :, :, :], images_classwise[secondImageClass, randomImgNums[1], :, :, :]])
		similarity.append(0)
		queryLabs.append(dissimLabels[i])
		databaseLabs.append(secondImageClass)
	imagePairs = np.array(imagePairs)
	similarity = np.array(similarity)
	return imagePairs, similarity, np.asarray(queryLabs), np.asarray(databaseLabs)


def prepareData(dataset='CIFAR10'):
	trainData, trainLabels, queryData, queryLabels, galleryData, galleryLabels = getData(dataset=dataset)
	return trainData, trainLabels, queryData, queryLabels, galleryData, galleryLabels

def multiLabelGetVectors(data, dim=300, nClasses=81, nTags=1000, method='mean'):
	vecMat = np.zeros((len(data), dim))
	labels = np.zeros((len(data), nClasses))
	images = np.zeros((len(data), 1))
	tags = np.zeros((len(data), nTags))
	for i in range(len(data)):
		curRec = data[i]
		curVector = np.zeros((300, ))
		images[i] = curRec[0]
		labels[i] = curRec[1]
		if method == 'mean':
			for j in range(len(curRec[2])):
				curVector = curVector + curRec[2][j]
				tags[i][int(curRec[2][j][1])] = 1
			vecMat[i][:] = curVector/float(len(curRec[2]))
		elif method == 'idf':
			#pdb.set_trace()
			avg = 0
			for j in range(len(curRec[2])):
				curVector = curVector +[x* curRec[2][j][2] for x in curRec[2][j][0]]
				tags[i][int(curRec[2][j][1])] = 1
				avg = avg + curRec[2][j][2]
			vecMat[i][:] = curVector/float(avg)
		elif method == 'minFreq':
			minFreq = 100
			minFrqIndex = -100
			for j in range(len(curRec[2])):
				if curRec[2][j][2] < minFreq:
					tags[i][int(curRec[2][j][1])] = 1
					minFreq = curRec[2][j][2]
					minFreqIndex = j
			vecMat[i][:] = curRec[2][minFreqIndex][0]
		elif method == 'cutFreq':
			avg = 0.00001
			for j in range(len(curRec[2])):
				if curRec[2][j][2] > 5.3 and curRec[2][j][2] < 8.2:
					curVector = curVector +[x* curRec[2][j][2] for x in curRec[2][j][0]]
					tags[i][int(curRec[2][j][1])] = 1
					avg = avg + curRec[2][j][2]
			vecMat[i][:] = curVector/float(avg)
	images = np.array(images, dtype='uint32')			
	return (images, labels, vecMat, tags)

def multiLabelGetVectorsNUS(data, dim=300, nClasses=81, nTags=1000, method='mean'):
	vecMat = np.zeros((len(data), dim))
	labels = np.zeros((len(data), nClasses))
	images = np.zeros((len(data), 1))
	tags = []
	for i in range(len(data)):
		curRec = data[i]
		curVector = np.zeros((300, ))
		images[i] = curRec[0]
		labels[i] = curRec[1]
		if method == 'mean':
			for j in range(len(curRec[2])):
				curVector = curVector + curRec[2][j]
			vecMat[i][:] = curVector/float(len(curRec[2]))
			tags.append(curRec[3])
		elif method == 'idf':
			avg = 0
			for j in range(len(curRec[2])):
				curVector = curVector +[x* curRec[2][j][2] for x in curRec[2][j][0]]
				avg = avg + curRec[2][j][2]
			vecMat[i][:] = curVector/float(avg)
		elif method == 'minFreq':
			minFreq = 100
			minFrqIndex = -100
			for j in range(len(curRec[2])):
				if curRec[2][j][2] < minFreq:
					minFreq = curRec[2][j][2]
					minFreqIndex = j
			vecMat[i][:] = curRec[2][minFreqIndex][0]
		elif method == 'cutFreq':
			avg = 0.00001
			for j in range(len(curRec[2])):
				if curRec[2][j][2] > 5.3 and curRec[2][j][2] < 8.2:
					curVector = curVector +[x* curRec[2][j][2] for x in curRec[2][j][0]]
					avg = avg + curRec[2][j][2]
			vecMat[i][:] = curVector/float(avg)
	images = np.array(images, dtype='uint32')			
	return (images, labels, vecMat, tags)

def multiLabelGetVectorsDelete(data, dim=300, nClasses=81, nTags=1000, method='mean'):
	vecMat = np.zeros((len(data), dim))
	labels = np.zeros((len(data), nClasses))
	images = np.zeros((len(data), 1))
	tags = np.zeros((len(data), nTags))
	for i in range(len(data)):
		curRec = data[i]
		images[i] = curRec[0]
		labels[i] = curRec[1]
	images = np.array(images, dtype='uint32')			
	return (images, labels)

def getTotalWeights(weightsShape):
	totalWeights = 1
	for i in range(len(weightsShape)):
		totalWeights = totalWeights*weightsShape[i]
	return totalWeights


def checkIfWeightsAreNotLost(model_1, model_2, layerList):
	for i in range(len(layerList)):
		sameWeights = False
		weights1 = model_1.layers[layerList[i]].get_weights()[0]
		weights2 = model_2.layers[layerList[i]].get_weights()[0]
		if weights1.shape != weights2.shape:
			print("Weights Shapes did not match")
			break
		else:
			totalNumberOfWeights = getTotalWeights(weights1.shape)
			if np.sum(weights1 == weights2) != totalNumberOfWeights:
				print("Weights are different")
				break
			else:
				sameWeights = True
	return sameWeights

def preprocessLabels(labels):
	temp = np.sum(labels, axis =0)
	temp = np.argsort(temp)
	temp = temp[-21:]
	labels = labels[:,temp]
	temp = np.array(np.sum(labels, axis=-1) !=0, dtype='bool')
	labels = labels[temp]
	return labels

def computeSimilarityMatrix(queryLabels, databaseLabels, typeOfData='singleLabelled', type='interOverUnion'):
	count = 0
	groundTruthSimilarityMatrix = np.zeros((queryLabels.shape[0], databaseLabels.shape[0]))
	if typeOfData=='singleLabelled':
		for i in range(queryLabels.shape[0]):
			groundTruthSimilarityMatrix[i,:] = queryLabels[i] == databaseLabels
	elif typeOfData=='multiLabelled':
		for i in range(queryLabels.shape[0]):
			curQue = queryLabels[i][:]
			if sum(curQue) != 0:
				threshold = 1
				sim = np.sum(np.logical_and(curQue, databaseLabels), axis=-1)
				den = np.sum(np.logical_or(curQue, databaseLabels), axis=-1)
				count = count + np.sum(np.logical_and(sum(curQue) > 1, sim == 1))
				if type=='zeroOne':		
					groundTruthSimilarityMatrix[i][np.where(sim >= threshold)[0]] = 1
				elif type=='interOverUnion':
					groundTruthSimilarityMatrix[i][:] = np.divide(np.array(sim,dtype='float32'),(np.array(den,dtype='float32')+0.00001))						
				# for j in range(databaseLabels.shape[0]):
				# 	curDb = databaseLabels[j][:]
				# 	sim = np.sum(np.logical_and(curQue, curDb), axis=-1)
				# 	den = np.sum(np.logical_or(curQue, curDb), axis=-1)
				# 	if type=='zeroOne':
				# 		#pdb.set_trace()
				# 		if sim >= threshold:
				# 			groundTruthSimilarityMatrix[i][j] = 1
				# 	elif type=='interOverUnion':
				# 		groundTruthSimilarityMatrix[i][j] = float(sim)/(float(den)+0.00001)
	if type=='zeroOne':
		groundTruthSimilarityMatrix = np.asarray(groundTruthSimilarityMatrix, dtype='float32')
	elif type=='interOverUnion':
		groundTruthSimilarityMatrix = groundTruthSimilarityMatrix > 0.25
		groundTruthSimilarityMatrix = np.asarray(groundTruthSimilarityMatrix, dtype='float32')
	return groundTruthSimilarityMatrix


def calcHammingRank(queryHashes, databaseHashes, space='Hamming'):
	hammingDist = np.zeros((queryHashes.shape[0], databaseHashes.shape[0]))
	hammingRank = np.zeros((queryHashes.shape[0], databaseHashes.shape[0]))
	if space == 'Hamming':
		for i in range(queryHashes.shape[0]):
			hammingDist[i] = np.reshape(np.sum(np.abs(queryHashes[i] - databaseHashes), axis=1), (databaseHashes.shape[0], ))
			hammingRank[i] = np.argsort(hammingDist[i])
	elif space == 'RealValued':
		for i in range(queryHashes.shape[0]):
			if i % 100 == 0:
				print(i)
			hammingDist[i] = cdist(np.reshape(queryHashes[i], (1, 300)),databaseHashes ,  'cosine')
			hammingRank[i] = np.argsort(hammingDist[i])
	return hammingDist, hammingRank


def calcMAP(groundTruthSimilarityMatrix, hammingRank, hammingDist):
	[Q, N] = hammingRank.shape
	pos = np.arange(N)+1
	MAP = 0
	numSucc = 0
	for i in range(Q):
		ngb = groundTruthSimilarityMatrix[i, np.asarray(hammingRank[i,:], dtype='int32')]
		ngb = ngb[0:N]
		nRel = np.sum(ngb)
		if nRel > 0:
			prec = np.divide(np.cumsum(ngb), pos)
			prec = prec[0:5000]
			ngb = ngb[0:5000]
			ap = np.mean(prec[np.asarray(ngb, dtype='bool')])
			rec = np.array(np.cumsum(ngb)/float(np.sum(groundTruthSimilarityMatrix[i])), dtype='float32')
			if i == 0:
				precisions = prec
				recalls = rec
			else:
				precisions = precisions + prec
				recalls = recalls + rec
			MAP = MAP + ap
			numSucc = numSucc + 1
	precisions = precisions/float(Q)
	recalls = recalls/float(Q)
	MAP = float(MAP)/numSucc
	precisions = []
	recalls = []
	for j in range(8):
		countOrNot = np.array(hammingDist <= j, dtype='int32')
		newSim = np.multiply(groundTruthSimilarityMatrix, countOrNot)
		countOrNot = countOrNot + 0.000001
		prec = np.mean(np.divide(np.sum(newSim, axis=-1), np.sum(countOrNot, axis=-1)))# float(np.sum(np.sum(newSim)))/float(np.sum())
		rec = np.mean(np.divide(np.sum(newSim, axis=-1), np.sum(groundTruthSimilarityMatrix, axis=-1)))
		precisions.append(prec)
		recalls.append(rec)
	return MAP, precisions, recalls


def getMAP(queryLabels, databaseLabels, queryHashes, databaseHashes, curType, typeOfData='singleLabelled', space='Hamming'):
	if typeOfData == 'singleLabelled':
		groundTruthSimilarityMatrix = computeSimilarityMatrix(queryLabels, databaseLabels)
	elif typeOfData == 'multiLabelled':
		groundTruthSimilarityMatrix = computeSimilarityMatrix(queryLabels, databaseLabels, typeOfData='multiLabelled', type = curType)
	hammingDist, hammingRank = calcHammingRank(queryHashes, databaseHashes, space)
	MAP, precisions, recalls = calcMAP(groundTruthSimilarityMatrix, hammingRank, hammingDist)
	precisions = []
	recalls = []
	countOrNot = np.array(hammingDist <= 2, dtype='int32')
	newSim = np.multiply(groundTruthSimilarityMatrix, countOrNot)
	#pdb.set_trace()
	countOrNot = countOrNot + 0.000001
	prec = np.mean(np.divide(np.sum(newSim, axis=-1), np.sum(countOrNot, axis=-1)))# float(np.sum(np.sum(newSim)))/float(np.sum())
	rec = np.mean(np.divide(np.sum(newSim, axis=-1), np.sum(groundTruthSimilarityMatrix, axis=-1)))
	# for i in range(12):
	# 	countOrNot = np.array(hammingDist <= i, dtype='int32')
	# 	newSim = np.multiply(groundTruthSimilarityMatrix, countOrNot)
	# 	#pdb.set_trace()
	# 	countOrNot = countOrNot + 0.000001
	# 	prec = np.mean(np.divide(np.sum(newSim, axis=-1), np.sum(countOrNot, axis=-1)))# float(np.sum(np.sum(newSim)))/float(np.sum())
	# 	rec = np.mean(np.divide(np.sum(newSim, axis=-1), np.sum(groundTruthSimilarityMatrix, axis=-1)))
	# 	precisions.append(prec)
	# 	recalls.append(rec)
	return (MAP, prec, rec)

def computeMAPRealValuedSpace(queryLabels, databaseLabels, queryVectors, databaseVectors):
	pass

def computeConfusion(simMat, dist):
	#temp = np.array(np.exp(-1*dist) > 0.5, dtype='int32')
	temp = np.array(3.0 - dist > 0, dtype='int32')
	#pdb.set_trace()
	tps = np.sum(np.logical_and(temp == 1, simMat == 1))
	tns = np.sum(np.logical_and(temp == 0, simMat == 0))
	fps = np.sum(np.logical_and(temp == 1, simMat == 0))
	fns = np.sum(np.logical_and(temp == 0, simMat == 1))
	return (tps, tns, fps, fns)



def precisionAtK(queryLabels, databaseLabels, queryHashes, databaseHashes, k, curType, typeOfData='singleLabelled'):
	if typeOfData == 'singleLabelled':
		groundTruthSimilarityMatrix = computeSimilarityMatrix(queryLabels, databaseLabels)
	elif typeOfData == 'multiLabelled':
		groundTruthSimilarityMatrix = computeSimilarityMatrix(queryLabels, databaseLabels, typeOfData='multiLabelled', type = curType)
	hammingDist, hammingRank = calcHammingRank(queryHashes, databaseHashes)
	countOrNot = np.array(hammingDist == k, dtype='int32')
	newSim = np.multiply(groundTruthSimilarityMatrix, countOrNot)
	precAtK = float(np.sum(np.sum(newSim)))/float(np.sum(np.sum(countOrNot)))
	return precAtK

def getWeightShapesFromModel(model, library='Keras'):
	"""
	Desc:

	Args:

	Returns:


	"""
	# pdb.set_trace()
	weightShapes=[]
	if library == 'Keras':
		nLayers = len(model.layers)
		for i in range(nLayers):
			nParamSets = len(model.layers[i].get_weights())
			assert nParamSets%2 == 0
			for j in range(int(nParamSets/2)):
				weightShapes.append([model.layers[i].get_weights()[2*j].shape, model.layers[i].get_weights()[2*j+1].shape])
				print(weightShapes[-1])
	return weightShapes
