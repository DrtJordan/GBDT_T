import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import cross_validation
from sklearn.cluster import KMeans
from xgboost import XGBRegressor
from sklearn import cross_validation, metrics   
from sklearn.grid_search import GridSearchCV 
import pandas as pd
import math

# 返回start到stop以step为步长的浮点数排列
def float_range(start,stop,step):
	flist = []
	flag = start
	while (flag + step * 0.9) < stop:
		flist.append(flag)
		flag = flag + step
	return flist

#搜索最佳参数
def GS_GBDT(train_x,train_y):
	best_param = [0,0,100000000,0]
	for n_esti in [800]:
		for lea_rate in [0.005,0.01,0.02,0.03]:# [0.06,0.07,0.08,0.09,0.10,0.12,0.13,0.14,0.15,0.16][0.0142,0.0144,0.0146]:#,0.07,0.09,0.11,0.13,0.15,0.18,0.21]:
			for depth in [3,4,5]:
				clr = GradientBoostingRegressor(n_estimators=n_esti, learning_rate=lea_rate,max_depth=depth, random_state=0, loss='ls')
				clr.fit(train_x,train_y)
				model = SelectFromModel(clr, prefit=True)
				train_xx = model.transform(train_x)
				preresult = cross_val_predict(clr, train_xx, train_y, cv=5)
				prmse = math.sqrt((((preresult - train_y)**2).sum())/len(preresult))
				if prmse < best_param[2]:
					best_param[0] = lea_rate
					best_param[1] = depth
					best_param[2] = n_esti
					best_param[3] = prmse
	return best_param
#搜索最佳参数
def GS_XGBDT(train_x,train_y):
        best_param = [0,0,0,0]
        param_set = {'learning_rate':list(float_range(0.01,0.05,0.01)),'n_estimators':list(range(100,200,100)),'max_depth':list(range(3,5,1)),'min_child_weight':list(range(1,6,2))}
        gsearchp = GridSearchCV(estimator = XGBRegressor(learning_rate =0.1, n_estimators=100, max_depth=5,
        min_child_weight=1, gamma=0, subsample=0.8,colsample_bytree=0.8, nthread=4, seed=27),param_grid = param_set, cv=5)
        gsearchp.fit(train_x,train_y)
        print(gsearchp.grid_scores_)
        print(gsearchp.best_params_)
        print(gsearchp.best_score_)
        best_param[0] = gsearchp.best_params_['learning_rate']
        best_param[1] = gsearchp.best_params_['n_estimators']
        best_param[2] = gsearchp.best_params_['max_depth']
        best_param[3] = gsearchp.best_params_['min_child_weight']
        return best_param
#先聚类后分类再回归
def CAR(train_x,train_y,test_x,n_cluster = 4):
	kmeans = KMeans(n_clusters = n_cluster, random_state=0).fit(train_x)
	preLables = kmeans.predict(test_x)
	preResult = np.zeros(len(test_x))
	for c in set(preLables):
		trainX = np.array([])
		trainY = np.array([])
		testX = np.array([])
		for index in range(len(kmeans.labels_)):
			if kmeans.labels_[index] == c:
				if len(trainX) < 1:
					trainX = train_x[index]
					trainY = train_y[index]
				else:
					# print(trainX.shape)
					# print(np.array(train_x[index]).shape)
					trainX = np.vstack([trainX , np.array(train_x[index])])
					trainY = np.vstack([trainY , np.array(train_y[index])])
		trainY = trainY.ravel()
		for index in range(len(preLables)): 
			if preLables[index] == c:
				if len(train_x) < 1:
					testX = test_x[index]
				else:
					testX = np.vstack([trainX , np.array(test_x[index])])
		## 过滤不存在的类别
		if len(testX) < 1:
			continue

		##回归
		best_param = GS_GBDT(trainX,trainY)
		clf = GradientBoostingRegressor(n_estimators=best_param[3], learning_rate=best_param[0],max_depth=best_param[1], random_state=0, loss='ls')
		clf.fit(trainX,trainY)
		model = SelectFromModel(clf, prefit=True)
		trainX = model.transform(trainX)	    
		testX =  model.transform(testX)
		clf.fit(trainX,trainY)
		preresult = clf.predict(testX)
		i = 0
		for index in range(len(preLables)): 
			if preLables[index] == c:
				preResult[index] = preresult[i]
				i += 1 
		scores = cross_val_score(clf, trainX, trainY, cv=10)
		print(scores.mean())
	return preResult
if __name__ == '__main__':
	print(float_range(0.1,0.5,0.1))
