import random
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import cross_validation
from GirdSearch_GBDT import GS_GBDT#CAR
from xgboost import XGBRegressor
from sklearn import preprocessing
from sklearn.decomposition import PCA
import math
f= open('./CIKM2017_train/train.txt','r')
i = 0
choiceSampleId = random.sample(range(1,101*101*60),600)
train_x = []
train_y = []
for r in f:
	i += 1
	nums = r.split(',')
	train_y.append(float(nums[1]))
	xnum = nums[2].split(' ')
	tx = []
	width = 100
	for num in choiceSampleId:
		tx.append(float(xnum[num]))
	train_x.append(tx)
	if i%1000 == 0:
		print(str(i/1000)+'/10')
f.close()
train_x = np.array(train_x)
train_y = np.array(train_y)
# print(train_x.shape)
# print(train_y.shape)
# print(i)
test_x = []
i = 0
f= open('./CIKM2017_testA/testA.txt','r')
i = 0
for r in f:
	i += 1
	nums = r.split(',')
	xnum = nums[2].split(' ')
	tx = []
	width = 100
	for num in choiceSampleId:
		tx.append(float(xnum[num]))
	test_x.append(tx)
	if i%1000 == 0:
		print(str(i/1000)+'/10')
f.close()
##print(i)
normalizer = preprocessing.Normalizer().fit(train_x)
test_x = normalizer.transform(np.array(test_x))
pca = PCA(n_components=200)
train_x = pca.fit_transform(train_x)
test_x = pca.transform(test_x)



# ##xgboost
# best_param = GS_XGBDT(train_x,train_y)#[0.03,100,3,5]#
# print('best_param')
# print(best_param)
# clr = XGBRegressor(learning_rate =best_param[0], n_estimators=best_param[1], max_depth=best_param[2],
# min_child_weight=best_param[3], gamma=0,subsample=0.8,colsample_bytree=0.8, nthread=4, seed=27)
# clr.fit(train_x,train_y)
# res = clr.predict(test_x)


# best_param = [0,0,100000000]
# for lea_rate in [0.013,0.0142,0.0144,0.0146,0.0148,0.0149,0.0141,0.015]:#,0.07,0.09,0.11,0.13,0.15,0.18,0.21]:
#   for depth in [2,3,4,5,6]:
#       clr = GradientBoostingRegressor(n_estimators=800, learning_rate=lea_rate,max_depth=depth, random_state=0, loss='ls')
#       clr.fit(train_x,train_y)
#       model = SelectFromModel(clr, prefit=True)#SelectKBest(chi2, k=20).fit(train_x, train_y)
#       train_xx = model.transform(train_x)
#       preresult = cross_val_predict(clr, train_xx, train_y, cv=5)
#       prmse = math.sqrt((((preresult - train_y)**2).sum())/len(preresult))
#       if prmse < best_param[2]:
#           best_param[0] = lea_rate
#           best_param[1] = depth
#           best_param[2] = prmse
best_param = GS_GBDT(train_x,train_y)
print('best_param')
print(best_param)#GS_XGBDT
clr = GradientBoostingRegressor(n_estimators=best_param[2], learning_rate=best_param[0],max_depth=best_param[1], random_state=0, loss='ls')
clr.fit(train_x,train_y)
##model = SelectFromModel(clr, prefit=True)#SelectKBest(chi2, k=20).fit(train_x, train_y)
##train_x = model.transform(train_x)
##test_x = model.transform(test_x)
##clr.fit(train_x,train_y)
res = clr.predict(test_x)#CAR(train_x,train_y,test_x,2)
f = open('gbdtpreRes.csv','w',encoding='UTF-8')
for i in range(2000):
    #num = random.random()*6
    if res[i] <= 0.001:
        res[i] = train_y.min()
    f.write(str(res[i])+'\n')
f.close()
# scores = cross_val_score(clr, train_x, train_y, cv=5)
# print(scores.mean())
