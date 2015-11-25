#Damir Jajetic, 2015, MIT licence

def predict (LD, output_dir, basename):
	import copy
	import os
	import numpy as np
	import libscores
	import data_converter
	from sklearn import preprocessing, ensemble
	from sklearn.utils import shuffle

	
	LD.data['X_train'], LD.data['Y_train'] = shuffle(LD.data['X_train'], LD.data['Y_train'] , random_state=1)
	
	Y_train = LD.data['Y_train']
	X_train = LD.data['X_train']
	
	Xta = np.copy(X_train)

	X_valid = LD.data['X_valid']
	X_test = LD.data['X_test']
	
	
	Xtv = np.copy(X_valid)
	Xts = np.copy(X_test)
	

	import xgboost as xgb
	if LD.info['name']== 'albert':
		model = xgb.XGBClassifier(max_depth=6, learning_rate=0.05, n_estimators=1800, silent=True, 
				objective='binary:logistic', nthread=6, gamma=0.6, 
				min_child_weight=0.7, max_delta_step=0, subsample=1, 
				colsample_bytree=1, base_score=0.5, seed=0, missing=None)

	if LD.info['name']== 'dilbert':
		model = xgb.XGBClassifier(max_depth=4, learning_rate=0.1, n_estimators=1000, silent=True, 
				objective='multi:softprob', nthread=-1, gamma=0, 
				min_child_weight=0, max_delta_step=0, subsample=1, 
				colsample_bytree=1, base_score=0.5, seed=0, missing=None)
	if LD.info['name']== 'fabert':
		model = xgb.XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=1200, silent=True, 
				objective='multi:softprob', nthread=-1, gamma=0, 
				min_child_weight=1, max_delta_step=0, subsample=1, 
				colsample_bytree=1, base_score=0.5, seed=0, missing=None)
	if LD.info['name']== 'robert':
		model = xgb.XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=600, silent=True, 
				objective='multi:softprob', nthread=-1, gamma=0, 
				min_child_weight=1, max_delta_step=0, subsample=1, 
				colsample_bytree=1, base_score=0.5, seed=0, missing=None)
	if LD.info['name']== 'volkert':
		from sklearn import  ensemble, preprocessing
		
		p = preprocessing.PolynomialFeatures()
		prep = ensemble.RandomForestRegressor(n_estimators=24, n_jobs=-1, random_state=0, verbose=1)
		
		prep.fit(Xta,Y_train)		
		Xta = Xta [:, prep.feature_importances_.argsort()[-50:][::-1]]
		Xtv = Xtv [:, prep.feature_importances_.argsort()[-50:][::-1]]
		Xts = Xts [:, prep.feature_importances_.argsort()[-50:][::-1]]
		
		
		Xta = p.fit_transform(Xta)
		Xtv = p.fit_transform(Xtv)
		Xts = p.fit_transform(Xts)
		
		prep.fit(Xta,Y_train)		
		Xta = Xta [:, prep.feature_importances_.argsort()[-800:][::-1]]
		Xtv = Xtv [:, prep.feature_importances_.argsort()[-800:][::-1]]
		Xts = Xts [:, prep.feature_importances_.argsort()[-800:][::-1]]
							
		X_train = np.hstack([X_train, Xta])
		X_valid = np.hstack([X_valid, Xtv])
		X_test = np.hstack([X_test, Xts])
		
		model = xgb.XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=350, silent=True, 
				objective='multi:softprob', nthread=-1, gamma=0, 
				min_child_weight=1, max_delta_step=0, subsample=1, 
				colsample_bytree=1, base_score=0.5, seed=0, missing=None)
		

	model.fit(X_train, Y_train)
	
	preds_valid = model.predict_proba(X_valid)
	preds_test = model.predict_proba(X_test)
				
	
	import data_io
	if  LD.info['target_num']  == 1:
		preds_valid = preds_valid[:,1]
		preds_test = preds_test[:,1]
								
	preds_valid = np.clip(preds_valid,0,1)
	preds_test = np.clip(preds_test,0,1)
	
	data_io.write(os.path.join(output_dir, basename + '_valid_000.predict'), preds_valid)
	data_io.write(os.path.join(output_dir,basename + '_test_000.predict'), preds_test)

