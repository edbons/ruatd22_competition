
import numpy as np, pandas as pd, regex as re

from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix


def _calculate_auc(predictions,groundtruth,thresholds):

  results = {}
  
  results['auc'] = roc_auc_score(groundtruth, predictions)
  results['threshold'] = {}

  for thresh in thresholds:

    labels = predictions > thresh    

    cm = confusion_matrix(groundtruth, labels)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (tp + fn)
    acc = (tp + tn) / (tp + tn + fp + fn) 

    results['threshold'][np.round(thresh,2)] = {}
    results['threshold'][np.round(thresh,2)]['fpr'] = fpr
    results['threshold'][np.round(thresh,2)]['fnr'] = fnr
    results['threshold'][np.round(thresh,2)]['acc'] = acc
 
  return results


def _evaluate_validation(data,classifier,parameters):
    
    # validation: find best parameters
    best_param = (0,0)
    
    validation_results = {}
    
    for p in parameters:
        
        if classifier == 'LogisticRegression':
            model = LogisticRegression(solver='lbfgs',C=p)
        elif classifier == 'SVM':
            model = LinearSVC(C=p,random_state=175)
        elif classifier == 'NeuralNet':
            model = MLPClassifier(hidden_layer_sizes=[100],activation='logistic',solver='adam',alpha=p,learning_rate='adaptive',
                                 learning_rate_init=0.001,max_iter=100,random_state=175)
        elif classifier == 'RandomForest':
            model = RandomForestClassifier(n_estimators=100, max_depth=p,random_state=175)
        
        model.fit(data['train']['features'],data['train']['labels'])
        
        val_acc = model.score(data['valid']['features'],data['valid']['labels'])
        if val_acc > best_param[1]:
            best_param = (p,val_acc)
            
        validation_results[p] = [val_acc, model]
            
    # testing: get results of best classifier
    best_model = validation_results[best_param[0]][1]

    accuracies = {}
    accuracies['train_accuracy'] = best_model.score(data['train']['features'],data['train']['labels'])
    accuracies['valid_accuracy'] = best_model.score(data['valid']['features'],data['valid']['labels'])
    accuracies['test_accuracy'] = best_model.score(data['test']['features'],data['test']['labels'])

    if classifier != 'SVM':
      predictions = best_model.predict_proba(data['test']['features'])[:,1]
      groundtruth = data['test']['labels']
      full_results = _calculate_auc(predictions,groundtruth,np.arange(0.05,1,0.05))
    else:
      full_results = 'Not Defined for SVM'
                
    return full_results, accuracies, best_model