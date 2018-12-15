from sklearn import svm as sk_svm
from fresh_config import cfg,cfgForCaching
from fresh_util import compute_accuracy,Cache
import numpy as np

def compute_separability(train_data,test_data,clusters):
    if cfg.verbose: print("-> preparing data for svm")
    prepare_data_for_svm(train_data,test_data,clusters) # overwrites 'exp_samples' field
    if cfg.verbose: print("-> compute_separability_by_dataset_class")
    sep_by_dsclass = compute_separability_by_dataset_class(train_data,test_data,clusters)
    if cfg.verbose: print("-> compute_separability_by_correctness")
    sep_by_correct = compute_separability_by_correctness(train_data,test_data,clusters)
    train_seps = [sep_by_dsclass[0],sep_by_correct[0]]
    test_seps = [sep_by_dsclass[1],sep_by_correct[1]]
    return train_seps,test_seps
    
def compute_separability_by_dataset_class(train_data,test_data,clusters):
    return compute_separability_template(train_data,test_data,clusters,"svm_by_dataset_class_cache.pkl",'dataset')

def compute_separability_by_correctness(train_data,test_data,clusters):
    return compute_separability_template(train_data,test_data,clusters,"svm_by_correctness_cache.pkl",'correctness')

def compute_separability_template(train_data,test_data,clusters,filename,label_field):
    svmCache = Cache(filename,cfgForCaching,"svm")
    sets_separability = svmCache.load()
    if svmCache.is_valid: return sets_separability[0],sets_separability[1]

    if label_field == 'correctness': train_data.exp_labels,test_data.exp_labels = train_data.correct,test_data.correct
    elif label_field == 'dataset': train_data.exp_labels,test_data.exp_labels = train_data.ds_labels,test_data.ds_labels
        
    train_separability,test_separability = compute_new_separability(train_data,test_data,clusters)

    svmCache.save([train_separability,train_separability])
    return train_separability,test_separability

def compute_new_separability(train_data,test_data,clusters):

    train_samples = train_data.exp_samples
    train_labels = train_data.exp_labels
    test_samples = test_data.exp_samples
    test_labels = test_data.exp_labels

    clf = sk_svm.LinearSVC(class_weight='balanced')
    if cfg.verbose: print("[compute_new_sep] pre-sk_svm.fit")
    svm = clf.fit(train_samples,train_labels)
    if cfg.verbose: print("[compute_new_sep] post-sk_svm.fit")

    preds = svm.predict(train_samples)
    train_separability = compute_accuracy(train_labels,preds)
    preds = svm.predict(test_samples)    
    test_separability = compute_accuracy(test_labels,preds)
    print(train_separability,test_separability,cfg.density_estimation.nClusters)
    return [train_separability,test_separability]

def prepare_data_for_svm(train_data,test_data,clusters): # overwrites 'exp_samples' field

    combo_order = [ comboID for comboID in train_data.samples.keys() ]
    prepare_new_data_for_svm(train_data,clusters,combo_order,'train')
    prepare_new_data_for_svm(test_data,clusters,combo_order,'test')

    return train_data,test_data

def prepare_new_data_for_svm(combo_data,clusters,combo_order,setID):
    exp_samples_list = []
    for comboID in combo_order:
        # svmDataCache = Cache("svm_data_cache.pkl",cfg,comboID+'_'+setID)
        # exp_samples = svmDataCache.load()
        # if svmDataCache.is_valid:
        #     exp_samples_list.append(exp_samples)
        #     continue
        exp_samples = prepare_new_data_for_svm_by_combo(combo_data.samples[comboID],clusters[comboID])
        exp_samples_list.append(exp_samples)
        # svmDataCache.save(exp_samples)
    combo_data.exp_samples = np.hstack(exp_samples_list)    
    
def prepare_new_data_for_svm_by_combo(samples,clusters):
    numberOfClusters = clusters.cluster_centers_.shape[0]
    numberOfSamples = samples.shape[0]
    clusterIdForEachSample = clusters.predict(samples)
    clusterDifferencesFromCenter = computeDifference(samples,clusters.cluster_centers_[clusterIdForEachSample])
    # ^this^ is the ONLY place were a "route" notion would change the output
    reformSamples = np.zeros((numberOfSamples,numberOfClusters+1))
    reformSamples[np.arange(numberOfSamples)[:np.newaxis],clusterIdForEachSample+1] = 1
    reformSamples[:,0] = clusterDifferencesFromCenter
    return reformSamples

def computeDifference(sampleA,sampleB):
    return np.mean(np.abs(sampleA - sampleB),axis=1)

