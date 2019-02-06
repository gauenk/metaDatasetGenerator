import cPickle,os
import numpy as np
from easydict import EasyDict as edict
from cache.test_results_cache import TestResultsCache

class aggregateActivations():

    def __init__(self,activationsCfg,cfg,imdb):
        self.save_dir = activationsCfg.GET_SAVE_DIR(cfg.EXP_DIR)
        self.save_bool = activationsCfg.SAVE_BOOL
        self.agg_type = activationsCfg.SAVE_OBJ
        self.layer_names = activationsCfg.LAYER_NAMES
        self.agg_obj = self._init_new_agg_obj()
        self.save_cache = TestResultsCache(self.save_dir,cfg,imdb.config,None,'agg_acitvations')

    def _init_new_agg_obj(self):
        agg_obj = dict.fromkeys(self.layer_names)
        if self.agg_type == 'order':
            for key in agg_obj.keys():
                agg_obj[key] = []
        elif self.agg_type == 'image_id':
            for key in agg_obj.keys():
                agg_obj[key] = {}
        return agg_obj

    def aggregate(self,activations,image_id):
        if self.save_bool is False:
            return
        if self.agg_type == 'order':
            self.aggregateByOrder(activations)
        elif self.agg_type == 'image_id':
            self.aggregateByImageId(activations,image_id)        
        else:
            raise ValueError ("unknown [agg_activity_values.py] self.agg_type: {}".format(self.agg_type))

    def aggregateByOrder(self,activations_by_layer):
        for layer_name in self.layer_names:
            layer_activations = activations_by_layer[layer_name].ravel()
            self.agg_obj[layer_name].append(layer_activations)
        
    def aggregateByImageId(self,activations_by_layer,image_id):
        for layer_name in self.layer_names:
            layer_activations = activations_by_layer[layer_name]
            self.agg_obj[layer_name][image_id] = layer_activations

    def numpyifyAggregateByOrder(self):
        for layer_name in self.layer_names:
            self.agg_obj[layer_name] = np.array(self.agg_obj[layer_name])

    def clear(self):
        self.agg_obj = self._init_new_agg_obj()

    def load(self,layers_to_load):
        ## LOADING THE ACTIVATIONS FROM A DIFFERENT EXPERIMENT IS CHALLENGING?
        # MAYBE "EXPORT" THE CONFIG FROM AN EXPERIMENT TO LOAD OTHER INFORMATION IN THE FUTURE?
        # (I THINK) THIS SHOULD BE A "CONFIG" FUNCTIONALITY.
        net_name = self.save_cache.test_cache_config.modelInfo.name
        load_information = edict()
        load_information.mode = 'regex'
        print("activations from loaded from")
        if self.agg_type == 'order':
            for layer_name in layers_to_load:
                self.save_cache.test_cache_config.id = layer_name
                load_information.regex = '.*{}_{}.*npy'.format(net_name,layer_name)
                print(load_information.regex)
                self.agg_obj[layer_name] = self.save_cache.load(saveType='ndarray',load_information=load_information)
        elif self.agg_type == 'image_id':
            for layer_name in layers_to_load:
                self.save_cache.test_cache_config.id = layer_name
                load_information.regex = '.*{}_{}.*pkl'.format(net_name,layer_name)
                print(load_information.regex)
                self.agg_obj[layer_name] = self.save_cache.load(saveType='pickle',load_information=load_information)
        else:
            raise ValueError ("unknown [agg_activations.py] self.agg_type: {}".format(self.agg_type))        

        return self.agg_obj
        
    def load_and_verify(self,layers_to_load):
        self.load(layers_to_load)
        return self.verify()

    def verify(self):
        all_loaded_bool = True
        for layer_name,activations in self.agg_obj.items():
            if activations is None:
                print("[agg_activations] layer_name [{}] is None".format(layer_name))
                all_loaded_bool = False
        if all_loaded_bool is False:
            self.agg_obj = self._init_new_agg_obj()
        return all_loaded_bool

    def save(self):
        if self.save_bool is False: return
        net_name = self.save_cache.test_cache_config.modelInfo.name
        print("activations from model saved @")
        if self.agg_type == 'order':
            self.numpyifyAggregateByOrder()
            for layer_name,activations in self.agg_obj.items():
                self.save_cache.test_cache_config.id = layer_name
                fn_prefix = '{}_{}'.format(net_name,layer_name)
                print(fn_prefix)
                self.save_cache.save(activations,filenamePrefix=fn_prefix,saveType='ndarray')
        elif self.agg_type == 'image_id':
            for layer_name,activations in self.agg_obj.items():
                self.save_cache.test_cache_config.id = layer_name
                fn_prefix = '{}_{}'.format(net_name,layer_name)
                print(fn_prefix)
                self.save_cache.save(activations,filenamePrefix=fn_prefix,saveType='ndarray')
        else:
            raise ValueError ("unknown [agg_activations.py] self.agg_type: {}".format(self.agg_type))        


