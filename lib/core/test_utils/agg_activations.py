import numpy as np
import cPickle

class aggregateActivations():

    def __init__(self,activationsCfg,save_dir):
        self.save_dir = save_dir
        self.save_bool = activationsCfg.SAVE_BOOL
        self.agg_type = activationsCfg.SAVE_OBJ
        self.layer_names = activationsCfg.LAYER_NAMES
        self.agg_obj = self._init_new_agg_obj()

    def _init_new_agg_obj(self):
        agg_obj = dict.fromkeys(self.layer_names)
        if self.agg_type == 'order':
            for key in agg_obj.keys():
                agg_obj[key] = {}
        elif self.agg_type == 'image_id':
            for key in agg_obj.keys():
                agg_obj[key] = []
        return agg_obj

    def aggregate(self,activations,image_id):
        if self.save_bool is False: return
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
            self.agg_obj[layer_name] = np.arrray(self.agg_obj[layer_name])

    def save(self,net_name):
        if self.save_bool is False: return
        print("activitions from model saved @")
        if self.agg_type == 'order':
            self.numpyifyAggregateByOrder()
            for layer_name,activations in self.agg_obj:
                fn = os.path.join(self.save_dir,"{}_{}.npy".format(layer_name,net_name))
                print(fn)
                np.save(fn,activations)
        elif self.agg_type == 'image_id':
            for layer_name,activations in self.agg_obj:
                fn = os.path.join(self.save_dir,"{}_{}.pkl".format(layer_name,net_name))
                print(fn)
                with open(fn, 'wb') as f:
                    cPickle.dump(activations, f, cPickle.HIGHEST_PROTOCOL)
        else:
            raise ValueError ("unknown [agg_activitions.py] self.agg_type: {}".format(self.agg_type))        


