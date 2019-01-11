from core.configBase import checkEdictEquality
from utils.base import readPickle,writePickle
import uuid

class Cache():

    """
    -> filename: the location of the cache
    -> stores within the filename by a 'uuid'
    -> search by comparing the config
    """

    def __init__(self,filename,config,cache_bool=True,fieldname=None):
        self.filename = filename
        self.fieldname = fieldname
        self.config = config
        self.cache_bool = True
        self.is_valid = False

    def load(self):
        if self.cache_bool is False: return None
        cache = readPickle(self.filename)
        if cache is None or len(cache) == 0: return None
        self.is_valid = True
        for uuID,expData in cache.items():
            self.is_valid = True
            isEqual = checkEdictEquality(expData['config'],self.config)
            #isEqual = checkEdictEquality(self.config,expData['config'])
            if isEqual:
                if self.fieldname is not None:
                    if self.fieldname in expData['data'].keys():
                        return expData['data'][self.fieldname]
                    else:
                        self.is_valid = False
                        return None
                else:
                    return expData['data']
        self.is_valid = False
        return None

    def load_all_matches(self):
        if self.cache_bool is False: return None
        match_list = []

        cache = readPickle(self.filename)
        if cache is None or len(cache) == 0: return match_list

        self.is_valid = True
        for uuID,expData in cache.items():
            self.is_valid = True
            isEqual = checkEdictEquality(expData['config'],self.config)
            #isEqual = checkEdictEquality(self.config,expData['config'])
            if isEqual:
                if self.fieldname is not None:
                    if self.fieldname in expData['data'].keys():
                        match_list.append(expData['data'][self.fieldname])
                    elif len(match_list) > 0:
                        print("weirdness in formatting: some cache values are okay and some are not")
                        self.is_valid = False
                        return match_list
                    else:
                        self.is_valid = False
                        return match_list
                else:
                    match_list.append(expData['data'])
        if len(match_list) > 0:
            self.is_valid = True
        else:
            self.is_valid = False
        return match_list

    def save(self,payload,saveField=""):
        if self.cache_bool is False:
            print("Did not save. Caching is turned off.")
            return None
        if saveField == "": saveField = self.fieldname
        cache = readPickle(self.filename)
        if cache is None: cache = {}
        uuID = str(uuid.uuid4())
        blob = {'config':self.config}
        blob['data'] = {}
        if saveField is not None: blob['data'][saveField] = payload
        else: blob['data'] = payload
        cache[uuID] = blob
        writePickle(self.filename,cache)

    def view(self):
       cache = readPickle(self.filename)
       print(cache.keys())
       for key,value in cache.items():
           print(key,value['config'].expName,checkEdictEquality(value['config'],cfg))
