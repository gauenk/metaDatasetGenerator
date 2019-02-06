from core.configBase import checkEdictEquality
from utils.base import readPickle,writePickle
import uuid

class Cache():

    """
    -> filename: the location of the cache
    -> stores within the filename by a 'uuid'
    -> search by comparing the config
    """

    def __init__(self,filename,config,fieldname,cache_bool=True):
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
        if self.cache_bool is False:
            return [],[]
        match_list = []
        uuid_list = []

        cache = readPickle(self.filename)
        if cache is None or len(cache) == 0:
            return match_list,uuid_list

        self.is_valid = True
        for uuID,expData in cache.items():
            self.is_valid = True
            isEqual = checkEdictEquality(expData['config'],self.config)
            #isEqual = checkEdictEquality(self.config,expData['config'])
            if isEqual:
                if self.fieldname is not None:
                    if self.fieldname in expData['data'].keys():
                        match_list.append(expData['data'][self.fieldname])
                        uuid_list.append(uuID)
                    """
                    This case is not an issue; says the current cache doesn't have the same field another match does.
                    """
                    # elif len(match_list) > 0:
                    #     print(len(match_list))
                    #     print("weirdness in formatting: some cache values are okay and some are not")
                    #     self.is_valid = False
                    #     return match_list,uuid_list
                    # else:
                    #     self.is_valid = False
                    #     return match_list,uuid_list
                else:
                    match_list.append(expData['data'])
                    uuid_list.append(uuID)
        if len(match_list) > 0:
            self.is_valid = True
        else:
            self.is_valid = False
        return match_list,uuid_list

    def save(self,payload,saveField="",enforceOneMatch=True):
        if self.cache_bool is False:
            print("Did not save. Caching is turned off.")
            return None
        matching_saves,uuids = self.load_all_matches()
        if len(matching_saves) > 1 and enforceOneMatch:
            print("[one_level_cache.py] ERROR: more than one config match. quittng")
            exit()
        if saveField == "":
            saveField = self.fieldname
        cache = readPickle(self.filename)
        if cache is None:
            cache = {}
        if enforceOneMatch and len(matching_saves) == 1:
            print("overwriting old match.")
            uuID = uuids[0]
        elif enforceOneMatch and len(matching_saves) == 0: # separated out for clarity
            uuID = str(uuid.uuid4())
        elif not enforceOneMatch:
            uuID = str(uuid.uuid4())            
        else:
            print("[one_level_cache] unknown issue. quitting.")
            exit()
        blob = {'config':self.config}
        blob['data'] = {}
        if saveField is not None:
            blob['data'][saveField] = payload
        else:
            blob['data'] = payload
        cache[uuID] = blob
        writePickle(self.filename,cache)

    def view(self):
        cache = readPickle(self.filename)
        print(cache.keys())
        for key,value in cache.items():
            print(key,value['data'])#,checkEdictEquality(value['config'],cfg))

    def remove_entry(self,config_to_remove):
        # TODO: test this function
        # set config for removal
        original_config = self.config
        self.config = config_to_remove

        # load cache into memory
        cache = readPickle(self.filename)
        if cache is None:
            return None

        # find all caches to delete & delete them 
        _,uuid_list = self.load_all_matches()
        for uuid_str in uuid_list:
            del cache[uuid_str]

        # restore original config
        self.config = original_config
        
    def update_config(self,new_config):
        # load cache into memory
        cache = readPickle(self.filename)
        if cache is None:
            return False

        # find all caches to re-configure
        _,uuid_list = self.load_all_matches()
        if len(uuid_list) == 0:
            return False

        # update the config for the caches
        for uuid_str in uuid_list:
            cache[uuid_str]['config'] = new_config
        self.config = new_config
        return True
        
