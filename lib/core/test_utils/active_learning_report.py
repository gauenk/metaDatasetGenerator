from utils.misc import computeEntropyOfNumpyArray

class activeLearningReportAppendActivationValueData():

    """
    records activity values from a list of preselected layers to augment the original active learning report
    """

    def __init__(self,net,imdb,activeLearningCfg,recordBool):
        self.recordBool = activeLearningCfg.REPORT
        self.layerNameList = activeLearningCfg.LAYER_NAMES # list of layers to save data from
        self.allModelLayerNames = [str(l.name) for l in  net.layer]
        # start the results file
        self.alReportFilename = "alReport_{}_{}.csv".format(imdb.name,net.name)
        if recordBool:
            self.startAlReportCsvFile(imdb.num_classes)
            # get the old results
            self.alResultsFromPreviousExperiment = readAlResultsFromPreviousExperiment(activeLearningCfg)

    def startAlReportCsvFile(self,num_classes):
        fidAlReport = open(self.alReportFilename,"w+")
        # this is gross to use 'self' when i don't use any class variables...
        self.createAlReportCsvHeader(fidAlReport,num_classes):
        fidAlReport.close()

    def createAlReportCsvHeader(self,fidAlReport,num_classes):
        headerStr = "image_index,"
        cls_probs_prefix = "cls_prob"
        for idx in range(num_classes):
            headerStr += "{}_{},".format(cls_probs_prefix,str(idx))
        headerStr += "cls_prob_entropy"
        av_prefix = "av"
        for av_blobName in cfg.SAVE_ACTIVITY_VECTOR_BLOBS:
            headerStr += "{}_{},".format(av_prefix,av_blobName)
        headerStr += "%errorReduction\n"
        fidAlReport.write(headerStr)

    def readAlResultsFromPreviousExperiment(self,activeLearningCfg):
        alResults = {}
        fn = "resultsAL_{}_{}_{}_{}.csv".format(
            activeLearningCfg.N_ITERS,
            activeLearningCfg.VAL_SIZE,
            activeLearningCfg.SUBSET_SIZE,
            activeLearningCfg.N_COVERS)
        with open(fn,"r") as f:
            csv_reader = csv.reader(f,delimiter=',')
            for line_num,row in enumerate(csv_reader):
                if line_num == 0: continue
                image_index = row[0]
                if row[1] == "":
                    pctErrorRed_ave = -1
                    pctErrorRed_std = -1
                else:
                    pctErrorRed_ave = float(row[1]) # average
                    pctErrorRed_std = float(row[2]) # std (error or deviation?)
                alResults[image_index] = {}
                alResults[image_index]['ave'] = pctErrorRed_ave
                alResults[image_index]['std'] = pctErrorRed_std
        return alResults

    def get_layer_index_from_name(self,layerName):
        return self.allModelLayerNames.index(layerName) # the list of RETURNED activity values; not all are always returned

    def record(self,model_output,image_id):
        scores = model_output['scores']
        activity_vectors = model_output['activity_vectors']
        if self.recordBool is False: return
        # recordActivityVectorTransformsForAlReport
        fidAlReport = open(self.alReportFilename,'a+')
        scores = np.squeeze(scores)
        imageStr = "{},".format(image_id)
        for score in scores:
            imageStr += "{:.5f},".format(score)
        scoreEntropy = computeEntropyOfNumpyArray(scores)
        imageStr += "{:.5f},".format(scoreEntropy)
        for layerName in self.layerNameList:
            # not correct at all...
            # the "idx" and the correct index for "activity_vectors" is not the same since "SAVE_ACTIVITY..." is not necessarly equivalent to the net's activity values
            activity_vector_index = self.get_layer_index_from_name(layerName)
            layerValue = self.transformActivationValuesByLayerName(layerName,activity_vectors[idx])
            imageStr += "{:.5f},".format(layerValue)
        imageStr += "{:.5f}\n".format(self.alResultsFromPreviousExperiment[image_index]['ave'])
        fidAlReport.write(imageStr)
        fidAlReport.close()


    def transformActivationValuesByLayerName(self,field,value):
        if field == 'conv1_1':
            return computeEntropyOfNumpyArray(value)
        elif field == 'cls_prob':
            return computeEntropyOfNumpyArray(value)
        return value
