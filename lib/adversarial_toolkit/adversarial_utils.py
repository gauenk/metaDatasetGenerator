def adversarialPerturbation(img,net,inputBlob,inputBlobImgFieldName,inputBlobLabelFieldName,outputFieldName,targetOutput,alpha=0.1):

    inputBlob[inputBlobImgFieldName] = img
    inputBlob[inputBlobLabelFieldName] = targetOutput
    
    ## L-BFGS Method

    # get current output info
    clsOriginal,gradCurrent = networkForwardBackward(net,inputBlob,outputFieldName)
    
    # Loop variables
    clsCurrent = clsOriginal
    historyQuit = False # a quit var flagged by too many of the same, but not moving toward target

    # Primary loop
    imgCurrent = img.copy()
    clsCurrent = clsOriginal
    direction = np.rand(gradCurrent.shape)
    # Bk = 0
    while( (targetOutput != clsCurrent) or historyQuit ):
        # propose new image
        sk = alpha * direction
        imgCurrentProp = imgCurrent + sk
        # get new gradient and output class
        inputBlob[inputBlobImgFieldName] = imgCurrentProp
        clsCurrentProp,gradCurrentProp = networkForwardBackward(net,inputBlob,outputFieldName)
        diff = gradCurrentProp - gradCurrent
        # Bk += ( diff * npt(diff) ) / ( npt(diff) * sk ) - ( Bk * sk * npt(sk) * Bk ) / ( npt(sk) * Bk * npt(sk) )

