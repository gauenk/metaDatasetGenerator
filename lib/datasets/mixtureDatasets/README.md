# mixtureDatasets

The "mixture datasets" folder saves the ids of the images used from each of the different datasets for each mixture

The directory structure is as follows:

```
./mixtureDatasets/<binary_code>/<repetition#>/<datasetSize>.txt
```

The "binary code" specifies which dataset is used in the mixture dataset where the indicies' meaning are specified in the "mixtureKey.yml" file. For example, a mixture dataset for 11000000 indicates that datasets VOC and ImageNet are used. 

An example is below:

./mixtureDatasets/
	11111111/
		1/
			100
			200
			500
			1000
			2000
			5000
			10000
		2/	
			100
			200
			500
			1000
			2000
			5000
			10000
	00000001/
		1/
			100
			200
			500
			1000
			2000
			5000
			10000
		2/	
			100
			200
			500
			1000
			2000
			5000
			10000
	01111000/
		1/
			100
			200
			500
			1000
			2000
			5000
			10000
		2/	
			100
			200
			500
			1000
			2000
			5000
			10000
		