
***************************************************
****************** #  Word2vec.ipynb **************
***************************************************


1- How to generate w2vec embeddings? 

Open w2vec.ipynb
	- Run all the cells with relevant functions to create w2vec model i.e., (getWord_model)
	- Run the helper functions to convert DNA sequences to KMERS (DNAToWOrd)
		- Split the two seqs in different lists (getDNASplit)
		- (GetAvgFeatureVecs) to get the mean of the embeddings  
	- Some function just help you to write to appropriate npy and csv files



Cell 6: (the cell where these functions are used)
	- Use appropriate "KMER SIZE" (here it is 6)

		kmer = 6
	
	- Declare the path for train or test data
		- seqfile = 'cross_test_data.fa'
		
	- read the data
		- DNAseq = pd.read_csv(seqfile,sep = "\t",error_bad_lines=False)

	- Define the number of positive samples in the file
		- # pos_number = 2096 # train pos samples
		- pos_number = 1658 # cross test
		
	- split pair of DNA sequences
		- words1,words2 = getDNA_split(DNAseq,kmer)
		
	- get the w2vec model, where Unfile represents the kmers present in the sequences
		- word_model = getWord_model(kmer,fea_num,min_fea,model,Unfile)


Cell 7: 
	- after training or loading the model
	- Avg out the features as discussed
		- dataDataVecs = getAvgFeatureVecs(words1,words2,word_model,fea_num)
	- convert embeddings to csv
		- A csv file contains label and the relevant feature vector of the seqs
		
			fea_svm = '%d_test_vecs.svm'%(kmer)
			fea_csv = '%d_test.csv'%(kmer)

			npyTosvm(fea_npy, fea_svm,pos_number)
			SVMtoCSV(fea_svm, fea_csv)
			


***************************************************
****************** CNN_model.ipynb ****************
***************************************************

Cell 1:
	- Load all libraries
cell 2:
	- Load the default evaluation measures
		- METRICS = [
			      keras.metrics.TruePositives(name='tp'),
			      keras.metrics.FalsePositives(name='fp'),
			      keras.metrics.TrueNegatives(name='tn'),
			      keras.metrics.FalseNegatives(name='fn'), 
			      keras.metrics.BinaryAccuracy(name='accuracy'),
			      keras.metrics.Precision(name='precision'),
			      keras.metrics.Recall(name='recall'),
			      keras.metrics.AUC(name='auc'),
			     ]
			     
	- Function to create the deep learning model (dnn_model())
	
			- Inputs
			- Conv1(inputs)
			- cc1 = concatenate(inputs, Conv1)
			
			- Conv2(Conv1)
			
			- Conv3(cc1)
			- cc2 = concatenate(inputs, Conv3)
			
			- dropout on cc1, cc2
			
			- Dense1(cc1)
			- Dense2(cc2)
			- cc3 = concatenate(Dense1, Dense2)
			
			- Dense3(Conv2)
			- cc4 = concatenate(cc3, Dense3)
			
			- Dense4, sigmoid => final prediction
			
			
			
			(ADAM, BCE)
			
Cell 3:
	- 5-Fold Validation


Cell 4:
	- Independent testing
	
	

	
(NOTE PLEASE DECLARE THE POS SAMPLES BY CROSS CHECKING WITH THE RELEVANT FILES PRESENT IN THE SUBSEQUENT DIRECTORIES)
			
			
			
			
		
	
	

