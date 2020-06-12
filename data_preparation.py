from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

#Vectorization parameters 
NGRAM_RANGE = (1,2)

# limit on the number of features. Here Top 20k features
TOP_K = 20000

# whether text should be split into word or character n-grams
#either 'word' or 'char'
TOKEN_MODE = 'word'

#Minimum document/corpus frequency below which a token will be discarded.
MIN_DOCUMENT_FREQUENCY = 2

def ngram_vectorize(train_texts, train_labels, val_texts):
	""""Vectrorizes texts as n-gram vectors.

	1 Text = 1 tf-idf vector the length of vocabulary of unigrams + bigrams

	# Arguments
		train_texts: list, training test strings
		train_labels: np.ndarray. training labels 
		val_texts: list, validation text strings

	# Returns
		x_train, x_val: vectorized training and validation texts

	"""
	kwargs = {
			'ngram_range': NGRAM_RANGE, 
			'dtype': 'int32',
			'strip_accents': 'unicode'
			'decode_error': 'replace'
			'analyzer': TOKEN_MODE
			'min_df': MIN_DOCUMENT_FREQUENCY

	}

	vectorizer = TfidfVectorizer(**kwargs)

	#Learn vocabulary from training texts and vectorize training texts
	x_train = vectorizer.fit_transform(train_texts)

	#vectorize validation texts 
	x_val = vectorizer.transform(val_texts)

	#select top 'k' of the vectorized features
	selector = SelectKBest(f_classif, k=min(TOP_K, x_train.shape[1]))
	selector.fit(x_train, train_labels)
	x_train = selector.transform(x_train).astype('float32')
	x_val = selector.transform(x_val).astype('float32')
	return x_train, x_val