from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing import text

TOP_K = 20000

MAX_SEQUENCE_LENGTH = 500

def sequence_vectorize(train_texts, val_texts):
	""" Vectorizes texts as sequence vectors

	1 text = 1 sequence vector with fixed length

	# Returns
		x_train, x_val, word_index: vectorized training and validation texts and word index dictionary

	"""
	tokenizer = text.Tokenizer(num_words = TOP_K)
	tokenizer.fit_on_texts(train_texts)

	x_train = tokenizer.texts_to_sequences(train_texts)
	x_val = tokenizer.texts_to_sequences(val_texts)

	max_length = len(max(x_train, key= len))
	if max_length > MAX_SEQUENCE_LENGTH:
		max_length = MAX_SEQUENCE_LENGTH

	# if shorter then padded in the beginning if longer then truncated

	x_train = sequence.pad_sequences(x_train, maxlen=max_length)
	x_val = sequence.pad_sequences(x_val, maxlen= max_length)

	return x_train, x_val, tokenizer.word_index