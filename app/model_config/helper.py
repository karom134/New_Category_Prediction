import numpy as np
import re
import pythainlp

stopwords = pythainlp.corpus.common.thai_stopwords()

def deEmojify(text):
	regrex_pattern = re.compile(pattern = "["
		u"\U0001F600-\U0001F64F"  # emoticons
		u"\U0001F300-\U0001F5FF"  # symbols & pictographs
		u"\U0001F680-\U0001F6FF"  # transport & map symbols
		u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
		"]+", flags = re.UNICODE)
	return regrex_pattern.sub(r'',text)

def clear_phone_number(line):
	return re.sub("([0-9]+-+)+[0-9]+|[0-9]{10}|[0-9]{9}|([0-9]+\s+)+[0-9]+"," ",line)

def clear_other_contact(line):
	return re.sub("Line|line|LINE|ติดต่อ|สาขา|\d{1,3}(?:[,]\d{3})*(?:[,]\d{2})"," ",line)

def clear_link(line):
	return re.sub(r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))|\S*@\S*\s?"," ",line)

def cleanse_text(text):
	keep=[]
	lines=text.split("\n")
	for line in lines:
		line=deEmojify(line)
		line=clear_phone_number(line)
		line=clear_other_contact(line)
		if len(line)!=0:
			words=pythainlp.tokenize.word_tokenize(line,keep_whitespace=True)
			for word in words:
				if word not in stopwords and pythainlp.util.isthai(word):
					keep.append(word)
	return " ".join(keep)

def cleanse_title(title):
	keep=[]
	lines=title.split("\n")
	for line in lines:
		line=deEmojify(line)
		if len(line)!=0:
			words=pythainlp.tokenize.word_tokenize(line,keep_whitespace=True)
			for word in words:
				keep.append(word)
	return " ".join(keep)






