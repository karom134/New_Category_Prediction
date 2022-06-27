import torch.nn as nn
import torch
import pandas as pd
from ..model_config.model import Category_classifier
from ..model_config.helper import cleanse_text,cleanse_title
import os 

file_path="/app/model/"
directory=os.getcwd()+file_path

word_dict=pd.read_csv(directory+"vocab_data_draft_3.csv")
category_dict=pd.read_csv(directory+"rebalanced_class_name.csv")

classifier=Category_classifier(len(word_dict)+1,100,100)
classifier.load_state_dict(torch.load(directory+"category_prediction_draft_3_MSE.pth",map_location=torch.device('cpu')))
classifier.eval()

def text_to_vec(text,max_len):
	vec=[]
	words=text.split(" ")
	for word in words:
		if word in word_dict["word"].tolist():
			vec.append(word_dict[word_dict["word"]==word].word_id.values[0])
		else:
			vec.append(word_dict[word_dict["word"]=="unk"].word_id.values[0])

		if len(vec)>=max_len:
			break
	while len(vec)<max_len:
		vec.append(0)
	return vec

def get_class_name(class_ids):

	classes=[]
	for class_id in class_ids:
		classes.append(category_dict[category_dict["class_id"]==class_id].name.values[0])
	
	return classes

def predict(title,desc,cut=0.8):
	c_title=cleanse_title(title)
	c_desc=cleanse_text(desc)

	title_vec=text_to_vec(c_title,10)
	desc_vec=text_to_vec(c_desc,20)

	vec=title_vec+desc_vec
	vec=torch.as_tensor([vec])

	prediction=classifier(vec)
	prediction=prediction.detach().numpy().squeeze()

	predict_class=[]
	for idx in range(len(prediction)):
		if prediction[idx]>cut:
			predict_class.append(idx)
	return get_class_name(predict_class)

		