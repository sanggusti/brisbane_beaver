'''
Author: Louis Owen (https://louisowen6.github.io/)
'''

import pandas as pd
import numpy as np
import re
from itertools import combinations
from tqdm import tqdm

import pylcs
from nltk import ngrams
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from .preprocessor import preprocessing

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()


def plagiarism_pipeline(doc_dict,model,candidate_doc_thres=0.25,paragraph_filtering_thres=2):
	'''
	Plagiarism Pipeline
	Return: plagiarism score dictionary for each candidate document pairs
	'''

	# Candidate Document Retrieval
	candidate_pairs = []
	doc_pair_combinations = list(combinations(list(doc_dict.keys()), 2))
	for pair1,pair2 in tqdm(doc_pair_combinations, desc="Candidate Document Retrieval"):
	    sim = extract_ngram_similarity(' '.join(doc_dict[pair1]),' '.join(doc_dict[pair2]),1)
	    
	    if sim > candidate_doc_thres:
	        candidate_pairs.append((pair1,pair2))

	# Paragraph Filtering
	candidate_doc_dict = {}
	for pair1,pair2 in tqdm(candidate_pairs, desc="Paragraph Filtering"):
	    candidate_doc_dict[pair1] = [x for x in doc_dict[pair1] if len(x.split('.'))>paragraph_filtering_thres]
	    candidate_doc_dict[pair2] = [x for x in doc_dict[pair2] if len(x.split('.'))>paragraph_filtering_thres]

	# Paragraph Preprocessing
	for doc in tqdm(candidate_doc_dict, desc="Paragraph Preprocessing"):
	    candidate_doc_dict[doc] = [preprocessing(x,stemmer) for x in candidate_doc_dict[doc]]

	# Paragraph Pairing
	paired_dict = {'cleaned_paragraph_1':[],'cleaned_paragraph_2':[],'type':[],'pairs':[]}
	for pair1,pair2 in tqdm(candidate_pairs, desc="Paragraph Pairing"):
	    for paragraph1 in candidate_doc_dict[pair1]:
	        for paragraph2 in candidate_doc_dict[pair2]:
	            paired_dict['cleaned_paragraph_1'].append(paragraph1)
	            paired_dict['cleaned_paragraph_2'].append(paragraph2)
	            paired_dict['type'].append('uni-paragraph')
	            paired_dict['pairs'].append((pair1,pair2))
	            
	    bi_paragraph1_list = generate_bigram(candidate_doc_dict[pair1])
	    bi_paragraph2_list = generate_bigram(candidate_doc_dict[pair2])
	    
	    for bi_paragraph1 in bi_paragraph1_list:
	        for bi_paragraph2 in bi_paragraph2_list:
	            paired_dict['cleaned_paragraph_1'].append(bi_paragraph1)
	            paired_dict['cleaned_paragraph_2'].append(bi_paragraph2)
	            paired_dict['type'].append('bi-paragraph')
	            paired_dict['pairs'].append((pair1,pair2))

	df_test = pd.DataFrame(paired_dict)

	# Feature Engineering
	df_train = pd.DataFrame(columns=['category','paragraph_1','paragraph_2','len_paragraph_1','is_plagiarism','plagiarism_type','cleaned_paragraph_1','cleaned_paragraph_2'])
	with open('../data/plagiarism_data_train.tsv','r') as f_in:
	    for i,line in tqdm(enumerate(f_in),desc="Feature Engineering"):
	        if i>0:
	            columns = line.split('\t')
	            columns[-1] = re.sub('\n','',columns[-1])
	            df_train.loc[i] = columns

	df_preprocessed_test = feature_engineering(df_train,df_test)

	print('============= Model Inference')
	df_preprocessed_test[['negative_prob','positive_prob']] = predict(df_preprocessed_test,model)
	df_preprocessed_test['prediction'] = np.argmax(df_preprocessed_test[['negative_prob','positive_prob']].values,axis=1)

	print('============= Score Aggregation')
	score_dict = {}
	for pairs in df_preprocessed_test['pairs'].unique():
	    df_preprocessed_pairs_i = df_preprocessed_test[df_preprocessed_test['pairs']==pairs]
	    
	    score = calculate_plagiarism_score(df_preprocessed_pairs_i,candidate_doc_dict,pairs)
	    
	    print(pairs,score)
	    score_dict[pairs] = score

	return score_dict


def extract_ngram_similarity(sentence_1,sentence_2,n=4):
    ngrams_set_1 = set()
    ngrams_set_2 = set()
    
    ngram1 = ngrams(sentence_1.split(), n)
    for grams in ngram1:
        ngrams_set_1.add(grams)
        
    ngram2 = ngrams(sentence_2.split(), n)
    for grams in ngram2:
        ngrams_set_2.add(grams)
    
    try:
        jaccard_sim = len(ngrams_set_1.intersection(ngrams_set_2)) / len(ngrams_set_1.union(ngrams_set_2))

        return jaccard_sim
    except:
        return 0


def generate_bigram(arr):
    return [arr[i]+' '+arr[i+1] for i in range(len(arr)-1)]


def feature_engineering(df_train,df_test):
    corpus = df_train['cleaned_paragraph_1'].to_list() + df_train['cleaned_paragraph_2'].to_list()
    
    # Word Pairs
    bigram_vectorizer = CountVectorizer(analyzer='word', ngram_range=(2, 2), min_df=0.002)
    bigram_vectorizer.fit(corpus)
    
    transformed_array_test = bigram_vectorizer.transform(df_test['cleaned_paragraph_1'] + ' '+ df_test['cleaned_paragraph_2']).toarray()
    num_feats = transformed_array_test.shape[1]

    df_preprocessed_test = pd.DataFrame(transformed_array_test,columns=[f'word_pairs_{i}' for i in range(num_feats)])
    
    # Words Similarity
    df_preprocessed_test['words_similarity'] = df_test.apply(lambda x: extract_ngram_similarity(x['cleaned_paragraph_1'],x['cleaned_paragraph_2'],n=1), axis=1)
    
    # Fingerprints Similarity
    df_preprocessed_test['fingerprint_similarity'] = df_test.apply(lambda x: extract_ngram_similarity(x['cleaned_paragraph_1'],x['cleaned_paragraph_2'],n=4), axis=1)
    
    # Longest Common Subsequence
    df_preprocessed_test['lcs'] = df_test.apply(lambda x: pylcs.lcs2(x['cleaned_paragraph_1'],x['cleaned_paragraph_2'])/max(len(x['cleaned_paragraph_1']),len(x['cleaned_paragraph_2'])), axis=1)
    
    # LSA Similarity
    num_component = 200
    tfidf_vectorizer = TfidfVectorizer(use_idf=True,smooth_idf=True)
    lsa = TruncatedSVD(num_component, algorithm = 'randomized',random_state=0)

    tfidf_vectorizer.fit(corpus)
    dtm = tfidf_vectorizer.transform(corpus)
    lsa.fit(dtm)
        
    dtm_lsa_test_1 = lsa.transform(tfidf_vectorizer.transform(df_test['cleaned_paragraph_1'].to_list()))
    dtm_lsa_test_2 = lsa.transform(tfidf_vectorizer.transform(df_test['cleaned_paragraph_2'].to_list()))
    
    cosine_sim_test = []
    for i in range(dtm_lsa_test_1.shape[0]):
        cosine_sim_test.append(cosine_similarity([dtm_lsa_test_1[i]],[dtm_lsa_test_2[i]])[0][0])
    
    df_preprocessed_test['lsa_similarity'] = cosine_sim_test
    
    # Add Supporting Features
    df_preprocessed_test[['type','pairs']] = df_test[['type','pairs']]
    
    return df_preprocessed_test


def predict(df_preprocessed_test,model):

    X_val = df_preprocessed_test.drop(columns=['type','pairs'])

    y_pred = model.predict_proba(X_val)
    
    return y_pred


def calculate_plagiarism_score(df_preprocessed_pairs_i,candidate_doc_dict,pairs):
	if len(df_preprocessed_pairs_i[df_preprocessed_pairs_i['prediction']==1])>0:
	    avg_pos_prob = df_preprocessed_pairs_i[df_preprocessed_pairs_i['prediction']==1]['positive_prob'].mean()
	    
	    uni_par_length = (len(candidate_doc_dict[pairs[0]]) + len(candidate_doc_dict[pairs[1]]))
	    bi_par_length = (len(candidate_doc_dict[pairs[0]]) - 1 + len(candidate_doc_dict[pairs[1]]) - 1)
	    freq = min(1,df_preprocessed_pairs_i['prediction'].value_counts().loc[1] / (uni_par_length+bi_par_length))

	    score = avg_pos_prob * freq
	else:
	    score = 0

	return score
