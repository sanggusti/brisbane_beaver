'''
Author: Louis Owen (https://louisowen6.github.io/)
'''

import re

stopwords = []
with open('../resources/freq_stopwords.txt','r') as f_in:
    for line in f_in:
        stopwords.append(re.sub('\n','',line))
        
with open('../resources/tata_stopwords.txt','r') as f_in:
    for line in f_in:
        stopwords.append(re.sub('\n','',line))


def deEmojify(inputString):
    '''
    Function to remove emoji
    '''
    return inputString.encode('ascii', 'ignore').decode('ascii')


def preprocessing(paragraph,stemmer):
    '''
    clean input sentence  
    '''
    http_pat = r'https?://[^ ]+'
    www_pat = r'www.[^ ]+'
    hashtag_pat = r'#[A-Za-z0-9_]+'
    linebreak_pat = r'\n'
    
    cleaned_sentence_list = []
    for sentence in paragraph.split('. '):
        #Remove Emoji
        stripped = deEmojify(sentence)

        #Remove url
        stripped = re.sub(http_pat, '', stripped)
        stripped = re.sub(www_pat, '', stripped)

        #Remove hashtag
        stripped = re.sub(hashtag_pat, '', stripped)

        #Remove Non Alphabet and Non Number Characters
        stripped = re.sub(' +',' ',re.sub(r'[^a-zA-Z-0-9]',' ',stripped)).strip()

        #Lowercase
        stripped = stripped.lower()

        #Remove Stopwords Before Stemming
        stripped = ' '.join([x for x in stripped.split() if x not in stopwords])

        #Stem each word
        stripped = stemmer.stem(stripped)

        #Remove Stopwords After Stemming
        stripped = ' '.join([x for x in stripped.split() if x not in stopwords])

        cleaned_sentence_list.append(re.sub(' +',' ',stripped).strip())
    
    return ' '.join(cleaned_sentence_list)