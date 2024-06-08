import os
import pandas as pd
import numpy as np
import re
import jieba
import pyLDAvis
import pyLDAvis.sklearn
import jieba.posseg as psg
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from common_function import save_LDA_result, plot_curve

def chinese_word_cut(my_text):
    jieba.load_userdict("./datase/dict.txt")
    jieba.initialize()
    stop_list = []
    try:
        stopword_list = open("./dataset/stopwords.txt", encoding='utf-8')
    except:
        stopword_list = []
        print("error in stop_file")
    for line in stopword_list:
        line = re.sub(u'\n', '', line)
        stop_list.append(line)
    word_list = []
    flag_list = ['n', 'nz', 'vn', 'a', 'v', 'f', 'eng']
    seg_list = psg.cut(my_text)
    for seg_word in seg_list:
        word = seg_word.word
        find = 0
        for stop_word in stop_list:
            if stop_word == word or len(word) < 2:
                find = 1
                break
        if find == 0 and seg_word.flag in flag_list:
            word_list.append(word)
    return " ".join(word_list)

data = pd.read_csv('./dataset/original_content.csv')
data["content_cutted"] = data['0'].apply(chinese_word_cut)

def print_top_words(model, feature_names, n_top_words):
    tword = []
    for topic_idx, topic in enumerate(model.components_):
        topic_w = " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        tword.append(topic_w)
        save_LDA_result("dataset/topic_words.csv", "Topic #%d:" % topic_idx)
        save_LDA_result("dataset/topic_words.csv", topic_w)
    return tword

n_features = 100
tf_vectorizer = CountVectorizer(strip_accents='unicode',
                                max_features=n_features,
                                max_df=1.0,
                                min_df=1)
tf = tf_vectorizer.fit_transform(data.content_cutted)

n_topics = 10
lda = LatentDirichletAllocation(n_components=n_topics, max_iter=50, learning_method='batch', learning_offset=50, doc_topic_prior=0.1, topic_word_prior=0.01, random_state=0)
lda.fit(tf)
n_top_words = 10
tf_feature_names = tf_vectorizer.get_feature_names()
topic_word = print_top_words(lda, tf_feature_names, n_top_words)

topics = lda.transform(tf)
topic_number = []
topic = []
for t in topics:
    topic.append("Topic #" + str(list(t).index(np.max(t))))
    topic_number.append(list(t).index(np.max(t)))
data['biggest_number'] = topic_number
data['biggest_topic'] = topic
data['topics_pro'] = list(topics)
for i in range(np.array(topics).shape[1]):
    data['topic' + str(i) + '_pro'] = np.array(topics)[:, i]
data.to_excel("./dataset/doc_topic.xlsx", index=False)
pic = pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer)
pyLDAvis.show(pic)
pyLDAvis.save_html(pic, 'lda_pass' + str(n_topics) + '.html')
pyLDAvis.display(pic)

plexs = []
scores = []
n_max_topics = 10
for i in range(1, n_max_topics):
    lda = LatentDirichletAllocation(n_components=i, max_iter=50, learning_offset=50, doc_topic_prior=0.1, topic_word_prior=0.01, random_state=0)
    lda.fit(tf)
    plexs.append(lda.perplexity(tf))
    scores.append(lda.score(tf))
n_t = n_max_topics - 1
x = list(range(1, n_t + 1))
plot_curve(x, plexs[0:n_t], "number of topics", "perplexity", "lda/")
plot_curve(x, scores[0:n_t], "number of topics", "scores", "lda/")
