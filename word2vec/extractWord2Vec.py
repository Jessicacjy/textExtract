
# coding: utf-8

# In[ ]:


from __future__ import print_function
from __future__ import unicode_literals
import sys
print("extract1")
try:
    reload(sys)
    sys.setdefaultencoding('utf-8')
except:
    pass

import sys
sys.path.append("../")


from snownlp import SnowNLP
import re
import codecs


DATA_SWAP = "/Users/daichanglin/Desktop/igoldenbeta/python/temp.txt";
DATA_SWAP2 = "/Users/daichanglin/Desktop/igoldenbeta/python/temp2.txt";
#load model
model = Word2Vec.load('/Users/Jesica/Documents/igolden/word2vec/word2vec-tutorial-master/w2v100.model.bin')

def extract():
    print("extract2")
    try:
        f = codecs.open(DATA_SWAP, 'r','utf-8')
        w = codecs.open(DATA_SWAP2, 'w','utf-8') # 若是'wb'就表示写二进制文件
        all_the_text = f.read()
        # print(all_the_text.decode('utf-8'))
        ret = extractSummary(all_the_text)
        # print(ret.encode('gbk'))
        for x in ret:
            # print(x.decode('utf-8'))
            w.write(x+'\r\n')
    finally:
        if f:
            f.close()
        if w:
            w.close()

def cut_sentences(sentence):  
    puns = frozenset(u'。！？')  
    tmp = []  
    for ch in sentence:  
        tmp.append(ch)  
        if puns.__contains__(ch):  
            yield ''.join(tmp)  
            tmp = []  
    yield ''.join(tmp)
    
def two_sentences_similarity(sents_1, sents_2):  
    ''''' 
    计算两个句子的相似性 
    相同词语的百分比
    :param sents_1: 
    :param sents_2: 
    :return: 
    '''  
    counter = 0  
    for sent in sents_1:  
        if sent in sents_2:  
            counter += 1  
    return counter / (math.log(len(sents_1) + len(sents_2)))  

def create_graph(word_sent):  
    """ 
    传入句子链表  返回句子之间相似度的图 
    :param word_sent: 
    :return: 
    """  
    num = len(word_sent)  
    board = [[0.0 for _ in range(num)] for _ in range(num)]  
  
    for i, j in product(range(num), repeat=2):  
        if i != j:  
            
            board[i][j] = compute_similarity_by_avg(word_sent[i], word_sent[j])  
    return board  
  
def cosine_similarity(vec1, vec2):  
    ''''' 
    计算两个向量之间的余弦相似度 
    :param vec1: 
    :param vec2: 
    :return: 
    '''  
    tx = np.array(vec1)  
    ty = np.array(vec2)  
    cos1 = np.sum(tx * ty)  
    cos21 = np.sqrt(sum(tx ** 2))  
    cos22 = np.sqrt(sum(ty ** 2))  
    cosine_value = cos1 / float(cos21 * cos22)  
    return cosine_value  

def compute_similarity_by_avg(sents_1, sents_2):  
    ''''' 
    对两个句子求平均词向量 
    :param sents_1: 
    :param sents_2: 
    :return: 
    '''  
    if len(sents_1) == 0 or len(sents_2) == 0:  
        return 0.0  
    vec1 = model[sents_1[0]]  
    for word1 in sents_1[1:]:  
        vec1 = vec1 + model[word1]  
  
    vec2 = model[sents_2[0]]  
    for word2 in sents_2[1:]:  
        print
        vec2 = vec2 + model[word2]  
  
    similarity = cosine_similarity(vec1 / len(sents_1), vec2 / len(sents_2))  
    return similarity  
  
def calculate_score(weight_graph, scores, i):  
    """ 
    计算句子在图中的分数 
    :param weight_graph: 
    :param scores: 
    :param i: 
    :return: 
    """  
    length = len(weight_graph)  
    d = 0.85  
    added_score = 0.0  
  
    for j in range(length):  
        fraction = 0.0  
        denominator = 0.0  
        # 计算分子  
        fraction = weight_graph[j][i] * scores[j]  
        # 计算分母  
        for k in range(length):  
            denominator += weight_graph[j][k]  
            if denominator == 0:  
                denominator = 1  
        added_score += fraction / denominator  
    # 算出最终的分数  
    weighted_score = (1 - d) + d * added_score  
    return weighted_score  

def weight_sentences_rank(weight_graph):  
    ''''' 
    输入相似度的图（矩阵) 
    返回各个句子的分数 
    :param weight_graph: 
    :return: 
    '''  
    # 初始分数设置为0.5  
    scores = [0.5 for _ in range(len(weight_graph))]  
    old_scores = [0.0 for _ in range(len(weight_graph))]  
  
    # 开始迭代  
    while different(scores, old_scores):  
        for i in range(len(weight_graph)):  
            old_scores[i] = scores[i]  
        for i in range(len(weight_graph)):  
            scores[i] = calculate_score(weight_graph, scores, i)  
    return scores  

def different(scores, old_scores):  
    ''''' 
    判断前后分数有无变化 
    :param scores: 
    :param old_scores: 
    :return: 
    '''  
    flag = False  
    for i in range(len(scores)):  
        if math.fabs(scores[i] - old_scores[i]) >= 0.0001:  
            flag = True  
            break  
    return flag 

def filter_symbols(sents):  
    stopwords = list(stopwordset)+ ['。', ' ', '.',', ','印']  
    _sents = []  
    for sentence in sents:  
        _sent = []
        for word in sentence:  
            if word not in stopwordset:  
                _sent.append(word)
        if _sent:  
            _sents.append(_sent)  
    return _sents  
def filter_model(sents):  
    _sents = []  
    for sentence in sents:  
        _sent = []
        for word in sentence:  
            if word in model: 
                _sent.append(word) 

        if _sent:  
            _sents.append(_sent)  
    return _sents  

def extractSummary(text, n=0):  
    text = text.replace('\n','')
    tokens = cut_sentences(text) 
    s_count = 0
    sentences = []  
    sents = []  
    ratio = 0.1 #取文本10%
    for sent in tokens:  
        s_count = s_count + 1
        sentences.append(sent)  
        sents.append([word for word in jieba.cut(sent) if word ])  
#    print(sents)
    sents_fs = filter_symbols(sents)  

    sents_fm = filter_model(sents_fs)  
    graph = create_graph(sents_fm)  
    if n==0:
        n = math.floor(s_count*ratio)

    scores = weight_sentences_rank(graph)  
    sent_selected = nlargest(n, zip(scores, count()))  
    sent_index = []  

    for i in range(n):  
        sent_index.append(sent_selected[i][1])  
    #return [sentences[i] for i in sent_index]  
    return [(sentences[i]) for i in sent_index] 


def test(string1,string2):
    return string1+string2


import sys
print(sys.path)
extract()

# if __name__=='__main__':
   

