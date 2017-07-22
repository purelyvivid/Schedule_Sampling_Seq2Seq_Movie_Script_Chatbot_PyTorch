
# coding: utf-8

# - try to implrment data.py

# In[1]:

corpus_path = 'corpus/'  #cornell movie-dialogs corpus
save_path = 'datasets/'

EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz ' 
EN_BLACKLIST = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\''

limit = {
        'maxq' : 25,
        'minq' : 2,
        'maxa' : 25,
        'mina' : 2
        }

UNK = 'unk'
VOCAB_SIZE = 8997

# idx2w[0]='_'    ...zero padding
# idx2w[1]='unk'
# idx2w[2]='<G0>'
# idx2w[3]='<EOS>'
# total 9000


import random

import nltk
import itertools
from collections import defaultdict

import numpy as np

import pickle


# In[16]:

''' 
    1. Read from 'movie_characters_metadata.txt'
    2. Create a dictionary with ( key = u_id, value = gender )
'''
def get_character2gender():
    lines=open(corpus_path+'movie_characters_metadata.txt', 
               ).read().split('\n') #encoding='utf-8', errors='ignore'
    c2g = {} #Create a dictionary
    for line in lines:
        _line = line.split(' +++$+++ ')
        if len(_line) == 6: #符合格式
            c2g[ _line[0] ] = _line[4] #Update the dictionary
            #_line[0]='u0', _line[4]='f'
    return c2g




# In[13]:

''' 
    1. Read from 'movie-lines.txt'
    2. Create a dictionary with ( key = line_id, value = text )
'''
def get_id2line():
    lines=open(corpus_path+'movie_lines.txt', 
               ).read().split('\n') #encoding='utf-8', errors='ignore'
    id2line = {} #Create a dictionary
    for line in lines:
        _line = line.split(' +++$+++ ')
        if len(_line) == 5: #符合格式
            id2line[ _line[0] ] = _line[4] #Update the dictionary
            #_line[0]='L1045', _line[4]='They do not!'
    return id2line




# In[21]:

''' 
    1. Read from 'movie-lines.txt'
    2. Create a dictionary with ( key = line_id, value = gender )
'''
def get_id2gender(u2gender):
    lines=open(corpus_path+'movie_lines.txt', 
               ).read().split('\n') #encoding='utf-8', errors='ignore'
    id2gender = {} #Create a dictionary
    for line in lines:
        _line = line.split(' +++$+++ ')
        if len(_line) == 5: #符合格式
            id2gender[ _line[0] ] = u2gender [ _line[1] ] #Update the dictionary
            #_line[0]='L1045', _line[4]='They do not!'
    return id2gender




# In[2]:

'''
    1. Read from 'movie_conversations.txt'
    2. Create a list of [list of line_id's]
'''
def get_conversations():
    conv_lines = open(corpus_path+'movie_conversations.txt', 
                      ).read().split('\n') #encoding='utf-8', errors='ignore'
    convs = [ ] #Create a list
    for line in conv_lines[:-1]:
        _line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
        #_line[-1]= "'L194', 'L195', 'L196', 'L197'"
        #[1:-1]: remove '[',']' 
        #_line = 'L194,L195,L196,L197'
        #_line.split(',')=['L194', 'L195', 'L196', 'L197']
        convs.append(_line.split(','))
    return convs




# In[4]:

'''
花時間!!暫時不需要!!!
    1. Get each conversation
    2. Get each line from conversation
    3. Save each conversation to file
'''
def extract_conversations(convs,id2line,path=''):
    idx = 0
    for conv in convs:
        f_conv = open(path + str(idx)+'.txt', 'w')#create file for each conv
        for line_id in conv:
            f_conv.write(id2line[line_id])
            f_conv.write('\n')
        f_conv.close()
        idx += 1
        


# In[15]:

'''
    Get lists of all conversations as Questions and Answers
    1. [questions]
    2. [answers]
'''
def gather_dataset(convs, id2line):
    questions = []; answers = []

    for conv in convs:
        '''
        #把['L194', 'L195', 'L196', 'L197']分成
        #['L194', 'L195']和[ 'L196', 'L197']
        #共13萬筆
        if len(conv) %2 != 0:
            conv = conv[:-1] #只取到偶數個
        for i in range(len(conv)):
            if i%2 == 0:
                questions.append(id2line[conv[i]])
            else:
                answers.append(id2line[conv[i]])
        '''
        for i in range(len(conv)-1):
            questions.append(id2line[conv[i]])
            answers.append(id2line[conv[i+1]])  
            #把['L194', 'L195', 'L196', 'L197']分成
            #['L194', 'L195'],['L195', 'L196']和[ 'L196', 'L197']
            #共22萬筆

    return questions, answers




# In[34]:

'''
    Get lists of Pratial conversations as Questions and Answers By Gender
    1. [questions]
    2. [answers]
'''
def gather_dataset_by_gender(convs, id2line, id2gender,
                             que_gender='?', ans_gender='?' ): 
    #if ans_gender = 'f', just female answer
    #if ans_gender = 'm', just male answer
    #if ans_gender = '?', both male/female answer, including gender unknown
    
    questions = []; answers = []

    for conv in convs:
        for i in range(len(conv)-1):
            # check gender
            bool_1 = que_gender == id2gender[conv[i]] 
            bool_2 = ans_gender == id2gender[conv[i+1]] 
            bool_3 = que_gender == '?' 
            bool_4 = ans_gender == '?' 
            bool_que = bool_1 or bool_3 
            bool_ans = bool_2 or bool_4 
            if ( bool_que and bool_ans ) :    
                questions.append(id2line[conv[i]])
                answers.append(id2line[conv[i+1]])  
                #把['L194', 'L195', 'L196', 'L197']分成
                #['L194', 'L195'],['L195', 'L196']和[ 'L196', 'L197']
                #共22萬筆
    return questions, answers


# In[39]:


'''
暫時沒用到!!!
    Shuffle and split to train/test dataset
    We need 4 files
        1. train.enc : Encoder input for training
        2. train.dec : Decoder input for training
        3. test.enc  : Encoder input for testing
        4. test.dec  : Decoder input for testing
    -> Encoder input is question, and Decoder input is answer 
'''
def prepare_seq2seq_files(questions, answers, gender = 'f' ,
                          path='',TESTSET_SIZE = 10000):
    
    # open files
    train_enc = open(path + gender+'_train.enc','w')
    train_dec = open(path + gender+'_train.dec','w')
    test_enc  = open(path + gender+'_test.enc', 'w')
    test_dec  = open(path + gender+'_test.dec', 'w')

    # choose 10,000 (TESTSET_SIZE) items to put into testset
    test_ids = random.sample([i for i in range(len(questions))],TESTSET_SIZE)

    for i in range(len(questions)):
        if i in test_ids:
            test_enc.write(questions[i]+'\n')
            test_dec.write(answers[i]+ '\n' )
        else:
            train_enc.write(questions[i]+'\n')
            train_dec.write(answers[i]+ '\n' )
        if i%10000 == 0:
            print('>> written {} lines'.format(i))

    # close files
    train_enc.close()
    train_dec.close()
    test_enc.close()
    test_dec.close()
    print('>>written finish')
            

# In[52]:

'''
 remove anything that isn't in the vocabulary
    return str(pure en)
'''
def filter_line(line, whitelist): # whitelist:安全名單  只留下whitelist
    return ''.join([ ch for ch in line if ch in whitelist ])

#filter_line('fweijwhf8328r8uAAAgerg19,,',EN_WHITELIST)


# In[43]:

'''
 filter too long and too short sequences
    return tuple( filtered_ta, filtered_en )
'''
def filter_data(qseq, aseq):
    filtered_q, filtered_a = [], []
    raw_data_len = len(qseq)

    assert len(qseq) == len(aseq)

    for i in range(raw_data_len):
        qlen, alen = len(qseq[i].split(' ')), len(aseq[i].split(' '))
        if qlen >= limit['minq'] and qlen <= limit['maxq']:
            if alen >= limit['mina'] and alen <= limit['maxa']:
                filtered_q.append(qseq[i])
                filtered_a.append(aseq[i])

    # print the fraction of the original data, filtered
    filt_data_len = len(filtered_q)
    filtered = int((raw_data_len - filt_data_len)*100/raw_data_len)
    print(str(filtered) + '% filtered from original data')

    return filtered_q, filtered_a


# In[51]:

'''
 read list of words, create index to word,
  word to index dictionaries
    return tuple( vocab->(word, count), idx2w, w2idx )
'''
def index_(tokenized_sentences, vocab_size):
    # get frequency distribution
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    # get vocabulary of 'vocab_size' most used words
    vocab = freq_dist.most_common(vocab_size)
    # index2word (as a list)
    index2word = ['_'] + [UNK] + [ x[0] for x in vocab ]
    # word2index (as a dict)
    word2index = dict([(w,i) for i,w in enumerate(index2word)] )
    return index2word, word2index, freq_dist


# In[55]:

'''
 filter based on number of unknowns (words not in vocabulary)
  filter out the worst sentences
'''
def filter_unk(qtokenized, atokenized, w2idx):
    data_len = len(qtokenized)

    filtered_q, filtered_a = [], []

    for qline, aline in zip(qtokenized, atokenized):
        unk_count_q = len([ w for w in qline if w not in w2idx ])
        unk_count_a = len([ w for w in aline if w not in w2idx ])
        if unk_count_a <= 2:
            if unk_count_q > 0:
                if unk_count_q/len(qline) > 0.2:
                    pass
            filtered_q.append(qline)
            filtered_a.append(aline)

    # print the fraction of the original data, filtered
    filt_data_len = len(filtered_q)
    filtered = int((data_len - filt_data_len)*100/data_len)
    print(str(filtered) + '% filtered from original data')

    return filtered_q, filtered_a


# In[56]:

'''
 create the final dataset : 
  - convert list of items to arrays of indices
  - add zero padding
      return ( [array_en([indices]), array_ta([indices]) )
 
'''
def zero_pad(qtokenized, atokenized, w2idx):
    # num of rows
    data_len = len(qtokenized)

    # numpy arrays to store indices
    idx_q = np.zeros([data_len, limit['maxq']], dtype=np.int32) 
    idx_a = np.zeros([data_len, limit['maxa']], dtype=np.int32)

    for i in range(data_len):
        q_indices = pad_seq(qtokenized[i], w2idx, limit['maxq'])
        a_indices = pad_seq(atokenized[i], w2idx, limit['maxa'])

        #print(len(idx_q[i]), len(q_indices))
        #print(len(idx_a[i]), len(a_indices))
        idx_q[i] = np.array(q_indices)
        idx_a[i] = np.array(a_indices)

    return idx_q, idx_a


# In[57]:

'''
 replace words with indices in a sequence
  replace with unknown if word not in lookup
    return [list of indices]
'''
def pad_seq(seq, lookup, maxlen):
    indices = []
    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[UNK])
    return indices + [0]*(maxlen - len(seq))



# In[59]:

'''
    main process
    for chatbot
'''
def process_data(): 
    
    u2gender = get_character2gender()
    print('>> gathered u2gender dictionary.\n')
    id2line = get_id2line()
    print('>> gathered id2line dictionary.\n')
    id2gender = get_id2gender(u2gender)
    print('>> gathered id2gender dictionary.\n')
    convs = get_conversations() # [ ['L750', 'L751'], [...] ,... ]
    print('>> gathered all conversations.\n')
    #questions, answers = gather_dataset(convs,id2line)
    f_questions, f_answers = gather_dataset_by_gender(convs, 
                                                  id2line, id2gender,
                                                  ans_gender='f')
    m_questions, m_answers = gather_dataset_by_gender(convs, 
                                                  id2line, id2gender,
                                                  ans_gender='m')
    print('\n Female dataset len : ' + str(len(f_questions)))
    print('\n Male dataset len : ' + str(len(m_questions)))
    
    
    # change to lower case (just for en)
    f_questions = [ line.lower() for line in f_questions ]
    f_answers = [ line.lower() for line in f_answers ]
    m_questions = [ line.lower() for line in m_questions ]
    m_answers = [ line.lower() for line in m_answers ]
    
    # filter out unnecessary characters #空白不能濾掉, 否則無法分字
    # 確保句子裡出現的所有字元, 都是在安全名單內
    print('\n>> Filter lines')
    f_questions = [ filter_line(line, EN_WHITELIST) for line in f_questions ]
    f_answers = [ filter_line(line, EN_WHITELIST) for line in f_answers ]
    m_questions = [ filter_line(line, EN_WHITELIST) for line in m_questions ]
    m_answers = [ filter_line(line, EN_WHITELIST) for line in m_answers ]
    
    # filter out too long or too short sequences
    print('\n>> Filter out too long or too short sequences')
    f_qlines, f_alines = filter_data(f_questions, f_answers)
    m_qlines, m_alines = filter_data(m_questions, m_answers)
    print('\n Female dataset len : ' + str(len(f_qlines)))
    print('\n Male dataset len : ' + str(len(m_qlines)))

    print('\n Before token : ')
    for q,a in zip(f_qlines[141:145], f_alines[141:145]):
        print('q : [{0}]; a : [{1}]'.format(q,a))

    # tokenize: convert list of [lines of text] into list of [list of words ]
    print('\n>> Segment lines into words')
    f_qtokenized = [ [w.strip() for w in wordlist.split(' ') if w] 
                  for wordlist in f_qlines ]
    f_atokenized = [ [w.strip() for w in wordlist.split(' ') if w] 
                  for wordlist in f_alines ]
    m_qtokenized = [ [w.strip() for w in wordlist.split(' ') if w] 
                  for wordlist in m_qlines ]
    m_atokenized = [ [w.strip() for w in wordlist.split(' ') if w] 
                  for wordlist in m_alines ]
    print('\n:: Sample from segmented list of words')
    
    print('\n After token : ')
    for q,a in zip(f_qtokenized[141:145], f_atokenized[141:145]):
        print('q : [{0}]; a : [{1}]'.format(q,a))

    # indexing -> idx2w, w2idx 
    all_tokenized = f_qtokenized + f_atokenized + m_qtokenized + m_atokenized
    print('\n >> Index words')
    idx2w, w2idx, freq_dist = index_( all_tokenized, 
                                     vocab_size=VOCAB_SIZE)
    
    # filter out sentences with too many unknown words
    print('\n >> Filter out sentences with too many unknown words')
    f_qtokenized, f_atokenized = filter_unk(f_qtokenized, f_atokenized, w2idx)
    m_qtokenized, m_atokenized = filter_unk(m_qtokenized, m_atokenized, w2idx)
    print('\n Final Female dataset len : ' + str(len(f_qtokenized)))
    print('\n Final Male dataset len : ' + str(len(m_qtokenized)))

    print('\n >> Zero Padding')
    f_idx_q, f_idx_a = zero_pad(f_qtokenized, f_atokenized, w2idx)
    m_idx_q, m_idx_a = zero_pad(m_qtokenized, m_atokenized, w2idx)
    
    print('\n >> Save numpy arrays to disk')
    # save them
    np.save(save_path+'f_idx_q.npy', f_idx_q)
    np.save(save_path+'f_idx_a.npy', f_idx_a)
    np.save(save_path+'m_idx_q.npy', m_idx_q)
    np.save(save_path+'m_idx_a.npy', m_idx_a)
    
    # let us now save the necessary dictionaries
    metadata = {
            'w2idx' : w2idx,
            'idx2w' : idx2w,
            'limit' : limit,
            'freq_dist' : freq_dist
                }

    # write to disk : data control dictionaries
    with open(save_path+'metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)

    # count of unknowns
    unk_count = (f_idx_q == 1).sum() + (f_idx_a == 1).sum() + (m_idx_q == 1).sum() + (m_idx_a == 1).sum()
    # count of words
    word_count = (f_idx_q > 1).sum() + (f_idx_a > 1).sum() + (m_idx_q > 1).sum() + (m_idx_a > 1).sum()

    print('% unknown : {0}'.format(100 * (unk_count/word_count)))
    print('Female Dataset count : ' + str(f_idx_q.shape[0]))
    print('Male Dataset count : ' + str(m_idx_q.shape[0]))

    #print '>> gathered questions and answers.\n'
    #prepare_seq2seq_files(f_idx_q,f_idx_a,'f')
    #prepare_seq2seq_files(m_idx_q,m_idx_a,'m')




# In[ ]:

if __name__ == '__main__':
    process_data()


# In[66]:

'''
    load_data
    for chatbot
'''
def load_data(PATH=''):
    # read data control dictionaries
    with open(PATH + 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    # read numpy arrays
    f_idx_q = np.load(PATH + 'f_idx_q.npy')
    f_idx_a = np.load(PATH + 'f_idx_a.npy')
    m_idx_q = np.load(PATH + 'm_idx_q.npy')
    m_idx_a = np.load(PATH + 'm_idx_a.npy')
    return metadata, f_idx_q, f_idx_a, m_idx_q, m_idx_a

def load_data_female(PATH=''):
    # read numpy arrays
    f_idx_q = np.load(PATH + 'f_idx_q.npy')
    f_idx_a = np.load(PATH + 'f_idx_a.npy')
    return f_idx_q, f_idx_a



