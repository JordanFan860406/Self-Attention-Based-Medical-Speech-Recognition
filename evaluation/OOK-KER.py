'''
Out of keyword(OOK) & OOK-WER
'''

import os
import re
import numpy as np
import pandas as pd
from collections import defaultdict


def sep_seq(seq):
    """
    return a list of sentence
    e.g. "Ada{asd}qe{qs}a" -> [A,s,a,{asd},q,e,{qs},a]

    :param seq: input sentence
    :return: list of sentence
    """
    temp = []
    is_eng_word = False
    word_temp = ""
    for c in seq:
        word_temp = word_temp + c
        if c == "{" or c == "}" or c == "<" or c == ">":
            is_eng_word = not is_eng_word
            if not is_eng_word:
                temp.append(word_temp)
                word_temp = ""
        elif not is_eng_word:
            temp.append(word_temp)
            word_temp = ""
    return temp

#############seperate keyword in sentence#############
next_word_idx=0
def sep_keyword(q,next_word_idx,keyword_data,sentence):
    '''
    return a list of sentence(keyword will be <keyword>)
    e.g. "LUS{asd}qe{vac}引流" -> [<LUS>,{asd},q,e,<{vac}引流>]

    q: input str of sentence 
    next_word_idx: word idx
    keyword_data: keyword list.txt
    return: list of sentence
    '''
    tmp=[]
    tmp_word=""
    last_current_keyword = ""
    conti = False   

    #"LUS{asd}qe{vac}引流" -> [L,U,S,{asd},q,e,{vac},引,流]
    q = sep_seq(q) 

    for i in q[next_word_idx:]:
        tmp_word += i  
        current_keyword=[]
        for keyword in keyword_data:
            if tmp_word in keyword:
                current_keyword.append(keyword)
                if len(keyword)!=1 and conti==False :
                    conti = not conti
                if i not in tmp:
                    tmp.append(i)
        if len(current_keyword) == 1: last_current_keyword = ''.join(current_keyword)
        if len(current_keyword)==0 and len(tmp_word)!=1  or next_word_idx==len(q)-1:
            if (tmp_word in keyword_data) & (tmp_word!=last_current_keyword):
                conti = not conti
            else:# tmp_word!=last_current_keyword:
                conti = False
            #if has current keyword & len(tmp_word)>2: tmp_word=LUS -> <LUS>
            if next_word_idx!=len(q)-1 and len(sep_seq(tmp_word))>2:  #!=q[-1]             
                tmp_word = '<'+tmp_word[:len(tmp_word)-len(i)]+'>'
            elif next_word_idx==len(q)-1 & len(sep_seq(tmp_word))==2:
                if '{' in tmp_word:
                    conti = not conti
                else:
                    tmp_word = tmp_word[:len(tmp_word) - len(i)]    #if two wards are both chinese
            elif len(sep_seq(tmp_word))==2:
                tmp_word = tmp_word[:len(tmp_word)-len(i)]

        if not conti and next_word_idx != len(q)-1: #append keyword or word
            sentence.append(tmp_word)
            sentence_sep_seq = sep_seq(''.join(sentence).replace('<','').replace('>',''))
            next_word_idx =len(sentence_sep_seq)
            tmp_word=""
            break
        elif next_word_idx == len(q)-1 and len(tmp_word)!=1: #tmp_word=q[-2]
            sentence.append('<'+tmp_word+'>')
            tmp_word=""
            break
        elif next_word_idx == len(q)-1: #last word
            sentence.append(tmp_word)
            tmp_word=""
            break
        else: #conti search next keyword
            next_word_idx +=1

    if next_word_idx < len(q)-1 and len(current_keyword)==0 :
        return sep_keyword(q,next_word_idx,keyword_data,sentence) 
    else:
        sentence.append(q[-1])
        return sentence

#############Find keyword in sentence#############
def find_keyword(gt_data,pd_data,keyword_data):
    '''
    input : gt & pd After sep_keyword       ex:[['a','<LUS>'],['b','c'],...]
    return : total_key_gt, total_key_pd     ex:[['<LUS>'],[],...]
    '''
    total_key_gt=[]          #how many keyword appear in whole result
    total_key_pd=[]
    for num,i in enumerate(gt_data):
        if num < len(pd_data):
            tmp_key_gt=[]
            tmp_key_pd=[]

            for label in i:
                if '<' in label:
                    l=''.join(label).replace('<','').replace('>','')
                    for k in keyword_data:
                        if l==k:
                            tmp_key_gt.append(label) #keyword in gt
            
            for label in pd_data[num]:
                if '<' in label:
                    l=''.join(label).replace('<','').replace('>','')
                    for k in keyword_data:
                        if l==k:
                            tmp_key_pd.append(label) #keyword in pd
            total_key_gt.append(tmp_key_gt)
            total_key_pd.append(tmp_key_pd)

    return total_key_gt, total_key_pd

def find_ook(gt_data,pd_data,ook):
    '''
    input : gt & pd After sep_keyword       ex:[['a','<LUS>'],['b','c'],...]
    return : total_key_gt, total_key_pd     ex:[['<LUS>'],[],...]
    '''
    total_key_gt=[]          #how many keyword appear in whole result
    total_key_pd=[]
    for num,i in enumerate(gt_data):
        tmp_key_gt=[]
        tmp_key_pd=[]
        for label in i:
            if ('<' in label) & (label in ook):
                tmp_key_gt.append(label) #keyword in gt
        for label in pd_data[num]:
            if ('<' in label) & (label in ook):
                tmp_key_pd.append(label) #keyword in pd
        total_key_gt.append(tmp_key_gt)
        total_key_pd.append(tmp_key_pd)

    return total_key_gt, total_key_pd

def levenshtein_distance(hypothesis: list, reference: list, total_num_SDI): 
    """
    C: correct
    W: wrong (S+D+I)
    I: insert
    D: delete
    S: substitution

    :param hypothesis
    :param reference
    :return: 1: S，D，I 
             2: ref and hyp 的所有對齊的index
             3: 返回 C、W、S、D、I 各自的數量
    """
    len_hyp = len(hypothesis)
    len_ref = len(reference)
    cost_matrix = np.zeros((len_hyp + 1, len_ref + 1), dtype=np.int16)

    # 记录所有的操作，0-equal；1-insertion；2-deletion；3-substitution
    ops_matrix = np.zeros((len_hyp + 1, len_ref + 1), dtype=np.int8)

    for i in range(len_hyp + 1):
        cost_matrix[i][0] = i
    for j in range(len_ref + 1):
        cost_matrix[0][j] = j

    # 生成 cost 矩阵和 operation矩阵，i:外层hyp，j:内层ref
    for i in range(1, len_hyp + 1):
        for j in range(1, len_ref + 1):
            if hypothesis[i-1] == reference[j-1]:
                cost_matrix[i][j] = cost_matrix[i-1][j-1]
            else:
                substitution = cost_matrix[i-1][j-1] + 1
                insertion = cost_matrix[i-1][j] + 1
                deletion = cost_matrix[i][j-1] + 1

                compare_val = [substitution, insertion, deletion]  

                min_val = min(compare_val)
                operation_idx = compare_val.index(min_val) + 1
                cost_matrix[i][j] = min_val
                ops_matrix[i][j] = operation_idx

    match_idx = []  # 保存 hyp and ref 中所有對齊的元素下標
    i = len_hyp
    j = len_ref
    nb_map = {"N": len_ref, "C": 0, "W": 0, "I": 0, "D": 0, "S": 0}
    total_num_SDI["N"]+=nb_map["N"]

    ref_map = []
    hyp_map = []
    while i >= 0 or j >= 0:
        i_idx = max(0, i)
        j_idx = max(0, j)

        if ops_matrix[i_idx][j_idx] == 0:     # correct
            if i-1 >= 0 and j-1 >= 0:
                match_idx.append((j-1, i-1))
                nb_map['C'] += 1
                total_num_SDI['C'] += 1
                ref_map.append(reference[j-1])
                hyp_map.append(hypothesis[i-1])

            # 出邊界後，這裡仍然使用，應為第一行與第一列必然是全零
            i -= 1
            j -= 1

        elif ops_matrix[i_idx][j_idx] == 2:   # insert
            i -= 1
            nb_map['I'] += 1
            total_num_SDI['I'] += 1
            ref_map.append('**')
            hyp_map.append(hypothesis[i_idx-1])

        elif ops_matrix[i_idx][j_idx] == 3:   # delete
            j -= 1
            nb_map['D'] += 1
            total_num_SDI['D'] += 1
            ref_map.append(reference[j_idx-1])
            hyp_map.append('**')

        elif ops_matrix[i_idx][j_idx] == 1:   # substitute
            i -= 1
            j -= 1
            nb_map['S'] += 1
            total_num_SDI['S'] += 1
            ref_map.append(reference[j_idx-1])
            hyp_map.append(hypothesis[i_idx-1])

        # 出邊界處理
        if i < 0 and j >= 0:
            nb_map['D'] += 1
            total_num_SDI['D'] += 1
            # id_num_SDI['D'] += 1
            ref_map.append(reference[j])
            hyp_map.append('**')
        elif j < 0 and i >= 0:
            nb_map['I'] += 1
            total_num_SDI['I'] += 1
            ref_map.append('**')
            hyp_map.append(hypothesis[i]) ######hypothesis[i-1]

    match_idx.reverse()
    ref_map.reverse()
    hyp_map.reverse()
    wrong_cnt = cost_matrix[len_hyp][len_ref]
    nb_map["W"] = wrong_cnt
    total_num_SDI['W'] += nb_map["W"]

    kind_of_error=[]
    ref_error_dict=defaultdict(dict)
    for i in range(len(ref_map)):
        count = 1
        D_count = 1
        I_count = 1
        tmp_dict = {}

        if ref_map[i] == '**': #多翻
            kind_of_error.append('I')
            if hyp_map[i] not in ref_error_dict['**insert']:
                tmp_dict[hyp_map[i]] = count
                ref_error_dict['**insert'].update(tmp_dict)
            else:
                I_count += 1
                ref_error_dict['**insert'][hyp_map[i]]=I_count

        elif hyp_map[i] == '**': #少翻
            kind_of_error.append('D')
            if hyp_map[i] not in ref_error_dict[ref_map[i]]:
                tmp_dict['**delete'] = count
                ref_error_dict[ref_map[i]].update(tmp_dict)
            else:
                D_count += 1
                ref_error_dict[ref_map[i]]['**delete']=D_count

        elif ref_map[i] != hyp_map[i]: #翻錯
            kind_of_error.append('S')
            if hyp_map[i] not in ref_error_dict[ref_map[i]].keys():
                tmp_dict[hyp_map[i]] = count
                ref_error_dict[ref_map[i]].update(tmp_dict)
            else:
                count += 1
                ref_error_dict[ref_map[i]][hyp_map[i]]=count

        tmp_dict.clear()
    return wrong_cnt, match_idx, nb_map,total_num_SDI,ref_error_dict

########################################
#keyword_list
with open('manifest/sort_keyword_list.txt', "r", encoding='utf8') as f:
    keyword_data = f.readlines()
keyword_data = [a.strip() for a in keyword_data]  #schimes

#asr_result.txt
asr_result = 'result/psChiMeS_14_speed_conformer.txt'

#為了算ook放使用keyword增量前的訓和測試集
train_file = 'manifest/train_manifest_14.csv' 
test_file = 'manifest/test_manifest_14.csv'
########################################

with open(train_file, "r", encoding="utf-8") as f:
    train_files = f.readlines()
train_files = [a.strip() for a in train_files]
train_c=[]
for t in train_files:
    t = t.split(",")[1]
    with open (t,'r', encoding="utf-8")as f:
        data=sep_seq(''.join(f.readlines()))
    for d in data:
        if d not in train_c:
            train_c.append(d)

with open(test_file, "r", encoding="utf-8") as f:
    data_files = f.readlines()
data_files = [a.strip() for a in data_files]

def keyword_in_train_test(train_files, data_files,next_word_idx,keyword_data):
    #keywords in training set
    Tkeywords = []
    for i in train_files:
        label = i.split(",")[1]
        with open(label, "r", encoding="utf-8") as f:
            content = f.readlines()[0].strip()
        train_sentence=[]
        train_sentence =  sep_keyword(content,next_word_idx,keyword_data,train_sentence)
        for j in train_sentence:
            if ('<' in j) & (j not in Tkeywords):
                sj=''.join(j.replace('<','').replace('>',''))
                for k in keyword_data:                
                    if sj==k:
                        Tkeywords.append(j)
    if 'keyword' in train_file:
        for k in keyword_data:
            k='<'+k+'>'
            if k not in Tkeywords:
                Tkeywords.append(k)
                        
    #keywords in testing set
    keywords = []
    for i in data_files:
        label = i.split(",")[1]
        with open(label, "r", encoding="utf-8") as f:
            content = f.readlines()[0].strip()
        test_sentence=[]
        test_sentence =  sep_keyword(content,next_word_idx,keyword_data,test_sentence)
        for j in test_sentence:
            if ('<' in j) & (j not in keywords):
                sj=''.join(j.replace('<','').replace('>',''))
                for k in keyword_data:                
                    if sj==k:
                        keywords.append(j)
                        if (len(sep_seq(sj))==1) & (sj in train_c):
                            Tkeywords.append(j)
    return Tkeywords,keywords
        
print('============Train============')
print('train_files:',len(train_files))
print('============Test============')
print('test_files:',len(data_files))

Tkeywords, keywords = keyword_in_train_test(train_files, data_files,next_word_idx,keyword_data)
# Compare two file to get difference
s1 = set(Tkeywords)
s2 = set(keywords)
before_keyaug_ook=[]
before_keyaug_ook = list(s2.difference(s1))

#################OOK KER in asr_result#######################
#test gt
test_manifest_list=[]
for i in data_files:
    i=i.split(',')[-1]              #/home/.../0530_03_25.txt
    test_manifest_list.append(i) 

n_list=[]
with open(asr_result,'r', encoding="utf-8")as f:
    data=f.readlines()
n_list=[a.strip().split('\t')[0] for a in data]

gt_list=[]
for i in n_list:
    df_file_path = i + '.txt' #ex:0530_03_25.txt
    for m in test_manifest_list:
        if df_file_path in m:
            with open(m,'r', encoding="utf-8")as f:
                data=f.readlines()
            gt_list.append(data)

#test pd
with open(asr_result,'r',encoding='utf-8')as f:
    data=f.readlines()
data = [a.strip().split('\t')[-1] for a in data]

#sep test: gt & pd, anotate keyword
gt_data=[]
pd_data=[]

for i in gt_list:
    gt_sentence=[]
    g = ''.join(i).replace(',','').replace('。','').replace('：','')
    gt_sentence =  sep_keyword(g,next_word_idx,keyword_data,gt_sentence)
    gt_data.append(gt_sentence)

for num in range(len(data)): 
    pd_sentence=[]  
    p = ''.join(data[num]).replace(',','').replace('。','').replace('：','')
    pd_sentence =  sep_keyword(p,next_word_idx,keyword_data,pd_sentence)
    pd_data.append(pd_sentence)

#################OOK-KER#####################
#list of keywords in testing gt & pd
oov_key_gt, oov_key_pd=find_ook(gt_data,pd_data,before_keyaug_ook)

oov_num_SDI={"N": 0, "C": 0, "W": 0, "I": 0, "D": 0, "S": 0}
oov_ref_error_dict=defaultdict(dict)  
for i in range(len(oov_key_pd)):
    wrong_cnt, match_idx, nb_map, oov_num_SDI, ref_error_dict = levenshtein_distance(
                        reference=oov_key_gt[i],
                        hypothesis=oov_key_pd[i],
                        total_num_SDI=oov_num_SDI,
                    )

#KER=(S+D+I)/# ref keywords
OOSDI=oov_num_SDI['S']+oov_num_SDI['D']+oov_num_SDI['I']
OOKER = OOSDI /oov_num_SDI['N']               #whole nume of keywords
print("Final OOKER:",OOKER)

#################KER#####################
total_key_gt, total_key_pd=find_keyword(gt_data,pd_data,keyword_data)

total_num_SDI={"N": 0, "C": 0, "W": 0, "I": 0, "D": 0, "S": 0}
total_ref_error_dict=defaultdict(dict)  
for i in range(len(total_key_pd)):
    wrong_cnt, match_idx, nb_map, total_num_SDI, ref_error_dict = levenshtein_distance(
                        reference=total_key_gt[i],
                        hypothesis=total_key_pd[i],
                        total_num_SDI=total_num_SDI,
                    )
#KER=(S+D+I)/# ref keywords
SDI=total_num_SDI['S']+total_num_SDI['D']+total_num_SDI['I']
KER = SDI /total_num_SDI['N']               #whole nume of keywords
print("Final KER:",KER)

#################Num of OOK#####################
ap=[]
for i in oov_key_gt:
    if len(i)!=0:
        for a in i:
            if a not in ap:
                ap.append(a)
eap=[]
for i in oov_key_pd:
    if len(i)!=0:
        for a in i:
            if a not in eap:
                eap.append(a)
print("*******Diff OOK in testing gt:",len(ap))
print("*******Correct num of diff OOK in testing pd:",len(eap))
print("Correct diff OOK:",eap)