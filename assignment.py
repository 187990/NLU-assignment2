#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 15:41:18 2021

@author: dimitri
"""
import pandas as pd
import conll
import spacy
import numpy

#EXERCISE 1 AND 3 UTILITY FUNCTION
def evaluate_tokes(ref, hyp, otag='O'):
    """
    evaluate in tokens level
    :param ref: reference sentences
    :param hyp: hypothesis sentences
    :return: accuracy for classes and total
    """
    
    #inside function to calculate accuracy
    def score(cor_cnt, ref_cnt):
        a = 1 if (cor_cnt == 0 and ref_cnt==0) else cor_cnt / ref_cnt
        return {"acc": a}
    #obtain align data
    aligned = conll.align_hyp(ref, hyp)
    
    #count without iob
    
    #retrive for total TP, not setted, total (TP+TN+FP+FN)
    tok = conll.stats()
    #define the dictionary
    cls = {}
    # token-level counts without iob
    for sent in aligned:
        for token in sent:
            #retrieve tags and iob for hyp and ref
            hyp_iob, hyp = conll.parse_iob(token[-1])
            ref_iob, ref = conll.parse_iob(token[-2])

            if(ref != None):#if ref is not none i found a tagged element
                #if dictionary is not setted
                #retrive TP, TP+FN, not sette
                if not cls.get(ref) and ref:
                    cls[ref] = conll.stats()

                #if true positive
                if ref == hyp:#if i have same tags 
                    tok['cor'] += 1 #one TP added
                    cls[ref]['cor'] += 1 #one TP added in class
                #total count for overall accuracy
                tok['ref'] += 1 #one added in total
                #partial count for partial accuracies
                if ref:
                    cls[ref]['ref'] += 1 #one TP or FN added in class
               
                
                
            else:
                
                if not cls.get(ref_iob):
                    cls[ref_iob] = conll.stats()
               
                if(hyp_iob=='O'):
                    tok['cor']+=1  #add one TP in total
                    cls[hyp_iob]['cor'] += 1 #add one TP in class O
                tok['ref']+=1 #add one TP or FN in total
                cls[ref_iob]['ref'] += 1 #one TP or FN added in class O

                     
    #the partial accuracies are the true positive (or correct prediction) for a class divided by the sum of TP and al FN for that class        
    res={lbl: score(cls[lbl]['cor'], cls[lbl]['ref']) for lbl in set(cls.keys())}  
    #the total or overall accuracy is the number of corrected prediction over all prediction
    res.update({"total without iob": score(tok.get('cor', 0), tok.get('ref', 0))})
    
    #count with iob
    
    #retrive for total TP, not setted, total (TP+TN+FP+FN)
    tok = conll.stats()
    #define the dictionary
    cls = {}
       # token-level counts without iob
    for sent in aligned:
        for token in sent:
            #retrieve tags and iob for hyp and ref
            hyp_iob, hyp = conll.parse_iob(token[-1])
            ref_iob, ref = conll.parse_iob(token[-2])

            if(ref != None):#if ref is not none i found a tagged element
               
                ref_key=str(ref_iob)+"-"+str(ref)

                #if dictionary is not setted
                #retrive TP, TP+FN, not sette
                if not cls.get(ref_key) and ref:
                    cls[ref_key] = conll.stats()

                #if true positive
                if ref == hyp and ref_iob == hyp_iob:#if i have same tags and same iob for ref and hyps
                    tok['cor'] += 1 #one TP added
                    cls[ref_key]['cor'] += 1 #one TP added in class
                #total count for overall accuracy
                tok['ref'] += 1 #one added in total
                #partial count for partial accuracies
                if ref:
                    cls[ref_key]['ref'] += 1 #one TP or FN added in class
               
                
                
            else:
                
                if not cls.get(ref_iob):
                    cls[ref_iob] = conll.stats()
               
                if(hyp_iob=='O'):
                    tok['cor']+=1  #add one TP in total
                    cls[hyp_iob]['cor'] += 1 #add one TP in class O
                tok['ref']+=1 #add one TP or FN in total
                cls[ref_iob]['ref'] += 1 #one TP or FN added in class O

    #the partial accuracies are the true positive (or correct prediction) for a class divided by the sum of TP and al FN for that class        
    res.update({lbl: score(cls[lbl]['cor'], cls[lbl]['ref']) for lbl in set(cls.keys())})  
    #the total or overall accuracy is the number of corrected prediction over all prediction
    res.update({"total with iob": score(tok.get('cor', 0), tok.get('ref', 0))})
    
    return res
            
            

def remap(value):
    """
    remap some named tags from spacy to conll
    :param value: value of the tag
    :return: new tag value
    """
    tag=""
    if(value=="PERSON"):
        tag="PER"
    elif(value=="EVENT"):
        tag="MISC"
    elif(value=="FAC"):
        tag="LOC"
    elif(value=="GPE"):
        tag="LOC"
    elif(value=="LAW"):
        tag="MISC"
    elif(value=="NORP"):
        tag="MISC"
    elif(value=="LANGUAGE"):
        tag="MISC"
    elif(value=="WORK_OF_ART"):
        tag="MISC"
    else: #mantain same tag
        tag=value
    return tag

def restructure_tokenisation(token, doc, new_list, problem='-', compound=False):   
    """
    resolve some tokenisation difference from spacy to connll
    :param token: a token
    :param doc: a parsed phrase
    :param new_list: list of tuples to be aggregated to the hyps
    :return: an index to jump in analizing tokens for creating hyps
    """
    jump_index=token.i #index to jump
    text=token.text #text of the token
    tag=""
    if problem=='-':  
        for num, t in enumerate(doc):
            if num<=token.i: #until you are not to the token index continue
                continue
            #concatenate text if you find a mismatch
            elif (num>token.i  and ((t.whitespace_=='' and t.text=='-') or doc[t.i-1].text=='-' )):
                text+=t.text
                jump_index+=1
            else:
                break
            
 
    #construct the label and add the tuple
        if(token.ent_type_!='') and (compound==False): 
            if(token.ent_type_!='DATE') and (token.ent_type_!='MONEY')  and (token.ent_type_!='PERCENT') and (token.ent_type_!='PRODUCT') and (token.ent_type_!='QUANTITY') and (token.ent_type_!='ORDINAL') and (token.ent_type_!='CARDINAL') and (token.ent_type_!='TIME') :
                tag=token.ent_iob_+'-'+remap(token.ent_type_)
            else:
                tag='O'               
        elif (token.ent_type_!='') and (compound[token.i] != None):
             if(token.ent_type_!='DATE') and (token.ent_type_!='MONEY')  and (token.ent_type_!='PERCENT') and (token.ent_type_!='PRODUCT') and (token.ent_type_!='QUANTITY') and (token.ent_type_!='ORDINAL') and (token.ent_type_!='CARDINAL') and (token.ent_type_!='TIME') :
                tag=compound[token.i]+'-'+remap(token.ent_type_)
             else:
                tag='O'  
        else:
            tag=token.ent_iob_
        new_list.append((text, tag))
        
    
    return jump_index              


                
             
#EXERCISE 1                
def test_spacy(path):
    """
    test spacy on conll
    :param path: the path in which ground truth is stored
    :return: a tuple containing pandas table for accuracy for tokens and other metrics for chunks
    """ 
    
    #PREPARING EVALUATION
    
    nlp = spacy.load('en_core_web_sm')
    refs=[]#refs list
    #extract the sentence as token list from corpus
    temporary_corpus=conll.read_corpus_conll(path)
    hyp_test_corpus=[]
    #recreate the corpus phrases and create the ground truth
    for sent in temporary_corpus:
        new_sent=[]
        new_list=[]
        for element in sent:
            word=element[0].split()[0]
            tag=element[0].split()[3]
            new_sent.append(word)
            new_tuple=(word, tag)
            new_list.append(new_tuple)
        if new_sent!=['-DOCSTART-']:
            hyp_test_corpus.append(" ".join(new_sent)) #add a phrase to recreated corpus
            refs.append(new_list) #add a tuple to ground truth
    
    
    
    #creating hyps list
    hyps=[]
    for sent in hyp_test_corpus:
        doc = nlp(sent)
        new_list=[]
        jump_index=-1#if you have to merge some tokens
        for token in doc:
            text=token.text
            #jump if you have to jump some already considered tokens
            if(token.i<=jump_index):
                continue
            #if find a token you have to merge with the followings
            elif token.whitespace_=='' and (doc.__len__()>(token.i+1)):
                if doc[token.i+1].text=='-':
                    jump_index=restructure_tokenisation(token, doc, new_list) 
            #else consider the label and text and add a tuple 
            elif(token.ent_type_!=''):#a named entity
                if(token.ent_type_!='DATE') and (token.ent_type_!='PERCENT') and (token.ent_type_!='MONEY') and (token.ent_type_!='TIME') and (token.ent_type_!='QUANTITY') and (token.ent_type_!='ORDINAL') and (token.ent_type_!='CARDINAL') and (token.ent_type_!='PRODUCT'):
                    tag=token.ent_iob_+'-'+remap(token.ent_type_)#note that you have to remap named tags
                    new_tuple=(text, tag)
                    new_list.append(new_tuple)
                else: #some tags are not considered in conll
                    tag='O'
                    new_tuple=(text, tag)
                    new_list.append(new_tuple)

            else:#for unnamed entity
                tag=token.ent_iob_
                new_tuple=(text, tag)
                new_list.append(new_tuple)
        hyps.append(new_list)
                
   
 

   
    
   #token eval
    results = evaluate_tokes(refs, hyps)
    pd_tbl_tok = pd.DataFrame().from_dict(results, orient='index')
    pd_tbl_tok.round(decimals=3)

   #chunk eval
    results = conll.evaluate(refs, hyps)
    pd_tbl_chunk = pd.DataFrame().from_dict(results, orient='index')
    pd_tbl_chunk.round(decimals=3)
   
    return((pd_tbl_tok, pd_tbl_chunk))

#EXERCISE 2 UTILITY FUNCTION

def extract_list_of_list(doc):
    """
    create a list of list formed by the named tag of noun chunks or single token if a token is not inside chunk
    :param doc: a parsed phrase
    :return: as described
    """ 
    outer_list=[] #result list
    
    #variable used to account for position and inserted chunk to cope with exlcuded single token chunk
    mask=numpy.full(doc.__len__(), False, dtype=bool)
    inserted_limit=[]
    
    #insert chunk in list
    for chunk in doc.noun_chunks:
        inner_list=[]
        sx=float('inf')
        for token in chunk:
            if token.i < sx:
                sx=token.i
            if(token.ent_type_!='') and not(token.ent_type_ in inner_list):
                inner_list.append(token.ent_type_)
            mask[token.i]=True
        inserted_limit.append(sx)
        if not inner_list:
            continue
        else:
            outer_list.append(inner_list)
                    

    #find chunks of single token and add in the correct place of the lists
    for i in range(doc.__len__()-1):
        if mask[i]==False:
            if doc[i].ent_type_!='':
                token=[]
                token.append(doc[i].ent_type_)
                insert_index=-1
                if not outer_list:
                    inserted_limit.append(i)
                    mask[i]=True
                else:
                    for element in outer_list:
                        index=outer_list.index(element) 
                        limit=inserted_limit[index]
                        if i<limit:
                            insert_index=index
                            break
                        else:
                            continue
                    inserted_limit.append(i)
                    inserted_limit.sort()
                    mask[i]=True
                outer_list.insert(insert_index, token)
    
    return outer_list        
       
def obtain_phrases(path): 
    """
    obtain phrases from document in conll format
    :param path: the path in which conll is stored
    :return: a list of string phrases
    """             
    
    temporary_corpus=conll.read_corpus_conll(path)
    corpus=[]
    #recreate the corpus phrases and create the ground truth
    for sent in temporary_corpus:
        new_sent=[]
        for element in sent:
            word=element[0].split()[0]
            new_sent.append(word)
        if new_sent!=['-DOCSTART-']:
                corpus.append(" ".join(new_sent)) #add a phrase to recreated corpus
    
    return corpus
   
#EXERCISE 2
def group_name_entities(phrases): #SAREBBE DA METTER UN NA NEI TOKENZ 
    """
    group name entities and report most frequent combos
    :param phrases: phrases list to be analised (can be a single phrase or a doc extracted for example)
    from conll2003 but in this case must be convertend in list of phrases as was done in the exercise 
    before
    :return: a class containing the analysis
    """ 
    class statistics:
        stat_dict={} #dictionary containing data for each possible chunks combos finding in phrases
        list_of_lists=[] #contain list of lists
        
        #class constructor
        def  __init__(self):
            self.total_groups_wouttoken=0;
            self.total_groups=0
        
        #update group counts
        def update_group(self, element):
            if len(element)>1:
                self.total_groups_wouttoken+=1
            self.total_groups+=1
            
        #update relative frequencies
        def update_dict(self):
            for key in self.stat_dict:
                if(len(key.split())>1):
                    self.stat_dict[key][2]='n/a' if self.total_groups_wouttoken==0 else self.stat_dict[key][0]/self.total_groups_wouttoken 
                else:
                    self.stat_dict[key][2]='n/a'
                self.stat_dict[key][1]='n/a' if self.total_groups==0 else self.stat_dict[key][0]/self.total_groups
            
        #add a phrase , insert in the lists of lists and for each combos of token tags insert in the dictionary
        def add_phrase(self, new_list):
            self.list_of_lists.append(new_list)
            for sub_list in new_list:
                
                self.update_group(sub_list)
                element=""
                for sub_element in sub_list:
                    element=element+str(sub_element)+", "
                
                    
                if element in self.stat_dict:
                    self.stat_dict[element][0]+=1
                else: 
                    self.stat_dict[element]=[]
                    self.stat_dict[element].append(1)
                    self.stat_dict[element].append(0)
                    self.stat_dict[element].append(0)
            self.update_dict()
        
        #obtain list of lists for each phrase
        def get_list_of_lists(self):
            return self.list_of_lists
         
        #obtain frequencies absolute
        def get_freq_groups(self, reverse=False):
            list_freq=[]
            for key, value in self.stat_dict.items():
                list_freq.append((key, value[0]))
            
            if reverse:
                list_freq.sort(key=lambda tup: tup[1] , reverse=True)

                
            else:
                list_freq.sort(key=lambda tup: tup[1])
                
            return list_freq
        
        #obtain relative frquencies counting all sublists
        def get_rel_freq_groups(self, reverse=False):
            list_freq=[]
            for key, value in self.stat_dict.items():
                list_freq.append((key, value[1]))
            
            if reverse:
                list_freq.sort(key=lambda tup: tup[1] , reverse=True)

                
            else:
                list_freq.sort(key=lambda tup: tup[1])
                
            return list_freq
        
        #obtain relative frequences without counting single token sublists
        def get_rel_freq_groups_wouttokens(self, reverse=False):
            list_freq=[]
            for key, value in self.stat_dict.items():
                if value[2]!='n/a':    
                    list_freq.append((key, value[2]))
            
            
            if reverse:
                list_freq.sort(key=lambda tup: tup[1] , reverse=True)

                
            else:
                list_freq.sort(key=lambda tup: tup[1])
                
            return list_freq
               

            
    
    nlp = spacy.load('en_core_web_sm')
    stats=statistics()
    
    #parsing of the sentences, extract the list of list as requested in assignment and pass to class
    for sent in phrases:
        doc = nlp(sent)
        outer_list=extract_list_of_list(doc)
        stats.add_phrase(outer_list)
        
    return stats #return the class
 
 
#EXERCISE 3
#there are the new function implemented for improving performance as exercise3_function 
# and test_spacy_exercise3 to test the improvement
def exercise3_function(doc):
    """
    extra the compunds and the element involved in them. check if you have to chang iob tag and NER tag and modifies them
    :param doc: parsed phrase
    :return: the modified doc and a list of iob to change
    """ 
    compound=[None]*doc.__len__()

    for token in doc:
        if (token.ent_type_=='') and (token.dep_=='compound')  and (token.head.ent_type_!=''):
            ent_root=token.head.ent_type_
            #if the token is before a noun in compound modifiy the B tag of the following
            if doc[token.i+1].ent_type_==ent_root and doc[token.i+1].ent_iob_=='B' and ((doc[token.i+1].dep_=='compound' and token.head==doc[token.i+1].head) or (token.head==doc[token.i+1])) : #the tokens is before another token with same NER tag
                compound[token.i+1]='I'
                compound[token.i]='B'
                token.ent_type_=ent_root
            #if the token is after a noun of its compound
            elif doc[token.i-1].ent_type_==ent_root  and doc[token.i-1].dep_=='compound'  and ((doc[token.i-1].dep_=='compound' and token.head==doc[token.i-1].head) or (token.head==doc[token.i-1])):
                compound[token.i]='I'
                token.ent_type_=ent_root
            else:#the token is alone
                compound[token.i]='B'
                token.ent_type_=ent_root
                   

    return (doc, compound)
    

    
def test_spacy_exercise3(path):
    """
    test spacy on conll
    :param path: the path in which ground truth is stored
    :return: a tuple containing pandas table for accuracy for tokens and other metrics for chunks
    """ 
    
    #PREPARING EVALUATION
    
    nlp = spacy.load('en_core_web_sm')
    refs=[]#refs list
    #extract the sentence as token list from corpus
    temporary_corpus=conll.read_corpus_conll(path)
    hyp_test_corpus=[]
    #recreate the corpus phrases and create the ground truth
    for sent in temporary_corpus:
        new_sent=[]
        new_list=[]
        for element in sent:
            word=element[0].split()[0]
            tag=element[0].split()[3]
            new_sent.append(word)
            new_tuple=(word, tag)
            new_list.append(new_tuple)
        if new_sent!=['-DOCSTART-']:
            hyp_test_corpus.append(" ".join(new_sent)) #add a phrase to recreated corpus
            refs.append(new_list) #add a tuple to ground truth
    
    
    
    #creating hyps list
    hyps=[]
    for sent in hyp_test_corpus:
        doc = nlp(sent)
        doc, compound = exercise3_function(doc) #call to the new function
        new_list=[]
        jump_index=-1#if you have to merge some tokens
        for token in doc:
            text=token.text
            #jump if you have to jump some already considered tokens
            if(token.i<=jump_index):
                continue
            #if find a token you have to merge with the followings
            elif token.whitespace_=='' and (doc.__len__()>(token.i+1)):
                if doc[token.i+1].text=='-':
                    jump_index=restructure_tokenisation(token, doc, new_list, compound) 
            #else consider the label and text and add a tuple 
            elif(token.ent_type_!='') and (compound[token.i]==None):#a named entity
                if(token.ent_type_!='DATE') and (token.ent_type_!='PERCENT') and (token.ent_type_!='MONEY') and (token.ent_type_!='TIME') and (token.ent_type_!='QUANTITY') and (token.ent_type_!='ORDINAL') and (token.ent_type_!='CARDINAL') and (token.ent_type_!='PRODUCT'):
                    tag=token.ent_iob_+'-'+remap(token.ent_type_)#note that you have to remap named tags
                    new_tuple=(text, tag)
                    new_list.append(new_tuple)
                else: #some tags are not considered in conll
                    tag='O'
                    new_tuple=(text, tag)
                    new_list.append(new_tuple)
            #else consider the label and text and add a tuple 
            elif(token.ent_type_!='') and (compound[token.i]!=None):#a named entity
                if(token.ent_type_!='DATE') and (token.ent_type_!='PERCENT') and (token.ent_type_!='MONEY') and (token.ent_type_!='TIME') and (token.ent_type_!='QUANTITY') and (token.ent_type_!='ORDINAL') and (token.ent_type_!='CARDINAL') and (token.ent_type_!='PRODUCT'):
                    tag=compound[token.i]+'-'+remap(token.ent_type_)#note that you have to remap named tags
                    new_tuple=(text, tag)
                    new_list.append(new_tuple)
                else: #some tags are not considered in conll
                    tag='O'
                    new_tuple=(text, tag)
                    new_list.append(new_tuple)
            else:#for unnamed entity
                tag=token.ent_iob_
                new_tuple=(text, tag)
                new_list.append(new_tuple)
        hyps.append(new_list)
                
   
 

   
    
   #token eval
    results = evaluate_tokes(refs, hyps)
    pd_tbl_tok = pd.DataFrame().from_dict(results, orient='index')
    pd_tbl_tok.round(decimals=3)

   #chunk eval
    results = conll.evaluate(refs, hyps)
    pd_tbl_chunk = pd.DataFrame().from_dict(results, orient='index')
    pd_tbl_chunk.round(decimals=3)
   
    return((pd_tbl_tok, pd_tbl_chunk))








#OTHER FUNCTIONS USED
#see report for details
def restructure_tokenisation_alternative(token, doc, new_list, problem='-'):   

    jump_index=token.i #index to jump
    text=token.text #text of the token
    tag=""
    if problem=='-':  
        for num, t in enumerate(doc):
            if num<=token.i: #until you are not to the token index continue
                continue
            #concatenate text if you find a mismatch
            elif (num>token.i  and ((t.whitespace_=='' and t.text=='-') or doc[t.i-1].text=='-' )):
                text+=t.text
                jump_index+=1
            else:
                break
 
    if(token.ent_type_!=''): 
        tag=token.ent_iob_+'-'+token.ent_type_
                         
    else:
        tag=token.ent_iob_
    
    new_list.append((text, tag))
        
    
        
    return jump_index              

def check_label_freq(path):

    nlp = spacy.load('en_core_web_sm')
    refs=[]#refs list
    #extract the sentence as token list from corpus
    temporary_corpus=conll.read_corpus_conll(path)
    hyp_test_corpus=[]
    #recreate the corpus phrases and create the ground truth
    for sent in temporary_corpus:
        new_sent=[]
        new_list=[]
        for element in sent:
            word=element[0].split()[0]
            tag=element[0].split()[3]
            new_sent.append(word)
            new_tuple=(word, tag)
            new_list.append(new_tuple)
        if new_sent!=['-DOCSTART-']:
            hyp_test_corpus.append(" ".join(new_sent)) #add a phrase to recreated corpus
            refs.append(new_list) #add a tuple to ground truth
    
    
    
    #creating hyps list
    hyps=[]
    for sent in hyp_test_corpus:
        doc = nlp(sent)
        new_list=[]
        jump_index=-1#if you have to merge some tokens
        for token in doc:
            text=token.text
            #jump if you have to jump some already considered tokens
            if(token.i<=jump_index):
                continue
            #if find a token you have to merge with the followings
            elif token.whitespace_=='' and (doc.__len__()>(token.i+1)):
                if doc[token.i+1].text=='-':
                    jump_index=restructure_tokenisation_alternative(token, doc, new_list) 
            elif(token.ent_type_!=''): 
                tag=token.ent_iob_+'-'+token.ent_type_
                new_tuple=(text, tag)
                new_list.append(new_tuple)
            else:#for unnamed entity
                new_tuple=(text, tag)
                new_list.append(new_tuple)
        hyps.append(new_list)
    
    result_dict={}    
    data=conll.align_hyp(refs, hyps)        
    for sent in data:
        for token in sent:
            #iob and tag for hyp and ref
            hyp_iob, hyp = conll.parse_iob(token[-1])
            ref_iob, ref = conll.parse_iob(token[-2])

            if hyp != None:
               
                if not result_dict.get(hyp) and hyp:
                    result_dict[hyp] = {}
                if not result_dict[hyp].get(ref) and ref:
                    result_dict[hyp][ref]=1
                elif not result_dict[hyp].get('O') and not ref:
                    result_dict[hyp]['O']=1
                else:
                    if ref:
                         result_dict[hyp][ref]+=1
                    else:
                        result_dict[hyp]['O']+=1

  
   
   
    return result_dict  


