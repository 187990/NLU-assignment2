#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 01:56:57 2021

@author: dimitri
"""

import assignment

#TEST EXERCISE 1
path="./CoNLL(2003)/test.txt"
result=assignment.test_spacy(path)
print("TEST EXERCISE 1 \n\n")
print("chunk statistics \n")
print(result[0])
print("\n token statistics \n")
print(result[1])


#TEST EXERCISE 2
phrases=[]
print("\n\n TEST EXERCISE 2 \n\n")
print("test with a single phrase \n")
phrases.append("Apple's Steve Jobs died in 2011 in Palo Alto, California.")
result=assignment.group_name_entities(phrases)
print("our list of lists \n")
print(result.get_list_of_lists())
print("\n absolute freq of chunks \n")
print(result.get_freq_groups())
print("\n relative freq of chunks \n")
print(result.get_rel_freq_groups())
print("\n relative freq of chunks not counting single word chunks \n")
print(result.get_rel_freq_groups_wouttokens())
print("\n test with oura conll2003 corpus\n")
phrases=assignment.obtain_phrases("./CoNLL(2003)/test.txt")
result=assignment.group_name_entities(phrases)
print("our list of lists \n")
print(result.get_list_of_lists())
print("\n absolute freq of chunks \n")
print(result.get_freq_groups())
print("\n relative freq of chunks \n")
print(result.get_rel_freq_groups())
print("\n relative freq of chunks not counting single word chunks \n")
print(result.get_rel_freq_groups_wouttokens())

#TEST EXERCISE 3
path="./CoNLL(2003)/test.txt"
result=assignment.test_spacy_exercise3(path)
print("\n\n TEST EXERCISE 3 \n\n")
print("chunk statistics \n")
print(result[0])
print("\n token statistics \n")
print(result[1])