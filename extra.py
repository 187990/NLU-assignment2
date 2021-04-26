#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 22:34:54 2021

@author: dimitri
"""
import assignment


path="./CoNLL(2003)/test.txt"
result=assignment.check_label_freq(path)

print(result.items())