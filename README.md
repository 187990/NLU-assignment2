# NLU-assignment2
## Outline:
This repository contains the second assignment of NLU.
the files contained are:
- [assignment.py](./assignment.py)
  it contains the code of the functions implemented for thesecond assignment
- [test.py](./test.py)
  it contains a script for testing the functions
- [conll.py](./conll.py)
  it contains the code of conll as implemented by instructor plus some added comment that I used to understand its implementation
- [result](./result)
  it contains the result of the function on test data

  
## Requirements:
The requirements are:
- Python 3.6 or following
- Spacy v3 (to install spacy run `pip install spacy` , to install english language model  run `python -m spacy download en` )



## Description:
a brief description of the contents of the files

### mandatory part functions and test file
the implemented functions relating the three points of the assignment that are inside [assignment.py](./assignment.py) are
- `test_spacy(path)` take as argument a path pointing to the test.txt file and return two pandas table. the first one contains the accuracy for tokens per class and total not considering difference in IOB plus  the accuracy for tokens per class and total  considering difference in IOB. the second one contains all the statistics requested for the chunk statistics.
- `group_name_entities(phrases)` take as argument a list of strings rapresenting phrases and output a class Statistics that has methods to extract the list of lists, absolute, relative and relative without single token chunk frequencies
- `test_spacy_exercise3(path)` take as argument a path pointing to the test.txt file and return two pandas table. the first one contains the accuracy for tokens per class and total not considering difference in IOB plus  the accuracy for tokens per class and total  considering difference in IOB. the second one contains all the statistics requested for the chunk statistics. this time is done extending labels of the root of a compound to all it's element that have not already tagged with a NER tag and contestually adjust IOB tags. this is done using `exercise3_function(doc)` as subroutine that find the compounds of a phrase and find how to update IOB and NER tag to do as described before.

all theese functions make use of other utility functions described briefly in [report.pdf](./report.pdf) , with also other details of implementation.
it is possible to test the functions with [test.py](./test.py) 

