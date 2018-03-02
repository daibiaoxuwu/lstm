
from __future__ import print_function
##!!changed output to 'parseword11'
import os

# Text file containing words for training
training_path = 'D:\djl\p2r'


def list_tags():
    mlist=os.listdir(training_path)
    print(len(mlist))
    with open('D:\djl\list','w') as g:
        for inputfile in mlist:#[:7000]:
            if inputfile[-4:]=='.txt':
                with open(os.path.join(training_path,inputfile)) as f:
                    ss=f.readlines()
                    g.write(str(len(ss)))
list_tags()
print('end')
