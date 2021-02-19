# coding=UTF-8

import  os

file_path = os.path.abspath('.')

file_list =  []
for root, dirs, files in os.walk(file_path+'/Annotations'):
    file_list.append(files)
print(file_list)
with open('ImageSets/Main/train.txt','w') as file:
    for i in file_list[0]:
        i=i.replace('.xml','')
        file.write(i+'\n')