#coding=utf-8  
  
import os
file_path = os.path.abspath('.')
count = 501
for file in os.listdir(file_path):
    os.rename(os.path.join(file_path,file),os.path.join(file_path,str(count)+".jpg"))
    count+=1