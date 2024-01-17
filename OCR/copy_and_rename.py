import os
import shutil

dir="/work/xiaolu/alldata"
l=[x[0]+"/"+f for x in os.walk(dir) for f in x[2] if f.endswith(".jpg")]
print(l)

for i in range(len(l)):
    if i %100==0:
        print(i)
    shutil.copyfile(l[i],"/work/xiaolu/number_data/"+str(i)+".jpg")

