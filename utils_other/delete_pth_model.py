import cmd, sys
from turtle import *
import os 
import shutil
import asyncio

#需要删除的文件所属的后缀
needDelFilesuffixs = ['.pth']

#需要删除的model_num范围
number_start = 0
number_end = 869000
#需要排除的文件夹，不去遍历的文件夹及其子集
excludeDirNames = ['assets']

#查看两个列表的元素是否有交集
def inter(a,b):
    return list(set(a)&set(b))

# 批量删除指定的文件
async def delect_allocate_file(file_dir):   

    dle_number = 0
    # 获取这个路径下所有的文件和文件夹
    for root, dirs, files in os.walk(file_dir, topdown=True):
        isExclude = False
        for excludedir in excludeDirNames:
            if not (excludedir in root):
                isExclude = True
        
        if isExclude:
            for filename in files:
                file_name_only, file_extension = os.path.splitext(filename)
                if(len(needDelFilesuffixs)>0):
                    for del_suffix in needDelFilesuffixs:
                        if(file_extension == del_suffix):
                            fileFullName = os.path.join(root, filename)
                            number = int(filename.split('_')[-2])
                            if number in range(number_start, number_end):
                                os.remove(fileFullName)  
                                print("删除 %s" % (fileFullName))
                                #删除这个后缀的文件
                                dle_number+=1
                        # if(file_extension == del_suffix):
                        #     fileFullName = os.path.join(root, filename)
                        #     os.remove(fileFullName)  
                        #     print("删除 %s" % (fileFullName))
                        #     #删除这个后缀的文件
                        #     dle_number+=1
                #print(os.path.join(root, filename))


    print("总共删除了 %s 个文件 "%(dle_number)) 

async def files_pos():  
   await delect_allocate_file(r'/home/zju/github/zdk/recognition-rl/results/IndependentSACsupervise-EnvInteractiveSingleAgent/2022-11-26-12:32:44----Nothing--supervise-hrz30-act10/saved_models_method')


asyncio.run(files_pos())
