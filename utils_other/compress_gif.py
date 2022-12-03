
import os


'''
sudo apt-get install gifsicle
pip install pygifsicle

https://zhuanlan.zhihu.com/p/46259590
'''


from pygifsicle import optimize
optimize(os.path.join('./results/tmp', 'pw-all.gif'))
