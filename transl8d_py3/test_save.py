import datetime
import shutil
import os


obj_path='./assets/cust.obj'
with open(obj_path, 'w') as f:
  f.write('txt\n')
timestamp   = datetime.datetime.now().strftime('__%Y_%B_%d____%H:%M_%p__')
dated_obj_dir='/home/nathanbendich/MultiGarmentNetwork/assets/MGN_obj{}/'.format(timestamp)
os.makedirs(dated_obj_dir)
shutil.copy2(obj_path, dated_obj_dir + 'cust.obj')
