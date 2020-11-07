import pathlib
data_dir = pathlib.Path('imgs/full')
import imghdr
import os
for file in list(data_dir.glob('*.jpg')):
    header = imghdr.what(file)
    if header != 'jpeg':
        os.remove(file)
        print(str(file)+' is not a valid jpeg. Deleting...')

print('We got '+str(len(list(data_dir.glob('*.jpg'))))+' files.')
