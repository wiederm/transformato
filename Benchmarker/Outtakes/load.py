from urllib.request import urlretrieve
import os
from zipfile import ZipFile


def loader(input_location):
    try:
        os.makedirs(input_location)
        print ('Directory ' , input_location ,  ' Created')
    except FileExistsError:
        print('Directory ', input_location , ' already exists')
        pass
    
    print ("Downloading benchmark molecules...")
    url = 'https://github.com/wiederm/transformato-systems/archive/master.zip'
    dest_file = os.path.join(input_location, 'transformato-systems-master.zip')
    urlretrieve(url, dest_file)
    
    print ("Extracting...")
    with ZipFile(dest_file, 'r') as zipObj:
        zipObj.extractall(input_location)

    try:
        os.makedirs(input_location + '/transformato-systems-master/output')
    except FileExistsError:
        pass
    
    print ("Cleaning...")
    os.remove(dest_file)
    print ("Finished!")