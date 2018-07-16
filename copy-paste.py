import os
import shutil
source = 'C:/Users/martin/Desktop/TestProject/Students'
target = 'C:/Users/martin/Desktop/TestProject/Python for App/UseCV/training-data'

sourcefile=os.listdir(source)

for file in sourcefile:
    try:
        os.makedirs(target+"/"+file.split(".")[0])
        shutil.copy(source+"/"+file, target+"/"+file.split(".")[0]+"/"+file)
    except IOError as e:
        print("Unable to copy file. %s" % e)
    except:
        print("Unexpected error:", sys.exc_info())
# assert not os.path.isabs(source)
# target = os.path.join(target, os.path.dirname(source))

# # create the folders if not already exists
# os.makedirs(target)

# # adding exception handling
# try:
#     shutil.copy(source, target)
# except IOError as e:
#     print("Unable to copy file. %s" % e)
# except:
#     print("Unexpected error:", sys.exc_info())