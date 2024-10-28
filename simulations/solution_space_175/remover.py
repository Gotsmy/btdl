#!/home/users/mgotsmy/.conda/envs/2302test3.10/bin/python

import os


cwd = os.getcwd()
valid_names = []
valid_dirs = []
for i in os.walk(cwd):
    for name in i[2]:
        if ".jl" in name and not "template" in name and not "xx" in name and not "yy" in name and "variables" not in name:
            valid_names.append(f"rm {os.path.join(cwd,name)}")
            valid_dirs.append(f"rm -r {os.path.join(cwd,name[:-3])}/")
#print(valid_names)
#print(valid_dirs)
for i in valid_names:
    os.system(i)
for j in valid_dirs:
    os.system(j)
