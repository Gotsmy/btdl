#!/home/users/mgotsmy/.conda/envs/2302test3.10/bin/python

import os
import itertools as it
import copy
import numpy as np

objectives = ["N[4,end,end]/V[end,end]*1e+0","N[4,end,end]/t_end*1e+1"]
t_maxes    = np.linspace(10,70,21) # 19
t_maxes    = np.array([10., 11.5, 13., 14.5, 16.,17.5, 19.,20.5, 22.,23.5, 25., 28., 31., 34., 37., 40., 43., 46.,
       49., 52., 55., 58., 61., 64., 67., 70.])
qXes       = np.linspace(0.0053424912486709+1e-6,0.1894000099986384-1e-6,11) # 11

with open("template_001.jl", "r") as file:
    text = file.read()

nr = 0
bash_prefix  = "/bin/bash"
julia_prefix = "/home/users/mgotsmy/julia/julia-1.8.5/bin/julia -Cnative -J/home/users/mgotsmy/julia/julia-1.8.5/lib/julia/sys.so -g1 --color=yes"
commands = []

print(f"{'file':7} {'objective':25} {'length':5} {'qX_min':7}")

for t, obj, qX in it.product(t_maxes,objectives,qXes):
    nr += 1
    tmp = text.replace("##OBJECTIVE##",obj)
    tmp = tmp.replace("##TMAX##",str(t))
    tmp = tmp.replace("##QXMIN##",str(qX))
    
    fname = f"{nr:03d}.jl"
    print(f"{fname:7} {obj:25} {t:5.0f} {qX:7.2f}")
    commands.append(f"{julia_prefix} {fname} \n")
    with open(fname,"w") as file:
        file.write(tmp)
    # break
    
nrH = 40
commands = np.array(commands)
np.random.shuffle(commands)
split_commands = np.array_split(commands, nrH)
print("Create Helper & Run Files",[len(i) for i in split_commands])
run_file = "#!"
run_file += bash_prefix + " \n"
for i, tmp_commands in enumerate(split_commands):
    helper = '#!'
    helper += bash_prefix + "\n"
    for command in tmp_commands:
        helper += command
    helper_name = f"helper_{i+1:d}.sh"
    with open(helper_name,"w") as file:
        file.write(helper)
    run_file += f"{bash_prefix} {helper_name} &\n"
    


with open("run.sh","w") as file:
        file.write(run_file)
os.system("chmod +x run.sh")
print("DONE")
