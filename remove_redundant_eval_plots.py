import os
folders = ['20', '30', '40', '50', '60', '70']
num_agents = 6
try:
    for name in folders:
        path = os.path.join(os.getcwd(), 'eval_plots_fst',name)
        files = os.listdir(path)
        for f in files:
            if '99000' in f or 'perform' in f:
                print(f)
            else:
                os.remove(os.path.join(path,f))
except:
    print("fst files absent")

try:
    path = os.path.join(os.getcwd(), 'eval_plots')
    files = os.listdir(path)
    for f in files:
        if '99000' in f or 'perform' in f:
            print(f)
        else:
            os.remove(os.path.join(path,f))
except :
    print("eval_plots don't exist")

try:
    path = os.path.join(os.getcwd(), 'plots')
    for i in range(1, num_agents+1):
        l_path = os.path.join(path, "Agent_" + str(i))
        files = os.listdir(os.path.join(l_path))
        ac_files = [f for f in files if 'actor.png' in f ]
        cric_files = [f for f in files if 'critic.png' in f ]
        ac_files.sort(key=len)
        cric_files.sort(key = len)

        int_list = []
        rem_list = []
        for f in ac_files:
            if len(f) < len(ac_files[-1]):
                os.remove(os.path.join(l_path, f))
            else:
                int_list.append(int(f[:len(ac_files[-1])-len('actor.png')]))
                rem_list.append(f)
        int_list.sort()
        for f in rem_list:
            if not str(int_list[-1]) in f:
                os.remove(os.path.join(l_path,f))

        int_list = []
        rem_list = []
        for f in cric_files:
            if len(f) < len(cric_files[-1]):
                os.remove(os.path.join(l_path, f))
            else:
                int_list.append(int(f[:len(cric_files[-1]) - (len('critic.png'))]))
                rem_list.append(f)
        int_list.sort()
        for f in rem_list:
            if not str(int_list[-1]) in f:
                os.remove(os.path.join(l_path, f))
except :
    print("plots don't exist")
