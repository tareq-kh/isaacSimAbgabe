import math
import numpy as np

#
def get_lines_positions(id):
    f = open("/home/windos/.local/share/ov/pkg/IsaacSim/OmniIsaacGymEnvsNew/omniisaacgymenvs/tasks/utils/roblearn/karten/0-"+str(id)+".world", "r")
    arr_x1y1 = []
    arr_x2y2 = []


    for line in f:
        x1y1 = []
        x2y2 = []
        if line[0] == 'l':
            words = line.split()
            x1y1.append(words[1])
            x1y1.append(words[2])
            x2y2.append(words[3])
            x2y2.append(words[4])
            arr_x1y1.append(x1y1)
            arr_x2y2.append(x2y2)
    np1 = np.array(arr_x1y1,dtype=float)
    np2 = np.array(arr_x2y2, dtype= float)


    #print(arr_x1y1)
    #print(np1)
    f.close()
    return (np1, np2)


def get_line_length(x1y1, x2y2):
    return round(math.sqrt((x2y2[0] - x1y1[0]) ** 2 + (x2y2[1] - x1y1[1]) ** 2), 2)


def get_line_pos(x1y1, x2y2):
    return [(x1y1[0] + x2y2[0]) / 2.0, (x1y1[1] + x2y2[1]) / 2.0]


# print(get_line_length([-2,1],[4,-3]))
# print(get_line_pos([2,3],[8,6]))

def get_lines_poss_lengths(arr_x1y1, arr_x2y2):
    lenths=[]
    poss= []
    for x1y1, x2y2 in zip(arr_x1y1, arr_x2y2):

        #get_line_pos(x1y1,x2y2)
        lenths.append( get_line_length(x1y1,x2y2))
        poss.append((get_line_pos(x1y1,x2y2)))
    print(poss)
    print(lenths)
    return (poss,lenths)




#arrs = get_lines_positions()
#poss = get
#poss_lenths = get_lines_poss_lengths(arrs[0], arrs[1])
#print(poss_lenths)
