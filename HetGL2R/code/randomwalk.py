import collections
import random


#Get the node to start wandering
od = []
link = []
with open(r'D:\Project\PythonProject\HetGL2R\data\SJ\od.txt') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        od.append(line.lower())

#print(od)


def get_node_info(file,column):
    with open(file, 'r') as file:
        info = [line.split()[column].lower() for line in file.readlines() if line.split()]
    return info

#from_node = set(get_node_info(r'D:\Project\PythonProject\HetGL2R\data\SY\graph_entity.txt', 0))
from_node = set(get_node_info(r'D:\Project\PythonProject\HetGL2R\data\SJ\graph_entity.txt', 0))
#to_node = set(get_node_info(r'D:\Project\PythonProject\HetGL2R\data\methodOne\graph_entity_01.txt', 1))
to_node = set(get_node_info(r'D:\Project\PythonProject\HetGL2R\data\SJ\graph_entity.txt', 1))
from_node_feature = set(get_node_info(r'D:\Project\PythonProject\HetGL2R\data\SJ\graph_entity_feature.txt', 0))
to_node_feature = set(get_node_info(r'D:\Project\PythonProject\HetGL2R\data\SJ\graph_entity_feature.txt', 1))
#from_node_feature = set(get_node_info(r'../data/methodTwo/graph_entity_feature_02.txt', 0))
#to_node_feature = set(get_node_info(r'../data/methodTwo/graph_entity_feature_02.txt', 1))
#from_node_feature = set(get_node_info(r'D:\Project\PythonProject\HetGL2R\data\methodThree\graph_entity_feature_03.txt', 0))
#to_node_feature = set(get_node_info(r'D:\Project\PythonProject\HetGL2R\data\methodThree\graph_entity_feature_03.txt', 1))


#Get entity link relationships
from_to = collections.defaultdict(list)
from_to_weight = collections.defaultdict(list)
to_from = collections.defaultdict(list)
to_from_weight = collections.defaultdict(list)
with open(r'D:\Project\PythonProject\HetGL2R\data\SJ\graph_entity.txt', 'r') as file:
    for line in file.readlines():
        conn = line.strip()
        conn = conn.split('\t')
        if conn[1].lower() in to_node:
            from_to[conn[0].lower().strip()].append(conn[1].lower())
            from_to_weight[conn[0].lower().strip()].append(float(conn[2])) #1
            to_from[conn[1].lower()].append(conn[0].lower().strip())
            to_from_weight[conn[1].lower()].append(float(conn[2])) #1

#print(from_to_weight)
#print(to_from_weight)

with open(r'D:\Project\PythonProject\HetGL2R\data\SJ\link_forward_link.txt', 'r') as f:
    # 读取所有行
    lines = f.readlines()
    # 分割每一行，得到二维数组
    data = [line.strip().split() for line in lines]
#print(data)
for i in range(len(data)):
    for j in range(len(data[i])-1,0,-1):
        weight = 2
        for k in range(j-1,-1,-1):
            if data[i][j] in from_to.keys():
                from_to[data[i][j].lower()].append(data[i][k].lower())
                from_to_weight[data[i][j].lower()].append(weight**(-(j-k-1)))
                to_from[data[i][k].lower()].append(data[i][j].lower())
                to_from_weight[data[i][k].lower()].append(weight**(-(j-k-1)))
            else:
                from_to[data[i][j].lower()] = [data[i][k].lower()]
                from_to_weight[data[i][j].lower()] = [weight ** (-(j-k-1))]
                to_from[data[i][k].lower()].append(data[i][j].lower())
                to_from_weight[data[i][k].lower()].append(weight ** (-(j - k - 1)))
#print(from_to)
#print(from_to_weight)

#Obtain entity and its feature link relationships
from_feature_to = collections.defaultdict(list)
from_feature_to_weight = collections.defaultdict(list)
to_feature_from = collections.defaultdict(list)
to_feature_from_weight = collections.defaultdict(list)
with open(r'D:\Project\PythonProject\HetGL2R\data\SJ\graph_entity_feature.txt', 'r') as file:
    for line in file.readlines():
        conn = line.strip()
        conn = conn.split('\t')
        if conn[1].lower() in to_node_feature:
            from_feature_to[conn[0].lower().strip()].append(conn[1].lower())
            from_feature_to_weight[conn[0].lower().strip()].append(float(conn[2]))
            to_feature_from[conn[1].lower()].append(conn[0].lower().strip())
            to_feature_from_weight[conn[1].lower()].append(float(conn[2]))

#print(random.choices(from_feature_to['lj172'], weights=from_feature_to_weight['lj172']))
#print(random.choices(from_to['od001']))



#Get node attribute
node_attr = {}
with open(r'D:\Project\PythonProject\HetGL2R\data\SJ\node_attribute.txt', 'r') as file:
    for line in file.readlines():
        key, value = line.strip().lower().split('\t')
        node_attr[key] = int(value)
#print(node_attr)


#Random walk starts from the OD node
attributes = [0,1]
#attributes = [0,0]
def random_walk_od(node,wandering_sequence):
    if(len(wandering_sequence)==20): #30
        return wandering_sequence
    else:
        current_attr = node_attr[node]
        if(current_attr == 0):
            next_node_attr = random.choices(attributes,weights=[0.6,0.4])[0]#weights=[0.6,0.4]
            if(next_node_attr == current_attr):
                if(node in from_to.keys()):
                    #next_node = random.choice(from_to[node])
                    next_node = random.choices(from_to[node], weights=from_to_weight[node])
                    next_node = ''.join(next_node)
                    wandering_sequence.append(next_node)
                    return random_walk_od(next_node,wandering_sequence)
                else:
                    #next_node = random.choice(to_from[node])
                    print(node)
                    next_node = random.choices(to_from[node], weights=to_from_weight[node])
                    next_node = ''.join(next_node)
                    wandering_sequence.append(next_node)
                    return random_walk_od(next_node, wandering_sequence)
            else:
                if (node in from_feature_to.keys()):
                    if(sum(from_feature_to_weight[node])>0):
                        next_node = random.choices(from_feature_to[node],weights=from_feature_to_weight[node])
                        next_node = ''.join(next_node)
                    else:
                        next_node = random.choice(from_feature_to[node])
                    wandering_sequence.append(next_node)
                    return random_walk_od(next_node, wandering_sequence)
                else:
                    if (sum(to_feature_from_weight[node]) > 0):
                        next_node = random.choices(to_feature_from[node],weights=to_feature_from_weight[node])
                        next_node = ''.join(next_node)
                    else:
                        next_node = random.choice(to_feature_from[node])
                    wandering_sequence.append(next_node)
                    return random_walk_od(next_node, wandering_sequence)
        else:
            if (node in from_feature_to.keys()):
                if (sum(from_feature_to_weight[node]) > 0):
                    next_node = random.choices(from_feature_to[node], weights=from_feature_to_weight[node])
                    next_node = ''.join(next_node)
                else:
                    next_node = random.choice(from_feature_to[node])
                wandering_sequence.append(next_node)
                return random_walk_od(next_node, wandering_sequence)
            else:
                if (sum(to_feature_from_weight[node]) > 0):
                    next_node = random.choices(to_feature_from[node], weights=to_feature_from_weight[node])
                    next_node = ''.join(next_node)
                else:
                    next_node = random.choice(to_feature_from[node])
                wandering_sequence.append(next_node)
                return random_walk_od(next_node, wandering_sequence)

total_walk = []
for o in od:
    for i in range(20):#10
        wandering_sequence = [o]
        wander = random_walk_od(o,wandering_sequence)
        total_walk.append(wander)

with open(r'D:\Project\PythonProject\HetGL2R\code\sequence_sj.txt','w') as f:
    for line in total_walk:
        link_numbers = set()
        for item in line:
            if item.startswith('link'):
                number = item[len('link'):]
                link_numbers.add(number)
        if len(link_numbers) >= 5:#4
            for node in line:
                f.write(node +' ')
            f.write('\n')
        else:
            continue
