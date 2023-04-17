# -*- comyding: utf-8 -*-
"""
Created on Tue Apr 11 15:43:56 2023

@author: Ravindranath Nemani
"""

import numpy as np

import math


from functools import reduce
 
 
def unique(lst):
    return reduce(lambda re, x: re+[x] if x not in re else re, lst, [])
    
    
def number_of_empty_bodies(clauses_data):
    num = clauses_data.count('-;')
    if clauses_data.endswith('-'):
        num = num + 1
    return num


def move_empty_body_clauses_to_the_end(clauses_data, num_empty_bodies, empty_body_processed_count):
    if num_empty_bodies == 1 and clauses_data.endswith('-'):
        return clauses_data
    elif num_empty_bodies > 1:
        for i in range(len(clauses_data)):
            if clauses_data[i] == ';' and clauses_data[i-1] == '-':
                clauses_data_left = clauses_data[:i-3]
                clauses_data_right = clauses_data[i+1:] + clauses_data[i]
                extracted_empty_rule = clauses_data[i-3] + clauses_data[i-2] + clauses_data[i-1]
                clauses_data = clauses_data_left + clauses_data_right + extracted_empty_rule
                empty_body_processed_count = empty_body_processed_count + 1
                if empty_body_processed_count < num_empty_bodies:
                    clauses_data = move_empty_body_clauses_to_the_end(clauses_data, num_empty_bodies, empty_body_processed_count)        
                print(clauses_data)
                return clauses_data
    
    
def num_clauses(rules):
    l = rules.split(";")
    return len(l)
    

def clauses(rules):
    l = rules.split(";")
    return l


def rule_parser(rule):
    l = rule.split("<-")
    head = l[0]
    body = l[1]
    return head, body


def get_clauses_dict(clauses_data):
    clauses_list = clauses(clauses_data)
    clauses_dict = {}    
    for clause in clauses_list:
        head, body = rule_parser(clause)
        clauses_dict[body] = head
    return clauses_dict 


def get_common_heads_dict(clauses_data):
    clauses_dict = get_clauses_dict(clauses_data)
    unique_heads = list(set(clauses_dict.values()))
    common_heads_dict = {}
    for head in unique_heads:
        common_heads_dict[head] = []
        for body in clauses_dict:
            if len(body) > 0 and clauses_dict[body] == head:
                common_heads_dict[head].append(body)
    return common_heads_dict


def empty_bodies_and_their_positions(clauses_data):
    clauses_list = clauses(clauses_data)
    empty_body_position = {}
    for c in range(len(clauses_list)):
        head, body = rule_parser(clauses_list[c])
        if len(body) == 0:
            empty_body_position[head] = c+1
    return empty_body_position


def num_literals_in_body_of_clause(body_list):
    num_literals = []
    for body in body_list:
        body = body.replace('~', '')
        num_literals.append(len(body))
    return num_literals


def num_positive_literals_in_body_of_clause(body_list):
    num_positive_literals = list()
    for item1, item2 in zip(num_literals_in_body_of_clause(body_list), num_negative_literals_in_body_of_clause(body_list)):
        num_positive_literals.append(item1 - item2) 
    return num_positive_literals


def num_negative_literals_in_body_of_clause(body):
    num_negative_literals = []
    for body in body_list:
        num_negative_literals.append(body.count('~'))
    return num_negative_literals


def num_literals_in_bodies_of_clauses(P, N):
    return sum(P) + sum(N)
    

def get_mu(head_list):
    mu = []
    for head in head_list:
        count_of_clauses_with_same_head = 0
        for elem in head_list:
            if elem == head:
                count_of_clauses_with_same_head += 1
        mu.append(count_of_clauses_with_same_head)
    return mu


def maxc_k_mu(k, mu):
    MAXC_k_and_mu = []
    for i in range(len(k)):
        elem = max(k[i], mu[i])
        MAXC_k_and_mu.append(elem)
    return MAXC_k_and_mu


def maxp_k_mu(k, mu):
    max_k = max(k)
    max_mu = max(mu)
    MAXP_k_and_mu = max(max_k, max_mu)
    return MAXP_k_and_mu 


def get_Amin(k, mu):
    MAXP_k_and_mu = maxp_k_mu(k, mu)
    Amin = ((MAXP_k_and_mu - 1) / (MAXP_k_and_mu + 1))
    Amin = Amin*(1 + 0.3)
    return Amin


def get_W(k, mu, beta):
    Amin = get_Amin(k, mu)
    MAXP_k_and_mu = maxp_k_mu(k, mu)
    numerator = math.log(1 + Amin) - math.log(1 - Amin)
    denominator = MAXP_k_and_mu * (Amin - 1) + Amin + 1
    W = (2 * numerator) / (beta * denominator)
    W = abs(W * 2)
    return W


def initialize_and_return_weight_matrices_and_bias_vectors(clauses_data, k, mu, beta, head_list, body_list):
    W = get_W(k, mu, beta)
    input_to_hidden_connections = {}
    input_to_hidden_dict = {}

    num_hidden_layer_neurons = num_clauses(clauses_data)
    
    body_list_copy = body_list
    for body in body_list_copy:
        input_to_hidden_dict[body] = []
        
    hidden_to_output_connections = {}
    
    empty_body_encountered = False
    num_empty_bodies = 0    
    lb = 0
    
    for b in range(len(body_list)):
        body_in_list_form = list(body_list[b])
        if len(body_in_list_form) > 0:
            for i in range(len(body_in_list_form)-1):
                print(i)
                if body_in_list_form[i] == '~':
                    new_elem = body_in_list_form[i] + body_in_list_form[i+1]
                    body_in_list_form.append(new_elem)
                    x1 = body_in_list_form.pop(i)
                    x2 = body_in_list_form.pop(i)
            if empty_body_encountered == True:
                c = b-num_empty_bodies
            else:
                c = b
            for j in range(len(body_in_list_form)):
                if '~' not in body_in_list_form[j]:
                    input_to_hidden_connections[(lb+j+1, c+1)] = W
                    input_to_hidden_dict[body_list[b]].append((lb+j+1, c+1))
                else:
                    input_to_hidden_dict[body_list[b]].append((lb+j+1, c+1))
                    input_to_hidden_connections[(lb+j+1, c+1)] = -W
            print(input_to_hidden_connections)
        else:
            empty_body_encountered = True
            num_empty_bodies += 1
        lb = lb + len(body_list[b]) - body_list[b].count('~')

    for key in input_to_hidden_dict.keys():
        input_to_hidden_dict[key] = list(set(input_to_hidden_dict[key]))

    common_heads_dict = get_common_heads_dict(clauses_data)
    unique_head_list = list(common_heads_dict.keys())
    output_neurons_connected = []
    hidden_neurons_connected = []
    for h in range(len(common_heads_dict)):
        for b in common_heads_dict[unique_head_list[h]]:
            if len(b) > 0:
                r = input_to_hidden_dict[b][0][1]
                t = tuple([r, h+1])
                hidden_to_output_connections[(r, h+1)] = W
                hidden_neurons_connected.append(r)
                output_neurons_connected.append(h+1)
    hidden_neurons_connected = list(set(hidden_neurons_connected))

    #collect all rules which are facts
    output_neurons_not_connected = empty_bodies_and_their_positions(clauses_data)
    output_neurons = list(set(list(output_neurons_not_connected.keys()) + list(common_heads_dict.keys())))

    for key in output_neurons_not_connected.keys():
        val = output_neurons_not_connected[key]
        for num in range(len(output_neurons)):
            if output_neurons[num] == key:
                hidden_to_output_connections[(val, num+1)] = W

    num_input_layer_neurons = lb
    
    num_output_layer_neurons = len(output_neurons)

    #Create all other connections to make it a Fully Connected Network
    #and initialize all such remaining connections to 0 weights
    
    #step1 - input to hidden
    global_input_to_hidden_connections = {}
    for n in range(1, num_input_layer_neurons+1):
        for o in range(1, num_hidden_layer_neurons+1):
            t = tuple([n, o])
            global_input_to_hidden_connections[t] = 0
    
    for conn1 in global_input_to_hidden_connections:
        if conn1 in input_to_hidden_connections:
            global_input_to_hidden_connections[conn1] = input_to_hidden_connections[conn1]
        
    #step2 - hidden to output
    global_hidden_to_output_connections = {}
    for p in range(1, num_hidden_layer_neurons+1):
        for q in range(1, num_output_layer_neurons+1):
            t = tuple([p, q])
            global_hidden_to_output_connections[t] = 0
    
    for conn2 in global_hidden_to_output_connections:
        if conn2 in hidden_to_output_connections:
            global_hidden_to_output_connections[conn2] = hidden_to_output_connections[conn2]
                        
    Weight_Input2Hidden = np.zeros([num_input_layer_neurons, num_hidden_layer_neurons], dtype = float)

    #convert to weight matrices
    for key in global_input_to_hidden_connections.keys():
        Weight_Input2Hidden[key[0]-1, key[1]-1] = global_input_to_hidden_connections[key]

    Weight_Hidden2Output = np.zeros([num_hidden_layer_neurons, num_output_layer_neurons], dtype = float)

    for key in global_hidden_to_output_connections.keys():
        Weight_Hidden2Output[key[0]-1, key[1]-1] = global_hidden_to_output_connections[key]

    #create and initialize bias vectors
    Bias_Hidden = []
        
    for bh in range(num_hidden_layer_neurons):
        print(bh)
        Bias_Hidden.append((W/2)*(1 + get_Amin(k, mu)) * (num_literals_in_body_of_clause(body_list)[bh] - 1))
    Bias_Hidden = np.array(Bias_Hidden)
        
    Bias_Output = []

    for bo in range(num_output_layer_neurons):
        Bias_Output.append((W/2)*(1 + get_Amin(k, mu)) * (1 - get_mu(head_list)[bo]))
    Bias_Output = np.array(Bias_Output)
        
    return Weight_Input2Hidden, Weight_Hidden2Output, Bias_Hidden, Bias_Output


def get_activation_g(x):
    return x


def get_activation_h(beta, x):
    return (2 / (1 + math.exp(-beta*x))) - 1


def get_valuation_act(beta, k, mu, x):
    if get_activation_h(beta, x) > get_Amin(k, mu):
        return 1
    else:
        return -1


def name(body_list, head_list):
    name_inputs = ''.join(body_list)
    name_inputs  = name_inputs.replace('~', '')
    name_inputs = list(name_inputs)
    name_outputs = ''.join(head_list)
    name_outputs  = name_outputs.replace('~', '')
    name_outputs = list(name_outputs)
    name_outputs = list(set(name_outputs))
    return name_inputs, name_outputs
                  

def check(name_inputs, name_outputs, input_vector_list, val_A2_list, k, mu, beta):
    pairs = []
    val_act_input = []
    val_act_output = []
    for o1 in range(len(name_outputs)):
        for i1 in range(len(name_inputs)):
            if name_outputs[o1] == name_inputs[i1]:
                pairs.append((i1, o1))
    
    for t in pairs:
        i2 = t[0]
        o2 = t[1]
        val_act_input.append(get_valuation_act(beta, k, mu, input_vector_list[i2]))
        val_act_output.append(get_valuation_act(beta, k, mu, val_A2_list[o2]))
    
    flag = True
    for s in range(len(val_act_input)):
        if val_act_input[s] != val_act_output[s]:
            flag = False

    return flag


def get_NN_output(body_list, head_list, beta, k, mu, Weight_Input2Hidden, Weight_Hidden2Output, Bias_Hidden, Bias_Output):
    
    input_vector_list = [-1]*num_literals_in_bodies_of_clauses(num_positive_literals_in_body_of_clause(body_list), num_negative_literals_in_body_of_clause(body_list))
    input_vector = np.array(input_vector_list)
    name_inputs, name_outputs = name(body_list, head_list)
    print(name_inputs, name_outputs)
    input("names")
    
    iterations = 100
    for itr in range(iterations):    
        
        # feedforward propagation on hidden layer
        Z1 = np.add(np.dot(Weight_Input2Hidden.T, input_vector), Bias_Hidden)
        A1 = [get_activation_h(beta, z) for z in Z1]
        A1 = np.array(A1)
        # feedforward propagation on output layer
        Z2 = np.add(np.dot(Weight_Hidden2Output.T, A1), Bias_Output)
        A2_list = [get_activation_h(beta, z) for z in Z2]
        val_A2_list = [get_valuation_act(beta, k, mu, x) for x in A2_list]

        for o in range(len(name_outputs)):
            for i in range(len(name_inputs)):
                if name_outputs[o] == name_inputs[i]:
                    input_vector_list[i] = get_valuation_act(beta, k, mu, A2_list[o])
                    
                    flag = check(name_inputs, name_outputs, input_vector_list, val_A2_list, k, mu, beta)
                    if flag == True:
                        print(input_vector_list)
                        input("input list")
                        return itr+1, val_A2_list
                                                

#############################################################################


#main

beta = -3.0

#Input a general Logic Program

clauses_data = 'A<-BC~D;A<-EF;B<-'

#clauses_data = 'A<-BC~D;B<-;A<-EF'

#clauses_data = 'A<-BC~D;B<-;C<-;A<-EF'

#clauses_data = 'A<-BC~D;B<-;A<-EF;G<-;F<-E'

num_empty_bodies = number_of_empty_bodies(clauses_data)

empty_body_processed_count = 0

clauses_data = move_empty_body_clauses_to_the_end(clauses_data, num_empty_bodies, empty_body_processed_count)

clauses_list = clauses(clauses_data)
print(clauses_list)
input("clauses list after all empty bodies moved to right")

head_list = []
body_list = []

for clause in clauses_list:
    head, body = rule_parser(clause)
    head_list.append(head)
    body_list.append(body)

head_list = unique(head_list)

MAXC_k_and_mu = maxc_k_mu(num_literals_in_body_of_clause(body_list), get_mu(body_list))

MAXP_k_and_mu = maxp_k_mu(num_literals_in_body_of_clause(body_list), get_mu(body_list))

print("number of clauses is : ", num_clauses(clauses_data))

print("head_list is : ", head_list)

print("body_list is : ", body_list)

print("number of literals is : ", num_literals_in_body_of_clause(body_list))

print("number of positive literals is : ", num_positive_literals_in_body_of_clause(body_list))
    
print("number of negative literals is : ", num_negative_literals_in_body_of_clause(body_list))

print("number of literals in bodies of all clauses us : ", num_literals_in_bodies_of_clauses(num_positive_literals_in_body_of_clause(body_list), num_negative_literals_in_body_of_clause(body_list)))

print("counts are : ", get_mu(head_list))
    
print("MAXC_k_and_mu is : ", MAXC_k_and_mu)

print("MAXP_k_and_mu is : ", MAXP_k_and_mu)

print("Amin is : ", get_Amin(num_literals_in_body_of_clause(body_list), get_mu(body_list)))

print("W is : ", get_W(num_literals_in_body_of_clause(body_list), get_mu(body_list), beta))

print("common heads dict : ", get_common_heads_dict(clauses_data))
input("common")

Weight_Input2Hidden, Weight_Hidden2Output, Bias_Hidden, Bias_Output = initialize_and_return_weight_matrices_and_bias_vectors(clauses_data, num_literals_in_body_of_clause(body_list), get_mu(body_list), beta, head_list, body_list)

print(Weight_Input2Hidden)
input("Weight_Input2Hidden")

print(Weight_Hidden2Output)
input("Weight_Hidden2Output")

print(Bias_Hidden)
input("Bias_Hidden")

print(Bias_Output)
input("Bias_Output")

num_iter, val_A2_list = get_NN_output(body_list, head_list, beta, num_literals_in_body_of_clause(body_list), get_mu(body_list), Weight_Input2Hidden, Weight_Hidden2Output, Bias_Hidden, Bias_Output)

print(num_iter)
input("number of itertions required to converge")

print(val_A2_list)
print("stable model obtained from logic program P using the Neural Network N")
