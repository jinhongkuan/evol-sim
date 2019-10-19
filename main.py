import random
from copy import copy, deepcopy 
import numpy as np 
import matplotlib.pyplot as plt 
import csv 
import os 
class MDP_Learner:

    @staticmethod
    def _normalize(row):
        return row / sum(row)

    def make_matrix_consistent(self):
        self.matrix = np.concatenate((self.matrix[:,0:1], 
        np.apply_along_axis(MDP_Learner._normalize, 1, self.matrix[:,1:self.state_size+1]),
        np.apply_along_axis(MDP_Learner._normalize, 1, self.matrix[:,self.state_size+1:])), axis=1) 

    def __init__(self, state_size):
        # This matrix is a Nx(2N+1) representation of learning strategy 
        # Each row represents a decision node, expressed in: action, trans. prob | opp. cooperate | opp. defect
        self.matrix = np.concatenate((np.asarray([[0],[1]]), #np.random.randint(low=0, high=2, size=(state_size,1))
        np.random.rand(state_size, state_size),
        np.random.rand(state_size, state_size)), axis=1)
        self.state_size = state_size
        self.make_matrix_consistent()
        self.state_index = 0
        self.prev_state_observation = (0,1)
        self.age = 0
        

    def play(self):
        return self.matrix[self.state_index,0]

    def observe(self, observation):
        if observation is not None:
            if observation == 0:
                trans_prob = self.matrix[self.state_index, 1:self.state_size+1]
            else:
                trans_prob = self.matrix[self.state_index, self.state_size+1:]
            
            next_state = np.random.choice(range(self.state_size),p=trans_prob)
            self.prev_state_observation = (self.state_index, int(next_state+1+observation*self.state_size))
            self.state_index = next_state

    def mutate(self, rate):
        for i in range(self.matrix.shape[0]):
            for j in range(1,self.matrix.shape[1]):
                self.matrix[i,j] *= 1 + np.random.random() * rate - rate/2
        self.make_matrix_consistent() 

           
    def lamarck(self, result, growth):
        prev_state, prev_cell = self.prev_state_observation
        self.matrix[prev_state, prev_cell] *= growth[result]
        self.make_matrix_consistent()


    def learn(self, result):
        if result is None:
            return 
        self.lamarck(result, [1.3,1.1,0.8,0.6]) # DC CC DD CD Prisoner's Dilemma
    

        
class Population:
    def __init__(self, population_size, state_size):
        self.pops = [MDP_Learner(state_size) for i in range(population_size)]

    def aggregate_selection(self, payoff_matrix):
        np.random.shuffle(self.pops)
        scores = [0]*len(self.pops)
        combinations = copy(template_combinations)
        
        for i in range(len(self.pops)):
            for j in range(i+1, len(self.pops)):
                i_play = int(self.pops[i].play())
                j_play = int(self.pops[j].play())
                self.pops[i].observe(j_play)
                self.pops[j].observe(i_play)
                scores[i] += payoff_matrix[i_play,j_play,0]
                scores[j] += payoff_matrix[i_play,j_play,1]
                play = play_to_str[i_play+j_play]
                if play not in combinations:
                    combinations[play] = 0 
                combinations[play] += 1

            
        
        # Kill off half and clone the rest 
        new_pops = sorted([(scores[i], self.pops[i]) for i in range(len(scores))], key=lambda x:x[0], reverse=True) 
        new_pops = new_pops[:len(scores)//2]
        new_pops = [x[1] for x in new_pops]
        for i in range(len(scores)//2):
            new_pops[i].age += 1
            new_pops += [deepcopy(new_pops[i])]
            new_pops[-1].age = 0
            new_pops[-1].mutate(0.15) 

        self.pops = new_pops
        return combinations

os.chdir(os.path.dirname(__file__))

# Change these
name = "chicken2"
p_matrix = [[[2,4],[3,3]],
                [[1,1],[4,2]]]
iterations = 10000
window = 50

play_to_str = {2 : "DD", 1 : "C/D", 0 : "CC"}
template_combinations = {"DD":0, "C/D":0, "CC":0}
Pops = Population(10,2)
name += "_" + str(len(Pops.pops)) + "players"
np.set_printoptions(precision=2, suppress=True)
tally = {"CC":[], "C/D":[], "DD":[]}

for i in range(iterations):
    combinations = (Pops.aggregate_selection(np.asarray([[p_matrix[0][1],p_matrix[0][0]],
                                                        [p_matrix[1][1],p_matrix[1][0]]])))
    for key in combinations:
        tally[key] += [combinations[key]]

for key in tally:
    plt.plot([sum(tally[key][i*window:(i+1)*window])/window for i in range(iterations//window)])
plt.legend(tally.keys())

plt.savefig(fname=name+".png")
plt.show()
with open(name+".csv", "w", newline="") as f:
    f_writer = csv.writer(f)
    f_writer.writerow(["Action", "C|C", "D|C", "C|D", "D|D"])
    for i in range(len(Pops.pops)):
        mat = Pops.pops[i].matrix
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                mat[r,c] = round(mat[r,c]*100)/100
        f_writer.writerows(mat)
        f_writer.writerow([])
        



