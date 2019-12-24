import random
from copy import copy, deepcopy 
import numpy as np 
import matplotlib.pyplot as plt 
import xlsxwriter 
import os 
import time
from collections import Counter
class MDP_Learner:

	def get_submatrix(self, key, obs):
		return self.matrix[key][:, 2+obs*self.state_size: 2+(obs+1)*self.state_size]

	def compute_similarity(self, other):
		if self.state_size != other.state_size:
			return 0
		behavioral_matrix_i = np.zeros((2, 2**(self.state_size+1) - 1))
		behavioral_matrix_j = np.zeros((2, 2**(self.state_size+1) - 1))
		current_step_i = [copy(self.matrix[:,1])]
		current_step_j = [copy(other.matrix[:,1])]
		for i in range(self.state_size+1):
			current_index = 2**i-1
			for j in range(len(current_step_i)):
				behavioral_matrix_i[:,current_index+j] = current_step_i[j]
				behavioral_matrix_j[:,current_index+j] = current_step_j[j]
			next_step_i = []
			next_step_j = []
			for j in range(len(current_step_i)):
				next_step_i += [np.transpose(np.matmul(np.transpose(current_step_i[j]),self.matrix[:,2:2+self.state_size]))]
				next_step_i += [np.transpose(np.matmul(np.transpose(current_step_i[j]),self.matrix[:,2+self.state_size:]))]
				next_step_j += [np.transpose(np.matmul(np.transpose(current_step_j[j]),other.matrix[:,2:2+self.state_size]))]
				next_step_j += [np.transpose(np.matmul(np.transpose(current_step_j[j]),other.matrix[:,2+self.state_size:]))]
			current_step_i = next_step_i
			current_step_j = next_step_j
		mse = ((behavioral_matrix_i - behavioral_matrix_j)**2).mean(axis=None) 
		return 1-mse 

	@staticmethod
	def _normalize(row):

		row = abs(row)
		return row/sum(row)

	def make_matrix_consistent(self):
		for key in type_space:
			self.matrix[key] = np.concatenate((self.matrix[key][:,0:1], 
			abs(self.matrix[key][:,1:2]/sum(self.matrix[key][:,1:2])),
			np.apply_along_axis(MDP_Learner._normalize, 1, self.matrix[key][:,2:self.state_size+2]),
			np.apply_along_axis(MDP_Learner._normalize, 1, self.matrix[key][:,self.state_size+2:])), axis=1) 

	def get_type(self):
		global type_threshold 
		return max([i for i in range(len(type_threshold)) if type_threshold[i] <= self.history])

	def __init__(self, state_size, set_strategy=None):
		# This matrix is an array of Nx(2N+1) representation of learning strategy 
		# Each row represents a decision node, expressed in: action, trans. prob | opp. cooperate | opp. defect
		self.history = 1 
		self.matrix = {} 
		self.action_matrix = {}
		for key in type_space:
			if set_strategy is None:
				self.matrix[key] = np.concatenate(([[i%2] for i in range(state_size)],
				[[1/state_size] for i in range(state_size)],
				np.random.rand(state_size, state_size),
				np.random.rand(state_size, state_size)), axis=1)
			else:
				self.matrix[key] = set_strategy

			self.action_matrix[key] = np.asarray([[1 if i == self.matrix[key][n,0] else 0 for i in range(2)] for n in range(state_size)])

		


		
		self.state_size = state_size
		self.make_matrix_consistent()
		self.state_index = {}
		for key in type_space:
			self.state_index[key] = np.random.choice(a=self.matrix[key].shape[0], p=self.matrix[key][:,1])
		self.prev_state_observation = (0,1)
		self.age = 0
		
	def refresh(self, key):
		self.state_index[key] = np.random.choice(a=self.matrix[key].shape[0], p=self.matrix[key][:,1])

	def play(self, key):
		return self.matrix[key][self.state_index[key],0]

	def observe(self, observation, key):
		if observation is not None:
			if observation == 0:
				trans_prob = self.matrix[key][self.state_index[key], 2:self.state_size+2]
			else:
				trans_prob = self.matrix[key][self.state_index[key], self.state_size+2:]
	
			next_state = np.random.choice(range(self.state_size),p=trans_prob)
			self.prev_state_observation = (self.state_index[key], int(next_state+2+observation*self.state_size))
			self.state_index[key] = next_state

	


	def mutate(self, rate):
		for key in type_space:
			# Mutate mixed strategy
			self.matrix[key][:,1] += Mutation.pairwise_gaussian(rate, self.matrix[key][:,1])
			
			# Mutate transition probabilities 
			for i in range(self.matrix[key].shape[0]):
				for obs_shift in range(2): # Do calculation separately for the Obs(Coop) and Ops(Defect) sets of columns
					shift = 2 + obs_shift * self.state_size
					mutation = Mutation.pairwise_gaussian(rate,self.matrix[key][i,shift:shift+self.state_size])
					self.matrix[key][i,shift:shift+self.state_size] += mutation

		self.make_matrix_consistent() # Only to fix floating-point error

class Mutation:
	@staticmethod
	def pairwise_gaussian(rate, array):
		normal_values = np.zeros(len(array))
		projected = copy(array)
		for i in range(len(projected)):
			j = (i+1) % len(projected)
			alpha = np.random.normal(loc=0, scale=rate)
			normal_values[i] += alpha 
			normal_values[j] -= alpha  
			projected[i] += alpha 
			projected[j] -= alpha 
		min_scaling = 1.0
		for i in range(len(normal_values)):
			i_after = projected[i]
			excess = i_after - 1 if i_after > 1 else abs(min(0, i_after))
			scaling = 1 - excess/abs(normal_values[i])
			min_scaling = min(min_scaling, scaling)
		normal_values *= min_scaling

		return normal_values

class Selection:
	@staticmethod
	def truncation(kwargs):
		population = kwargs["population"]
		scores = kwargs["scores"]
		cutoff = kwargs["cutoff"]

		# Kill off half and clone the rest 
		new_pops = sorted([(scores[i], population[i]) for i in range(len(scores))], key=lambda x:x[0], reverse=True) 
		new_pops = new_pops[:cutoff]
		new_pops = [x[1] for x in new_pops]
		return new_pops

	@staticmethod
	def uniform_random(kwargs):
		population = kwargs["population"]
		cutoff = kwargs["cutoff"]

		return np.random.choice(population, size=cutoff)

	@staticmethod 
	def roulette_wheel(kwargs):
		population = kwargs["population"]
		score = kwargs["scores"]
		cutoff = kwargs["cutoff"]

		return np.random.choice(population, size=cutoff, p=score)
	
	@staticmethod 
	def rank_selection(kwargs):
		population = kwargs["population"]
		score = kwargs["scores"]
		cutoff = kwargs["cutoff"]

		sorted_score = sorted(score)
		for i in range(len(score)):
			score[i] = sorted_score.index(score[i])
		
		score /= sum(score)

		return np.random.choice(population, size=cutoff, p=score)

	@staticmethod 
	def tournament_selection(kwargs):
		population = kwargs["population"]
		score = kwargs["scores"]
		k = kwargs["k"]
		p_ = kwargs["p"]
		cutoff = kwargs["cutoff"]
		probabilities = [p_*(1-p_)**i for i in range(k)] 
		probabilities = [x/sum(probabilities) for x in probabilities]
		new_pops = []
		for i in range(cutoff):
			pool = []
			indices = np.random.choice(a=len(population), size=k)
			while len(indices) > 0:
				arg = np.argmax([score[x] for x in indices])
				pool += [population[arg]]
				indices = np.concatenate((indices[:arg], indices[arg+1:]))
			new_pops += [np.random.choice(pool, p=probabilities)]
		
		return new_pops

	@staticmethod 
	def get_selection(command, **kwargs):
		if command == "truncation":
			return Selection.truncation(kwargs)
		elif command == "uniform_random":
			return Selection.uniform_random(kwargs)
		elif command == "roulette_wheel":
			return Selection.roulette_wheel(kwargs)
		elif command == "rank_selection":
			return Selection.rank_selection(kwargs)
		elif command == "tournament_selection":
			return Selection.tournament_selection(kwargs)

class Interaction:

	@staticmethod
	def pairwise(kwargs):
		population = kwargs["population"]
		fresh_mind = kwargs["fresh_mind"]
		payoff_matrix = kwargs["payoff_matrix"]

		len_sigma = population[0].action_matrix[0].shape[1]
		scores = [0]*len(population)
		combinations = copy(template_combinations)
		total_action = np.zeros((len_sigma, len(population))) # Keeps track of each agent's last moves, used to build reputation 

		for i in range(len(population)):
			for j in range(i+1, len(population)):
				# Lin Alg approach 
				P_i = population[i].matrix[population[j].get_type()][:,1]
				P_j = population[j].matrix[population[i].get_type()][:,1]
				# print('Contest between --')
				# print(population[i].matrix[population[j].get_type()])
				# print('&')
				# print(population[j].matrix[population[i].get_type()])
				for k in range(repetition):
					'''
					add total actions mechanism
					i_play = int(population[i].play(population[j].get_type()))
					j_play = int(population[j].play(population[i].get_type()))
					population[i].observe(j_play, population[j].get_type())
					population[j].observe(i_play, population[i].get_type())
					scores[i] += payoff_matrix[i_play,j_play,0]
					scores[j] += payoff_matrix[i_play,j_play,1]
					play = play_to_str[i_play+j_play]
					if play not in combinations:
						combinations[play] = 0 
					combinations[play] += 1
					
					'''
					A_i = np.matmul(np.transpose(P_i), population[i].action_matrix[population[j].get_type()])
					A_j = np.matmul(np.transpose(P_j), population[j].action_matrix[population[i].get_type()])
					
					
					# print('Score I: {0}'.format(np.sum(combination_matrix * payoff_matrix[:,:,0])))
					# print('Score J: {0}'.format(np.sum(combination_matrix * payoff_matrix[:,:,1])))
					# We can do some eigenvector shiz to get the infinite horizon eq.
					# input()
					TA_i = sum([population[i].get_submatrix(population[j].get_type(), k) * A_j[k] for k in range(2)])
					TA_j = sum([population[j].get_submatrix(population[i].get_type(), k) * A_i[k] for k in range(2)])
					
					
					P_i = np.matmul(np.transpose(TA_i), P_i) 
					P_j = np.matmul(np.transpose(TA_j), P_j) 
					
					# Prevent floating-point error
					P_i /= np.sum(P_i)
					P_j /= np.sum(P_j)
					# print('TA I: {0}'.format(TA_i))
					# print('TA J: {0}'.format(TA_j))
					# print()
					# print('P I: {0}'.format(P_i))
					# print('P J: {0}'.format(P_j))
				if fresh_mind:
					population[i].refresh(population[j].get_type())
					population[j].refresh(population[i].get_type())
				combination_matrix = np.outer(A_i, A_j) 
				# print(combination_matrix)
				total_action[:, i] += A_i
				total_action[:, j] += A_j
				combinations["CC"] += combination_matrix[0,0]
				combinations["C/D"] += combination_matrix[0,1] + combination_matrix[1,0]
				combinations["DD"] += combination_matrix[1,1]
				scores[i] += np.sum(combination_matrix * payoff_matrix[:,:,0])
				scores[j] += np.sum(combination_matrix * payoff_matrix[:,:,1])
		return scores, combinations, total_action

	@staticmethod
	def get_interaction(command, **kwargs):
		if command == "pairwise":
			return Interaction.pairwise(kwargs)
   

class Population:
	def __init__(self, population_size, state_size, repetition, fresh_mind = False, set_strategy=None):
		self.pops = [MDP_Learner(state_size, set_strategy) for i in range(population_size)]
		self.repetition = repetition 
		self.fresh_mind = fresh_mind
		self.population_size = population_size
		self.state_size = state_size

	def aggregate_selection(self, payoff_matrix_, interaction, parent_sel, survivor_sel, overlap, num_offsprings, k_=0, p_=0):

		np.random.shuffle(self.pops)
		for pop in self.pops:
			pop.age += 1

		parent_scores, combinations, total_actions = Interaction.get_interaction(interaction, 
			population=self.pops, 
			fresh_mind=self.fresh_mind, 
			payoff_matrix=payoff_matrix_)

		# Update post-interaction history
		for i in range(len(self.pops)):
			self.pops[i].history = self.pops[i].history * (1-reputation_update) + total_actions[0, i] / (np.sum(total_actions[:, i])) * reputation_update

		parents = Selection.get_selection(parent_sel, 
			population=self.pops, 
			scores=parent_scores, 
			cutoff=num_offsprings,
			k = k_,
			p = p_)

		offsprings = []
		for parent in parents:
			offsprings += [deepcopy(parent)]
			offsprings[-1].age = 0
			offsprings[-1].mutate(0.02) 

		if overlap:
			self.pops += offsprings
		
		scores_, _, _ = Interaction.get_interaction(interaction, 
			population=self.pops, 
			fresh_mind=self.fresh_mind, 
			payoff_matrix=payoff_matrix_)

		self.pops = Selection.get_selection(survivor_sel, 
			population=self.pops, 
			scores=scores_, 
			cutoff=self.population_size - 0 if overlap else num_offsprings,
			k = k_,
			p = p_)
		
		if not overlap:
			self.pops += offsprings

		
		return combinations

rand_seed = time.time()
random.seed(rand_seed)
# Change these
name = "data/Prisoner/prisoner_ep2states4_o"
prisoner = [[[1,4],[3,3]],
				[[2,2],[4,1]]]

no_conflict = [[[3,2],[4,4]],
				[[1,1],[2,3]]]

chicken = [[[2,4],[3,3]],
				[[1,1],[4,2]]]

battle = [[[3,4],[2,2]],
				[[1,1],[4,3]]]
p_matrix = prisoner
iterations = 500
type_space = range(1)
type_threshold = [0.0]
window = 25
repetition = 5
reputation_update = 0.5 
play_to_str = {2 : "DD", 1 : "C/D", 0 : "CC"}
template_combinations = {"DD":0, "C/D":0, "CC":0}
tit_for_tat = np.asarray([[0,1.0,1.0,0.0,0.0,1.0], [1,0.0,1.0,0.0,0.0,1.0]])
midway = np.asarray([[0,1.0,0.5,0.5,0.5,0.5], [1,0.0,0.5,0.5,0.5,0.5]]) 
Pops = Population(10,2, repetition, fresh_mind=True)
name += "_" + str(len(Pops.pops)) + "players"
np.set_printoptions(precision=2, suppress=True)
tally = {"CC":[], "C/D":[], "DD":[]}

fossils = []
type_fossils = dict([(key, []) for key in type_space])

for i in range(iterations):

	combinations = Pops.aggregate_selection(
		payoff_matrix_= np.asarray([[p_matrix[0][1],p_matrix[0][0]],
						[p_matrix[1][1],p_matrix[1][0]]]),
		interaction = "pairwise",
		parent_sel = "truncation",
		survivor_sel = "truncation",
		overlap = True,
		num_offsprings = len(Pops.pops),
		k_ = 3,
		p_ = 0.5)

	for key in combinations:
		tally[key] += [combinations[key]]

	type_distribution = Counter([pop.get_type() for pop in Pops.pops])
	for key in type_distribution:
		type_fossils[key] += [type_distribution[key]]

	if i % window == 0:
		print(i)
		all_matrices = []
		for pop in Pops.pops:
			all_matrices += copy([(pop.matrix, pop.get_type())])
		fossils += [all_matrices]

interactions_per_generation = sum([tally[key][0] for key in tally])
print(interactions_per_generation)
for key in tally:
	plt.plot([(sum(tally[key][i*window:(i+1)*window])/window)/(interactions_per_generation) for i in range(iterations//window)])

 

plt.legend(list(tally.keys()))
plt.savefig(fname=name+".png")
plt.show()

plt.figure()
for key in type_space:
	plt.plot(type_fossils[key])
plt.legend(["Type {0}".format(key) for key in type_space])
plt.savefig(fname=name+"_typedist.png")
plt.show()

workbook = xlsxwriter.Workbook(name + '.xlsx')
header = ["Action", "Start. P"]
borderline = [1, Pops.state_size+1] 
border = workbook.add_format({'right': 2})

for i, generation in enumerate(fossils):
	generation_sheet = workbook.add_worksheet(str(i))
	cells_array = []
	cells_array += [header]
	for j, matrix in enumerate(generation):
		matrix, agent_type = matrix
		cells_array += [[""]]
		cells_array += [["Agent {0}, type {1}".format(j, agent_type)]]
		for key in matrix:
			cells_array += [["Type {0}".format(key)]]
			for row in matrix[key]:
				cells_array += [row]
	for r, row in enumerate(cells_array):
		for k, cell in enumerate(row):
			generation_sheet.write(r, k, cell, border if k in borderline else None)
workbook.close()

