import random
from copy import copy, deepcopy 
import numpy as np 
import matplotlib.pyplot as plt 
import xlsxwriter 
import os 
import time
import argparse
class MDP_Learner:

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
		self.matrix = np.concatenate((self.matrix[:,0:1], 
		abs(self.matrix[:,1:2]/sum(self.matrix[:,1:2])),
		np.apply_along_axis(MDP_Learner._normalize, 1, self.matrix[:,2:self.state_size+2]),
		np.apply_along_axis(MDP_Learner._normalize, 1, self.matrix[:,self.state_size+2:])), axis=1) 

	def __init__(self, state_size, set_strategy=None):
		# This matrix is a Nx(2N+1) representation of learning strategy 
		# Each row represents a decision node, expressed in: action, trans. prob | opp. cooperate | opp. defect
		if set_strategy is None:
			self.matrix = np.concatenate(([[i%2] for i in range(state_size)],
			[[1/state_size] for i in range(state_size)],
			np.random.rand(state_size, state_size),
			np.random.rand(state_size, state_size)), axis=1)
		else:
			self.matrix = set_strategy

		
		self.state_size = state_size
		self.make_matrix_consistent()
		self.state_index = np.random.choice(a=self.matrix.shape[0], p=self.matrix[:,1])
		self.prev_state_observation = (0,1)
		self.age = 0
		
	def refresh(self):
		self.state_index = np.random.choice(a=self.matrix.shape[0], p=self.matrix[:,1])

	def play(self):
		return self.matrix[self.state_index,0]

	def observe(self, observation):
		if observation is not None:
			if observation == 0:
				trans_prob = self.matrix[self.state_index, 2:self.state_size+2]
			else:
				trans_prob = self.matrix[self.state_index, self.state_size+2:]
	
			next_state = np.random.choice(range(self.state_size),p=trans_prob)
			self.prev_state_observation = (self.state_index, int(next_state+2+observation*self.state_size))
			self.state_index = next_state

	


	def mutate(self, rate):
		# Mutate mixed strategy
		self.matrix[:,1] += Mutation.pairwise_gaussian(rate, self.matrix[:,1])
		
		# Mutate transition probabilities 
		for i in range(self.matrix.shape[0]):
			for obs_shift in range(2): # Do calculation separately for the Obs(Coop) and Ops(Defect) sets of columns
				shift = 2 + obs_shift * self.state_size
				mutation = Mutation.pairwise_gaussian(rate,self.matrix[i,shift:shift+self.state_size])
				self.matrix[i,shift:shift+self.state_size] += mutation

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

		return np.random.choice(population, size=cutoff).tolist()

	@staticmethod 
	def roulette_wheel(kwargs):
		population = kwargs["population"]
		score = kwargs["scores"]
		cutoff = kwargs["cutoff"]

		return np.random.choice(population, size=cutoff, p=score/sum(score)).tolist()
	
	@staticmethod 
	def rank_selection(kwargs):
		population = kwargs["population"]
		score = kwargs["scores"]
		cutoff = kwargs["cutoff"]

		sorted_score = sorted(score)
		for i in range(len(score)):
			score[i] = sorted_score.index(score[i])
		
		score /= sum(score)

		return np.random.choice(population, size=cutoff, p=score).tolist()

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
		scores = [0]*len(population)
		combinations = copy(template_combinations)
		for i in range(len(population)):
			for j in range(i+1, len(population)):
				for k in range(repetition):
					i_play = int(population[i].play())
					j_play = int(population[j].play())
					population[i].observe(j_play)
					population[j].observe(i_play)
					scores[i] += payoff_matrix[i_play,j_play,0]
					scores[j] += payoff_matrix[i_play,j_play,1]
					
					play = play_to_str[i_play+j_play]
					if play not in combinations:
						combinations[play] = 0 
					combinations[play] += 1
				if fresh_mind:
					population[i].refresh()
					population[j].refresh()
		return scores, combinations

	@staticmethod
	def pairwise_matrix(kwargs):
		population = kwargs["population"]
		fresh_mind = kwargs["fresh_mind"]
		payoff_matrix = kwargs["payoff_matrix"]

		scores = [0]*len(population)
		combinations = copy(template_combinations)
		for i in range(len(population)):
			for j in range(i+1, len(population)):
				i_action = population[i].matrix[:,0]
				j_action = population[i].matrix[:,0]
				i_dist = population[i].matrix[:,1]
				j_dist = population[i].matrix[:,1]
				i_matrix = [population[i].matrix[:,2:2+population[i].state_size], population[i].matrix[:,2+population[i].state_size:]]
				j_matrix = [population[j].matrix[:,2:2+population[j].state_size], population[j].matrix[:,2+population[j].state_size:]]
				for k in range(repetition):
					i_play = np.asarray([sum(np.take(i_dist, np.where(i_action==i)[0])) for i in range(2)])
					j_play = np.asarray([sum(np.take(j_dist, np.where(j_action==i)[0])) for i in range(2)])
					i_dist = np.matmul(i_dist.T, (i_matrix[0] * j_play[0] + i_matrix[1] * j_play[1])).T
					j_dist = np.matmul(j_dist.T, (j_matrix[0] * i_play[0] + j_matrix[1] * i_play[1])).T
				
				comb = np.outer(i_play, j_play)
				# This works because CC + DD = 2CD and thus iterative score is sequence invariant
				scores[i] = np.sum(comb * payoff_matrix[:,:,0])
				scores[j] = np.sum(comb * payoff_matrix[:,:,1])
				
				combinations["CC"] += comb[0,0] * repetition
				combinations["C/D"] += (comb[0,1] + comb[1,0]) * repetition
				combinations["DD"] += comb[1,1] * repetition
	
		return scores, combinations
	@staticmethod
	def get_interaction(command, **kwargs):
		if command == "pairwise":
			return Interaction.pairwise(kwargs)
		elif command == "pairwise_matrix":
			return Interaction.pairwise_matrix(kwargs)
   

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


		parent_scores, combinations = Interaction.get_interaction(interaction, 
			population=self.pops, 
			fresh_mind=self.fresh_mind, 
			payoff_matrix=payoff_matrix_)


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
			offsprings[-1].mutate(0.03) 

		if overlap:
			self.pops.extend(offsprings)
		
		scores_, _ = Interaction.get_interaction(interaction, 
			population=self.pops, 
			fresh_mind=self.fresh_mind, 
			payoff_matrix=payoff_matrix_)

		self.pops = Selection.get_selection(survivor_sel, 
			population=self.pops, 
			scores=scores_, 
			cutoff=self.population_size - (0 if overlap else num_offsprings),
			k = k_,
			p = p_)
		
		if not overlap:
			self.pops.extend(offsprings)

		
		return combinations

def display_heatmap(data_2d, x_axis, y_axis):
	fig, ax = plt.subplots()
	im = ax.imshow(data_2d, cmap='jet')
	# ax.set_xticks(x_axis)
	plt.colorbar(im)
	plt.show() 
	
if __name__ == "__main__":
	rand_seed = time.time()
	random.seed(rand_seed)

	parser = argparse.ArgumentParser()
	parser.add_argument("game")
	parser.add_argument("--name")
	parser.add_argument("--iter", type=int, default=1000)
	parser.add_argument("--states", type=int, default=2)
	parser.add_argument("--pop", type=int, default=10)
	parser.add_argument("--window", type=int, default=25)
	parser.add_argument("--repetition", type=int, default=10)
	parser.add_argument("--overlap", action='store_true')
	parser.add_argument("--heatmap", type=int, default=1)
	parser.add_argument("--mean_score_dist", nargs=2, type=int, default=[0,0])
	parser.add_argument("--par_sel", default="uniform_random")
	parser.add_argument("--surv_sel", default="truncation")
	parser.add_argument("--interaction", default="pairwise")
	args = parser.parse_args() 
	
	prisoner = [[[1,4],[3,3]],
					[[2,2],[4,1]]]

	no_conflict = [[[10,2],[10,10]],
					[[1,1],[2,10]]]
	
	games = {
		"prisoner" : prisoner,
		"no_conflict" : no_conflict
	}
	if args.game not in games:
		raise ValueError("Game not defined")

	p_matrix = games[args.game]
	iterations = args.iter 
	states = args.states 
	pop_size = args.pop
	heatmap = args.heatmap
	mean_score_dist = args.mean_score_dist[0]
	mean_score_discard = args.mean_score_dist[1]

	window = args.window
	repetition = args.repetition
	interactions_per_generation = pop_size * (pop_size-1) /2 * repetition
	play_to_str = {2 : "DD", 1 : "C/D", 0 : "CC"}
	str_to_index = {"DD" : [1,0], "C/D" : [0,0], "CC" : [0,1]}
	template_combinations = {"DD":0, "C/D":0, "CC":0}
	tit_for_tat = np.asarray([[0,1.0,1.0,0.0,0.0,1.0], [1,0.0,1.0,0.0,0.0,1.0]])
	midway = np.asarray([[0,1.0,0.5,0.5,0.5,0.5], [1,0.0,0.5,0.5,0.5,0.5]]) 




	
	np.set_printoptions(precision=2, suppress=True)

	
	num_runs = 1 
	if heatmap > 1:
		num_runs = heatmap 
	elif mean_score_dist > 0:
		num_runs = mean_score_dist

	# if command == "truncation":
	# 	return Selection.truncation(kwargs)
	# elif command == "uniform_random":
	# 	return Selection.uniform_random(kwargs)
	# elif command == "roulette_wheel":
	# 	return Selection.roulette_wheel(kwargs)
	# elif command == "rank_selection":
	# 	return Selection.rank_selection(kwargs)
	# elif command == "tournament_selection":
	# 	return Selection.tournament_selection(kwargs)

	ps = args.par_sel
	ss = args.surv_sel
	ol = args.overlap
	name = "data/%s/%s" % (args.game, args.name) if args.name else "data/%s/pop%d_iter%d_rep%d_state%d" % (args.game, pop_size, iterations, repetition, states)
	
	
	if not os.path.exists(os.path.dirname(name)):
		os.makedirs(os.path.dirname(name))
	
	def run_simulation(ps, ss, ol, name):	
		Pops = Population(pop_size,states, repetition, fresh_mind=True)
		tallies = []
		mean_scores = [] 		
		for k in range(num_runs):
			print("-- Run %d --" % k)
			fossils = []
			tally = {"CC":[], "C/D":[], "DD":[]}
			for i in range(iterations):
				combinations = Pops.aggregate_selection(
					payoff_matrix_= np.asarray([[p_matrix[0][1],p_matrix[0][0]],
									[p_matrix[1][1],p_matrix[1][0]]]),
					interaction = args.interaction,
					parent_sel = ps,
					survivor_sel = ss,
					overlap = ol,
					num_offsprings = len(Pops.pops)//2,
					k_ = 3,
					p_ = 0.5)

				for key in combinations:
					tally[key] += [combinations[key]]
				if i % window == 0 and num_runs == 1: 
					print(i)
					all_matrices = []
					for pop in Pops.pops:
						all_matrices += copy([pop.matrix])
					fossils += [all_matrices]

			

			tallies += [{key:np.asarray(tally[key])/interactions_per_generation for key in tally}]
			interaction_scores = [np.mean(tallies[-1][key][mean_score_discard:]) * sum(p_matrix[str_to_index[key][0]][str_to_index[key][1]])/2  for key in tallies[-1]]
			mean_scores += [sum(interaction_scores)]
			print("Mean score: %.2f" % mean_scores[-1])

		# If this is a single-run, save
		if num_runs == 1:
			tally = tallies[0]
			for key in tally:
				plt.plot([(sum(tally[key][i*window:(i+1)*window])/window) for i in range(iterations//window)])
			plt.legend(list(tally.keys()))

			plt.savefig(fname=name+".png")
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
					cells_array += [[""]]
					cells_array += [["Agent {0}".format(j)]]
					for row in matrix:
						cells_array += [row]
				for r, row in enumerate(cells_array):
					for k, cell in enumerate(row):
						generation_sheet.write(r, k, cell, border if k in borderline else None)
			workbook.close()
		elif heatmap > 1:
			percentages = np.linspace(1,0,100)
			x_res = 100 
			x_step = iterations // x_res 
			hmap = np.zeros((len(percentages), x_res))
			for t in tallies:
				for i in range(x_res):
					perc = np.average(t['CC'][i*x_step:(i+1)*x_step])
					for j,p in enumerate(percentages):
						if p <= perc:
							hmap[j, i] += 1 
			display_heatmap(hmap, range(iterations), percentages)
		elif mean_score_dist > 0:
			plt.figure()
			plt.hist(mean_scores, bins=20, range=(2,3))
			plt.savefig(name+".png")
			
	for ps in ["truncation", "uniform_random", "roulette_wheel"]:
		for ss in ["truncation", "uniform_random"]:
			for ol in [True, False]:
				name = "data/%s/%s+%s+%s" % (args.game, ps, ss, str(ol))
				run_simulation(ps, ss, ol, name)
