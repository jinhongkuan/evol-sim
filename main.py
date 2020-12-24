import time
import pyximport; pyximport.install()
import random
import argparse
import ess

def main():
  start = time.time()
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
  parser.add_argument("--interaction", default="pairwise_matrix")
  parser.add_argument("--default", action='store_true')
  args = parser.parse_args() 

  ess.do_everything(args)
  print("Amount of time = ", time.time() - start)

if __name__ == "__main__":
  main()