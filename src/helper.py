import argparse

def parse_arguments():
    parse = argparse.ArgumentParser(description='Dino IA run')
    parse.add_argument('--pop_size', help='Population size', type=int)
    parse.add_argument('--max_iterations', help='Maximum number of iterations', type=int)
    parse.add_argument('--elitism_percent', help='Percentual of individuals to be selected to elite set [0,1]', type=float)
    parse.add_argument('--mutation_ratio', help='Mutation ratio [0,1]', type=float)
    parse.add_argument('--crossover_ratio', help='CrossOver ratio [0,1]', type=float)
    parse.add_argument('--max_time', help='Maximum execution time in seconds', type=float)
    args = parse.parse_args()
    return vars(args)