from geneticAlgorithm import GeneticAlgorithm
import numpy as np
from helper import parse_arguments

def compute_args(args):
    pop_size = 100
    elitism_percent = 0.1
    crossover_ratio = 0.8
    mutation_ratio = 0.4
    max_iterations = 1000
    max_time = 100000

    if args['pop_size']:
        pop_size = args['pop_size']    

    if args['elitism_percent']:
        elitism_percent = args['elitism_percent']
    
    if args['crossover_ratio']:
        crossover_ratio = args['crossover_ratio']
    
    if args['mutation_ratio']:
        mutation_ratio = args['mutation_ratio']
    
    if args['max_iterations']:
        max_iterations = args['max_iterations']

    if args['max_time']:
        max_time = args['max_time']

    return pop_size, elitism_percent, crossover_ratio, mutation_ratio, max_iterations, max_time

def main():
    args = compute_args(parse_arguments())
    
    genAlg = GeneticAlgorithm(solution_size= 25, population_size=args[0], 
                              elitism_percent=args[1], cross_ratio= args[2], 
                              mutation_ratio= args[3], max_iterations = args[4],
                              max_time= args[5])
    
    best_solution, _, best_values = genAlg.run() # roda Dino conforme especificado    
    results = genAlg.test(best_solution= best_solution) # testa a melhor solução retornada
    
    print('Média após teste:', np.mean(results) - np.std(results))


if __name__ == '__main__':
    main()


