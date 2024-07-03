from geneticAlgorithm import GeneticAlgorithm
from FNN import FNN
import matplotlib.pyplot as plt

def main():
    genAlg = GeneticAlgorithm(solution_size= 37, population_size= 100, 
                              elitism_percent= 0.2, cross_ratio= 0.7, 
                              mutation_ratio= 0.6, max_iterations = 1000, 
                              max_time= 10000)
    
    best_solution, _, best_values = genAlg.run()
    results, mean = genAlg.test(best_solution= best_solution)
    print(results)
    print('Média após teste:', mean)

main()


