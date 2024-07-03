from geneticAlgorithm import GeneticAlgorithm
from FNN import FNN
import matplotlib.pyplot as plt

def main():
    genAlg = GeneticAlgorithm(solution_size= 25, population_size= 100, 
                              elitism_percent= 0.2, cross_ratio= 0.9, 
                              mutation_ratio= 0.4, max_iterations = 100000, 
                              max_time= 10000)
    
    best_solution, _, best_values = genAlg.run()
    results, mean = genAlg.test(best_solution= best_solution)
    print(results)
    print('Média após teste:', mean)

main()


