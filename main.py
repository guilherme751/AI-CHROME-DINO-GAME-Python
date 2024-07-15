from geneticAlgorithm import GeneticAlgorithm
import numpy as np

def main():
    genAlg = GeneticAlgorithm(solution_size= 25, population_size= 100, 
                              elitism_percent= 0.2, cross_ratio= 0.8, 
                              mutation_ratio= 0.6, max_iterations = 1000,
                              max_time= 100000)
    
    best_solution, _, best_values = genAlg.run() # roda Dino conforme especificado
    
    results = genAlg.test(best_solution= best_solution) # testa a melhor solução retornada
    
    print('Média após teste:', np.mean(results))


if __name__ == 'main':
    main()


