import numpy as np
from dinoAIParallel import manyPlaysResultsTrain, manyPlaysResultsTest
import math, time
from tqdm import tqdm

class GeneticAlgorithm():
    def __init__(self, solution_size, population_size, elitism_percent, cross_ratio, 
                mutation_ratio, max_iterations, max_time):
        self.population = [
            np.random.uniform(-1,1, solution_size) for _ in range(population_size)]
        self.solution_size = solution_size
        self.population_size = population_size
        self.elitism_percent = elitism_percent
        self.cross_ratio = cross_ratio
        self.mutation_ratio = mutation_ratio
        self.max_iterations = max_iterations
        self.max_time = max_time        
    

    def evaluate_population(self):
        results = manyPlaysResultsTrain(3, self.population)
        self.val_pop = []
        self.results = results
        for i in range(self.population_size):
            if results[i] < 0: 
                print('NEGATIVO|||||||', results[i])
                results[i] = 1
            self.val_pop.append((results[i], self.population[i]))
        self.val_pop = sorted(self.val_pop, reverse=True, key=lambda x: x[0])

    def mutation(self, sol):
        new_sol = sol.copy()
        rands = np.random.randint(0, self.solution_size - 1, int(self.solution_size/3))
        for rand in rands:
            new_sol[rand] = np.random.uniform(-1, 1, 1)[0]
        return new_sol
    def mutation_step(self, crossed):
        for i, sol in enumerate(crossed):
            rand = np.random.uniform(0, 1)
            if rand <= self.mutation_ratio:            
                mutated = self.mutation(sol)
                crossed[i] = mutated
        return crossed
    
    def elitism(self):
        n = math.floor(self.population_size * self.elitism_percent)
        if n < 1: n = 1
        elite = self.val_pop[:n]
        self.new_pop = [sol for _, sol in elite]
                
        return elite[0]
    
    def crossover(self, parent1, parent2):
        crossover_point = np.random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]

        return np.array(child1), np.array(child2)

    def crossover_step(self, selected):
        temp_pop = []  

        for _ in range(round(len(selected)/2)):
            
            fst_ind = np.random.randint(0, len(selected) - 1)
            scd_ind = np.random.randint(0, len(selected) - 1)
            parent1 = selected[fst_ind] 
            parent2 = selected[scd_ind]

            rand = np.random.uniform(0,1)
            if rand <= self.cross_ratio:
                child1, child2 = self.crossover(list(parent1), list(parent2))
            else:
                child1, child2 = parent1, parent2

            temp_pop.append(child1)
            temp_pop.append(child2)
        return temp_pop
    
    def selection(self):
        prob_values = self.results/sum(self.results)
        index = np.random.choice(range(0, self.population_size), 
                                p=prob_values, replace=False, 
                                size=self.population_size - len(self.new_pop))
        
        return [self.population[i] for i in index]
    
    def convergent(self):
        base = self.population[0]
        for sol in self.population:
            if base != sol:
                return False
        return True

    def run(self):
        self.evaluate_population()
        start = time.process_time()
        end = 0
        iter = 0     
        iternotbetter = 0   
        self.best_score = 0
        bestvalues = []
        with tqdm(total=self.max_iterations) as pbar:
            while iter < self.max_iterations and end-start <=self.max_time:
            
                self.evaluate_population()
                best = self.elitism()  
                bestvalues.append(best[0])

                if best[0] > self.best_score:
                    iternotbetter = 0
                    self.best_score = best[0]
                    self.best_solution = best[1]
                
                selected = self.selection()        
                crossed = self.crossover_step(selected)
                mutated = self.mutation_step(crossed)
            
                self.population = self.new_pop + mutated  
                pbar.set_description(
                    f'Current Best: {best[0]:.2f}, Best Score: {self.best_score:.2f}'
                )
                
                pbar.update()
                # conv = convergent(pop)

                iter += 1
                iternotbetter += 1
                end = time.process_time()

        return self.best_solution, self.best_score, bestvalues
    
    def test(self, best_solution):
        res, value = manyPlaysResultsTest(30, best_solution)
        
        return res, value