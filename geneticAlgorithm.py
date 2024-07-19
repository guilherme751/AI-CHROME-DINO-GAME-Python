import numpy as np
from dinoAIParallel import manyPlaysResultsTrain, manyPlaysResultsTest
import math, time
from tqdm import tqdm
from dinoAIParallel import RENDER_GAME

class GeneticAlgorithm():
    def __init__(self, solution_size, population_size, elitism_percent, cross_ratio, 
                mutation_ratio, max_iterations, max_time):
        # população inicial com valores aleatórios
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
        '''
        Avalia os valores de cada indivíduo da população com base na média de 3 execuções do jogo.
        '''
        results = manyPlaysResultsTrain(3, self.population) # roda 3 vezes o jogo e calcula a média dessas execuções
        self.val_pop = []
        self.results = results
        for i in range(self.population_size):
            if results[i] < 0:               
                results[i] = 1
            self.val_pop.append((results[i], self.population[i]))
        # salva um vetor com os valores de cada solução ordenados
        self.val_pop = sorted(self.val_pop, reverse=True, key=lambda x: x[0]) 

    def mutation(self, sol):
        '''
        São escolhidos pontos aleatórios da solução para sofrerem mutação. Troca-se o valor por um
        outro gerado aleatoriamente de uma distribuição uniforme que varia de -1 a 1.
        '''
        new_sol = sol.copy()        
        rands = np.random.randint(0, self.solution_size - 1, int(self.solution_size/3))
        for rand in rands:
            new_sol[rand] = np.random.uniform(-1, 1, 1)[0]
        return new_sol
    
    def mutation_step(self, crossed):
        '''
        Escolhe indivíduos com base no mutation_ratio para sofrer mutação.
        '''
        for i, sol in enumerate(crossed):
            rand = np.random.uniform(0, 1) # fator aleatório para se realizar a mutação
            if rand <= self.mutation_ratio:            
                mutated = self.mutation(sol)
                crossed[i] = mutated
        return crossed
    
    def elitism(self):
        '''
        Computa o conjunto elite com os melhores indivíduos da população
        '''
        n = math.floor(self.population_size * self.elitism_percent)
        if n < 1: n = 1
        elite = self.val_pop[:n]
        self.new_pop = [sol for _, sol in elite]
                
        return elite[0]
    
    def crossover(self, parent1, parent2):
        '''
        Realiza o crossover "um ponto".
        '''
        crossover_point = np.random.randint(1, len(parent1) - 1)        
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]

        return np.array(child1), np.array(child2)

    def crossover_step(self, selected):
        '''
        Escolhe indivíduos com base no crossover_ratio para realizar crossover.
        '''
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
        '''
        Seleciona indivíduos com o método da roleta em que a chance de um indivíduo ser selecionado depende
        do seu valor no jogo.
        '''
        prob_values = self.results/sum(self.results)
        index = np.random.choice(range(0, self.population_size), 
                                p=prob_values, replace=False, 
                                size=self.population_size - len(self.new_pop))
        
        return [self.population[i] for i in index]
    
    def convergent(self):
        '''
        Verifica se a população convergiu.
        '''
        base = self.population[0]
        for sol in self.population:
            if base != sol:
                return False
        return True

    def run(self):
        '''
        Executa o algoritmo completo conforme especificado.
        '''
        start = time.process_time()
        end = 0; iter = 0; self.best_score = -1
        bestvalues = []
        
        with tqdm(total=self.max_iterations) as pbar:
            while iter < self.max_iterations and end-start <=self.max_time:
            
                self.evaluate_population() # avalia os valores de cada indivíduo com base nos scores no jogo
                best = self.elitism()      # computa o conjunto elite com os melhores indivíduos da população
                bestvalues.append(best[0])  # salva o valor do melhor indivíduo da população 

                if best[0] > self.best_score:   # atualiza a melhor solução global                   
                    self.best_score = best[0]
                    self.best_solution = best[1]               
                
                selected = self.selection()   # seleção por meio da roleta     
                crossed = self.crossover_step(selected)  # crossover
                mutated = self.mutation_step(crossed)   # mutação
            
                self.population = self.new_pop + mutated  # nova população
                pbar.set_description(
                    f'Population Best: {best[0]:.2f}, Best Score: {self.best_score:.2f}'
                )
                
                pbar.update()

                iter += 1
                
                end = time.process_time()

        return self.best_solution, self.best_score, bestvalues
    
    def test(self, best_solution):
        '''
        Testa a melhor solução encontrada na fase de treinamento a partir de 30 execuções do jogo.
        São retornados os valores de cada execução.
        '''
        res, _ = manyPlaysResultsTest(30, best_solution)
        
        return res