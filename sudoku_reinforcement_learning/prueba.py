import sys
import time
import numpy as np
import random as rnd
from random import shuffle, sample, randint
from copy import deepcopy
from math import exp
import pandas as pd
import matplotlib.pyplot as plt
import heapq

def get_column_indices(i, type="data index"):
	if type=="data index":
		column=1%9
	elif type=="column index":
		column = i
	indices = [column + 9 * j for j in range(9)]
	return indices
	
def get_row_indices(i, type="data index"):
    if type=="data index":
        row = i // 9
    elif type=="row index":
        row = i
    indices = [j + 9*row for j in range(9)]
    return indices
	
def get_block_indices(k,initialEntries,ignore_originals=False):
	row_offset = (k//3)*3
	col_offset= (k%3)*3
	indices=[col_offset+(j%3)+9*(row_offset+(j//3)) for j in range(9)]
	if ignore_originals:
		#indices = filter(lambda x:x not in initialEntries, indices)
		indices = [x for x in indices if x not in initialEntries]
	return indices

def randomAssign(puzzle, initialEntries):
	for num in range(9):
		block_indices=get_block_indices(num, initialEntries)
		block= puzzle[block_indices]
		zero_indices=[ind for i,ind in enumerate(block_indices) if block[i] == 0]
		to_fill = [i for i in range(1,10) if i not in block]
		shuffle(to_fill)
		for ind, value in zip(zero_indices, to_fill):
			puzzle[ind]=value

def score_board(puzzle):
	score = 0
	for row in range(9): # por cada fila obtiene la cantidad de numeros diferentes
		score-= len(set(puzzle[get_row_indices(row, type="row index")]))
	for col in range(9): # por cada columna obtiene la cantidad de numeros diferentes
		score -= len(set(puzzle[get_column_indices(col,type="column index")]))
	return score

def make_neighborBoard(puzzle, initialEntries):
    new_data = deepcopy(puzzle)
    block = randint(0,8)  # escoje un bloque aleatoriamente
    num_in_block = len(get_block_indices(block,initialEntries,ignore_originals=True)) #cantidad de ´posiciones que se puede mover en el bloque 
    random_squares = sample(range(num_in_block),2)
    square1, square2 = [get_block_indices(block,initialEntries,ignore_originals=True)[ind] for ind in random_squares]
    new_data[square1], new_data[square2] = new_data[square2], new_data[square1]
    return new_data

def showPuzzle(puzzle):
	def checkZero(s):
		if s != 0: return str(s)
		if s == 0: return "0"
	results = np.array([puzzle[get_row_indices(j, type="row index")] for j in range(9)])
	s=""
	for i, row in enumerate(results):
		if i%3==0:
			s +="-"*25+'\n'
		s += "| " + " | ".join([" ".join(checkZero(s) for s in list(row)[3*(k-1):3*k]) for k in range(1,4)]) + " |\n"
	s +="-"*25+''
	print (s)

#######################################################################
# Clases y metodos para el algoritmo genetico
#######################################################################
## CLASE INDIVIDUO
class Individual(object):   
   
    def __init__(self, chromosome, puzzle):
            self.chromosome = chromosome[:]
            self.puzzle = puzzle
            self.fitness = -1  # -1 indica que el individuo no ha sido evaluado
            
    def crossover_onepoint(self, other):
        "Retorna dos nuevos individuos del cruzamiento de un punto entre individuos self y other "
        c = rnd.randrange(len(self.chromosome))
        ind1 = Individual(self.chromosome[:c] + other.chromosome[c:],self.puzzle)
        ind2 = Individual(other.chromosome[:c] + self.chromosome[c:],self.puzzle)
        return [ind1, ind2]   
    
    
    def crossover_uniform(self, other):
        chromosome1 = []
        chromosome2 = []
        "Retorna dos nuevos individuos del cruzamiento uniforme entre self y other "
        for i in range(len(self.chromosome)):
            if rnd.uniform(0, 1) < 0.5:
                chromosome1.append(self.chromosome[i])
                chromosome2.append(other.chromosome[i])
            else:
                chromosome1.append(other.chromosome[i])
                chromosome2.append(self.chromosome[i])
        ind1 = Individual(chromosome1,self.puzzle)
        ind2 = Individual(chromosome2,self.puzzle)
        return [ind1, ind2] 

    def mutate_position(self): 
        """       Bit flip
        Cambia aleatoriamente un alelo de un gen."""
        mutated_chromosome = deepcopy(self.chromosome)
        initialEntries = np.arange(81)[self.puzzle > 0]
        new_puzzle, changed_block = make_neighborBoard_mod(create_puzzle(self.chromosome,self.puzzle),initialEntries)
        mutGene = changed_block
        newAllele = new_puzzle[get_block_indices(mutGene,initialEntries)]
        mutated_chromosome[mutGene] = newAllele
        return Individual(mutated_chromosome,self.puzzle)

## METODO PARA MOSTRAR LA POBLACION
def display(population):
    listaAG=[]
    for i in range(len(population)):
        listaAG.append([population[i].chromosome,population[i].fitness])

    data=pd.DataFrame(listaAG)
    data.columns = ['Poblacion','fitness']
    return data

## METODO SELECCION RULETA
def select_parents_roulette(population):
    popsize = len(population)
    
    # Escoje el primer padre
    sumfitness = sum([indiv.fitness for indiv in population])  # suma total del fitness de la poblacion
    pickfitness = rnd.uniform(0, sumfitness)   # escoge un numero aleatorio entre 0 y sumfitness
    cumfitness = 0     # fitness acumulado
    for i in range(popsize):
        cumfitness += population[i].fitness
        if cumfitness > pickfitness: 
            iParent1 = i
            break
     
    # Escoje el segundo padre, desconsiderando el primer padre
    sumfitness = sumfitness - population[iParent1].fitness # retira el fitness del padre ya escogido
    pickfitness = rnd.uniform(0, sumfitness)   # escoge un numero aleatorio entre 0 y sumfitness
    cumfitness = 0     # fitness acumulado
    for i in range(popsize):
        if i == iParent1: continue   # si es el primer padre 
        cumfitness += population[i].fitness
        if cumfitness > pickfitness: 
            iParent2 = i
            break        
    return (population[iParent1], population[iParent2])

## METODO SELECCION TORNEO

def select_parents_torneo(population,size_torneo):
    
    # Escoje el primer padre
    list_indiv=[]
    x1 = np.random.permutation(len(population) )
    y1= x1[0:size_torneo]
    for i in range(size_torneo):
        list_indiv.append(population[y1[i]].fitness)
    
    iParent1=np.argmax(list_indiv)
    
    # Escoje el segundo padre, desconsiderando el primer padre   
    x2 = np.delete(x1, iParent1)
    x2 = np.random.permutation(x2)
    list_indiv=[]
    y2= x2[0:size_torneo]
    for i in range(size_torneo):
        list_indiv.append(population[y2[i]].fitness)
    iParent2=np.argmax(list_indiv)
    
    return (population[x1[iParent1]],population[x2[iParent2]])

## METODO FUNCION FITNESS SCORE
def score_fun_puzzle(chromosome,puzzle):
    """Función que evalua el fitness para sudoku"""
    new_puzzle = deepcopy(puzzle)
    
    initialEntries = np.arange(81)[new_puzzle > 0]
    
    for n in range(len(chromosome)):
        block_indices = get_block_indices(n,initialEntries)
        for m in range(9):
            new_indice = block_indices[m]
            new_puzzle[new_indice] =  chromosome[n][m]
    
    fitness = 0
    
    for row in range(9): # por cada fila obtiene la cantidad de numeros diferentes
        fitness += len(set(new_puzzle[get_row_indices(row, type="row index")]))
    for col in range(9): # por cada columna obtiene la cantidad de numeros diferentes
        fitness += len(set(new_puzzle[get_column_indices(col,type="column index")]))
    return fitness

## METODO PARA CREAR UN ARRAY PUZZLE DESDE UN CROMOSOMA
def create_puzzle(chromosome,puzzle):
    """CREAR UN ARRAY PUZZLE DESDE CHROMOSONA"""
    new_puzzle = deepcopy(puzzle)
    
    initialEntries = np.arange(81)[new_puzzle > 0]
    
    for n in range(len(chromosome)):
        block_indices = get_block_indices(n,initialEntries)
        for m in range(9):
            new_indice = block_indices[m]
            new_puzzle[new_indice] =  chromosome[n][m]
            
    return new_puzzle

## METODO PARA EVALUAR EL FITNESS DE LA POBLACION
def evaluate_population(population,fitness_fn,puzzle): 
    """Retorna el fitness de un cromosoma como el cuadrado del numero recibido en binario"""
   
    for i in range(len(population)):
        if population[i].fitness == -1:    # evalua solo si el individuo no esta evaluado
            population[i].fitness = fitness_fn(population[i].chromosome,puzzle)
    return population

## METODO randomAssign MODIFICADO
def randomAssign_mod(puzzle, initialEntries):
    new_puzzle = deepcopy(puzzle)
    for num in range(9):
        block_indices=get_block_indices(num, initialEntries)
        block= new_puzzle[block_indices]
        zero_indices=[ind for i,ind in enumerate(block_indices) if block[i] == 0]
        to_fill = [i for i in range(1,10) if i not in block]
        shuffle(to_fill)
        for ind, value in zip(zero_indices, to_fill):
            new_puzzle[ind]=value
    return new_puzzle

## METODO PARA CREAR UNA POBLACION DE INDIVIDUOS

def init_population(pop_number,puzzle):
    
    population = []
    initialEntries = np.arange(81)[puzzle > 0]
    for i in range(pop_number):
        new_puzzle = randomAssign_mod(puzzle,initialEntries)
        genes = []
        for num in range(9):
            block_indices = get_block_indices(num, initialEntries)
            block= new_puzzle[block_indices]
            genes.append(block)
        new_chromosome = genes
        population.append( Individual(new_chromosome,puzzle) )
    return population

## METODO make_neighBoard MODIFICADO

def make_neighborBoard_mod(puzzle, initialEntries):
    new_data = deepcopy(puzzle)
    block = randint(0,8)  # escoje un bloque aleatoriamente
    num_in_block = len(get_block_indices(block,initialEntries,ignore_originals=True)) #cantidad de ´posiciones que se puede mover en el bloque 
    random_squares = sample(range(num_in_block),2)
    square1, square2 = [get_block_indices(block,initialEntries,ignore_originals=True)[ind] for ind in random_squares]
    new_data[square1], new_data[square2] = new_data[square2], new_data[square1]
    return new_data, block

## METODO PARA SELECCIONAR SOBREVIVIENTES

def select_survivors(population, offspring_population, numsurvivors):
    next_population = []
    population.extend(offspring_population) # une las dos poblaciones
    isurvivors = sorted(range(len(population)), key=lambda i: population[i].fitness, reverse=True)[:numsurvivors]
    for i in range(numsurvivors): next_population.append(population[isurvivors[i]])
    return next_population


def sa_solver(puzzle, strParameters):
	""" Simulating annealing solver. 
		puzzle: is a np array of 81 elements. The first 9 are the first row of the puzzle, the next 9 are the second row ...    
		strParameters: a string of comma separated parameter=value pairs. Parameters can be:
				T0: Initial temperatura
				DR: The decay rate of the schedule function: Ti = T0*(DR)^i (Ti is the temperature at iteration i). For efficiecy it is calculated as Ti = T(i-1)*DR
				maxIter: The maximum number of iterations
	"""
	import shlex
	parameters = {'T0': .5,	'DR': .99999, 'maxIter': 100000} # Dictionary of parameters with default values
	parms_passed = dict(token.split('=') for token in shlex.split(strParameters.replace(',',' '))) # get the parameters from the parameter string into a dictionary
	parameters.update(parms_passed)  # Update  parameters with the passed values
	
	start_time = time.time()
	print ('Simulated Annealing intentará resolver el siguiente puzzle: ')
	showPuzzle(puzzle)
	
	initialEntries = np.arange(81)[puzzle > 0]  # las posiciones no vacias del puzzle
	randomAssign(puzzle, initialEntries)  # En cada box del puzzle asigna numeros aleatorios en pociciones vacias, garantizando que sean los 9 numeros diferentes 
	best_puzzle = deepcopy(puzzle)
	current_score = score_board(puzzle)
	best_score = current_score
	T = float(parameters['T0'])  # El valor inicial de la temperatura
	DR = float(parameters['DR']) # El factor de decaimiento de la temperatura
	maxIter = int(parameters['maxIter']) # El maximo numero de iteraciones
	t = 0
	while (t < maxIter):
		try:
			if (t % 10000 == 0): 
				print('Iteration {},\tTemperaure = {},\tBest score = {},\tCurrent score = {}'.format(t, T, best_score, current_score))
			neighborBoard = make_neighborBoard(puzzle, initialEntries)
			neighborBoardScore = score_board(neighborBoard)
			delta = float(current_score - neighborBoardScore)
			if (exp((delta/T)) - rnd() > 0):
				puzzle = neighborBoard
				current_score = neighborBoardScore 
			if (current_score < best_score):
				best_puzzle = deepcopy(puzzle)
				best_score = score_board(best_puzzle)
			if neighborBoardScore == -162:   # -162 es el score optimo
				puzzle = neighborBoard
				break
			T = DR*T
			t += 1
		except:
			print("Numerical error occurred. It's a random algorithm so try again.")         
	end_time = time.time() 
	if best_score == -162:
		print ("Solution:")
		showPuzzle(puzzle)
		print ("It took {} seconds to solve this puzzle.".format(end_time - start_time))
	else:
		print("Couldn't solve! ({}/{} points). It's a random algorithm so try again.".format(best_score,-162))
		
def ga_solver(puzzle, strParameters):
	
	import shlex
	parameters = {'w': 10,	'Cx': "single", 'm': 0.1,'maxGener': 10000} # Dictionary of parameters with default values
	parms_passed = dict(token.split('=') for token in shlex.split(strParameters.replace(',',' '))) # get the parameters from the parameter string into a dictionary
	parameters.update(parms_passed)  # Update  parameters with the passed values

	start_time = time.time()
	print ('Genetic Algorithm for this puzzle: ')
	showPuzzle(puzzle)

	# NUESTRO ALGORITMO GENETICO
	# PARAMETROS A COLOCAR:
	# PUZZLE = puzzle
	# POPULATION SIZE = w
	# CRUZAMIENTO = SINGLE O UNIFOMR
	# RATIO DE MUTACION = m
	# MAX_GEN = maximo numero de iteraciones

	#Parametros a colocar:

	num_individuals = int(parameters['w'])
	crossover = str(parameters['Cx'])
	pmut = float(parameters['m'])
	MAX_GEN = int(parameters['maxGener'])

	# parametros que usaremos dentro de nuestra funcion:
	# Ya se les asigna valores
	size_torneo = int(1*num_individuals/10)
	metodoSeleccion = "roulette"
	fitness_fn = score_fun_puzzle

	# Inicializa una poblacion inicial de forma aleatoria
	population = init_population(num_individuals,puzzle)

	popsize = len(population)
	evaluate_population(population,score_fun_puzzle,puzzle)
	# display(population) #Imprime la primera poblacion 
	ibest = sorted(range(len(population)), key=lambda i:population[i].fitness,reverse=True)[:1] # mejor individuo
	bestfitness = [population[ibest[0]].fitness] #mejor fitness del mejor individuo
	print("Poblacion inicial, best_fitness = {}".format(population[ibest[0]].fitness))

	buffer1 = 0
	buffer2 = 0

	for g in range(MAX_GEN): # para cada iteracion
	
		# 1. Seleccion de parejas de padres para cruzamiento
		mating_pool=[]
		if metodoSeleccion == 'roulette':
			for i in range(int(popsize/2)): mating_pool.append(select_parents_roulette(population))
		elif metodoSeleccion == 'torneo':
			for i in range(int(popsize/2)): mating_pool.append(select_parents_torneo(population,size_torneo))
				
		# 2. Creacion de poblacion descendencia cruzando parejas de mating_pool
		offspring_population = []
		for i in range(len(mating_pool)):
			if crossover == "single":
				offspring_population.extend(mating_pool[i][0].crossover_onepoint(mating_pool[i][1])) # cruzamiento un punto
			elif crossover == "uniform":
				offspring_population.extend(mating_pool[i][0].crossover_uniform(mating_pool[i][1])) # cruzamiento uniforme
				
		# 3. Operador de mutacion con probabilidad pmut en cada hijo generado
		for i in range(len(offspring_population)):
			if rnd.uniform(0,1) < pmut:
				offspring_population[i] = offspring_population[i].mutate_position()   #mutacion de posicion
		
		# 4. Evaluacion de poblacion descendencia creada
		evaluate_population(offspring_population,fitness_fn,puzzle) #evalua la poblacion descendencia
		
		# 5. Seleccion de popsize individuos para la siguiente generacion de la union de la poblacion actual y pob descendencia
		population = select_survivors(population, offspring_population, popsize)
		
		# 6. Almacena la historia del fitness del mejor individuo
		ibest = sorted(range(len(population)),key=lambda i:population[i].fitness, reverse=True)[:1]
		bestfitness.append(population[ibest[0]].fitness)
		
		# VISUALIZACION DE FITNESS
		if g % 200 == 0:
			print("generacion {}, fitness = {}".format(g,population[ibest[0]].fitness))

		
		if g % 800 == 0:
			buffer2 = population[ibest[0]].fitness

			if buffer2 > buffer1:
				buffer1 = buffer2
			else:
				break
			
		
	showPuzzle(create_puzzle(population[ibest[0]].chromosome,puzzle))
	end_time = time.time()

	print ("It took {} seconds to solve this puzzle.".format(end_time - start_time))

	"""
	if population[ibest[0]].fitness == 162:
		print ("Solution:")
		showPuzzle(puzzle)
		print ("It took {} seconds to solve this puzzle.".format(end_time - start_time))
	else:
		print("Couldn't solve! ({}/{} points). It's a random algorithm so try again.".format(population[ibest[0]].fitness,162))
	"""
	
	""" Genetic Algorithm solver. 
		puzzle: is a np array of 81 elements. The first 9 are the first row of the puzzle, the next 9 are the second row ...    
		strParameters: a string of comma separated parameter=value pairs. Parameters can be:
				w: Population size
				Cx: Crossover ( single  or uniform )
				m: Mutation rate
				maxGener: The maximum number of generations
	"""
	

def default(str):
    return str + ' [Default: %default]'

def readCommand( argv ):
	"""
	Processes the arguments  used to run sudokusolver from the command line.
	"""
	from optparse import OptionParser
	usageStr = """
	USAGE:      python sudokusolver.py <options>
	EXAMPLES:   (1) python sudokusolver.py -p my_puzzle.txt -s sa -a T0=0.5,DR=0.9999,maxIter=100000
	"""
	parser = OptionParser(usageStr)
	parser.add_option('-p', '--puzzle', dest='puzzle', help=default('the puzzle filename'), default=None)
	parser.add_option('-s', '--solver', dest='solver', help=default('name of the solver (sa or ga)'), default='sa')
	parser.add_option('-a', '--solverParams', dest='solverParams', help=default('Comma separated pairs parameter=value to the solver. e.g. (for sa): "T0=0.5,DR=0.9999,nIter=100000"'))
	
	options, otherjunk = parser.parse_args(argv)
	if len(otherjunk) != 0:
		raise Exception('Command line input not understood: ' + str(otherjunk))
	args = dict()
	
	fd = open(options.puzzle,"r+")    # Read the Puzzle file
	puzzle = eval(fd.readline())
	array = []
	for row in puzzle:
		for col in row:
			array.append(col)
	args['puzzle'] = np.array(array)  # puzzle es un vector con todas las filas del puzzle concatenadas (vacios tiene valor 0) 
	args['solver'] = options.solver
	args['solverParams'] =  options.solverParams
	return args	

if __name__=="__main__":
	"""
	The main function called when sudokusolver.py is run from the command line:
	> python sudokusolver.py

	See the usage string for more details.

	> python sudokusolver.py --help
    """
	args = readCommand( sys.argv[1:] ) # Get the arguments from the command line input
	solvers = {'sa': sa_solver,	'ga': ga_solver }  # Dictionary of available solvers
	
	solvers[args['solver']]( args['puzzle'], args['solverParams'] )  # Call the solver method passing the string of parameters
	
	pass