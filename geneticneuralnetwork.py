from bridge import Bridge
import random
import copy

def randomPositiveNegitive():
    return 1 if random.random() < 0.5 else -1

class Population:
    def __init__(self, populationSize=100, startingGenome=None):
        self.populationSize = populationSize
        self.winnerGeneration = 0
        self.highestFitness = 0.0
        self.highestLastChanged = 0
        self.organisms = []
        self.species = []

        self.currentNodeId = 0
        self.currentInnovationNumber = 0

        if startingGenome == None:
            startingGenome = Genome()
        self.spawn(startingGenome, populationSize)


        self.species = []
        self.generation = 0
        self.innovation = 4 # TODO: Change this to be the number of outputs in the NN
        self.currentSpecies = 0
        self.currentGenome = 0
        self.currentFrame = 0
        self.maxFitness = 0

    def spawn(self, genome, populationSize):
        for organism in range(0, populationSize):
            print("Creating Organism: " + str(organism))
            newGenome = copy.deepcopy(genome) #TODO: Check to see if this is a true copy or just reference
            newGenome.linkWeightMutate()
            newOrganism = Organism(0.0, newGenome, 1)
            self.organisms.append(newOrganism)

        # self.currentNodeId #TODO: Set current Node Id
        # self.currentInnovationNumber #TODO: Set current Innovation Number

        self.speciate()
            
    def speciate(self):
        speciesCount = 1
        for i in range(0, len(self.organisms)):
            if len(self.species) == 0:
                newSpecies = Species(speciesCount)
                speciesCount += 1
                self.species.append(newSpecies)
                #TODO: Add organism to species ???? Dive deep here
                self.organisms[i].species = newSpecies
            else:
                for x in range(i + 1, len(self.species)):
                    self.organisms[i].genome.compatibility(self.genome


class Organism:

    def __init__(self, fitness, genome, generationNumber):
        self.fitness = fitness
        self.originalFitness = fitness
        self.genome = genome
        self.network = None # TODO: Set this value
        self.species = None
        self.expectedOffspring = 0
        self.generation = generationNumber
        self.eliminate = False
        self.error = 0
        self.winner = False
        self.champion = False
        self.superChampionOffspring = 0

        self.timeAlive = 0

        self.populationChamp = False
        self.populationChampChild = False
        self.highFit = 0
        self.mutateStructureBaby = 0
        self.mateBaby = 0

        self.modified = True

class Species:
    def __init__(self, idVal):
        self.id = idVal
        self.age = 1
        averageFitness = 0.0
        expectedOffspring = 0
        novel = False
        ageOfLastImprovement = 0
        maxFitness = 0
        maxFitnessEver = 0
        obliterate = False

        averageEstamte = 0


        self.topFitness = 0
        self.staleness = 0
        self.genomes = []
        self.averageFitness = 0

        # Used to see how close species are
        self.deltaDisjoint = 2.0
        self.deltaWeights = 0.4
        self.deltaThreshold = 1.0

    def sameSpecies(self, genome1, genome2):
        dd = self.deltaDisjoint*self.disjoint(genome1.genes, genome2.genes)
        dw = self.deltaWeights*self.weights(genome1.genes, genome2.genes) 
        return dd + dw < self.deltaThreshold

    def disjoint(self, genes1, genes2):
        i1 = []
        for i in range(0, len(genes1)):
            gene = genes1[i]
            i1[gene.innovation] = True

        i2 = []
        for i in range(0,len(genes2)):
            gene = genes2[i]
            i2[gene.innovation] = True

        disjointGenes = 0
        for i in range(0, len(genes1)):
            gene = genes1[i]
            if not i2[gene.innovation]:
        	disjointGenes += 1

        for i in range(0,len(genes2)):
            gene = genes2[i]
            if not i1[gene.innovation]:
                disjointGenes += 1

        n = max(len(genes1), len(genes2))

        print("Max Number Of Genes: " + str(n))
        if n == 0:
            n=10000000 #TODO: Ajust this value if needed
        return disjointGenes / n

    def weights(self, genes1, genes2):
        i2 = []
        for i in range(0, len(genes2)):
            gene = genes2[i]
            i2[gene.innovation] = gene
        
        sum = 0
        coincident = 0
        for i in range(0, len(genes1)):
            gene = genes1[i]
            if i2[gene.innovation] != None:
                gene2 = i2[gene.innovation]
                sum = sum + math.abs(gene.weight - gene2.weight)
                coincident = coincident + 1

        if coincident == 0:
            coincident=10000000 #TODO: Ajust this value if needed
        return sum / coincident

class Genome:
    def __init__(self):
        self.genes = []
        self.fitness = 0
        self.adjustedFitness = 0
        self.network = []
        self.maxneuron = 0
        self.globalRank = 0

        self.mutateConnectionsChance = 0.25
        self.linkMutationChance = 2.0
        self.biasMutationChance = 0.40
        self.nodeMutationChance = 0.50
        self.disableMutationChance = 0.4
        self.enableMutationChance = 0.2
        self.stepSize = 0.1
        PerturbChance = 0.90

        self.mutationRates = {}
        self.mutationRates["connections"] = self.mutateConnectionsChance
        self.mutationRates["link"] = self.linkMutationChance
        self.mutationRates["bias"] = self.biasMutationChance
        self.mutationRates["node"] = self.nodeMutationChance
        self.mutationRates["enable"] = self.enableMutationChance
        self.mutationRates["disable"] = self.disableMutationChance
        self.mutationRates["step"] = self.stepSize

    def compatibility(self, genome):
        excessNumber = 0
        disjoingNumber = 0
        matchNumber = 0

        genes1end = len(self.genes)
        genes2end = len(genome.genes)

        maxGenomeSize = max(genes1end, genes2end)

        genes1current = 0
        genes2current = 0
        while genes1current < genes1end and genes2current < genes2end:
            if genes1current == genes1end-1:
                genes2current += 1
                excessNumber += 1
            elif genes2current == genes2end-1:
                genes1current += 1
                excessNumber += 1
            else:
                genes1InnovationNumber = self.genes[genes1current].innocationNumber
                genes2InnovationNumber = genome.genes[genes2current].innocationNumber

                if genes1InnovationNumber == genes2InnovationNumber:
                    matchNumber += 1
                    mutationDifference = self.genes[genes1current].mutationNumber - genome.genes[genes2current].mutationNumber
                    mutationDifference = abs(mutationDifference)

                    mutationDifferenceTotal += mutationDifference
                    genes1current += 1
                    genes2current += 1

                elif genes1InnovationNumber < genes2InnovationNumber:
                    genes1current += 1
                    disjointNumber += 1

                elif genes2InnovationNumber < genes1InnovationNumber:
                    genes2current += 1
                    disjointNumber += 1

        disjointCoefficient = 1.0
        excessCoefficient = 1.0
        mutationDifferenceCoefficient = 0.4

        return  (disjointCoefficient*disjointNumber + excessCoefficient*excessNumber + mutationDifferenceCoefficent*mutationDifferenceTotal)

    def mutate(self, genome):
        for mutation, rate in genome.mutationRates.iteritems():
            if random.randrange(0, 2, 1) == 1:
                genome.mutationRates[mutation] = 0.95*rate
            else:
                genome.mutationRates[mutation] = 1.05263*rate
        
        if random.random() < genome.mutationRates["connections"]:
            self.pointMutate(genome)
        
        p = genome.mutationRates["link"]
        while p > 0:
            if random.random() < p:
                self.linkMutate(genome, False)
            p = p - 1
        
        p = genome.mutationRates["bias"]
        while p > 0:
            if random.random() < p:
                self.linkMutate(genome, True)
            p = p - 1
        
        p = genome.mutationRates["node"]
        while p > 0:
            if random.random() < p:
                self.nodeMutate(genome)
            p = p - 1
        
        p = genome.mutationRates["enable"]
        while p > 0:
            if random.random() < p:
                self.enableDisableMutate(genome, True)
            p = p - 1

        p = genome.mutationRates["disable"]
        while p > 0:
            if random.random() < p:
                self.enableDisableMutate(genome, False)
            p = p - 1

    def pointMutate(self, genome):
        step = genome.mutationRates["step"]

        for i in range(0, len(genome.genes)):
            gene = genome.genes[i]
            if random.random() < PerturbChance:
                gene.weight = gene.weight + random.random() * step*2 - step
            else:
                gene.weight = random.random()*4-2
        
    def linkWeightMutate(self):
        power = 1.0
        for gene in range(0, len(self.genes)):
            weight = randomPositiveNegitive()*random.random()*power
            gene.link.weight = weight

            #TODO: Add Gaussian mutation not just coldgaussian
            #TODO: Cap weights??

    def nodeMutate(self, genome):
        if len(genome.genes) == 0:
            return
        # 	end
        # 
        # 	genome.maxneuron = genome.maxneuron + 1
        # 
        # 	local gene = genome.genes[math.random(1,#genome.genes)]
        # 	if not gene.enabled then
        # 		return
        # 	end
        # 	gene.enabled = false
        # 	
        # 	local gene1 = copyGene(gene)
        # 	gene1.out = genome.maxneuron
        # 	gene1.weight = 1.0
        # 	gene1.innovation = newInnovation()
        # 	gene1.enabled = true
        # 	table.insert(genome.genes, gene1)
        # 	
        # 	local gene2 = copyGene(gene)
        # 	gene2.into = genome.maxneuron
        # 	gene2.innovation = newInnovation()
        # 	gene2.enabled = true
        # 	table.insert(genome.genes, gene2)
        # end
        # 
    def enableDisableMutate(self, genome, enable):
        candidates = {}
        # 	for _,gene in pairs(genome.genes) do
        # 		if gene.enabled == not enable then
        # 			table.insert(candidates, gene)
        # 		end
        # 	end
        # 	
        # 	if #candidates == 0 then
        # 		return
        # 	end
        # 	
        # 	local gene = candidates[math.random(1,#candidates)]
        # 	gene.enabled = not gene.enabled
        # end

    def randomNeuron(self, genes, nonInput, inputsSize, outputsSize): # TODO: Put this in the network class???
        neurons = []
        if not nonInput:
            for i in range(0, inputSize):
                neurons[i] = True

        for o in range(0, outputsSize):
           neurons[MaxNodes+o] = True

        for i in range(0, len(genes)):
            if (not nonInput) or genes[i].into > inputsSize:
         	neurons[genes[i].into] = True

            if (not nonInput) or genes[i].out > inputsSize:
         	neurons[genes[i].out] = True
         
        count = 0
        for neuron in neurons:
            if neuron != None:
                count = count + 1

        n = random.random(1, count, 1) #TODO: is the actually working correctly???
         	
        for k,v in neurons:
            if neuron != None:
                n = n-1
            if n == 0:
               return k

        return 0

class Gene:
    def __init__(self):
        self.link = Link()
        self.innovationNumber = 0
        self.mutaionNumber = 0
        self.enabled = True

class Link:
    def __init__(self):
        self.weight = 0.0
        self.into = Node()
        self.out = Node()

class Node:
    def __init__(self):
        self.temp = 0


class Network:
    def __init__(self, inputs, outputs):
        self.maxNodes = 1000000

        self.inputs = inputs + 1
        self.outputs = outputs
        self.neurons = [None] * (self.maxNodes * self.outputs)

    def generateNetwork(self, genome):
        network = Network(self.inputs, self.outputs)

        for i in range(0, self.inputs):
            network.neurons[i] = Neuron()

        for o in range(0, self.outputs):
            network.neurons[self.maxNodes+o] = Neuron()

        genome.genes.sort(key=lambda x: x.out, reverse=True)
        print("Genome with size: " + str(len(genome.genes)) + ", Sorted genes: " + ''.join(genome.genes))

        for i in range(0, len(genome.genes)):
            gene = genome.genes[i]
            if gene.enabled:
                if network.neurons[gene.out] == None:
                    network.neurons[gene.out] = Neuron()

                neuron = network.neurons[gene.out]
                neuron.incoming.append(gene)
                if network.neurons[gene.into] == None:
                    network.neurons[gene.into] = Neuron()
        
        #genome.network = network
        return network

    def evaluateNetwork(inputs):
        inputs.append(1) #Bias
        if len(inputs) != self.inputs:
            print("Incorrect number of neural network inputs.")
            return []
    	
        for i in range(0, self.inputs):
            self.neurons[i].value = inputs[i]
    	
        for neuron in self.neurons:
            sum = 0
            for j in range(0, len(neuron.incoming)):
                incoming = neuron.incoming[j]
                other = self.neurons[incoming.into]
                sum = sum + incoming.weight * other.value
    		
            if len(neuron.incoming) > 0:
                neuron.value = sigmoid(sum)
    	
    	outputs = []
        for o in range(0, self.outputs):
            button = "key:" + ButtonNames[o]
            if network.neurons[MaxNodes+o].value > 0:
                outputs[button] = True
            else:
                outputs[button] = False

        return outputs

class Neuron:
    def __init__(self):
        self.incoming = []
        self.value = 0.0

class NEAT:
    def __init__(self):
        # Nerual Network paramaters
        self.nn_inputs = 1000 #TODO: Change this to the screen size
        self.nn_outputs = 4 #TODO: Set the output to be equal to the inputs of the game
        self.network = Network(self.nn_inputs, self.nn_outputs)

        # Set Genetic Algorithm Paramaters
        self.ga_population = 300

        # 
        # StaleSpecies = 15
        # 
        # CrossoverChance = 0.75
        # 
        self.timeoutConstant = 20
        # 
        self.population = None

        #self.bridge = Bridge()
        #self.bridge.connectToSocket()
        #self.getInputs()
        #self.bridge.sendAndForget("close:connection")


        if self.population == None:
            self.population = Population(self.ga_population)

    def getRomName(self):
        return "Ms. Pac-Man (U) [!]"

    def getInputs(self):
        self.inputs = ["up", "down", "left", "right"]
        return self.inputs
    
    def getNetworkInputs(self):
        return self.bridge.getScreen()

    def sigmoid(self, x):
        return 2/(1+math.exp(-4.9*x))-1

    def newInnovation(self):
        self.pool.innovation = self.pool.innovation + 1
        return self.pool.innovation


        # function copyGenome(genome)
        # 	local genome2 = newGenome()
        # 	for g=1,#genome.genes do
        # 		table.insert(genome2.genes, copyGene(genome.genes[g]))
        # 	end
        # 	genome2.maxneuron = genome.maxneuron
        # 	genome2.mutationRates["connections"] = genome.mutationRates["connections"]
        # 	genome2.mutationRates["link"] = genome.mutationRates["link"]
        # 	genome2.mutationRates["bias"] = genome.mutationRates["bias"]
        # 	genome2.mutationRates["node"] = genome.mutationRates["node"]
        # 	genome2.mutationRates["enable"] = genome.mutationRates["enable"]
        # 	genome2.mutationRates["disable"] = genome.mutationRates["disable"]
        # 	
        # 	return genome2
        # end

    def basicGenome(self):
        genome = Genome()
        innovation = 1

        genome.maxneuron = self.nn_inputs
        genome.mutate(genome)
        
        return genome

        # function copyGene(gene)
        # 	local gene2 = newGene()
        # 	gene2.into = gene.into
        # 	gene2.out = gene.out
        # 	gene2.weight = gene.weight
        # 	gene2.enabled = gene.enabled
        # 	gene2.innovation = gene.innovation
        # 	
        # 	return gene2
        # end
        # 

        # 
        # function crossover(g1, g2)
        # 	-- Make sure g1 is the higher fitness genome
        # 	if g2.fitness > g1.fitness then
        # 		tempg = g1
        # 		g1 = g2
        # 		g2 = tempg
        # 	end
        # 
        # 	local child = newGenome()
        # 	
        # 	local innovations2 = {}
        # 	for i=1,#g2.genes do
        # 		local gene = g2.genes[i]
        # 		innovations2[gene.innovation] = gene
        # 	end
        # 	
        # 	for i=1,#g1.genes do
        # 		local gene1 = g1.genes[i]
        # 		local gene2 = innovations2[gene1.innovation]
        # 		if gene2 ~= nil and math.random(2) == 1 and gene2.enabled then
        # 			table.insert(child.genes, copyGene(gene2))
        # 		else
        # 			table.insert(child.genes, copyGene(gene1))
        # 		end
        # 	end
        # 	
        # 	child.maxneuron = math.max(g1.maxneuron,g2.maxneuron)
        # 	
        # 	for mutation,rate in pairs(g1.mutationRates) do
        # 		child.mutationRates[mutation] = rate
        # 	end
        # 	
        # 	return child
        # end
        # 
        # function containsLink(genes, link)
        # 	for i=1,#genes do
        # 		local gene = genes[i]
        # 		if gene.into == link.into and gene.out == link.out then
        # 			return true
        # 		end
        # 	end
        # end
        # 
        # 
        # 
        # 
        # function rankGlobally()
        # 	local global = {}
        # 	for s = 1,#pool.species do
        # 		local species = pool.species[s]
        # 		for g = 1,#species.genomes do
        # 			table.insert(global, species.genomes[g])
        # 		end
        # 	end
        # 	table.sort(global, function (a,b)
        # 		return (a.fitness < b.fitness)
        # 	end)
        # 	
        # 	for g=1,#global do
        # 		global[g].globalRank = g
        # 	end
        # end
        # 
        # function calculateAverageFitness(species)
        # 	local total = 0
        # 	
        # 	for g=1,#species.genomes do
        # 		local genome = species.genomes[g]
        # 		total = total + genome.globalRank
        # 	end
        # 	
        # 	species.averageFitness = total / #species.genomes
        # end
        # 
        # function totalAverageFitness()
        # 	local total = 0
        # 	for s = 1,#pool.species do
        # 		local species = pool.species[s]
        # 		total = total + species.averageFitness
        # 	end
        # 
        # 	return total
        # end
        # 
        # function cullSpecies(cutToOne)
        # 	for s = 1,#pool.species do
        # 		local species = pool.species[s]
        # 		
        # 		table.sort(species.genomes, function (a,b)
        # 			return (a.fitness > b.fitness)
        # 		end)
        # 		
        # 		local remaining = math.ceil(#species.genomes/2)
        # 		if cutToOne then
        # 			remaining = 1
        # 		end
        # 		while #species.genomes > remaining do
        # 			table.remove(species.genomes)
        # 		end
        # 	end
        # end
        # 
        # function breedChild(species)
        # 	local child = {}
        # 	if math.random() < CrossoverChance then
        # 		g1 = species.genomes[math.random(1, #species.genomes)]
        # 		g2 = species.genomes[math.random(1, #species.genomes)]
        # 		child = crossover(g1, g2)
        # 	else
        # 		g = species.genomes[math.random(1, #species.genomes)]
        # 		child = copyGenome(g)
        # 	end
        # 	
        # 	mutate(child)
        # 	
        # 	return child
        # end
        # 
        # function removeStaleSpecies()
        # 	local survived = {}
        # 
        # 	for s = 1,#pool.species do
        # 		local species = pool.species[s]
        # 		
        # 		table.sort(species.genomes, function (a,b)
        # 			return (a.fitness > b.fitness)
        # 		end)
        # 		
        # 		if species.genomes[1].fitness > species.topFitness then
        # 			species.topFitness = species.genomes[1].fitness
        # 			species.staleness = 0
        # 		else
        # 			species.staleness = species.staleness + 1
        # 		end
        # 		if species.staleness < StaleSpecies or species.topFitness >= pool.maxFitness then
        # 			table.insert(survived, species)
        # 		end
        # 	end
        # 
        # 	pool.species = survived
        # end
        # 
        # function removeWeakSpecies()
        # 	local survived = {}
        # 
        # 	local sum = totalAverageFitness()
        # 	for s = 1,#pool.species do
        # 		local species = pool.species[s]
        # 		breed = math.floor(species.averageFitness / sum * Population)
        # 		if breed >= 1 then
        # 			table.insert(survived, species)
        # 		end
        # 	end
        # 
        # 	pool.species = survived
        # end
        # 
        # 
    def addToSpecies(self, child):
     	foundSpecies = False
        print("Species Count: " + str(len(self.pool.species)))
        for s in range(0, len(self.pool.species)):
     	    species = self.pool.species[s]
            print("Genomes Count: " + str(len(species.genomes)))
            if not foundSpecies and species.sameSpecies(child, species.genomes[0]):
     		self.pool.species[s].genomes.append(child)
     		foundSpecies = True

        if not foundSpecies:
            print("No Species OR Species NOT Found")
     	    childSpecies = Species()
     	    childSpecies.genomes.append(child)
     	    self.pool.species.append(childSpecies)
        # 
        # function newGeneration()
        # 	cullSpecies(false) -- Cull the bottom half of each species
        # 	rankGlobally()
        # 	removeStaleSpecies()
        # 	rankGlobally()
        # 	for s = 1,#pool.species do
        # 		local species = pool.species[s]
        # 		calculateAverageFitness(species)
        # 	end
        # 	removeWeakSpecies()
        # 	local sum = totalAverageFitness()
        # 	local children = {}
        # 	for s = 1,#pool.species do
        # 		local species = pool.species[s]
        # 		breed = math.floor(species.averageFitness / sum * Population) - 1
        # 		for i=1,breed do
        # 			table.insert(children, breedChild(species))
        # 		end
        # 	end
        # 	cullSpecies(true) -- Cull all but the top member of each species
        # 	while #children + #pool.species < Population do
        # 		local species = pool.species[math.random(1, #pool.species)]
        # 		table.insert(children, breedChild(species))
        # 	end
        # 	for c=1,#children do
        # 		local child = children[c]
        # 		addToSpecies(child)
        # 	end
        # 	
        # 	pool.generation = pool.generation + 1
        # 	
        # 	writeFile("backup." .. pool.generation .. "." .. forms.gettext(saveLoadFile))
        # end
        # 	
    def initializePool(self):
        self.pool = Pool()

        for i in range(0,self.ga_population):
            basic = self.basicGenome()
            self.addToSpecies(basic)

        self.initializeRun()
# 
# function clearJoypad()
# 	controller = {}
# 	for b = 1,#ButtonNames do
# 		controller["P1 " .. ButtonNames[b]] = false
# 	end
# 	joypad.set(controller)
# end
# 
    def initializeRun(self):
        self.pool.currentFrame = 0
        self.timeout = self.timeoutConstant

        species = self.pool.species[self.pool.currentSpecies]
        genome = species.genomes[self.pool.currentGenome]
        genome.network = self.network.generateNetwork(genome)
        evaluateCurrent()

    def evaluateCurrent(self):
 	species = self.pool.species[self.pool.currentSpecies]
 	genome = species.genomes[self.pool.currentGenome]
 
 	inputs = getInputs()
 	controller = evaluateNetwork(genome.network, inputs)
 	
 	# if controller["P1 Left"] and controller["P1 Right"] then
 	# 	controller["P1 Left"] = false
 	# 	controller["P1 Right"] = false
 	# if controller["P1 Up"] and controller["P1 Down"] then
 	# 	controller["P1 Up"] = false
 	# 	controller["P1 Down"] = false
 
        # joypad.set(controller) #TODO: Send signal to emulator
# 
# if pool == nil then
# 	initializePool()
# end
# 
# 
# function nextGenome()
# 	pool.currentGenome = pool.currentGenome + 1
# 	if pool.currentGenome > #pool.species[pool.currentSpecies].genomes then
# 		pool.currentGenome = 1
# 		pool.currentSpecies = pool.currentSpecies+1
# 		if pool.currentSpecies > #pool.species then
# 			newGeneration()
# 			pool.currentSpecies = 1
# 		end
# 	end
# end
# 
# function fitnessAlreadyMeasured()
# 	local species = pool.species[pool.currentSpecies]
# 	local genome = species.genomes[pool.currentGenome]
# 	
# 	return genome.fitness ~= 0
# end
# 
# function displayGenome(genome)
# 	local network = genome.network
# 	local cells = {}
# 	local i = 1
# 	local cell = {}
# 	for dy=-BoxRadius,BoxRadius do
# 		for dx=-BoxRadius,BoxRadius do
# 			cell = {}
# 			cell.x = 50+5*dx
# 			cell.y = 70+5*dy
# 			cell.value = network.neurons[i].value
# 			cells[i] = cell
# 			i = i + 1
# 		end
# 	end
# 	local biasCell = {}
# 	biasCell.x = 80
# 	biasCell.y = 110
# 	biasCell.value = network.neurons[Inputs].value
# 	cells[Inputs] = biasCell
# 	
# 	for o = 1,Outputs do
# 		cell = {}
# 		cell.x = 220
# 		cell.y = 30 + 8 * o
# 		cell.value = network.neurons[MaxNodes + o].value
# 		cells[MaxNodes+o] = cell
# 		local color
# 		if cell.value > 0 then
# 			color = 0xFF0000FF
# 		else
# 			color = 0xFF000000
# 		end
# 		gui.drawText(223, 24+8*o, ButtonNames[o], color, 9)
# 	end
# 	
# 	for n,neuron in pairs(network.neurons) do
# 		cell = {}
# 		if n > Inputs and n <= MaxNodes then
# 			cell.x = 140
# 			cell.y = 40
# 			cell.value = neuron.value
# 			cells[n] = cell
# 		end
# 	end
# 	
# 	for n=1,4 do
# 		for _,gene in pairs(genome.genes) do
# 			if gene.enabled then
# 				local c1 = cells[gene.into]
# 				local c2 = cells[gene.out]
# 				if gene.into > Inputs and gene.into <= MaxNodes then
# 					c1.x = 0.75*c1.x + 0.25*c2.x
# 					if c1.x >= c2.x then
# 						c1.x = c1.x - 40
# 					end
# 					if c1.x < 90 then
# 						c1.x = 90
# 					end
# 					
# 					if c1.x > 220 then
# 						c1.x = 220
# 					end
# 					c1.y = 0.75*c1.y + 0.25*c2.y
# 					
# 				end
# 				if gene.out > Inputs and gene.out <= MaxNodes then
# 					c2.x = 0.25*c1.x + 0.75*c2.x
# 					if c1.x >= c2.x then
# 						c2.x = c2.x + 40
# 					end
# 					if c2.x < 90 then
# 						c2.x = 90
# 					end
# 					if c2.x > 220 then
# 						c2.x = 220
# 					end
# 					c2.y = 0.25*c1.y + 0.75*c2.y
# 				end
# 			end
# 		end
# 	end
# 	
# 	gui.drawBox(50-BoxRadius*5-3,70-BoxRadius*5-3,50+BoxRadius*5+2,70+BoxRadius*5+2,0xFF000000, 0x80808080)
# 	for n,cell in pairs(cells) do
# 		if n > Inputs or cell.value ~= 0 then
# 			local color = math.floor((cell.value+1)/2*256)
# 			if color > 255 then color = 255 end
# 			if color < 0 then color = 0 end
# 			local opacity = 0xFF000000
# 			if cell.value == 0 then
# 				opacity = 0x50000000
# 			end
# 			color = opacity + color*0x10000 + color*0x100 + color
# 			gui.drawBox(cell.x-2,cell.y-2,cell.x+2,cell.y+2,opacity,color)
# 		end
# 	end
# 	for _,gene in pairs(genome.genes) do
# 		if gene.enabled then
# 			local c1 = cells[gene.into]
# 			local c2 = cells[gene.out]
# 			local opacity = 0xA0000000
# 			if c1.value == 0 then
# 				opacity = 0x20000000
# 			end
# 			
# 			local color = 0x80-math.floor(math.abs(sigmoid(gene.weight))*0x80)
# 			if gene.weight > 0 then 
# 				color = opacity + 0x8000 + 0x10000*color
# 			else
# 				color = opacity + 0x800000 + 0x100*color
# 			end
# 			gui.drawLine(c1.x+1, c1.y, c2.x-3, c2.y, color)
# 		end
# 	end
# 	
# 	gui.drawBox(49,71,51,78,0x00000000,0x80FF0000)
# 	
# 	if forms.ischecked(showMutationRates) then
# 		local pos = 100
# 		for mutation,rate in pairs(genome.mutationRates) do
# 			gui.drawText(100, pos, mutation .. ": " .. rate, 0xFF000000, 10)
# 			pos = pos + 8
# 		end
# 	end
# end
# 
# function writeFile(filename)
#         local file = io.open(filename, "w")
# 	file:write(pool.generation .. "\n")
# 	file:write(pool.maxFitness .. "\n")
# 	file:write(#pool.species .. "\n")
#         for n,species in pairs(pool.species) do
# 		file:write(species.topFitness .. "\n")
# 		file:write(species.staleness .. "\n")
# 		file:write(#species.genomes .. "\n")
# 		for m,genome in pairs(species.genomes) do
# 			file:write(genome.fitness .. "\n")
# 			file:write(genome.maxneuron .. "\n")
# 			for mutation,rate in pairs(genome.mutationRates) do
# 				file:write(mutation .. "\n")
# 				file:write(rate .. "\n")
# 			end
# 			file:write("done\n")
# 			
# 			file:write(#genome.genes .. "\n")
# 			for l,gene in pairs(genome.genes) do
# 				file:write(gene.into .. " ")
# 				file:write(gene.out .. " ")
# 				file:write(gene.weight .. " ")
# 				file:write(gene.innovation .. " ")
# 				if(gene.enabled) then
# 					file:write("1\n")
# 				else
# 					file:write("0\n")
# 				end
# 			end
# 		end
#         end
#         file:close()
# end
# 
# function savePool()
# 	local filename = forms.gettext(saveLoadFile)
# 	writeFile(filename)
# end
# 
# function loadFile(filename)
#         local file = io.open(filename, "r")
# 	pool = newPool()
# 	pool.generation = file:read("*number")
# 	pool.maxFitness = file:read("*number")
# 	forms.settext(maxFitnessLabel, "Max Fitness: " .. math.floor(pool.maxFitness))
#         local numSpecies = file:read("*number")
#         for s=1,numSpecies do
# 		local species = newSpecies()
# 		table.insert(pool.species, species)
# 		species.topFitness = file:read("*number")
# 		species.staleness = file:read("*number")
# 		local numGenomes = file:read("*number")
# 		for g=1,numGenomes do
# 			local genome = newGenome()
# 			table.insert(species.genomes, genome)
# 			genome.fitness = file:read("*number")
# 			genome.maxneuron = file:read("*number")
# 			local line = file:read("*line")
# 			while line ~= "done" do
# 				genome.mutationRates[line] = file:read("*number")
# 				line = file:read("*line")
# 			end
# 			local numGenes = file:read("*number")
# 			for n=1,numGenes do
# 				local gene = newGene()
# 				table.insert(genome.genes, gene)
# 				local enabled
# 				gene.into, gene.out, gene.weight, gene.innovation, enabled = file:read("*number", "*number", "*number", "*number", "*number")
# 				if enabled == 0 then
# 					gene.enabled = false
# 				else
# 					gene.enabled = true
# 				end
# 				
# 			end
# 		end
# 	end
#         file:close()
# 	
# 	while fitnessAlreadyMeasured() do
# 		nextGenome()
# 	end
# 	initializeRun()
# 	pool.currentFrame = pool.currentFrame + 1
# end
#  
# function loadPool()
# 	local filename = forms.gettext(saveLoadFile)
# 	loadFile(filename)
# end
# 
# function playTop()
# 	local maxfitness = 0
# 	local maxs, maxg
# 	for s,species in pairs(pool.species) do
# 		for g,genome in pairs(species.genomes) do
# 			if genome.fitness > maxfitness then
# 				maxfitness = genome.fitness
# 				maxs = s
# 				maxg = g
# 			end
# 		end
# 	end
# 	
# 	pool.currentSpecies = maxs
# 	pool.currentGenome = maxg
# 	pool.maxFitness = maxfitness
# 	forms.settext(maxFitnessLabel, "Max Fitness: " .. math.floor(pool.maxFitness))
# 	initializeRun()
# 	pool.currentFrame = pool.currentFrame + 1
# 	return
# end
# 
# function onExit()
# 	forms.destroy(form)
# end
# 
# writeFile("temp.pool")
# 
# event.onexit(onExit)
# 
# form = forms.newform(200, 260, "Fitness")
# maxFitnessLabel = forms.label(form, "Max Fitness: " .. math.floor(pool.maxFitness), 5, 8)
# showNetwork = forms.checkbox(form, "Show Map", 5, 30)
# showMutationRates = forms.checkbox(form, "Show M-Rates", 5, 52)
# restartButton = forms.button(form, "Restart", initializePool, 5, 77)
# saveButton = forms.button(form, "Save", savePool, 5, 102)
# loadButton = forms.button(form, "Load", loadPool, 80, 102)
# saveLoadFile = forms.textbox(form, Filename .. ".pool", 170, 25, nil, 5, 148)
# saveLoadLabel = forms.label(form, "Save/Load:", 5, 129)
# playTopButton = forms.button(form, "Play Top", playTop, 5, 170)
# hideBanner = forms.checkbox(form, "Hide Banner", 5, 190)
# 

#def main(self):
#while True:
# 	local species = pool.species[pool.currentSpecies]
# 	local genome = species.genomes[pool.currentGenome]
# 	
# 	if forms.ischecked(showNetwork) then
# 		displayGenome(genome)
# 	end
# 	
# 	if pool.currentFrame%5 == 0 then
# 		evaluateCurrent()
# 	end
# 
# 	joypad.set(controller)
# 
# 	getPositions()
# 	if marioX > rightmost then
# 		rightmost = marioX
# 		timeout = TimeoutConstant
# 	end
# 	
# 	timeout = timeout - 1
# 	
# 	
# 	local timeoutBonus = pool.currentFrame / 4
# 	if timeout + timeoutBonus <= 0 then
# 		local fitness = rightmost - pool.currentFrame / 2
# 		if gameinfo.getromname() == "Super Mario World (USA)" and rightmost > 4816 then
# 			fitness = fitness + 1000
# 		end
# 		if gameinfo.getromname() == "Super Mario Bros." and rightmost > 3186 then
# 			fitness = fitness + 1000
# 		end
# 		if fitness == 0 then
# 			fitness = -1
# 		end
# 		genome.fitness = fitness
# 		
# 		if fitness > pool.maxFitness then
# 			pool.maxFitness = fitness
# 			forms.settext(maxFitnessLabel, "Max Fitness: " .. math.floor(pool.maxFitness))
# 			writeFile("backup." .. pool.generation .. "." .. forms.gettext(saveLoadFile))
# 		end
# 		
# 		console.writeline("Gen " .. pool.generation .. " species " .. pool.currentSpecies .. " genome " .. pool.currentGenome .. " fitness: " .. fitness)
# 		pool.currentSpecies = 1
# 		pool.currentGenome = 1
# 		while fitnessAlreadyMeasured() do
# 			nextGenome()
# 		end
# 		initializeRun()
# 	end
# 
# 	local measured = 0
# 	local total = 0
# 	for _,species in pairs(pool.species) do
# 		for _,genome in pairs(species.genomes) do
# 			total = total + 1
# 			if genome.fitness ~= 0 then
# 				measured = measured + 1
# 			end
# 		end
# 	end
# 	if not forms.ischecked(hideBanner) then
# 		gui.drawText(0, 0, "Gen " .. pool.generation .. " species " .. pool.currentSpecies .. " genome " .. pool.currentGenome .. " (" .. math.floor(measured/total*100) .. "%)", 0xFF000000, 11)
# 		gui.drawText(0, 12, "Fitness: " .. math.floor(rightmost - (pool.currentFrame) / 2 - (timeout + timeoutBonus)*2/3), 0xFF000000, 11)
# 		gui.drawText(100, 12, "Max Fitness: " .. math.floor(pool.maxFitness), 0xFF000000, 11)
# 	end
# 		
# 	pool.currentFrame = pool.currentFrame + 1
# 
# 	emu.frameadvance();
# end


if __name__ == "__main__":
    neat = NEAT()
