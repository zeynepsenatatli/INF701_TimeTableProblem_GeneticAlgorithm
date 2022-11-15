import random as rnd
import numpy as np
import matplotlib.pyplot as plt

class lecture:
    def __init__(self, id, instructors, semester=None, elective=None):
        self.id = id
        self.instructors = instructors
        self.semester = semester
        self.elective = elective

    def __str__(self):
        return self.id + "by" + self.instructors

l1 = lecture("MAT103", "IG", "1")
l2 = lecture("INF101", "VG", "1")
l3 = lecture("INF103", "FB", "1")
l4 = lecture("INF107", "FB", "1")
l5 = lecture("DEU121", "DEU", "1")
l6 = lecture("ENG101", "ENG", "1")
l7 = lecture("TUR001", "TUR", "1")

l8 = lecture("INF201", "CY", "3")
l9 = lecture("INF203", "EMY", "3")
l10 = lecture("INF205", "CY", "3")
l11 = lecture("INF209", "FB", "3")
l12 = lecture("INF211", "CY", "3")

l13 = lecture("ENG201", "ENG", "3")
l14 = lecture("AIT001", "AIT", "3")

l14 = lecture("INF303", "OK", "5")
l15 = lecture("ISG001", "ISG", "5")
l16 = lecture("ENG301", "ENG", "5")

l17 = lecture("INF506", "EI", None , True)
l18 = lecture("INF523", "DG", None , True)
l19 = lecture("INF517", "SI", None , True)
l20 = lecture("INF701", "CY", None , True)
l21 = lecture("INF714", "EI", None , True)
l22 = lecture("INF905", "BB", None , True)

lectures = [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15, l16, l17, l18, l19, l20, l21, l22]
rooms =  ["R1", "R2", "R3", "R4"]
slots = ["T1", "T2", "T3", "T4", "T5"]
days = ["M", "TUE", "W", "TH", "F"]

class scheduledLecture:
    def __init__(self, lecture, room, slot, day):
        self.lecture = lecture
        self.room = room
        self.slot = slot
        self.day = day

    def __str__(self):
        return "[" + self.lecture.id + ", " + self.lecture.instructors + ", " + self.room + ", " + self.day + ", " + self.slot + "]"

class schedule:
    def __init__(self, scheduledLectures):
        self.scheduledLectures = scheduledLectures
        self.fitness = calculateFitness(scheduledLectures)

    def __str__(self):
        s = ""
        for i in self.scheduledLectures:
            s = s + "[" + i.lecture.id + ", " + i.lecture.instructors + ", " + i.room + ", " + i.day + ", " + i.slot + "] "
        return s

def createPopulation():
    # create random population
    for i in range(POPULATION_SIZE):
        global population
        scheduled = []
        for k in lectures:
            scheduled.append(scheduledLecture(k, rooms[rnd.randrange(0, len(rooms))], slots[rnd.randrange(0, len(slots))], days[rnd.randrange(0, len(days))]))

        population.append(schedule(scheduled))
    
    population = sorted(population, key=lambda a : a.fitness)


def calculateFitness(individual):
    # calculate fitness value for individuals
    fitness = 0
    for i in individual:
        for k in individual:
            if i == k: continue
            if i.day == k.day and i.slot == k.slot:
                if i.room == k.room:
                    fitness += 1
                if i.lecture.instructors == k.lecture.instructors:
                    fitness += 1
                   
                if (i.lecture.elective and k.lecture.elective) or (i.lecture.elective and k.lecture.semester == "5")  or (i.lecture.semester == "5" and k.lecture.elective):
                    fitness += 1
                    
                if i.lecture.semester == k.lecture.semester:
                    fitness += 1

    return fitness // 2

def selectParent():
    # two parent selection by Roulette Wheel Selection
    sum = 0
    for i in population:
        sum = sum + i.fitness
 
    p = []
    for i in population:
        p.append(i.fitness/sum)

    for i in range(1, len(p)):
        p[i] = p[i-1] + p[i]

    rand = rnd.random()

    index = 0
    for i in range(0,len(p)):
        if rand > p[i]:
            continue
        else:
            index = i
            break

    return population[index]

def crossover(p1, p2):
    # two point crossover
    point1 = rnd.randint(0, len(p1.scheduledLectures)-12)
    point2 = rnd.randint(point1, len(p1.scheduledLectures))

    off1 = schedule(p1.scheduledLectures[:point1] + p2.scheduledLectures[point1:point2] + p1.scheduledLectures[point2:])
    off2 = schedule(p2.scheduledLectures[:point1] + p1.scheduledLectures[point1:point2] + p2.scheduledLectures[point2:])

    return off1, off2

def mutation(offspring):
    #Change a random value if random number smaller than mutation rate
    for i in offspring.scheduledLectures:
        if rnd.random() < MUTATION_RATE:

            toChange = rnd.choice(["room", "day", "slot"])
            if toChange == "room":
                i.room = rooms[rnd.randrange(0, len(rooms))]
            elif toChange == "day":
                i.day = days[rnd.randrange(0, len(days))]
            elif toChange == "slot":
                i.slot = slots[rnd.randrange(0, len(slots))]

    return offspring

def updateGeneration(offsprings):
    global population
    # update Generation with the best individuals (best fitness score)
    buff = population + offsprings
    buff = sorted(buff, key=lambda a : a.fitness)
   
    population = buff[:len(buff)//2]


def geneticAlgorithm():

    offsprings = []

    while len(offsprings) != len(population):
        
        parent1 = selectParent()
        parent2 = selectParent()
        
        while parent1 == parent2:
            parent2 = selectParent()

        off1, off2 = crossover(parent1, parent2)

        off1 = mutation(off1)
        off2 = mutation(off2)

        # control if new offsprings is already created
        control1 = True
        for i in offsprings:
            if (i.__str__() == off1.__str__()) or (i.__str__() == off2.__str__()):
                control1 = False
                break
        
        # control if new offsprings exist already in population
        control2 = True
        for i in population:
            if (i.__str__() == off1.__str__()) or (i.__str__() == off2.__str__()):
                control2 = False
                break
            
        if control1 and control2:
        
            offsprings.append(off1)
            offsprings.append(off2)

    updateGeneration(offsprings)

# Start
POPULATION_SIZE = 8
MUTATION_RATE = 0.05

gens = []
runs = []
found = 0
for run in range(100):
    population = []

    createPopulation()
    print(f'Run: {run}')
    control = False
    solution = None

    gen = 0
    while gen < 100:
        print("GEN: " + str(gen))

        index = 1
        for i in population:
            
            print("Individual " + str(index) + " with fitness score " + str(i.fitness))
            print(i)

            if i.fitness == 0:
                control = True
                solution = i
            
            index += 1
        gen += 1

        if control: break
        elif control == False:
            geneticAlgorithm()

    if control:
        print("Solution found")
        print("Generation: " + str(gen))
        print(solution)
        found += 1
        gens.append(gen) 
        runs.append(run)
    else:
        print("Solution not found")

print(f'{found} solutions found')
print(f'Mean: {np.mean(gens)}')
print(f'St. dev: {np.std(gens)}')
print(f'Min: {min(gens)}')
print(f'Max: {max(gens)}')

x = np.array(runs)
y = np.array(gens)
#plt.scatter(runs,gens, c = "blue")
#create basic scatterplot
plt.plot(x, y, 'o')

#obtain m (slope) and b(intercept) of linear regression line
m, b = np.polyfit(x, y, 1)

#add linear regression line to scatterplot 
plt.plot(x, m*x+b)

plt.xlabel("runs")
plt.ylabel("gens")
plt.show()
