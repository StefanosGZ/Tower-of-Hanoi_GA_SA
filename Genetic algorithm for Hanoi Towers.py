import numpy as np
import math
from ypstruct import structure


# Function to generate a random solution for the Tower of Hanoi problem
def generate_solution(n_disks):
    # Calculate the optimal number of moves required to solve the puzzle
    optimal_amount_of_moves = 2 ** n_disks - 1
    # Set the length of the solution to a random value near the optimal number of moves
    solution_length = np.random.randint(optimal_amount_of_moves, math.ceil(optimal_amount_of_moves * 1.33 + 1))
    solution_list = []
    # Define possible moves between three rods (0, 1, and 2)
    action_list = [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]]

    # Helper function, so the solution doesn't get stuck in the beginning
    #solution_list.append(action_list[0])
    #solution_list.append(action_list[1])
    # Generate moves until the optimal number of moves is reached

    # Generate moves until the solution length is reached
    while len(solution_list) < solution_length:
        random_idx = np.random.randint(0, len(action_list))
        solution_list.append(action_list[random_idx])

    return solution_list


# Function to execute the moves of a solution and count illegal moves
def moves(solution, num_disks, optimal="No"):
    stick_position_list = []  # List to track positions of disks on rods
    middle_list = []  # List to represent initial state of disks on the first rod
    counter = 10  # Counter for tracking additional conditions

    # Initialize the rods with disks in starting positions
    for disk in range(1, num_disks + 1):
        middle_list.append(disk)
    stick_position_list.append(middle_list)
    stick_position_list.append([])
    stick_position_list.append([])

    illegal_moves = 0  # Counter for illegal moves

    # Execute each move in the solution sequence
    for move in solution:
        from_stick = stick_position_list[move[0]]
        to_stick = stick_position_list[move[1]]

        # If from_stick is empty, increment illegal move counter
        if not from_stick:
            illegal_moves += 1
            continue

        # Move disk if the to_stick is empty or top disk is larger than moving disk
        if not to_stick:
            counter += 1
            moving = from_stick.pop(0)
            to_stick.insert(0, moving)
        else:
            # Check if move is legal; otherwise, increment illegal moves
            if from_stick[0] > to_stick[0]:
                illegal_moves += 1
                continue
            moving = from_stick.pop(0)
            to_stick.insert(0, moving)

        # Condition to break loop if all disks are on the goal rod
        if 5 in stick_position_list[2]:
            break

    # If optimal flag is set to "YES," print state and illegal moves
    if optimal == "YES":
        print(stick_position_list, illegal_moves)

    return stick_position_list, illegal_moves


# Fitness function to evaluate the quality of a solution
def fitness_func(solution, num_disks, best_solution="No", iteration=0):
    # Execute moves and get final state and illegal moves count
    stick_position_list, illegal_moves = moves(solution, num_disks)

    starting_stick = stick_position_list[0]
    auxiliary_stick = stick_position_list[1]
    goal_stick = stick_position_list[2]

    num_disks_on_goal = len(goal_stick)

    # Calculate optimal number of moves
    optimal_moves = 2 ** num_disks - 1
    num_moves = len(solution)

    # Check if solution is optimal and reward with high fitness value
    if num_disks_on_goal == num_disks and illegal_moves == 0 and num_moves == optimal_moves:
        return 9999

    # Calculate fitness based on disk position and illegal moves
    fitness = 0
    if goal_stick:
        for disk in goal_stick:
            fitness += disk * 10

    if starting_stick:
        fitness -= sum(starting_stick) * 10
    else:
        fitness += 3

    if auxiliary_stick:
        fitness -= sum(auxiliary_stick) * 10
    else:
        fitness += 3
    # Penalize illegal moves
    fitness -= illegal_moves * 1

    # Reward shorter solutions that are closer to optimal
    fitness += (optimal_moves - num_moves) * 2
    if best_solution == "Yes":
        return goal_stick

    return float(fitness)


# Roulette wheel selection for selecting parents based on fitness
def roulette_wheel_selection(probabilities):
    cumulative_sum = np.cumsum(probabilities)
    r = sum(probabilities) * np.random.rand()
    ind = np.argwhere(r <= cumulative_sum)
    return ind[0][0]


# Tournament selection for selecting parents based on fitness
def tournament_selection(pop: structure, k=4):
    empty_solution = structure()
    empty_solution.solution = None
    empty_solution.fitness = None
    selected = empty_solution.repeat(k)
    for i in range(k):
        random_idx = np.random.randint(0, len(pop))
        selected[i].solution = pop[random_idx].solution
        selected[i].fitness = pop[random_idx].fitness
    selected = sorted(selected, key=lambda x: -x.fitness)
    return selected[0]


# Crossover function to create new solutions by combining parts of two parent solutions
def crossover(p1, p2):
    child1 = p1.deepcopy()
    child2 = p2.deepcopy()

    child1.solution = []
    child2.solution = []

    parents = [p1.solution, p2.solution]
    length = min(len(p1.solution), len(p2.solution))
    # Perform crossover by randomly selecting moves from each parent
    for i in range(length):
        selection_of_parent = np.random.randint(0, 2)
        child1.solution.append(parents[selection_of_parent][i])

        reverse_selection_of_parent = 1 - selection_of_parent
        child2.solution.append(parents[reverse_selection_of_parent][i])

    return child1, child2

# Basic mutation function
"""def mutate(x, mutation_rate, it = 0):
    y = x.deepcopy()
    flag = np.random.rand(len(y.solution)) <= mutation_rate
    indices_to_mutate = np.where(flag)[0]
    action_list = [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]]

    # Randomly change selected moves
    for ind in indices_to_mutate:

        random_idx = np.random.randint(0, len(action_list))
        while y.solution[ind] == action_list[random_idx]:
            random_idx = np.random.randint(0, len(action_list))
        y.solution[ind] = action_list[random_idx]

    return y"""

# Evolving mutation function
def mutate(x, mutation_rate, it=0, maxit=100):
    if it % 5 == 0 and it != 0:
        y = x.deepcopy()
        if np.random.rand() < mutation_rate:
            # Select two points in the solution to create a subsequence
            start_idx = np.random.randint(0, len(y.solution) - 1)
            end_idx = np.random.randint(start_idx + 1, len(y.solution))
            # Scramble the subsequence
            subsequence = y.solution[start_idx:end_idx]
            np.random.shuffle(subsequence)
            y.solution[start_idx:end_idx] = subsequence
    else:
        y = x.deepcopy()
        indices_to_mutate = []
        i = math.floor(len(y.solution) * (it / maxit))
        while i < len(y.solution):
            random = np.random.uniform(0, 1)
            if random < mutation_rate:
                indices_to_mutate.append(i)
            i += 1
        action_list = [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]]

        # Randomly change selected moves
        for ind in indices_to_mutate:
            random_idx = np.random.randint(0, len(action_list))
            while y.solution[ind] == action_list[random_idx]:
                random_idx = np.random.randint(0, len(action_list))
            y.solution[ind] = action_list[random_idx]

    return y


# Main function to run the genetic algorithm
def run(problem, params, best_solutions=None):
    fitnessfunc = problem.fitnessfunc
    num_disks = problem.num_disks
    npop = params.npop
    maxit = params.maxit
    pc = params.pc
    nc = int(np.round(pc * npop / 2) * 2)
    mutation_rate = params.mutation_rate
    crossover_rate = params.crossover_rate
    new_solution_rate = params.new_solution_rate

    empty_solution = structure()
    empty_solution.solution = None
    empty_solution.fitness = None

    best_solution = empty_solution.deepcopy()
    best_solution.fitness = -np.inf

    # Initialize population with random solutions if not nested GA
    if best_solutions is None:
        best_fitness = np.empty(maxit)
        incest_rate = 0.2
        pop = empty_solution.repeat(npop)
        for i in range(npop):
            pop[i].solution = generate_solution(num_disks)
            pop[i].fitness = fitnessfunc(pop[i].solution, num_disks)
            if pop[i].fitness > best_solution.fitness:
                best_solution = pop[i].deepcopy()
        pop = sorted(pop, key=lambda x: -x.fitness)
    else:
        # Preparation if it's the nested GA
        pop = sorted(best_solutions, key=lambda x: -x.fitness)
        best_solution.fitness = best_solutions[0].fitness
        best_solution.solution = best_solutions[0].solution
        incest_rate = 0.8

    # Reduce population size by half for nested GA
    pop = pop[0:int(npop / 2)]

    # Evolution loop
    for it in range(maxit):
        popc = []

        # This is used inorder to do roulette wheel selection. Uncomment if you want to use it
        """fitnesses = np.array([x.fitness for x in pop])
        avg_fitness = float(np.mean(fitnesses))
        if avg_fitness != 0:
            fitnesses /= avg_fitness
        probs = np.exp(-beta * fitnesses)"""
        # Generate new solutions through crossover and mutation
        for _ in range(int(round(nc * crossover_rate))):
            # Anti-incest breeding check
            incest = True
            counter = 0
            while incest and counter < 3:
                similarity_counter = 0
                # This is used inorder to do roulette wheel selection. Uncomment this and comment the if else if you
                # Want to use roulette wheel selection
                #p1 = pop[roulette_wheel_selection(probs)]
                #p2 = pop[roulette_wheel_selection(probs)]
                # Parent selection with tournament
                if best_solutions is None:
                    p1 = tournament_selection(pop)
                    p2 = tournament_selection(pop)
                else:
                    p1 = tournament_selection(pop, k=2)
                    p2 = tournament_selection(pop, k=2)
                for move_in_p1, move_in_p2 in zip(p1.solution, p2.solution):
                    if move_in_p1 == move_in_p2:
                        similarity_counter += 1

                similarity_rate = similarity_counter / len(p1.solution)

                if similarity_rate < incest_rate:
                    incest = False

                counter += 1

            c1, c2 = crossover(p1, p2)

            c1 = mutate(c1, mutation_rate, it, maxit)
            c2 = mutate(c2, mutation_rate, it, maxit)

            c1.fitness = fitnessfunc(c1.solution, num_disks, it)
            c2.fitness = fitnessfunc(c2.solution, num_disks, it)

            popc.append(c1)
            popc.append(c2)

        pop += popc
        if best_solutions is None:
            new_population = empty_solution.repeat(int(round(nc * new_solution_rate)))
            for i in range(int(round(nc * new_solution_rate))):
                new_population[i].solution = generate_solution(num_disks)
                new_population[i].fitness = fitnessfunc(new_population[i].solution, num_disks, it)
            pop += new_population

        pop = sorted(pop, key=lambda x: -x.fitness)
        if best_solution != pop[0]:
            best_solution = pop[0].deepcopy()
        pop = pop[0:int(npop)]
        if best_solutions is None:
            best_fitness[it] = best_solution.fitness

        goal_stick = fitnessfunc(best_solution.solution, num_disks, best_solution="Yes")
        # Uncomment this if you want to print out the best fitness and goal stick status every 5th iteration
        if best_solutions is None:
            if it % 5 == 0:
                print(f"Iteration: {it}, Best fitness = {best_fitness[it]}")
        # Uncomment to print goal stick during the run
        # print(goal_stick)
        #goal_stick = moves(best_solution.solution, num_disks, "Yes")
        #print(goal_stick)
        if float(best_solution.fitness) == 9999:
            print(f"Optimal solution found in iteration: {it}\n"
                  f"Optimal solution is:\n"
                  f"{best_solution.solution}")
            break

        # Adjust parameters dynamically if best_solutions is None
        if best_solutions is None:
            if mutation_rate < 0.5:
                mutation_rate *= 1.01
            if crossover_rate > 0.3:
                crossover_rate *= 0.99
            if it % 10 == 0 and it != 0:
                incest_rate += 0.1

    # Return structure with final population and best solution
    out = structure()
    out.pop = pop
    out.bestsol = best_solution
    out.bestfitness = best_solution.fitness
    return out


def main():
    # Problem and parameter definitions
    problem = structure()
    problem.fitnessfunc = fitness_func
    problem.num_disks = 3  # Number of disks in the game

    params = structure()
    params.maxit = 101  # Maximum number of iterations
    params.npop = 200  # The population that will be used
    params.pc = 1  #
    params.mutation_rate = 0.1  # Mutation rate
    params.crossover_rate = 0.5  # Crossover-rate
    params.beta = 1  # Roulette wheel selection parameter
    params.new_solution_rate = 0.5  # How many new solutions will be created compared to npop

    best_solutions = []

    # Run the genetic algorithm and print the best solution
    for i in range(20):
        out = run(problem, params)
        best_solutions.append(out.bestsol)
        print(f"Iteration of {i} genetic algorithm")
        print(f"Best solution")
        moves(out.bestsol.solution, problem.num_disks, "YES")
        print(f"Best fitness of iteration")
        print(out.bestsol.fitness)
        print("\n")

    print("______________________________________________")
    print("Starting the nested GA")
    print("\n")

    final_nested = run(problem, params, best_solutions)
    moves(final_nested.bestsol.solution, problem.num_disks, "YES")
    print(f"Best fitness of nested GA")
    print(final_nested.bestsol.fitness)


if __name__ == "__main__":
    main()
