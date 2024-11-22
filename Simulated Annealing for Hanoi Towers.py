import copy
import numpy as np


# Function to generate a random solution for the Tower of Hanoi problem
def generate_solution(n_disks):
    # Calculate the optimal number of moves required for the problem
    optimal_amount_of_moves = 2 ** n_disks - 1
    solution_list = []  # List to store the sequence of moves
    # Define all possible moves between the three rods
    action_list = [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]]

    # Generate a random sequence of moves until the optimal number is reached
    while len(solution_list) < optimal_amount_of_moves:
        random_idx = np.random.randint(0, len(action_list))
        solution_list.append(action_list[random_idx])

    return solution_list


# Function to execute the moves of a solution and count illegal moves
def moves(solution, num_disks, optimal="No"):
    stick_position_list = []  # List to track the state of each rod
    middle_list = []  # Temporary list to initialize the first rod with disks

    # Initialize the rods: all disks start on the first rod
    for disk in range(1, num_disks + 1):
        middle_list.append(disk)
    stick_position_list.append(middle_list)
    stick_position_list.append([])  # Second rod
    stick_position_list.append([])  # Third rod

    illegal_moves = 0  # Counter for illegal moves

    # Execute each move in the solution
    for move in solution:
        from_stick = stick_position_list[move[0]]  # Source rod
        to_stick = stick_position_list[move[1]]  # Destination rod

        # Check if the source rod is empty
        if not from_stick:
            illegal_moves += 1  # Increment illegal move counter
            continue

        # If the destination rod is empty or the move is valid, perform the move
        if not to_stick:
            moving = from_stick.pop(0)
            to_stick.insert(0, moving)
        else:
            # Check if the move violates the rules (larger disk on smaller disk)
            if from_stick[0] > to_stick[0]:
                illegal_moves += 1
                continue
            moving = from_stick.pop(0)
            to_stick.insert(0, moving)

    # Optionally print the rod states and illegal move count
    if optimal == "YES":
        print(stick_position_list, illegal_moves)

    return stick_position_list, illegal_moves


# Fitness function to evaluate the quality of a solution
def fitness_func(solution, num_disks, best_solution="No", iteration=0):
    stick_position_list, illegal_moves = moves(solution, num_disks)
    # Extract the state of the rods
    starting_stick = stick_position_list[0]
    auxiliary_stick = stick_position_list[1]
    goal_stick = stick_position_list[2]

    # Count the number of disks on the goal rod
    num_disks_on_goal = len(goal_stick)

    # Calculate the optimal number of moves
    optimal_moves = 2 ** num_disks - 1
    num_moves = len(solution)

    # Return a high fitness score if the solution is optimal
    if num_disks_on_goal == num_disks and illegal_moves == 0 and num_moves == optimal_moves:
        return [9999, 0]

    # Calculate fitness based on the state of the rods
    fitness = 0
    if goal_stick:
        for disk in goal_stick:
            fitness += disk * 10

    # Penalize based on the remaining disks on the starting and auxiliary rods
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

    # Return the goal rod if the best solution is requested
    if best_solution == "Yes":
        return goal_stick

    return float(fitness), goal_stick


# Function to create a modified solution by randomly changing moves
def change(solution, num_idx=2):
    candidate = copy.deepcopy(solution)  # Create a copy of the solution
    idx_to_change = []  # List to track indices to be modified
    random_idx = -100  # Placeholder for a random index

    # Randomly select indices to modify
    for i in range(num_idx):
        while random_idx in idx_to_change or random_idx == -100:
            random_idx = np.random.randint(0, len(solution))
        idx_to_change.append(random_idx)

    # Modify the selected indices with new random moves
    action_list = [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]]
    for idx in idx_to_change:
        random_idx = np.random.randint(0, len(action_list))
        while candidate[idx] == action_list[random_idx]:
            random_idx = np.random.randint(0, len(action_list))
        candidate[idx] = action_list[random_idx]
    return candidate


# Main function to run the simulated annealing algorithm
def run(num_disks, max_iterations, temperature, cooling_rate):
    solution = generate_solution(num_disks)  # Generate an initial random solution

    best_fitness = fitness_func(solution, num_disks)  # Evaluate the initial solution
    goal_stick = best_fitness[1]
    best_fitness = best_fitness[0]

    # Initialize history tracking
    fitness_history = [best_fitness]
    solution_history = [solution]
    goal_stick_history = [goal_stick]

    counter = 0  # Counter for stagnation
    last_fitness = 0  # Track the last fitness value

    # Perform iterative optimization
    for i in range(max_iterations):
        num_idx = np.random.randint(0, len(solution))  # Random number of changes
        candidate = change(solution, num_idx)  # Generate a new candidate solution
        candidate_fitness = fitness_func(candidate, num_disks)
        temporary_goal_stick = candidate_fitness[1]
        candidate_fitness = candidate_fitness[0]

        # Detect stagnation in fitness
        if last_fitness == candidate_fitness:
            counter += 1
            if counter == 5000:
                print("No change in fitness for 5000 iterations, deleting")
                for i in range(-1, -6, -1):
                    print(solution_history[i])
                break

        last_fitness = candidate_fitness

        # Calculate acceptance probability for simulated annealing
        acceptance_probability = np.exp((candidate_fitness - best_fitness) / temperature)

        if candidate_fitness > best_fitness or np.random.rand() < acceptance_probability:
            solution = candidate
            best_fitness = candidate_fitness
            goal_stick = temporary_goal_stick
        temperature *= cooling_rate  # Reduce temperature
        fitness_history.append(best_fitness)
        solution_history.append(solution)
        goal_stick_history.append(goal_stick)

        # Break the loop and return if optimal solution is found
        if candidate_fitness == 9999:
            print("Optimal solution found")
            return solution
        if i % 1 == 0:
            print(f"Iteration {i}: Fitness {best_fitness}")

    print(f"Goal stick: {goal_stick}")
    return solution


# Main entry point to run the algorithm
def main():
    num_disks = 3  # Number of disks in the Tower of Hanoi
    max_iterations = 20201  # Maximum number of iterations
    temperature = 100.0  # Initial temperature for simulated annealing
    cooling_rate = 0.999  # Cooling rate for temperature decay
    solution = run(num_disks, max_iterations, temperature, cooling_rate)
    print(f"Solution is: {solution}")


# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()
