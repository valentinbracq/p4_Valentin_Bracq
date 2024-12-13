import pandas as pd
import numpy as np
from scipy.optimize import linprog

def load_transportation_problem(file_path):
    data = pd.read_excel(file_path, header=0)
    cost_matrix = data.iloc[:-1, 1:-1].values
    supply = pd.to_numeric(data.iloc[:-1, -1].values, errors='coerce')
    demand = pd.to_numeric(data.iloc[-1, 1:-1].values, errors='coerce')
    print("Cost Matrix:\n", cost_matrix)
    print("Supply:", supply)
    print("Demand:", demand)
    if np.any(np.isnan(supply)) or np.any(np.isnan(demand)):
        raise ValueError("Non-numeric values detected in supply or demand")
    return cost_matrix, supply, demand

def balance_problem(cost_matrix, supply, demand):
    if sum(supply) != sum(demand):
        if sum(supply) > sum(demand):
            demand = np.append(demand, sum(supply) - sum(demand))
            cost_matrix = np.column_stack((cost_matrix, np.zeros(len(supply))))
        else:
            supply = np.append(supply, sum(demand) - sum(supply))
            cost_matrix = np.vstack((cost_matrix, np.zeros(len(demand))))
    return cost_matrix, supply, demand

def minimum_cost_method(cost_matrix, supply, demand):
    cost_matrix = cost_matrix.astype(float)
    allocation = np.zeros_like(cost_matrix)
    supply = supply.copy()
    demand = demand.copy()

    while np.any(supply > 0) and np.any(demand > 0):
        i, j = np.unravel_index(np.argmin(cost_matrix, axis=None), cost_matrix.shape)
        allocation[i, j] = min(supply[i], demand[j])
        supply[i] -= allocation[i, j]
        demand[j] -= allocation[i, j]
        cost_matrix[i, j] = np.inf

    return allocation

def solve_transportation_simplex(cost_matrix, supply, demand):
    num_sources = len(supply)
    num_destinations = len(demand)
    c = cost_matrix.flatten()
    A_eq = []
    b_eq = []

    for i in range(num_sources):
        constraint = [0] * len(c)
        for j in range(num_destinations):
            constraint[i * num_destinations + j] = 1
        A_eq.append(constraint)
        b_eq.append(supply[i])

    for j in range(num_destinations):
        constraint = [0] * len(c)
        for i in range(num_sources):
            constraint[i * num_destinations + j] = 1
        A_eq.append(constraint)
        b_eq.append(demand[j])

    result = linprog(c, A_eq=A_eq, b_eq=b_eq, method='highs')

    if result.success:
        return result.x.reshape(num_sources, num_destinations)
    else:
        raise ValueError("No feasible solution found.")

def display_solution(solution, supply, demand):
    supply_labels = [f"Source {i+1}" for i in range(len(supply))]
    demand_labels = [f"Destination {j+1}" for j in range(len(demand))]
    solution_df = pd.DataFrame(solution, index=supply_labels, columns=demand_labels)
    print("Transportation Plan:")
    print(solution_df)

def main():
    file_path = "parameters.xlsx"  # Path to your Excel file
    cost_matrix, supply, demand = load_transportation_problem(file_path)
    cost_matrix, supply, demand = balance_problem(cost_matrix, supply, demand)
    initial_solution = minimum_cost_method(cost_matrix.copy(), supply, demand)
    print("Initial Feasible Solution (Minimum Cost Method):")
    display_solution(initial_solution, supply, demand)
    optimal_solution = solve_transportation_simplex(cost_matrix, supply, demand)
    print("\nOptimal Solution (Simplex Method):")
    display_solution(optimal_solution, supply, demand)

if __name__ == "__main__":
    main()
