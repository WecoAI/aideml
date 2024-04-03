import pandas as pd
import numpy as np
from math import exp

# Load the data
family_data = pd.read_csv("./input/family_data.csv")
sample_submission = pd.read_csv("./input/sample_submission.csv")

# Constants
N_DAYS = 100
N_FAMILY = len(family_data)
MAX_OCCUPANCY = 300
MIN_OCCUPANCY = 125


# Cost function components
def preference_cost_matrix(family_data):
    cost_matrix = np.zeros((N_FAMILY, N_DAYS + 1), dtype=np.int64)
    for i in range(N_FAMILY):
        family = family_data.iloc[i]
        n_people = family["n_people"]
        for j in range(10):
            day = family[f"choice_{j}"]
            if j == 0:
                cost_matrix[i, day] = 0
            elif j == 1:
                cost_matrix[i, day] = 50
            elif j == 2:
                cost_matrix[i, day] = 50 + 9 * n_people
            elif j == 3:
                cost_matrix[i, day] = 100 + 9 * n_people
            elif j == 4:
                cost_matrix[i, day] = 200 + 9 * n_people
            elif j == 5:
                cost_matrix[i, day] = 200 + 18 * n_people
            elif j == 6:
                cost_matrix[i, day] = 300 + 18 * n_people
            elif j == 7:
                cost_matrix[i, day] = 300 + 36 * n_people
            elif j == 8:
                cost_matrix[i, day] = 400 + 36 * n_people
            elif j == 9:
                cost_matrix[i, day] = 500 + 36 * n_people + 199 * n_people
        cost_matrix[i, 0] = 500 + 36 * n_people + 398 * n_people
    return cost_matrix


def accounting_penalty(occupancy):
    penalties = np.zeros(N_DAYS + 1)
    for day in range(N_DAYS - 1, -1, -1):
        Nd = occupancy[day]
        Nd_next = occupancy[day + 1]
        penalties[day] = max(0, (Nd - 125) / 400 * Nd ** (0.5 + abs(Nd - Nd_next) / 50))
    return penalties.sum()


# Simulated annealing
def simulated_annealing(family_data, sample_submission, cost_matrix):
    best = sample_submission["assigned_day"].values
    occupancy = np.zeros(N_DAYS + 1, dtype=int)
    for i, day in enumerate(best):
        occupancy[day] += family_data.iloc[i]["n_people"]
    occupancy[0] = occupancy[N_DAYS]  # Occupancy for the "zeroth" day
    best_score = cost_matrix[np.arange(N_FAMILY), best].sum() + accounting_penalty(
        occupancy
    )
    temperature = 1.0
    alpha = 0.99
    for step in range(10000):
        # Create new candidate solution
        family_id = np.random.choice(range(N_FAMILY))
        old_day = best[family_id]
        new_day = np.random.choice(range(1, N_DAYS + 1))
        best[family_id] = new_day

        # Calculate the cost
        new_occupancy = occupancy.copy()
        new_occupancy[old_day] -= family_data.iloc[family_id]["n_people"]
        new_occupancy[new_day] += family_data.iloc[family_id]["n_people"]
        new_occupancy[0] = new_occupancy[N_DAYS]  # Occupancy for the "zeroth" day
        if any((new_occupancy < MIN_OCCUPANCY) | (new_occupancy > MAX_OCCUPANCY)):
            best[family_id] = old_day  # Revert changes
            continue
        new_score = cost_matrix[np.arange(N_FAMILY), best].sum() + accounting_penalty(
            new_occupancy
        )

        # Acceptance probability
        if new_score < best_score or np.random.rand() < exp(
            -(new_score - best_score) / temperature
        ):
            best_score = new_score
            occupancy = new_occupancy
        else:
            best[family_id] = old_day  # Revert changes

        # Cool down
        temperature *= alpha

    return best, best_score


# Run the optimization
cost_matrix = preference_cost_matrix(family_data)
best_schedule, best_score = simulated_annealing(
    family_data, sample_submission, cost_matrix
)

# Output the result
print(f"Best score: {best_score}")
sample_submission["assigned_day"] = best_schedule
sample_submission.to_csv("./working/submission.csv", index=False)
