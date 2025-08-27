
# True if should stop, false if not
def get_decision(stopping_rule, trial_history):

    if stopping_rule == 'no_improvment_iterations':
        return no_improvement_stopping_rule(trial_history,5)


def no_improvement_stopping_rule(trial_history, iterations, mode="min"):
    """
    Stops the optimization run if there has been no improvement for 'iterations' iterations.

    Parameters:
        trial_history (list): The result values of the optimization run up to the current iteration.
        iterations (int): Number of consecutive iterations without improvement allowed.
        mode (str): "min" for minimizing objective, "max" for maximizing objective.

    Returns:
        bool: True if the optimization run should stop, False otherwise.
    """
    if len(trial_history) <= iterations:
        return False  # Not enough data to check for improvement

    # Determine the best value seen so far
    if mode == "min":
        best_value = min(trial_history[:-iterations])
        # Check if the recent values show no improvement
        return all(result >= best_value for result in trial_history[-iterations:])
    elif mode == "max":
        best_value = max(trial_history[:-iterations])
        # Check if the recent values show no improvement
        return all(result <= best_value for result in trial_history[-iterations:])
