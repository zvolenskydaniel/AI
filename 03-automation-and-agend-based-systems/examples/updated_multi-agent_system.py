#
# 2026.03 AI: Learning Path
# zvolensky.daniel@gmail.com
#

class PlannerAgent:
    def plan(self, goal):
        if "sum" in goal:
            return "calculate_sum"
        return "unknown"


class ExecutorAgent:
    def execute(self, task, data):
        if task == "calculate_sum":
            return sum(data)
        return None


class CriticAgent:
    def validate(self, goal, result):
        # Simple logic: if the goal was a sum, the result should be a number
        if "sum" in goal and isinstance(result, (int, float)):
            return True, "Result passed validation."
        return False, "Result seems incorrect for the given goal."


# --- Orchestration ---
planner = PlannerAgent()
executor = ExecutorAgent()
critic = CriticAgent()

goal = "calculate sum"
data = [1, 2, 3]

# 1. Planning
task = planner.plan(goal)

# 2. Execution
result = executor.execute(task, data)

# 3. Validation (The "Critic" step)
is_valid, message = critic.validate(goal, result)

if is_valid:
    print(f"Final Verified Result: {result}")
else:
    print(f"Error: {message}")
