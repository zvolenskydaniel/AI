#
# 2026.03 AI: Learning Path
# zvolensky.daniel@gmail.com
#

class PlannerAgent:
    def plan(self, goal):
        if "sum" in goal:
            return {"task": "calculate_sum"}
        return {"task": "unknown"}


class ExecutorAgent:
    def execute(self, task, data):
        if task == "calculate_sum":
            return sum(data)
        return None


planner = PlannerAgent()
executor = ExecutorAgent()

goal = "calculate sum"
plan = planner.plan(goal)
result = executor.execute(plan["task"], [1, 2, 3])

print(result)
