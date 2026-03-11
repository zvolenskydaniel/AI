#
# 2026.03 AI: Learning Path
# zvolensky.daniel@gmail.com
#

class SimpleAgent:
    def __init__(self):
        self.memory = []

    def think(self, goal):
        if "sum" in goal:
            return "use_math_tool"
        return "unknown"

    def use_tool(self, tool_name, input_data):
        if tool_name == "use_math_tool":
            return sum(input_data)

    def run(self, goal, input_data):
        decision = self.think(goal)
        result = self.use_tool(decision, input_data)
        self.memory.append((goal, result))
        return result


agent = SimpleAgent()
result = agent.run("calculate sum", [1, 2, 3])
print(f"output: {result}")
print(f"memory: {agent.memory}")
