#
# 2026.03 AI: Learning Path
# zvolensky.daniel@gmail.com
#
# Mimic CrewAI’s Sequential Process
#

class Tools:

    @staticmethod
    def validate(change_request):
        if "interface" in change_request:
            return {"status": "valid"}
        return {"status": "invalid", "reason": "missing interface"}

    @staticmethod
    def generate_config(change_request):
        interface = change_request["interface"]
        ip = change_request.get("ip", "dhcp")
        return f"set interfaces {interface} unit 0 family inet address {ip}"

    @staticmethod
    def deploy(config):
        # simulate deployment
        return {"deployment": "success"}

    @staticmethod
    def verify(interface):
        # simulate verification
        return {"state": "up"}

    @staticmethod
    def rollback(change_request):
        # simulate rollback
        interface = change_request["interface"]
        ip = change_request.get("ip", "dhcp")
        return f"delete interfaces {interface} unit 0 family inet address {ip}"


class NetworkChangeAgent:

    def __init__(self):
        self.memory = []
        self.state = {
            "plan": [],
            "config": None,
            "result": None,
            "rollback": None
        }

    def planner(self, goal):
        """
        The Planning Layer: Maps Natural Language to a 
        Sequence of technical capabilities.
        """
        goal = goal.lower()

        # Scenario A: Full Deployment
        if "deploy" in goal:
            return ["validate", "generate_config", "deploy", "verify"]
        
        # Scenario B: Health Check / Audit / Troubleshoot
        elif "check" in goal or "verify" in goal or "troubleshoot" in goal:
            return ["validate", "verify"]
        
        # Scenario C: Emergency Rollback
        elif "rollback" in goal:
            return ["validate", "rollback", "verify"]
            
        return []

    def run(self, goal, change_request):
        self.state["plan"] = self.planner(goal)
        
        print(f"--- Strategy Phase ---")
        print(f"Goal identified: {goal}")
        print(f"Executing Plan: {' -> '.join(self.state['plan'])}\n")

        # Execute tasks dynamically from the plan list
        for task in self.state["plan"]:
            print(f"Executing: {task}...")

            if task == "validate":
                res = Tools.validate(change_request)
                if res["status"] != "valid":
                    self.state["result"] = f"Failed at validation: {res.get('reason')}"
                    break
            
            elif task == "generate_config":
                self.state["config"] = Tools.generate_config(change_request)
                res = self.state["config"]

            elif task == "deploy":
                res = Tools.deploy(self.state["config"])
                if res["deployment"] != "success":
                    self.state["result"] = "Deployment failed"
                    break

            elif task == "verify":
                res = Tools.verify(change_request["interface"])
                self.state["result"] = f"Completed. State: {res['state']}"

            elif task == "rollback":
                self.state["rollback"] = Tools.rollback(change_request)
                res = self.state["rollback"]

            # Keep memory logging
            self.memory.append((task, res))

        return self.state["result"]

# --- Execution ---
agent = NetworkChangeAgent()

change = {
    "interface": "ge-0/0/1",
    "ip": "10.0.0.1/24"
}

result = agent.run(
    goal = "troubleshoot",
    change_request = change
)

print(f"\nFinal Result: {result}")
print("\n[REASONING TRACE]")
for i, (task, output) in enumerate(agent.memory, 1):
    print(f"{i}. >> {task.title()}\n   └─ Output: {output}")
