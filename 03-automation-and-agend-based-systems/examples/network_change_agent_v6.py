#
# 2026.03 AI: Learning Path
# zvolensky.daniel@gmail.com
#
# Mimnic LangGraph: Dynamic Replanning Loop with Reliability and Tiered Escalation Patterns
#

# ---- Import libraries ----
import sys

class Tools:
    @staticmethod
    def validate(change_request):
        return {"status": "valid"} if "interface" in change_request else {"status": "invalid"}

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
        # Still forced failure to demonstrate the retry-then-escalate logic
        return {"state": "down"}

    @staticmethod
    def rollback(change_request):
        return {"rollback": "executed", "status": "reverted"}

class NetworkChangeAgent:
    def __init__(self):
        self.memory = []
        self.max_retries = 2
        self.state = {
            "plan": [],
            "config": None,
            "result": None,
            "retry_count": 0
        }

    def planner(self, goal):
        goal = goal.lower()
        if "deploy" in goal:
            return ["validate", "generate_config", "deploy", "verify"]
        elif "rollback" in goal:
            return ["rollback", "verify"]
        return []

    def run(self, goal, change_request):
        self.state["plan"] = self.planner(goal)
        
        print(f"--- Strategy Phase ---")
        print(f"Goal: {goal} | Max Retries: {self.max_retries}\n")

        while self.state["plan"]:
            task = self.state["plan"].pop(0)
            print(f"Executing: {task}...")

            if task == "validate":
                res = Tools.validate(change_request)
                if res["status"] != "valid":
                    self.state["result"] = "Validation failed."
                    break
            
            elif task == "generate_config":
                res = self.state["config"] = Tools.generate_config(change_request)

            elif task == "deploy":
                res = Tools.deploy(self.state["config"])

            elif task == "verify":
                res = Tools.verify(change_request["interface"])
                
                if res["state"] == "up":
                    self.state["result"] = "Success"
                    self.state["retry_count"] = 0
                else:
                    # Logic: Retry before escalating
                    if self.state["retry_count"] < self.max_retries:
                        self.state["retry_count"] += 1
                        print(f"!!! Verification Failed. Automated Retry {self.state['retry_count']}/{self.max_retries}...")
                        # Re-add deploy and verify to the plan
                        state['plan'].extend(["deploy", "verify"])
                    else:
                        # Escalation to Human
                        print("\n" + "!"*40)
                        print("MAX RETRIES REACHED. SYSTEM REQUIRES HUMAN INTERVENTION.")
                        user_choice = input("AGENT PROMPT: Retries failed. Attempt ROLLBACK? (yes/no): ")
                        print("!"*40 + "\n")
                        
                        if user_choice.lower() == 'yes':
                            recovery_tasks = self.planner("rollback")
                            self.state["plan"].extend(recovery_tasks)
                            self.state["result"] = "Manual Rollback initiated."
                        else:
                            self.state["result"] = "Manual intervention required. Plan aborted."
                            break

            elif task == "rollback":
                res = Tools.rollback(change_request)

            self.memory.append((task, res))

        return self.state["result"]

# --- Execution ---
agent = NetworkChangeAgent()

change = {
    "interface": "ge-0/0/1",
    "ip": "10.0.0.1/24"
}

result = agent.run(
    goal = "deploy interface",
    change_request = change
)

print(f"\nFinal Result: {result}")
print("\n[REASONING TRACE]")
for i, (task, output) in enumerate(agent.memory, 1):
    print(f"{i}. >> {task.title()}\n   └─ Output: {output}")
