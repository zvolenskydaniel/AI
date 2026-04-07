#
# 2026.03 AI: Learning Path
# zvolensky.daniel@gmail.com
#
# Mimnic LangGraph's Dynamic Replanning Loop
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
        # Forced failure to trigger the HITL Rollback scenario
        return {"state": "down"}

    @staticmethod
    def rollback(change_request):
        # simulate rollback
        interface = change_request["interface"]
        ip = change_request.get("ip", "dhcp")
        return {"rollback": "executed", "status": "reverted to previous state", "command": f"delete interfaces {interface} unit 0 family inet address {ip}"}


class NetworkChangeAgent:

    def __init__(self):
        self.memory = []
        self.state = {
            "plan": [],
            "config": None,
            "result": None
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
            return ["rollback", "verify"]
            
        return []

    def run(self, goal, change_request):
        self.state["plan"] = self.planner(goal)
        
        print(f"--- Strategy Phase ---")
        print(f"Goal identified: {goal}")
        print(f"Executing Plan: {' -> '.join(self.state['plan'])}\n")

        # Use a while loop to allow the plan to change dynamically
        while self.state["plan"]:
            task = self.state["plan"].pop(0) # Take the first task
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

                if res["state"] != "up":
                    print(f"!!! CRITICAL: Verification Failed.")
                    
                    # --- HUMAN-IN-THE-LOOP BREAKPOINT ---
                    print("\n" + "!"*40)
                    user_choice = input("AGENT PROMPT: Failure detected. Should I attempt ROLLBACK? (yes/no): ")
                    print("!"*40 + "\n")
                    
                    if user_choice.lower() == 'yes':
                        recovery_tasks = self.planner("rollback")
                        self.state["plan"].extend(recovery_tasks)
                        self.state["result"] = "User approved recovery."
                    else:
                        self.state["result"] = "User aborted recovery. Manual intervention required."
                        break # Stop execution

            elif task == "rollback":
                res = Tools.rollback(change_request)

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
    goal = "deploy interface",
    change_request = change
)

print(f"\nFinal Result: {result}")
print("\n[REASONING TRACE]")
for i, (task, output) in enumerate(agent.memory, 1):
    print(f"{i}. >> {task.title()}\n   └─ Output: {output}")
