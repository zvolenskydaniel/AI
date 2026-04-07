#
# 2026.03 AI: Learning Path
# zvolensky.daniel@gmail.com
#
# Multi-Agent Implementation
#

# ---- Import libraries ----
import sys

# 1. THE TOOLS (The physical capabilities)
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
        return {"deployment": "success"}

    @staticmethod
    def verify(interface):
        # Forced failure to trigger recovery logic
        return {"state": "down"}

    @staticmethod
    def rollback(change_request):
        return {"rollback": "executed", "status": "reverted"}

# 2. THE SPECIALIZED AGENTS (The Roles)
class PlannerAgent:
    def create_plan(self, goal):
        goal = goal.lower()
        if "deploy" in goal:
            return ["validate", "generate_config", "deploy", "verify"]
        if "rollback" in goal:
            return ["rollback", "verify"]
        return []

class ExecutorAgent:
    def execute_task(self, task, context):
        if task == "validate":
            return Tools.validate(context)
        if task == "generate_config":
            return Tools.generate_config(context)
        if task == "deploy":
            return Tools.deploy(context.get('config'))
        return None

class VerifierAgent:
    def check_health(self, interface):
        return Tools.verify(interface)

class RecoveryAgent:
    def __init__(self):
        self.max_retries = 2
        self.retry_count = 0

    def should_retry(self):
        if self.retry_count < self.max_retries:
            self.retry_count += 1
            return True
        return False

    def ask_human(self):
        print("\n" + "!"*40)
        print("RECOVERY AGENT: Automated retries exhausted.")
        choice = input("Escalation to Human: Attempt Rollback? (yes/no): ")
        print("!"*40 + "\n")
        return choice.lower() == 'yes'

# 3. THE ORCHESTRATOR (The Graph / Crew Management)
class Orchestrator:
    def __init__(self):
        self.planner = PlannerAgent()
        self.executor = ExecutorAgent()
        self.verifier = VerifierAgent()
        self.recovery = RecoveryAgent()
        self.memory = []
        self.state = {"plan": [], "config": None, "result": None}

    def run_workflow(self, goal, change_request):
        self.state["plan"] = self.planner.create_plan(goal)
        
        while self.state["plan"]:
            task = self.state["plan"].pop(0)
            print(f"--- Agent Handoff: {task.upper()} ---")

            if task in ["validate", "generate_config", "deploy"]:
                res = self.executor.execute_task(task, change_request if task != 'deploy' else {'config': self.state['config']})
                if task == "generate_config": self.state["config"] = res
                if task == "validate" and res["status"] != "valid":
                    self.state["result"] = "Aborted: Invalid Request"
                    break
            
            elif task == "verify":
                res = self.verifier.check_health(change_request["interface"])
                if res["state"] == "up":
                    self.state["result"] = "Workflow Successful"
                else:
                    if self.recovery.should_retry():
                        print(f"Recovery Agent triggered retry {self.recovery.retry_count}")
                        self.state["plan"] = ["deploy", "verify"] + self.state["plan"]
                    elif self.recovery.ask_human():
                        self.state["plan"] = self.planner.create_plan("rollback")
                    else:
                        self.state["result"] = "Halted by Human"
                        break

            elif task == "rollback":
                res = Tools.rollback(change_request)
                self.state["result"] = "System Safely Reverted"

            self.memory.append((task, res))
        
        return self.state["result"]

# --- Execution ---
orchestrator = Orchestrator()

request = {
    "interface": "ge-0/0/1",
    "ip": "10.0.0.1/24"
}

final_status = orchestrator.run_workflow(
    goal = "deploy interface", 
    change_request = request
)

print(f"\nFinal Status: {final_status}")
