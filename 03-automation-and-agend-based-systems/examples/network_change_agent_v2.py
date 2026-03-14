#
# 2026.03 AI: Learning Path
# zvolensky.daniel@gmail.com
#
# Phase-Based State Machine Agent
#

class Tools:

    @staticmethod
    def validate_change(change_request):
        if "interface" in change_request:
            return {"status": "valid"}
        return {"status": "invalid", "reason": "missing interface"}

    @staticmethod
    def generate_config(change_request):
        interface = change_request["interface"]
        ip = change_request.get("ip", "dhcp")
        return f"set interfaces {interface} unit 0 family inet address {ip}"

    @staticmethod
    def deploy_config(config):
        # simulate deployment
        return {"deployment": "success"}

    @staticmethod
    def verify_state(interface):
        # simulate verification
        return {"state": "up"}

    @staticmethod
    def rollback_config(change_request):
        # simulate rollback
        interface = change_request["interface"]
        ip = change_request.get("ip", "dhcp")
        return f"delete interfaces {interface} unit 0 family inet address {ip}"


class NetworkChangeAgent:

    def __init__(self):
        self.memory = []
        self.state = {
            "phase": None,
            "retry": 0,
            "config": None,
            "result": None
        }

    def run(self, change_request):
        print("Starting change workflow...\n")

        self.state["phase"] = "validation"

        while self.state["phase"] != "done":

            if self.state["phase"] == "validation":
                self.handle_validation(change_request)

            elif self.state["phase"] == "deployment":
                self.handle_deployment(change_request)

            elif self.state["phase"] == "verification":
                self.handle_verification(change_request)

            elif self.state["phase"] == "rollback":
                self.handle_rollback(change_request)

        return self.state["result"]

    
    # --- Phase Handlers ---
    def handle_validation(self, change_request):
        # 1. Validate
        validation = Tools.validate_change(change_request)
        self.memory.append(("validation", validation))

        if validation["status"] != "valid":
            self.state["result"] = f"Validation failed: {validation.get('reason')}"
            self.state["phase"] = "done"
            return

        # 2. Generate config
        config = Tools.generate_config(change_request)
        self.memory.append(("config", config))

        self.state["config"] = config
        self.state["phase"] = "deployment"

    def handle_deployment(self, change_request):
        config = self.state["config"]
        # 3. Deploy
        deployment = Tools.deploy_config(config)
        self.state["retry"] += 1

        if deployment["deployment"] == "success":
            self.memory.append(("deployment", deployment))
            self.state["phase"] = "verification"
            return

        if self.state["retry"] < 3:
            print(f"Retry deployment ({self.state['retry']})...")
            return

        self.memory.append(("deployment", deployment))
        self.state["result"] = "Deployment failed after retries"
        self.state["phase"] = "done"

    def handle_verification(self, change_request):
        # 4. Verify
        verification = Tools.verify_state(change_request["interface"])
        self.memory.append(("verification", verification))

        if verification["state"] == "up":
            self.state["result"] = f"Change completed. Interface state: {verification['state']}"
            self.state["phase"] = "done"
        else:
            self.state["phase"] = "rollback"

    def handle_rollback(self, change_request):
        # 5. Rollback
        rollback = Tools.rollback_config(change_request)
        self.memory.append(("rollback", rollback))

        self.state["result"] = "Change rolled back due to failed verification."
        self.state["phase"] = "done"

# --- Execution ---
agent = NetworkChangeAgent()

change = {
    "interface": "ge-0/0/1",
    "ip": "10.0.0.1/24"
}

result = agent.run(change)

print(f"Result: {result}\n")

print(f"--- Memory: ")
for i, (task, result) in enumerate(agent.memory, 1):
    print(f"{i}. >> {task.title()}")
    print(f"   └─ Output: {result}")
print(f"{'='*30}")

print(f"--- State: ")
for i, (key, value) in enumerate(agent.state.items(), 1):
    label = key.replace('_', ' ').title()    
    print(f"{i}. >> {label}")
    print(f"   └─ {value}")
print(f"{'='*30}")
