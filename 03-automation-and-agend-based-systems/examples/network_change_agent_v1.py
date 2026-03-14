#
# 2026.03 AI: Learning Path
# zvolensky.daniel@gmail.com
#
# Linear Agent
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


class NetworkChangeAgent:

    def __init__(self):
        self.memory = []
        self.state = {}

    def run(self, change_request):
        print("Starting change workflow...\n")

        # 1. Validate
        validation = Tools.validate_change(change_request)
        self.memory.append(("validation", validation))

        if validation["status"] != "valid":
            return f"Validation failed: {validation.get('reason')}"

        # 2. Generate config
        config = Tools.generate_config(change_request)
        self.memory.append(("config", config))

        # 3. Deploy
        deployment = Tools.deploy_config(config)
        self.memory.append(("deployment", deployment))

        if deployment["deployment"] != "success":
            return "Deployment failed"

        # 4. Verify
        verification = Tools.verify_state(change_request["interface"])
        self.memory.append(("verification", verification))

        return f"Change completed. Interface state: {verification['state']}"

# --- Execution ---
agent = NetworkChangeAgent()

change = {
    "interface": "ge-0/0/1",
    "ip": "10.0.0.1/24"
}

result = agent.run(change)

print("\nResult:", result)

for i, (task, result) in enumerate(agent.memory, 1):
    print(f"{i}. >> {task.title()}")
    print(f"   └─ Output: {result}")
print(f"{'='*30}")
