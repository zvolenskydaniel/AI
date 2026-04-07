#
# 2026.04 AI: Learning Path
# zvolensky.daniel@gmail.com
#

# ---- Import libraries ----
from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import BaseTool
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional

# Load OPEN_AI_KEY
load_dotenv()

# Define LLM
llm = LLM(
    model = "gpt-4o-mini",
    temperature = 0
)

# --- Define Tools Classes ---
class ChangeRequest(BaseModel):
    interface: str = Field(description="The name of the interface, e.g., 'ge-0/0/1'")
    ip: Optional[str] = Field(default="dhcp", description="The IP address or 'dhcp'")

class ValidateInput(BaseModel):
    change_request: ChangeRequest

class Config(BaseModel):
    config: str

class VerifyInput(BaseModel):
    interface: str = Field(description="The name of the interface to verify, e.g., 'ge-0/0/1'")

class ValidateTool(BaseTool):
    name: str = "validate"
    description: str = "Validate a network change request."
    args_schema: type[BaseModel] = ValidateInput

    def _run(self, change_request: dict) -> dict:
        """validate"""
        print(f"DEBUG TOOL INPUT: {change_request}")
        if "interface" in change_request:
            return {"status": "valid"}
        return {"status": "invalid", "reason": "missing interface"}

class GenerateConfigTool(BaseTool):
    name: str = "generate_config"
    description: str = "Generate configuration on the base of provided network change request."
    args_schema: type[BaseModel] = ValidateInput

    def _run(self, change_request: dict) -> str:
        """generate_config"""
        interface = change_request.get("interface")
        ip = change_request.get("ip", "dhcp")
        return f"set interfaces {interface} unit 0 family inet address {ip}"

class DeployTool(BaseTool):
    name: str = "deploy_config"
    description: str = "Generate configuration on the base of provided network change request."
    args_schema: type[BaseModel] = Config

    def _run(self, config: str) -> dict:
        """deploy"""
        return {"deployment": "success"}

class VerifyTool(BaseTool):
    name: str = "verify"
    description: str = "Verify applied configuration, the modified interface must be UP."
    args_schema: type[BaseModel] = VerifyInput

    def _run(self, interface: str) -> dict:
        """verify"""
        print(f"DEBUG: Verifying interface state for: {interface}")
        # return {"interface": interface, "state": "up"}
        return {"interface": interface, "state": "down"}

class RollbackTool(BaseTool):
    name: str = "rollback"
    description: str = "Revert a change by generating the 'delete' command for the specified interface."
    args_schema: type[BaseModel] = ValidateInput

    def _run(self, change_request: dict) -> str:
        """generate_rollback_config"""
        interface = change_request.get("interface")
        ip = change_request.get("ip", "dhcp")
        return f"delete interfaces {interface} unit 0 family inet address {ip}"

validate_tool = ValidateTool()
generate_config_tool = GenerateConfigTool()
deploy_tool = DeployTool()
verify_tool = VerifyTool()
rollback_tool = RollbackTool()

# --- Define the Agents --
planner_agent = Agent(
    role="Network Planner",
    goal="Determine the correct sequence of steps to achieve the network goal",
    backstory="Expert network architect who plans safe execution steps",
    llm=llm,
    verbose=True
)

executor_agent = Agent(
    role="Network Executor",
    goal="Execute network configuration tasks",
    backstory="Automation engineer executing configurations",
    tools=[validate_tool, generate_config_tool, deploy_tool, rollback_tool],
    verbose=True
)

verifier_agent = Agent(
    llm=llm,
    role="Network Verifier",
    goal="Verify the state of network devices",
    backstory="Monitoring system ensuring network health",
    tools=[verify_tool, rollback_tool],
    verbose=True
)

# --- Define the Tasks ---
plan_task = Task(
    description="Create execution plan for: {change_request}",
    expected_output="Ordered steps",
    agent=planner_agent
)

validate_task = Task(
    description=(
        "Validate the change request.\n"
        "You MUST pass the full 'change_request' dictionary to the validate tool.\n"
        "Change request: {change_request}"
    ),
    expected_output="Validation result",
    agent=executor_agent
)

config_task = Task(
    description="Generate configuration for: {change_request}",
    expected_output="Network configuration",
    agent=executor_agent,
    context=[validate_task]
)

deploy_task = Task(
    description="Deploy the generated configuration",
    expected_output="Deployment result",
    agent=executor_agent,
    context=[config_task]
)

verify_task = Task(
    description=(
        "1. Verify the interface state for: {change_request} using the verify tool.\n"
        "2. If the state is 'down', you MUST use the rollback tool immediately to revert the changes.\n"
        "3. If the state is 'up', provide a final success report."
    ),
    expected_output="Verification result and confirmation of rollback if it was required.",
    agent=verifier_agent
)

# --- Create Crew ---
crew = Crew(
    agents=[planner_agent, executor_agent, verifier_agent],
    tasks=[validate_task, config_task, deploy_task, verify_task],
    process=Process.sequential,
    verbose=False
)

# --- Execute ---
result = crew.kickoff(
    inputs={
        "change_request": {
            "interface": "ge-0/0/1",
            "ip": "10.0.0.1/24"
        }
    }
)

print("\nFinal Result:")
print(result)

