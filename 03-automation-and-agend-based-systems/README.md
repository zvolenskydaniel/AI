# Automation & Agent-Based Systems
---

## Overview
---
The landscape of network and systems engineering is undergoing a fundamental paradigm shift. Automation is maturing across a clear evolutionary trajectory:
- **Deterministic Automation:** Linear script-based execution (*Ansible, Python, NSO*) where outcomes are predefined and rigid.
- **Intelligent Workflow Automation:** Integration of LLMs for basic intent recognition and data transformation.
- **Autonomous Agentic Systems:** Systems capable of recursive reasoning, strategic planning, dynamic delegation, and autonomous error recovery.

By leveraging frameworks like **CrewAI**, **LangGraph**, and **AutoGen**, we move beyond the "Prompt-Response" cycle. We are now designing distributed reasoning architectures capable of:
- **Role-Based Collaboration:** Specialized agents (e.g., Planner vs. Executor) working toward a shared state.
- **Stateful Cycles:** Moving from "chains" to "graphs" that allow for loops, retries, and conditional logic.
- **Self-Healing Pipelines:** Systems that observe their own output and execute rollbacks or corrections upon failure.

## Goal
---
This project aims to bridge the gap between high-level AI theory and practical systems engineering. By the conclusion of this chapter, you will transition from a **Prompt Engineer** to an **AI Systems Architect**, mastering the ability to:
- **Architect Complex Topologies:** Distinguish between Single-Agent and Multi-Agent designs, selecting the right pattern based on task complexity.
- **Master Orchestration Patterns:** Implement Role-Based delegation (CrewAI style) and Cyclic State Machines (LangGraph style).
- **Build for Resiliency:** Engineer "Hardened" agents with integrated Failure Detection, Exponential Backoff Retries, and Automated Rollbacks.
- **Implement Deep Observability:** Construct transparent "Reasoning Traces" to monitor agentic thoughts, tool usage, and state transitions.
- **Evaluate Trade-offs:** Identify when multi-agent overhead is a necessity versus when a streamlined single-agent logic is superior.

## Core Concepts
---
- [Single-agent vs. Multi-agent Design](#single-agent-vs-multi-agent-design)
- [Architectural Patterns: Single vs. Multi-Agent](#architectural-patterns-single-vs-multi-agent)
- [CrewAI, LangGraph and AutoGen Fundamentals](#crewai-langgraph-and-autogen-fundamentals)
- [Task Orchestration and Delegation](#task-orchestration-and-delegation)
- [Failure Handling and Observability](#failure-handling-and-observability)

## Single-Agent vs. Multi-agent Design
---
An **AI Agent** is an autonomous software entity designed to achieve specific goals by perceiving its environment, reasoning through complex problems, and executing actions. While a standard chatbot is reactive (*answering a prompt and stopping*), an agent is proactive - it breaks down high-level objectives into a series of iterative steps, selecting the appropriate tools until the task is complete.

> **Analyst vs. Actor:** A Large Language Model (LLM) is an analyst that can process and summarize data. An AI Agent uses the LLM as its "brain" but adds an execution layer, allowing it to translate analysis into independent action.


### The Anatomy of an Agent
---
An agent functions through the orchestration of four core pillars:
- **The Brain (Model):** The reasoning engine (*typically an LLM*) that handles planning, decision-making, and intent recognition.
- **The Toolbox (Capabilities):**
  - **Perception/RAG:** Access to static data (*Vector DBs, documents, web search*).
  - **Action:** Ability to interact with the world (*APIs, SSH to routers, sending emails, database writes*).
- **The Memory (Context):**
  - **Short-term:** Maintaining the current "thought trace" and conversation history within the context window.
  - **Long-term:** Persisting learnings and user preferences across multiple sessions using databases or dedicated storage.
- **The Persona (*Instructions*):** System prompts that define the agent's Role (*e.g., "Network Security Auditor"*), Constraints, and Operating Procedures.

## Architectural Patterns: Single vs. Multi-Agent
---
As system complexity grows, we must decide whether to centralize intelligence in one agent or distribute it across a team.

### 1. Single-Agent Design (*The Solo Specialist*)
---
In this pattern, a single agent manages the entire reasoning loop. It is responsible for planning, tool selection, and execution from start to finish.
- **Best for:** Linear, well-defined tasks (*e.g., refactoring a specific code file, triaging incoming emails*).
- **Pros:** Lower latency, simpler debugging, and unified context (*no information is lost during handoffs*).
- **Cons:** Limited by the "cognitive load" of the model; as toolsets grow, the agent may become confused or hallucinate.

### 2. Multi-Agent Design (*The Collaborative Team*)
---
Intelligence is divided among specialized agents who communicate and delegate tasks. This often follows an **Orchestrator-Worker** or **Peer-to-Peer** pattern.
- **Best for:** Complex, multi-domain, or parallelizable tasks (*e.g., a "Researcher" agent gathering data while a "Writer" agent drafts a report*).
- **Pros:** High modularity, parallel execution, and better accuracy due to specialization (*each agent has a smaller, focused prompt*).
- **Cons:** High architectural complexity, increased token costs, and potential coordination overhead (*agents "misunderstanding" each other*).

### Comparison summary
---
| **Feature** | Single-Agent | Multi-Agent |
| ------- | ------------ | ----------- |
| **Complexity** | Low (*straightforward design*) | High (*requires orchestration logic*) |
| **Context** | Unified (*full history in one place*) | Distributed (*context must be shared*) |
| **Cost** | Efficient (lower token usage) | Higher (multiple agents = multiple calls) |
| **Reliability** | High for narrow tasks | Higher for complex, modular tasks |
| **Best Use Case** | Sequential focused workflows | Parallel cross-domain projects |

### Single-Agent Architecure
---
In a Single-Agent architecture, a central reasoning engine (*the LLM*) serves as the "*Controller*". It is responsible for the entire lifecycle of a task, from initial decomposition to final verification. This pattern is often referred to as a **ReAct loop** (*Reasoning and Acting*).

#### The Cognitive Loop
Unlike a traditional script, the agent does not follow a fixed sequence. Instead, it operates in a continuous cycle:
- **Perception & Reasoning:** The agent analyzes the user input and the current context to form a "Thought."
- **Action Planning:** It selects the most appropriate tool from its inventory.
- **Observation:** It executes the tool and "observes" the raw output (e.g., a database result or an error message).
- **Reflection:** It evaluates the observation. If the goal isn't met, it loops back to Step 1 to refine its approach.

#### Architectural Components
```text
      ┌──────────────────────────┐
      │       User Request       │
      └────────────┬─────────────┘
                   ▼
       ┌──────────────────────────┐
       │    Reasoning Engine      │ <───┐
       │ (LLM + System Prompt)    │     │
       └────────────┬─────────────┘     │
                    ▼                   │
       ┌──────────────────────────┐     │
       │     Tool Selection       │     │ (Iteration / Reflection)
       └────────────┬─────────────┘     │
                    ▼                   │
       ┌──────────────────────────┐     │
       │     Tool Execution       │     │
       └────────────┬─────────────┘     │
                    ▼                   │
       ┌──────────────────────────┐     │
       │  Observation & Feedback  │ ────┘
       └────────────┬─────────────┘
                    ▼
       ┌──────────────────────────┐
       │     Final Response       │
       └──────────────────────────┘

```

#### Why Choose Single-Agent?
- **Context Integrity:** The agent has a "*monolithic*" view of the conversation, meaning it doesn't lose details during handoffs to other agents.
- **Reduced Latency:** There is no overhead from inter-agent communication or negotiation.
- **Predictability:** For focused tasks (*like querying a specific network device or calculating a sum*), a single agent is easier to audit and constrain.

> **Key Limitation:** As the number of tools increases, "*Tool Confusion*" can occur. The agent may struggle to choose the right tool if its instructions become too bloated, which is the primary signal to move toward a **Multi-Agent** design.

#### Single-Agent Examples
The below example provides a foundational, "*manual*" representation of an **Agentic Workflow**. It mimics how an AI might process a task by breaking it down into distinct phases: *reasoning*, *action*, and *logging*.

```python
# examples/simple_agentic_workflow.py
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

```

The script defines a `SimpleAgent` class that follows a basic **Sense-Think-Act** cycle. Instead of using a complex neural network, it uses simple string matching to "decide" what to do.

The agent operates through three primary stages when the `run()` method is called:
- **Step 1: Reasoning** (`think`)
  The agent inspects the `goal` string. If it detects the keyword "sum", it selects the internal tool `use_math_tool`. This is a manual version of an LLM choosing a function to call.
- **Step 2: Execution** (`use_tool`)
  Once a decision is made, the agent executes the corresponding logic. In this case, it takes a list of numbers (the `input_data`) and applies Python’s built-in `sum()` function.
- **Step 3: Persistence** (`memory`)
  The agent doesn't just return the answer; it stores a record of the goal and the resulting calculation in its `self.memory` list. This allows the agent to "remember" what it did in previous steps.

In next example, let's build "*raw python*"" AI-driven network change deployment system via single-agent named **Network Change Agent**, which:
- validates change request
- generates configuration
- simulates deployment
- verifies results

This code simulates an *Automated Network Configuration Pipeline*. It mimics the high-stakes workflow of a Network Engineer by wrapping specific technical actions (*Tools*) into a structured operational sequence (*the Agent*).

The code is split into two logical components:
- **The Tools Class (*The Capability Layer*):** These are static methods that perform the "*heavy lifting*". They handle the specific syntax of network configuration (*like Junos-style set commands*) and simulate interactions with hardware.
```python
# examples/network_change_agent_v1.py
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

```

- **The NetworkChangeAgent (*The Orchestrator*):** This is the "*brain*" that manages the state. It doesn't just run commands; it follows a strict protocol to ensure safety and logging.
```python
# examples/network_change_agent_v1.py
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

```

- **Execution:**
```python
# examples/network_change_agent_v1.py
agent = NetworkChangeAgent()

change = {
    "interface": "ge-0/0/1",
    "ip": "10.0.0.1/24"
}

result = agent.run(change)
print("\nResult:", result)

```

> This script `network_change_agent_v1.py` is currently **linear** and is missing *retry logic*, *rollback functionality*, *memory tracking*, *execution stages*.

Therefore, the second version (*Phase-Based State Machine Agent*) is going to provide the mentioned improvements:
- retry deployment up to 3 times
- explicit rollback stage
- memory tracking
- clear execution stages

```python
# examples/network_change_agent_v2.py
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

```

The second version of the script `network_change_agent_v2.py` serves as a manual conceptual equivalent to **LangGraph** because it transitions from a linear "*chain*" to a **cyclic state-aware graph**. Instead of a fixed sequence, the agent uses a central `state` object to decide the next `node` (*method*) to execute, allowing for complex behaviors like loops (*retries*) and conditional branching (*rollback*) based on real-time feedback.

```text
┌──────────────────────────┐
|       Validation         |
└────────────┬─────────────┘
             |
             ▼
┌──────────────────────────┐  ┌───────┐
|       Deployment         |──| Retry |──┐
└────────────┬─────────────┘  └───────┘  |
             |                           |
             |                           |
             |                           │
             ▼                           │
┌──────────────────────────┐             |
|       Verification       |             │
└────────────┬─────────────┘             |
             │                           │
         ┌───┴─────┐                     │
         ▼         ▼                     │
     ┌──────┐  ┌──────────┐              |
     | Done |  | Rollback |──────────────┘
     └──────┘  └────┬─────┘
                    │
                    ▼
                 ┌──────┐
                 | Done |
                 └──────┘

```

> This structure is exactly how high-level agents handle uncertainty. By checking the `state` at every turn, the agent can handle "*flaky*" network hardware (*via the retry loop*) or unexpected environment states (*via the rollback*) without crashing the entire script.

#### From State Machines to Graph-Based Agents
Instead of thinking "*the agent runs phases*" let's start thinking "*the system executes a graph of states*". This shift is exactly what frameworks like
**LangGraph** formalize.

Graphs allow things that linear scripts cannot easily do:
- parallel work
- conditional routing
- dynamic planning

In next example, let's introduce a planner that decides the next step, instead of fixed flow `validation → deployment → verification`.
This introduces the **concept of plans** instead of phases.

```python
# examples/network_change_agent_v3.py
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


class NetworkChangeAgent:

    def __init__(self):
        self.memory = []
        self.state = {
            "plan": [],
            "config": None,
            "result": None
        }

    def planner(self, goal):
        """Produces a dynamic task list based on the goal"""
        if "deploy interface" in goal:
            return ["validate", "generate_config", "deploy", "verify"]
        return []

    def run(self, goal, change_request):
        print(f"Goal: {goal}")

        # Create the plan
        self.state["plan"] = self.planner(goal)
        print(f"\nPlanner created plan:")
        for i, task in enumerate(self.state["plan"], 1):
            print(f"{i} {task}")
        print("-" * 30)

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
  goal = "deploy interface config",
  change_request = change
)

print(f"\nFinal Result: {result}")
print("\n[REASONING TRACE]")
for i, (task, output) in enumerate(agent.memory, 1):
    print(f"{i}. >> {task.title()}\n   └─ Output: {output}")

```

By introducing a separate **Planner** function, the system transitions from a hard-coded script to an **intent-based** architecture. The agent first maps the user's natural language goal to a sequence of discrete capabilities (*Tools*) before entering the execution phase.

This is where the agent becomes **goal-oriented** instead of **phase-oriented**.  
Frameworks like:
- CrewAI
- LangGraph
- AutoGen

..are built around this concept.

Before diving into multi-step reasoning systems, let's update the **Planner** to return different plans based on different goals (*e.g., a "troubleshoot" goal that only runs 'validate' and 'verify' without 'deploy'*).

```python
# examples/network_change_agent_v4.py
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

```

The updated code `network_change_agent_v4.py` mimics **CrewAI’s Sequential Process**. The **Planner** acts as the "*Manager*" who defines the *Tasks list*, and the **Agent** acts as the "*Worker*" who goes through them one by one.

Now, let's make this look like **LangGraph**. Add a "*Re-planner*" step inside the loop. If `verify` fails, the agent wouldn't just stop; it would call the `planner()` again with a new goal: `"rollback the interface"`. Moreover, integrating a **Human-in-the-Loop (HITL)** mechanism is a core feature of enterprise-grade agentic frameworks. It prevents the "*runaway agent*" problem, ensuring that high-risk actions (*like a network rollback or a database wipe*) require a literal "*thumbs up*" from a human operator.

> In *LangGraph*, this is known as a **Breakpoint**.

```python
# examples/network_change_agent_v5.py
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

```

- **The Stateful Schema:** Notice how `self.state` acts as a "*Shared Blackboard*". By modifying `self.state["plan"]` dynamically, the agent demonstrates that in a graph-based system, the roadmap is not fixed - it is a mutable schema that nodes update in real-time based on environmental feedback.

- **Conditional Routing:** The `if res["state"] == "up"` block acts as a **Conditional Edge**. It evaluates a condition and determines the next node to visit.

- **Cyclic Self-Healing:** The architecture moves away from linear execution to **Cyclic Behavior**. By "*looping*" back to the planner after a failure, the agent demonstrates a self-healing autonomous loop, continuing its operations until the goal is either met or safely rolled back.

- **Deterministic Guardrails:** The **Human-in-the-Loop (HITL)** step acts as a "*Control Gate*" for risk mitigation. This ensures that even an autonomous system cannot initiate destructive commands—like a global configuration delete - without a "*second pair of eyes*" validating the recovery strategy.

- **State Persistence & Interruption:** This mimics **LangGraph’s Checkpointing**. In production, the agent would save its current state to a database and enter a "*Wait*" state, effectively going to "*sleep*" until an external human signal wakes it up to resume the plan.

- **Hybrid Traceability:** Every decision - the AI's failure detection, the Planner's recovery strategy, and the Human's approval - is captured in the **Reasoning Trace**. This provides a unified audit trail of both machine logic and human oversight.

In a graph-based system, the agent doesn't just iterate through a static list; it evaluates the state after each node and decides if it needs to **re-route** or **re-plan**. The above examples transitioned from a **Sequential Chain** to a **Cyclic Graph**. By implementing a *Re-planner* within the execution loop, the agent gains "*reflexive*" capabilities: it can observe its own failure at the verification stage and dynamically modify its task queue to include a rollback, ensuring the system remains in a safe state.

#### Conclusion: The Architectural Shift
In a graph-based system, execution is no longer a static "*to-do list*". It is a dynamic journey where the agent evaluates its progress after every node to decide if it needs to re-route or re-plan.

By implementing a **Re-planner** and **Breakpoints** within the loop, the agent gains "*reflexive*" capabilities. It can observe its own failures at the verification stage and intelligently interact with its human counterpart to modify its task queue. This transition from a **Sequential Chain** to a **Cyclic Graph** is what transforms basic automation into a resilient, enterprise-ready agentic system.

#### Conclusion: Lessons Learned from "Raw Python" Agent Design
Building an agentic system from the ground up - without the abstractions of CrewAI or LangGraph - reveals the true mechanics of AI-driven automation. Here are the three fundamental pillars discovered during this development:

**The Power of Intent over Scripts**

Traditional automation (*Ansible/Python scripts*) is deterministic; if a network state changes unexpectedly, the script breaks. By moving to a **Planner-based architecture**, we shift to **intent-based automation**. The system doesn't just run commands; it understands the goal and dynamically generates the path to reach it.

**State is the "Source of Truth"**

In a multi-agent or cyclic system, the **State Object** is the most critical component. It serves as the thread that ties the *Planner*, *Executor*, and *Critic* together. Mastering state management—specifically how to handle **Breakpoints** and **Re-planning** - is what separates a simple "*AI wrapper*" from a resilient, production-ready agent.

**Safety through Reflexivity**

True autonomy requires **reflexivity** - the ability of a system to look at its own output and realize it made a mistake. By implementing a **Re-planner** and **Human-in-the-Loop** gates, we prove that agents can be both autonomous and safe. We use the AI for speed and scale, but use architectural guardrails for reliability and risk mitigation.

**Final Thought**

Building this in raw Python proves that frameworks like **LangGraph** aren't magic - they are structured patterns for managing state and flow. Understanding these patterns at the code level allows us to build systems that aren't just "*smart*", but are predictable, auditable, and safe for critical infrastructure.

### Multi-Agent Architecture
--- 
**Multi-Agent Architectures** (*MAS*) distribute intelligence across a network of specialized agents. By narrow-casting the scope of each agent, we reduce the "*cognitive load*" on individual models, leading to higher accuracy and more robust system behavior.

Instead of one agent trying to do everything, MAS operates like a professional team - each member is a *Subject Matter Expert* (*SME*).

#### Core Orchestration Patterns
The way agents are organized defines the system's behavior. There are three primary patterns:
- **Sequential (*Chain*):** Agents work in a "*pipeline*". `Agent A` finishes and hands off the result to `Agent B` (*e.g., Researcher → Writer → Editor*).
- **Hierarchical (*Manager-Worker*):** A "*Manager*" agent receives the goal, creates a plan, and delegates tasks to specialized workers. The workers report back to the manager.
- **Joint Collaboration (*Peer-to-Peer*):** Agents share a common state or "*blackboard*" and contribute whenever their specific expertise is required.

#### Functional Roles in Agentic Teams
By splitting responsibilities, we create a built-in system of checks and balances:
- **The Planner:** Decomposes complex goals into actionable steps.
- **The Executor:** Interacts with tools (*APIs, CLI, Databases*) to perform the work.
- **The Critic/Reviewer:** Validates the output of the Executor against the original requirements.
- **The Security Guard:** Specialized in checking configs or code for vulnerabilities before deployment.

#### Why move to Multi-Agent?
- **Specialization:** You can use a smaller, faster model for simple execution and a powerful, "*heavy*" model for complex planning.
- **Modular Scalability:** You can add a "*Rollback Agent*" to your system without changing the logic of the "*Deployment Agent*".
- **Enhanced Safety:** By separating the "*Creator*" from the "*Verifier*", you significantly reduce the risk of "*hallucinated success*".
- **Parallelism:** Multiple agents can tackle different sub-tasks simultaneously (*e.g., configuring five different routers at once*).

#### Comparison Matrix: Trade-offs at Scale
| **Feature** | Multi-Agent Benefit | Operational Cost |
| ----------- | ------------------- | ---------------- |
| **Modularity** | Easy to swap or upgrade specific roles. | High architectural complexity. |
| **Specialization** | Agents are less likely to get "*confused*" by too many tools. | Increased coordination overhead and "*handoff*" latency." |
| **Safety** | Independent verification layers (*human-in-the-loop ready*). | Significant increase in token consumption (*cost*). |
| **Observability** | Clear logs for which role failed and why. | Difficult to debug "*inter-agent*" misunderstandings. |

#### Multi-Agent Examples

Below code simulates a **decoupled multi-agent system** by separating the "*thinking*" (*strategy*) from the "*doing*" (*action*). It mimics a classic architecture where one specialized agent defines the roadmap and another carries out the specific technical labor.

- **The PlannerAgent (*The "Brain"*):** This agent is responsible for **intent recognition**. It looks at a high-level goal and maps it to a specific task name. It doesn't know how to do the math; it only knows what needs to be done.

- **The ExecutorAgent (*The "Hands"*):** This agent contains the **operational logic**. It waits for a specific task instruction and a set of data. It doesn't care about the original "*goal*" (*the "why"*); it only cares about the "*task*" (*the "how"*).

```python
# examples/multi-agent_system.py
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
```

The logic follows a linear pipeline:
- **Decomposition**: The `goal` ("*calculate sum*") is passed to the **Planner**.
- **Instruction Generation**: The **Planner** returns a dictionary (*a "plan"*) that translates the vague goal into a structured instruction: `{"task": "calculate_sum"}`.
- **Handoff**: The main script takes the `task` from the *Planner*’s output and hands it - along with the raw data `[1, 2, 3]` - to the **Executor**.
- **Final Output**: The *Executor* identifies the "*calculate_sum*" command, runs the arithmetic, and returns the final value of **6**.

> By splitting these into two classes, we've created a system that is **modular**.

To level up this simulation, let's introduce a **CriticAgent**. In a production multi-agent system, the "*Critic*" acts as a quality control layer, ensuring the *Executor* didn't hallucinate or make a calculation error before the final result is delivered to the user.

The flow now evolves from a simple handoff to a *Feedback Loop*:
- **Planner**: Defines the task.
- **Executor**: Performs the task.
- **Critic**: Verifies the result against the original goal.

```python
# examples/updated_multi-agent_system.py
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
```

Behavior of the New System:
- **Self-Correction Potential**: In this version, the **CriticAgent** provides a safety net. If the **Executor** had returned a string or a `None` value by mistake, the **Critic** would flag it as an error rather than letting the system output bad data.
- **Separation of Concerns**: Each agent has a single responsibility. The **Planner** handles intent, the **Executor** handles math, and the **Critic** handles logic/policy.
- **Scalability**: You could now easily add a "*Retry*" loop where, if the **Critic** says `False`, the **Planner** has to try a different approach.


## CrewAI, LangGraph and AutoGen Fundamentals
---
| Framework | Core Idea                          | Strength                       |
| --------- | ---------------------------------- | ------------------------------ |
| CrewAI    | Role-based agents with delegation  | Simple multi-agent modeling    |
| LangGraph | Graph-based stateful workflows     | Production-grade orchestration |
| AutoGen   | Conversational agent collaboration | Research-oriented flexibility  |


### CrewAI
---
**CrewAI** is an open-source **Python framework for orchestrating autonomous AI agents**, allowing them to collaborate as a "crew" to tackle complex, multi-step tasks by defining specific roles, goals, and tools for each specialized agent, much like a human project team. It provides a layer for agents to communicate, delegate, and work together, shifting from single, general-purpose AI to specialized teams, making complex workflows more manageable and automated.

#### Key Concepts
- **Crews:** Teams of AI agents formed to work on a specific project or task.
- **Agents:** Individual AIs with defined roles (e.g., researcher, analyst, writer), goals, backstories, and access to specific tools.
- **Orchestration:** The framework manages how these agents interact, delegate, and share information, ensuring the overall workflow is efficient.
- **LLM-Agnostic** Works with various large language models (LLMs) from different providers like OpenAI, Anthropic, and Mistral.

#### How it works
Imagine planning a complex event: instead of one person doing everything, *CrewAI* assigns tasks like a human team would.
- **Planner Agent:** Coordinates the overall event.
- **Food Agent:** Handles catering details.
- **Decorations Agent:** Manages venue aesthetics.

> Source: https://www.crewai.com/ & https://docs.crewai.com/en/introduction


### LangGraph
---
**LangGraph** is an open-source library (*part of the LangChain ecosystem*) designed for building **stateful**, **multi-agent applications using graphs**. While other frameworks focus on linear sequences or predefined "*crews*", **LangGraph** treats agentic workflows as a series of **nodes** (*actions*) and **edges** (*paths*). This allows for **cyclic transitions**, meaning an agent can loop back to a previous step to retry a failed task or refine an answer based on new feedback. It is the industry standard for creating complex, non-linear reasoning systems that require high degrees of control and persistence.

#### Key Concepts
- **State Management:** A shared "*State*" object acts as the system's memory. Every node reads from and writes to this state, ensuring information is never lost during handoffs.
- **Nodes & Edges:** 
  - *Nodes* are the functions or agents performing the work.
  - *Edges* define the logic for moving between nodes, including **Conditional Edges** (*e.g., "If the output is invalid, go to the Rollback node; otherwise, go to End"*).
- **Persistence (*Checkpoints*):** LangGraph can "*checkpoint*" the state at every step. This allows for *Human-in-the-loop* interaction, where a human can inspect the state, approve it, or even "*rewind*" the agent to a previous step.
- **Cycles & Loops:** Unlike linear chains, LangGraph natively supports loops, making it ideal for self-healing systems where an agent must repeatedly attempt a task until it succeeds.

#### How it Works
Think of LangGraph like a **smart flowchart** for a Network Operations Center (NOC). Instead of a simple "*start-to-finish*" script, it builds a map of possibilities:
- **Entry Point:** The request arrives at the Planner Node.
- **The Loop:** The Executor Node attempts a configuration change.
- **The Decision:** The Verifier Node checks the result.
  - *If success:* Move to the **End Node**.
  - *If failure:* The **Conditional Edge** routes the flow back to the **Rollback Node**, and then potentially back to the **Planner** to try a different strategy.

> Source: https://langchain-ai.github.io/langgraph/ & https://blog.langchain.dev/langgraph/


### AutoGen
---
**AutoGen** is an open-source **Microsoft framework for building multi-agent AI applications**, allowing different AI agents (*powered by LLMs*) to converse, collaborate, and use tools to solve complex tasks, mimicking human teamwork with customizable roles and workflows, and offering low-code tools like *AutoGen Studio* for easier development and management. It simplifies creating systems that handle complex workflows, code execution, data analysis, and automation, reducing the need for extensive human intervention by coordinating specialized agents.

#### Key Features & Concepts
- **Multi-Agent Collaboration:** Define multiple agents (e.g., a planner, an engineer, a critic) that communicate and work together to achieve a goal.
- **Conversational Nature:** Agents interact through human-readable dialogue, making complex workflows more intuitive.
- **Customization & Flexibility:** Easily extend agents with new tools, define behaviors with code or natural language, and create modular workflows.
- **Tool Integration:** Agents can use external tools and APIs (like search engines, code interpreters).
- **Low-Code Development:** AutoGen Studio offers a visual, flow-chart-based interface for building and sharing workflows.
- **Observability:** Provides tracing and visibility into agent interactions and decisions, powered by OpenTelemetry.

#### How it Works
- **Define Agents:** Create agents with specific roles (*e.g., User Proxy, Planner, Coder, Critic*).
- **Set Up Conversation:** Establish how agents will communicate and trigger each other.
- **Execute Tasks:** Agents collaborate, passing messages, generating code, using tools, and seeking human input as needed to complete complex tasks.

> Source: https://www.microsoft.com/en-us/research/project/autogen/

## Task Orchestration and Delegation
---
Focus: Control vs Autonomy

We’ll explore:
- Central orchestrator vs emergent collaboration
- Planning agent patterns
- Hierarchical delegation
- Dynamic task generation
- Conditional routing
- Parallel task execution
- Tool calling patterns
- Human-in-the-loop checkpoints

This is where automation meets AI reasoning.

## Failure Handling and Observability
---
This is where most AI demos break.

We’ll cover:
- LLM failure modes
- Tool call failures
- Hallucination containment
- Structured output validation
- Retries & fallback chains
- Guardrails
- Logging agent reasoning
- Traceability
- Metrics

## Suggested Final Outcome Project
---
Given my background (network automation + Docker + APIs), a perfect capstone would be:

> AI Multi-Agent Network Change Orchestrator

Example:
- Planner agent creates change steps
- Validation agent checks policy compliance
- Execution agent calls REST API
- Verification agent checks state
- Recovery agent handles rollback
