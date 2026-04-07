# Automation & Agent-Based Systems
---

## Overview
---
The landscape of network and systems engineering is undergoing a fundamental paradigm shift. Automation is maturing across a clear evolutionary trajectory:
- **Deterministic Automation:** Linear script-based execution (*Ansible, Python, NSO*) where outcomes are predefined and rigid.
- **Intelligent Workflow Automation:** Integration of LLMs for basic intent recognition and data transformation.
- **Autonomous Agentic Systems:** Systems capable of recursive reasoning, strategic planning, dynamic delegation, and autonomous error recovery.

By leveraging frameworks like **CrewAI**, **LangGraph**, and **AutoGen**, we move beyond the *"Prompt-Response"* cycle. We are now designing distributed reasoning architectures capable of:
- **Role-Based Collaboration:** Specialized agents (*e.g., Planner vs. Executor*) working toward a shared state.
- **Stateful Cycles:** Moving from "chains" to "graphs" that allow for loops, retries, and conditional logic.
- **Self-Healing Pipelines:** Systems that observe their own output and execute rollbacks or corrections upon failure.

## Goal
---
This project aims to bridge the gap between high-level AI theory and practical systems engineering. The objective is the transition from **Prompt Engineering** to **AI Systems Architecture** through the mastery of:
- **Architect Complex Topologies:** Distinguish between Single-Agent and Multi-Agent designs, selecting the right pattern based on task complexity.
- **Master Orchestration Patterns:** Implement Role-Based delegation (CrewAI style) and Cyclic State Machines (LangGraph style).
- **Build for Resiliency:** Engineer *"Hardened"* agents with integrated Failure Detection, Exponential Backoff Retries, and Automated Rollbacks.
- **Implement Deep Observability:** Construct transparent "Reasoning Traces" to monitor agentic thoughts, tool usage, and state transitions.
- **Evaluate Trade-offs:** Identify when multi-agent overhead is a necessity versus when a streamlined single-agent logic is superior.

## Core Concepts
---
- [Single-agent vs. Multi-agent Design](#single-agent-vs-multi-agent-design)
  - [The Anatomy of an Agent](#the-anatomy-of-an-agent)
- [Architectural Patterns: Single vs. Multi-Agent](#architectural-patterns-single-vs-multi-agent)
  - [Single-Agent Design](#single-agent-design-the-solo-specialist)
  - [Multi-Agent Design](#multi-agent-design-the-collaborative-team)
  - [Comparison summary](#comparison-summary)
  - [Single-Agent Architecture](#single-agent-architecture)
    - [The Cognitive Loop](#the-cognitive-loop)
    - [Architectural Components](#architectural-components)
    - [Why to Choose Single-Agent?](#why-to-choose-single-agent)
    - [Single-Agent Examples](#single-agent-examples)
      - [Simple Agentic Workflow](#simple-agentic-workflow)
      - [Linear Agent](#linear-agent)
      - [Phase-Based State Machine Agent](#phase-based-state-machine-agent)
    - [From State Machines to Graph-Based Agents](#from-state-machines-to-graph-based-agents)
      - [Plan-Based Execution Agent](#plan-based-execution-agent)
      - [Mimic CrewAI’s Sequential Process](#mimic-crewais-sequential-process)
      - [Mimnic LangGraph's Dynamic Replanning Loop](#mimnic-langgraphs-dynamic-replanning-loop)
    - [Reliability and Tiered Escalation Patterns](#reliability-and-tiered-escalation-patterns)
    - [Conclusion: The Architectural Shift](#conclusion-the-architectural-shift)
    - [Conclusion: Lessons Learned from *"Raw Python"* Agent Design](#conclusion-lessons-learned-from-raw-python-agent-design)
    - [Single-Agent: Summary](#single-agent-summary)
  - [Multi-Agent Architecture](#multi-agent-architecture)
    - [Core Orchestration Patterns](#core-orchestration-patterns)
    - [Functional Roles in Agentic Teams](#functional-roles-in-agentic-teams)
    - [Why move to Multi-Agent?](#why-move-to-multi-agent)
    - [Comparison Matrix: Trade-offs at Scale](#comparison-matrix-trade-offs-at-scale)
    - [Multi-Agent Examples](#multi-agent-examples)
    - [Mapping Manual Logic to Industry Frameworks](#mapping-manual-logic-to-industry-frameworks)
    - [Framework Philosophy: CrewAI vs. LangGraph](#framework-philosophy-crewai-vs-langgraph)
      - [CrewAI: Role-Based Collaboration](#crewai-role-based-collaboration)
      - [LangGraph: State-Based Cycles](#langgraph-state-based-cycles)
  - [Multi-Agent: Summary](#multi-agent-summary)
- [CrewAI, LangGraph and AutoGen Fundamentals](#crewai-langgraph-and-autogen-fundamentals)
  - [CrewAI](#crewai)
    - [Key Concepts](#key-concepts)
    - [How it works](#how-it-works)
    - [CrewAI Examples](#crewai-examples)
      - [Manual Orchestration]()
      - [Manual Orchestration with Pydantic Schema]()
      - [Streamlined Pipeline Orchestration]()
      - [Declarative Frameworks]()
      - [Network Change Agent via CrewAI]()
  - [LangGraph](#langgraph)
    - [Key Concepts](#key-concepts-1)
    - [How it Works](#how-it-works-1)
  - [AutoGen](#autogen)
    - [Key Features & Concepts](#key-features--concepts)
    - [How it Works](#how-it-works-2)
- [Task Orchestration and Delegation](#task-orchestration-and-delegation)
- [Failure Handling and Observability](#failure-handling-and-observability)

## Single-Agent vs. Multi-agent Design
---
An **AI Agent** is an autonomous software entity designed to achieve specific goals by perceiving its environment, reasoning through complex problems, and executing actions. While a standard chatbot is reactive (*answering a prompt and stopping*), an agent is proactive - it breaks down high-level objectives into a series of iterative steps, selecting the appropriate tools until the task is complete.

> **Analyst vs. Actor:** A Large Language Model (LLM) is an analyst that can process and summarize data. An AI Agent uses the LLM as its *"brain"* but adds an execution layer, allowing it to translate analysis into independent action.


### The Anatomy of an Agent
---
An agent functions through the orchestration of four core pillars:
- **The Brain (*Model*):** the reasoning engine (*typically an LLM*) that handles planning, decision-making, and intent recognition
- **The Toolbox (*Capabilities*):**
  - **Perception/RAG:** access to static data (*Vector DBs, documents, web search*)
  - **Action:** ability to interact with the world (*APIs, SSH to routers, sending emails, database writes*)
- **The Memory (*Context*):**
  - **Short-term:** maintaining the current *"thought trace"* and conversation history within the context window
  - **Long-term:** persisting learnings and user preferences across multiple sessions using databases or dedicated storage
- **The Persona (*Instructions*):** system prompts that define the agent's Role (*e.g., "Network Security Auditor"*), Constraints, and Operating Procedures

## Architectural Patterns: Single vs. Multi-Agent
---
As system complexity grows, the decision must be taken whether to centralize intelligence in one agent or distribute it across a team.

### Single-Agent Design (*The Solo Specialist*)
---
In this pattern, a single agent manages the entire reasoning loop. It is responsible for planning, tool selection, and execution from start to finish.
- **Best for:** linear, well-defined tasks (*e.g., refactoring a specific code file, triaging incoming emails*)
- **Pros:** lower latency, simpler debugging, and unified context (*no information is lost during handoffs*)
- **Cons:** limited by the *"cognitive load"* of the model; as toolsets grow, the agent may become confused or hallucinate

### Multi-Agent Design (*The Collaborative Team*)
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

### Single-Agent Architecture
---
In a Single-Agent architecture, a central reasoning engine (*the LLM*) serves as the *"Controller"*. It is responsible for the entire lifecycle of a task, from initial decomposition to final verification. This pattern is often referred to as a **ReAct loop** (*Reasoning and Acting*).

#### The Cognitive Loop
---
Unlike a traditional script, the agent does not follow a fixed sequence. Instead, it operates in a continuous cycle:
- **Perception & Reasoning:** the agent analyzes the user input and the current context to form a *"Thought"*
- **Action Planning:** it selects the most appropriate tool from its inventory
- **Observation:** it executes the tool and *"observes"* the raw output (*e.g., a database result or an error message*)
- **Reflection:** it evaluates the observation, if the goal isn't met, it loops back to Step 1 to refine its approach

#### Architectural Components
---
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

#### Why to Choose Single-Agent?
---
- **Context Integrity:** the agent has a *"monolithic"* view of the conversation, meaning it doesn't lose details during handoffs to other agents
- **Reduced Latency:** there is no overhead from inter-agent communication or negotiation
- **Predictability:** tor focused tasks (*like querying a specific network device or calculating a sum*), a single agent is easier to audit and constrain

> **Key Limitation:** As the number of tools increases, *"Tool Confusion"* can occur. The agent may struggle to choose the right tool if its instructions become too bloated, which is the primary signal to move toward a **Multi-Agent** design.

#### Single-Agent Examples
---
##### Simple Agentic Workflow
---
The below example provides a foundational, *"manual"* representation of an **Agentic Workflow**. It mimics how an AI might process a task by breaking it down into distinct phases: *reasoning*, *action*, and *logging*.

```python
# 03-automation-and-agend-based-systems/examples/simple_agentic_workflow.py
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

The script defines a `SimpleAgent` class that follows a basic **Sense-Think-Act** cycle. Instead of using a complex neural network, it uses simple string matching to *"decide"* what to do.

The agent operates through three primary stages when the `run()` method is called:
- **Step 1: Reasoning** (`think`)
  The agent inspects the `goal` string. If it detects the keyword *"sum"*, it selects the internal tool `use_math_tool`. This is a manual version of an LLM choosing a function to call.
- **Step 2: Execution** (`use_tool`)
  Once a decision is made, the agent executes the corresponding logic. In this case, it takes a list of numbers (the `input_data`) and applies Python’s built-in `sum()` function.
- **Step 3: Persistence** (`memory`)
  The agent doesn't just return the answer; it stores a record of the goal and the resulting calculation in its `self.memory` list. This allows the agent to *"remember"* what it did in previous steps.

##### Linear Agent
---
In next example, let's build *"raw python"* AI-driven network change deployment system via single-agent named **Network Change Agent**, which:
- validates change request
- generates configuration
- simulates deployment
- verifies results

This code simulates an *Automated Network Configuration Pipeline*. It mimics the high-stakes workflow of a Network Engineer by wrapping specific technical actions (*Tools*) into a structured operational sequence (*the Agent*).

The code is split into two logical components:
- **The Tools Class (*The Capability Layer*):** These are static methods that perform the *"heavy lifting"*. They handle the specific syntax of network configuration (*like Junos-style set commands*) and simulate interactions with hardware.
```python
# 03-automation-and-agend-based-systems/examples/network_change_agent_v1.py
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

- **The NetworkChangeAgent (*The Orchestrator*):** This is the *"brain"* that manages the state. It doesn't just run commands; it follows a strict protocol to ensure safety and logging.
```python
# 03-automation-and-agend-based-systems/examples/network_change_agent_v1.py
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
# 03-automation-and-agend-based-systems/examples/network_change_agent_v1.py
agent = NetworkChangeAgent()

change = {
    "interface": "ge-0/0/1",
    "ip": "10.0.0.1/24"
}

result = agent.run(change)
print("\nResult:", result)

```

> This script `network_change_agent_v1.py` is currently **linear** and is missing *retry logic*, *rollback functionality*, *memory tracking*, *execution stages*.

##### Phase-Based State Machine Agent
---
Therefore, the second version (*Phase-Based State Machine Agent*) is going to provide the mentioned improvements:
- retry deployment up to 3 times
- explicit rollback stage
- memory tracking
- clear execution stages

```python
# 03-automation-and-agend-based-systems/examples/network_change_agent_v2.py
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

The second version of the script `network_change_agent_v2.py` serves as a manual conceptual equivalent to **LangGraph** because it transitions from a linear *"chain"* to a **cyclic state-aware graph**. Instead of a fixed sequence, the agent uses a central `state` object to decide the next `node` (*method*) to execute, allowing for complex behaviors like loops (*retries*) and conditional branching (*rollback*) based on real-time feedback.

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

> This structure is exactly how high-level agents handle uncertainty. By checking the `state` at every turn, the agent can handle *"flaky"* network hardware (*via the retry loop*) or unexpected environment states (*via the rollback*) without crashing the entire script.

#### From State Machines to Graph-Based Agents
---
Instead of thinking *"the agent runs phases"* let's start thinking *"the system executes a graph of states"*. This shift is exactly what frameworks like
**LangGraph** formalize.

Graphs allow things that linear scripts cannot easily do:
- parallel work
- conditional routing
- dynamic planning

##### Plan-Based Execution Agent
---
In next example, let's introduce a planner that decides the next step, instead of fixed flow `validation → deployment → verification`.
This introduces the **concept of plans** instead of phases.

```python
# 03-automation-and-agend-based-systems/examples/network_change_agent_v3.py
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


##### Mimic CrewAI’s Sequential Process
---
Before diving into multi-step reasoning systems, let's update the **Planner** to return different plans based on different goals (*e.g., a "troubleshoot" goal that only runs 'validate' and 'verify' without 'deploy'*).

```python
# 03-automation-and-agend-based-systems/examples/network_change_agent_v4.py
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

The updated code `network_change_agent_v4.py` mimics **CrewAI’s Sequential Process**. The **Planner** acts as the *"Manager"* who defines the *Tasks list*, and the **Agent** acts as the *"Worker"* who goes through them one by one.

##### Mimnic LangGraph's Dynamic Replanning Loop
---
Now, let's make this look like **LangGraph**. Add a *"Re-planner"* step inside the loop. If `verify` fails, the agent wouldn't just stop; it would call the `planner()` again with a new goal: `"rollback the interface"`. Moreover, integrating a **Human-in-the-Loop (HITL)** mechanism is a core feature of enterprise-grade agentic frameworks. It prevents the *"runaway agent"* problem, ensuring that high-risk actions (*like a network rollback or a database wipe*) require a literal *"thumbs up"* from a human operator.

> In *LangGraph*, this is known as a **Breakpoint**.

```python
# 03-automation-and-agend-based-systems/examples/network_change_agent_v5.py
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

- **The Stateful Schema:** Notice how `self.state` acts as a *"Shared Blackboard"*. By modifying `self.state["plan"]` dynamically, the agent demonstrates that in a graph-based system, the roadmap is not fixed - it is a mutable schema that nodes update in real-time based on environmental feedback.

- **Conditional Routing:** The `if res["state"] == "up"` block acts as a **Conditional Edge**. It evaluates a condition and determines the next node to visit.

- **Cyclic Self-Healing:** The architecture moves away from linear execution to **Cyclic Behavior**. By *"looping"* back to the planner after a failure, the agent demonstrates a self-healing autonomous loop, continuing its operations until the goal is either met or safely rolled back.

- **Deterministic Guardrails:** The **Human-in-the-Loop (HITL)** step acts as a *"Control Gate"* for risk mitigation. This ensures that even an autonomous system cannot initiate destructive commands - like a global configuration delete - without a *"second pair of eyes"* validating the recovery strategy.

- **State Persistence & Interruption:** This mimics **LangGraph’s Checkpointing**. In production, the agent would save its current state to a database and enter a *"Wait"* state, effectively going to *"sleep"* until an external human signal wakes it up to resume the plan.

- **Hybrid Traceability:** Every decision - the AI's failure detection, the Planner's recovery strategy, and the Human's approval - is captured in the **Reasoning Trace**. This provides a unified audit trail of both machine logic and human oversight.

#### Reliability and Tiered Escalation Patterns
---
In enterprise-grade agentic systems, reliability is achieved through a multi-layered approach to failure. Rather than treating every deviation as a terminal error, the system is designed to distinguish between transient glitches and persistent state failures.

To accommodate a tiered recovery strategy, let's introduce a retry counter within the agent's state. This allows the system to remain autonomous during minor transient failures and only escalate to a **Human-in-the-Loop (HITL)** breakpoint when the automated recovery threshold is exceeded.

The key logic updates are:
- **Self-Healing Loop:** if verification fails, the agent first re-adds `deploy` and `verify` to the plan queue
- **Threshold Escalation:** a `retry_count` is tracked; once it hits a defined limit (*e.g., 2 attempts*), the agent triggers the `input()` prompt to ask for human guidance
- **State Reset:** if the human approves a rollback, the retry counter is typically reset or the flow is redirected to the recovery plan

```python
# 03-automation-and-agend-based-systems/examples/network_change_agent_v6.py
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
```

In a graph-based system, the agent doesn't just iterate through a static list; it evaluates the state after each node and decides if it needs to **re-route** or **re-plan**. The above examples transitioned from a **Sequential Chain** to a **Cyclic Graph**. By implementing a *Re-planner* within the execution loop, the agent gains *"reflexive"* capabilities: it can observe its own failure at the verification stage and dynamically modify its task queue to include a rollback, ensuring the system remains in a safe state.

#### Conclusion: The Architectural Shift
---
In a graph-based system, execution is no longer a static *"to-do list"*. It is a dynamic journey where the agent evaluates its progress after every node to decide if it needs to re-route or re-plan.

By implementing a **Re-planner** and **Breakpoints** within the loop, the agent gains *"reflexive"* capabilities. It can observe its own failures at the verification stage and intelligently interact with its human counterpart to modify its task queue. This transition from a **Sequential Chain** to a **Cyclic Graph** is what transforms basic automation into a resilient, enterprise-ready agentic system.

#### Conclusion: Lessons Learned from *"Raw Python"* Agent Design
---
Building an agentic system from the ground up - without the abstractions of CrewAI or LangGraph - reveals the true mechanics of AI-driven automation. Here are the three fundamental pillars discovered during this development:

**The Power of Intent over Scripts**
Traditional automation (*Ansible/Python scripts*) is deterministic; if a network state changes unexpectedly, the script breaks. By moving to a **Planner-based architecture**, we shift to **intent-based automation**. The system doesn't just run commands; it understands the goal and dynamically generates the path to reach it.

**State is the *"Source of Truth"***
In a multi-agent or cyclic system, the **State Object** is the most critical component. It serves as the thread that ties the *Planner*, *Executor*, and *Critic* together. Mastering state management - specifically how to handle **Breakpoints** and **Re-planning** - is what separates a simple *"AI wrapper"* from a resilient, production-ready agent.

**Safety through Reflexivity**
True autonomy requires **reflexivity** - the ability of a system to look at its own output and realize it made a mistake. By implementing a **Re-planner** and **Human-in-the-Loop** gates, we prove that agents can be both autonomous and safe. We use the AI for speed and scale, but use architectural guardrails for reliability and risk mitigation.

#### Single-Agent: Summary
---
Building this in raw Python proves that frameworks like **LangGraph** aren't magic - they are structured patterns for managing state and flow. Understanding these patterns at the code level allows to build systems that aren't just *"smart"*, but are predictable, auditable, and safe for critical infrastructure.

### Multi-Agent Architecture
--- 
**Multi-Agent Architectures** (*MAS*) distribute intelligence across a network of specialized agents. By narrow-casting the scope of each agent, we reduce the "*cognitive load*" on individual models, leading to higher accuracy and more robust system behavior.

Instead of one agent trying to do everything, MAS operates like a professional team - each member is a *Subject Matter Expert* (*SME*).

#### Core Orchestration Patterns
---
The way agents are organized defines the system's behavior. There are three primary patterns:
- **Sequential (*Chain*):** Agents work in a "*pipeline*". `Agent A` finishes and hands off the result to `Agent B` (*e.g., Researcher → Writer → Editor*).
- **Hierarchical (*Manager-Worker*):** A "*Manager*" agent receives the goal, creates a plan, and delegates tasks to specialized workers. The workers report back to the manager.
- **Joint Collaboration (*Peer-to-Peer*):** Agents share a common state or "*blackboard*" and contribute whenever their specific expertise is required.

#### Functional Roles in Agentic Teams
---
By splitting responsibilities, we create a built-in system of checks and balances:
- **The Planner:** Decomposes complex goals into actionable steps.
- **The Executor:** Interacts with tools (*APIs, CLI, Databases*) to perform the work.
- **The Verifier:** Validates the output of the Executor against the original requirements.
- **The Security Guard:** Specialized in checking configs or code for vulnerabilities before deployment.

#### Why move to Multi-Agent?
---
- **Specialization:** You can use a smaller, faster model for simple execution and a powerful, "*heavy*" model for complex planning.
- **Modular Scalability:** You can add a "*Rollback Agent*" to your system without changing the logic of the "*Deployment Agent*".
- **Enhanced Safety:** By separating the "*Creator*" from the "*Verifier*", you significantly reduce the risk of "*hallucinated success*".
- **Parallelism:** Multiple agents can tackle different sub-tasks simultaneously (*e.g., configuring five different routers at once*).

#### Comparison Matrix: Trade-offs at Scale
---
| **Feature** | Multi-Agent Benefit | Operational Cost |
| ----------- | ------------------- | ---------------- |
| **Modularity** | Easy to swap or upgrade specific roles. | High architectural complexity. |
| **Specialization** | Agents are less likely to get "*confused*" by too many tools. | Increased coordination overhead and "*handoff*" latency." |
| **Safety** | Independent verification layers (*human-in-the-loop ready*). | Significant increase in token consumption (*cost*). |
| **Observability** | Clear logs for which role failed and why. | Difficult to debug "*inter-agent*" misunderstandings. |

#### Multi-Agent Examples
---
To provide continuity of the single-agent examples, upcomming multi-agent example is decomposing the logic to multiple agents. Each agent is a discrete class with a single responsibility. An *Orchestrator* is used to manage the handoffs between them.

```python
# 03-automation-and-agend-based-systems/examples/multi_agent_implementation.py
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
request = {"interface": "ge-0/0/1", "ip": "10.0.0.1/24"}
final_status = orchestrator.run_workflow("deploy interface", request)

print(f"\nFinal Status: {final_status}")
```

The architecture has been evolved from a monolithic **Single-Agent** design into a **Multi-Agent Orchestration**. By decoupling the **Planner**, **Executor**, **Verifier**, and **Recovery** logic into specialized entities, the system gains modularity. This mirrors the design patterns of enterprise frameworks like *CrewAI* and *LangGraph*, where specialized *"workers"* collaborate under a central orchestration layer to manage complex, non-linear workflows.

#### Mapping Manual Logic to Industry Frameworks
---
The transition from *"Raw Python"* to a specialized framework involves moving from **imperative code** (*telling the system how to loop*) to **declarative configuration** (*defining the roles and relationships*).

| Component in Example Code | CrewAI Equivalent | LangGraph Equivalent |
| ------------------------- | ----------------- | -------------------- |
| `PlannerAgent` | `Manager LLM` / `Process.sequential` | `Router Node` / `Conditional Edge` |
| `ExecutorAgent` | `Agent(role='Configurator')` | `ToolNode` or `Action Node` |
| `VerifierAgent` | `Task(description='Validate...')` | `Validation Node` |
| `RecoveryAgent` | `Task(max_retries=n)` | `Retry Loop` / `Checkpointer` |
| `Orchestrator.state` | `Task Output` / `Context` | `State Schema` (The Graph State) |
| `input()` (HITL) | `human_input=True` | `Interrupt` / `Breakpoint` |

#### Framework Philosophy: CrewAI vs. LangGraph
---
While both frameworks can execute the multi-agent logic developed in this project, they approach the **Orchestration** differently:

##### CrewAI: Role-Based Collaboration
---
**CrewAI** excels at **top-down delegation**. In this framework, the "Crews" are treated like a human department.
  - **Philosophy:** *"Who is the best expert for this task?"*
  - **Example Code Map:** the `Orchestrator` acts as the `Crew`, and the specialized classes are the `Agents`
  - **Best Use Case:** high-level business processes, research, and content creation where specialized *"personas"* are required

##### LangGraph: State-Based Cycles
---
**LangGraph** excels at **low-level control** and **reliability**. It treats the workflow as a formal state machine (*a directed graph*).
  - **Philosophy:** *"Given the current state, what is the next logical node?"*
  - **Example Code Map:** the while loop and the `plan.pop(0)` logic are a manual implementation of a **LangGraph State Machine**
  - **Best Use Case:** technical workflows, network automation, and self-healing systems where loops, retries, and "Human-in-the-loop" breakpoints are critical.

#### Multi-Agent: Summary
---
The manual decomposition of the `NetworkChangeAgent` into specialized roles proves that multi-agent design is not merely a software convenience, but a **reliability strategy**. By isolating the `RecoveryAgent`, the system architecture allows for independent scaling of safety logic. Whether implemented via **CrewAI's role-based delegation** or **LangGraph's stateful cycles**, the core requirement remains the same: a shared state, clear handoffs, and deterministic guardrails for high-risk operations.

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
---
- **Crews:** Teams of AI agents formed to work on a specific project or task.
- **Agents:** Individual AIs with defined roles (e.g., researcher, analyst, writer), goals, backstories, and access to specific tools.
- **Orchestration:** The framework manages how these agents interact, delegate, and share information, ensuring the overall workflow is efficient.
- **LLM-Agnostic** Works with various large language models (LLMs) from different providers like OpenAI, Anthropic, and Mistral.

#### How it works
---
Imagine planning a complex event: instead of one person doing everything, *CrewAI* assigns tasks like a human team would.
- **Planner Agent:** Coordinates the overall event.
- **Food Agent:** Handles catering details.
- **Decorations Agent:** Manages venue aesthetics.

```
Goal → Planner → Fixed Plan → Execute → Done
```

CrewAI's Sequential Process a.k.a. *Static Task Orchestrator* style can be used for predictable workflows, with rare failures and the speed and simplicity are the main criteria.

> Source: https://www.crewai.com/ & https://docs.crewai.com/en/introduction

#### CrewAI Examples
---
The first CrewAI example is going to create two agents as 2 separate files:
- first agent to obtain coordinates for specific city
- second agent to obtain current weather forecast for specific city on the base of obtained `latitude` and `longitude`

Run each script and see the output. The `coordinates_agent` will provide the *latitude* and the *longitude* for the specific city. The `weather_agent` is going to provide the current *temperature* and the current *speed of the wind* for the provided latitude and longitude of the city.

```python
# 03-automation-and-agend-based-systems/examples/coordinates_agent.py

# Get coordinates: Coordinates API
# @tool decorator automatically does the input typing + metadata LangChain needs.
@tool
def geocode_city(city: str) -> dict:
    """
    Convert city name to latitude and longitude.
    """
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": city, 
        "format": "json",
        "limit": 1
    }
    headers = {
        "User-Agent": "langchain-agent-demo"
    }

    response = requests.get(url, params=params, headers=headers, timeout=10)
    data = response.json()

    if not data:
        return {"error": "City not found"}

    return {
        "latitude": float(data[0]["lat"]),
        "longitude": float(data[0]["lon"]),
    }

# Define LLM
llm = ChatOpenAI(
    model = "gpt-4o-mini",
    temperature = 0
)

# Create the Agent
agent = create_agent(
    model = llm,
    tools = [geocode_city]
)

# Run it
ai_message = agent.invoke({
    "messages": [
        HumanMessage(content="What's the coordinates for the city Bratislava?")
    ]
})

print(ai_message["messages"][-1].content)

```

```python
# 03-automation-and-agend-based-systems/examples/weather_agent.py

# Get weather: Weather API
# @tool decorator automatically does the input typing + metadata LangChain needs.
@tool
def get_weather(latitude: float, longitude: float) -> str:
    """
    Get current weather for given coordinates.
    """
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}&longitude={longitude}&current_weather=true"
    )
    response = requests.get(url, timeout=10)
    data = response.json()

    weather = data.get("current_weather", {})
    return (
        f"Temperature: {weather.get('temperature')}°C,\n"
        f"Wind: {weather.get('windspeed')} km/h"
    )

# Define LLM
llm = ChatOpenAI(
    model = "gpt-4o-mini",
    temperature = 0
)

# Create the Agent
agent = create_agent(
    model = llm,
    tools = [get_weather]
)

# Run it
ai_message = agent.invoke({
    "messages": [
        HumanMessage(content="What's the weather at latitude 48.1559 longitude 17.1314?")
    ]
})

print(ai_message["messages"][-1].content)

```

##### Manual Orchestration
---
Next, let's create *agent to agent* example, where both agents are manully orchestrated.

> **IMPORTANT:**
>
> - agents are not APIs
> - agents are LLMs that may call tools

To have deterministic behavior, the agents must:
- call the tool
- extract the tool result
- pass structured data forward

```python
# 03-automation-and-agend-based-systems/examples/a2a_v1.py

def weather_by_city(city: str) -> str:
    """
    Call each agent and return current weather for specific city.
    """
    # print(f"city: {city}")
    coords_raw = geo_agent.invoke({
        "messages": [
            HumanMessage(
                content = f"Get coordinates for the city {city}."
            )
        ]
    })
    # extract latitude & longitude received from coordinates_agent
    coords = extract_coords(coords_raw)
    # print(f"coords: {coords}")
    latitude = coords["latitude"]
    longitude = coords["longitude"]
    weather_raw = weather_agent.invoke({
        "messages": [
            HumanMessage(
                content=f"Provide the current weather for {city} based on these coordinates: "
                        f"latitude {latitude} and longitude {longitude}. "
                        f"Output ONLY the weather in this format: "
                        f"'The current weather in [City] is [Temp]°C with a wind speed of [Speed] km/h.' "
            )
        ]
    })
    weather = weather_raw["messages"][-1].content
    return weather

print(weather_by_city(city = "Bratislava"))
```

In a multi-agent architecture, each agent is an *"island"* with its own memory. When a task is hand of from the `GeoAgent` to the `WeatherAgent`, the original intent (*the city name*) is often lost unless the orchestrator explicitly passes that metadata forward. The simplest way to include the city name in response is to provide the city in the content of `HumanMessage`, when sent to the `weather_agent`. This gives the LLM the context it needs to format the final sentence (`Provide the current weather for {city} based on these coordinates:`).

The LLMs are naturally conversational. To get expected output, a formatting instruction must be included in the content of `HumanMessage` (`The current weather in [City] is [Temp]°C with a wind speed of [Speed] km/h.`). 


##### Manual Orchestration with Pydantic Schema
---
In professional AI systems, relying on an agent to *"speak"* the right format is risky. Instead, the library such as `Pydantic` is used to define a *"Schema"*. This forces the LLM to map its findings into a specific data structure. Therefore, the next example is going to use `Pydantic` to format the LLM responses of both agents.

```python
# 03-automation-and-agend-based-systems/examples/a2a_v2.py

from pydantic import BaseModel, Field

class GeoResponse(BaseModel):
    latitude: float = Field(description="Latitude of the city")
    longitude: float = Field(description="Longitude of the city")

class WeatherResponse(BaseModel):
    city: str = Field(description="The name of the city")
    temperature: float = Field(description="Current temperature in Celsius")
    wind_speed: float = Field(description="Current wind speed in km/h")
    summary: str = Field(description="A one-sentence summary of the weather")

def weather_by_city(city: str):
    """
    Call each agent and return current weather for specific city.
    """
    coords_raw = geo_agent.invoke({
        "messages": [
            HumanMessage(
                content = f"Use your tools to find coordinates for {city}"
            )
        ]
    })
    raw_content = tool_output(response = coords_raw)

     # Structured output
    structured_llm_geo = llm.with_structured_output(GeoResponse)

    # Pass raw API data to the LLM
    prompt_geo = f"Extract coordinates from this raw data: {raw_content}"
    coords = structured_llm_geo.invoke(prompt_geo)
    
    # Get Weather Data
    weather_raw = weather_agent.invoke({
        "messages": [
            HumanMessage(
                content = f"Get weather for latitude {coords.latitude} and longitude {coords.longitude}."
            )
        ]
    })
    weather_data = weather_raw["messages"][-1].content
    
    # Wrap the LLM with the structured output requirement
    structured_llm_wea = llm.with_structured_output(WeatherResponse)

    # Use the LLM as a "Parser" to fill the Pydantic object
    prompt_wea = f"Format the weather data for {city}. Data: {weather_data}"
    
    # Return WeatherResponse
    final_output = structured_llm_wea.invoke(prompt_wea)
    
    return final_output

# --- Execution ---
result = weather_by_city("Bratislava")

# Now you can access it like an object:
print(f"City: {result.city}")
print(f"Temp: {result.temperature}")

# Or convert to a clean JSON string:
print(result.model_dump_json(indent = 2))

```

The implementation in `a2a_v2.py` represents a **Manual Handoff Pattern**. In this model, the developer acts as the central router, explicitly extracting data from the `GeoAgent`’s message history, validating it against a Pydantic schema, and then manually injecting those attributes into the `WeatherAgent`’s prompt.


##### Streamlined Pipeline Orchestration
---
The next evolution of the system utilizes **LangChain Expression Language (LCEL)**. By transitioning from manual extraction to a **Functional Chain**, the architecture collapses the *"Action-Observation-Parsing"* loop into a single atomic operation.

In the forthcoming `a2a_v3.py`, the orchestration moves from **Imperative** to **Declarative**.

```python
# 03-automation-and-agend-based-systems/examples/a2a_v3.py

def weather_chain(city: str):
    """Unified Orchestration Logic"""
    geo_llm = llm.bind_tools([geocode_city]).with_structured_output(GeoResponse)
    weather_llm = llm.bind_tools([weather_city]).with_structured_output(WeatherResponse)

    coords = geo_llm.invoke(f"Find coordinates for {city}")
    weather_result = weather_llm.invoke(
        f"Get weather for lat: {coords.latitude}, lon: {coords.longitude}"
    )

    return {
        "location": city,
        "coordinates": coords.model_dump(),
        "weather": weather_result.model_dump()
    }

# --- Execution ---
result = weather_chain("Bratislava")
print(json.dumps(result, indent=2))
```

Frameworks like **CrewAI** are powerful, but they are often *"overkill"* for simple linear tasks. The above script ended up with this **LangChain/LCEL** approach because:
- **Reduced Latency:** CrewAI agents go through a *"Thought/Action/Observation"* loop; for a simple API call, this adds extra LLM tokens and seconds of waiting
- **Granular Control:** by using `bind_tools` and `with_structured_output`, we have 100% control over the data schema
- **Architectural Clarity:** it demonstrates the *"Core"* of an agent without the heavy lifting of a manager or a background process

##### Declarative Frameworks
---
To use **CrewAI**, there is shift from *"Functions"* to *"Roles"*. Instead of calling a tool, an Agent owns a specific job. Here is how to rebuild this into a formal `Crew`.

While **CrewAI** is compatible with **LangChain**, it performs best when using its own `LLM` class wrapper. This ensures that the **Pydantic** validation layers between the Agent and the Model are perfectly aligned, preventing *"TypeErrors"* during the initialization phase.

```python
# 03-automation-and-agend-based-systems/examples/a2a_v4.py

from crewai import LLM

llm = LLM(
    model = "gpt-4o-mini",
    temperature = 0
)
```

Standard decorators (*like LangChain's `@tool`*) can sometimes fail *Pydantic* validation in strict environments. The most robust solution is to define tools as classes inheriting from `crewai.tools.BaseTool`. By implementing a *Pydantic* `args_schema`, we create a deterministic *"Input Contract"*. This forces the LLM to use specific `JSON` keys, eliminating common errors like *Multiple Values for Argument* or *KeyErrors*.

```python
# 03-automation-and-agend-based-systems/examples/a2a_v4.py

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

class GeocodeInput(BaseModel):
    city: str = Field(description="The name of the city to search for (e.g., 'Bratislava').")

class GeocodeTool(BaseTool):
    name: str = "geocode_city"
    description: str = "Convert a city name into latitude and longitude coordinates."
    args_schema: type[BaseModel] = GeocodeInput

    # Get coordinates: Coordinates API
    def _run(self, city: str) -> str:
        """Convert city name to latitude and longitude."""
        print(f"DEBUG: Agent is searching for: {city}")

        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": city, 
            "format": "json",
            "limit": 1
        }
        headers = {
            "User-Agent": "langchain-agent-demo"
        }
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            data = response.json()

            if not data:
                return f"Error: City '{city}' not found. Please try a different name."

            return f"Latitude: {float(data[0]['lat'])}, Longitude: {float(data[0]['lon'])}"

        except Exception as e:
            return f"Error connecting to Geocoding service: {str(e)}"

class WeatherInput(BaseModel):
    latitude: float = Field(description="The latitude coordinate.")
    longitude: float = Field(description="The longitude coordinate.")

class WeatherTool(BaseTool):
    name: str = "weather_city"
    description: str = "Get current weather for given latitude and longitude coordinates."
    args_schema: type[BaseModel] = WeatherInput

    # Get weather: Weather API
    def _run(self, latitude: float, longitude: float) -> str:
        """Get current weather for given coordinates."""
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={latitude}&longitude={longitude}&current_weather=true"
        )

        try:
            response = requests.get(url, timeout=10)
            data = response.json()

            # Another safety check
            if "current_weather" not in data:
                return "Error: Weather data currently unavailable for these coordinates."

            weather = data.get("current_weather", {})
            return f"Temperature: {weather.get('temperature')}°C, Wind: {weather.get('windspeed')} km/h"

        except Exception as e:
            return f"Error connecting to Weather service: {str(e)}"

geocode_tool = GeocodeTool()
weather_tool = WeatherTool()
```

Agents are defined by **Roles**, **Goals**, and **Backstories**. This *"Triple-A" (Agent, Attribute, Assignment)* structure shifts the focus from writing logic to defining expertise, allowing the LLM to better understand its specific domain of responsibility.

```python
# 03-automation-and-agend-based-systems/examples/a2a_v4.py

from crewai import Agent

geo_researcher = Agent(
    role='Geographic Researcher',
    goal='Find accurate coordinates for {city}',
    backstory='Expert in geocoding and global coordinates.',
    tools=[geocode_tool],
    llm=llm,
    verbose=True
)

weather_analyst = Agent(
    role='Weather Analyst',
    goal='Provide weather data for coordinates provided by the researcher',
    backstory='Meteorological data specialist.',
    tools=[weather_tool],
    llm=llm,
    verbose=True
)
```

CrewAI manages the *"Chain of Thought"* between agents using **Task Context**. By setting `context=[coord_task]`, we enable a seamless data flow where the `Weather Analyst` automatically consumes the output of the `Geographic Researcher`. This replaces manual data parsing with **Natural Language Context**, mirroring how human teams collaborate.

```python
# 03-automation-and-agend-based-systems/examples/a2a_v4.py

from crewai import Crew, Task, Process

coord_task = Task(
    description='Find the latitude and longitude for the city: {city}',
    expected_output='A dictionary with latitude and longitude.',
    agent=geo_researcher
)

weather_task = Task(
    description='Look at the coordinates from the previous task and provide the current weather.',
    expected_output='A concise weather report including temperature and wind speed.',
    agent=weather_analyst,
    context=[coord_task] # This tells CrewAI to feed the output of task 1 into task 2
)

# --- Assemble the Crew ---
weather_crew = Crew(
    agents=[geo_researcher, weather_analyst],
    tasks=[coord_task, weather_task],
    process=Process.sequential # Task 1 must finish before Task 2 starts
)
```

The `kickoff` process initiates the orchestration. Because the tools are now schema-validated, if an agent makes a mistake, the framework provides a *"Semantic Error"* (*e.g., "City not found"*), allowing the agent to self-correct and try again without crashing the entire workflow.

```python
# 03-automation-and-agend-based-systems/examples/a2a_v4.py

result = weather_crew.kickoff(inputs={'city': 'Bratislava'})

print("\n--- FINAL CREW REPORT ---")
print(result)
```

Transitioning from **Manual Orchestration** to **Declarative Frameworks** trades low-level code simplicity for high-level system resilience. While the initial setup is more complex due to strict type-checking, the resulting system is far more capable of handling the inherent ambiguity of natural language inputs.

##### Network Change Agent via CrewAI
---
The transition of a *Network Change Agent* from raw Python `network_change_agent_v4.py` to the **CrewAI framework** requires a fundamental shift in how automation is structured. While raw Python scripts rely on explicit `if/else` loops and manual function calls to handle state, CrewAI introduces a **Declarative Role-Based** architecture.

The keys to this architectural rebuild include:
- **Atomic Tool Encapsulation:** Functions are wrapped in `BaseTool` classes with strict **Pydantic** `args_schema` definitions. This ensures that the *Large Language Model (LLM)* interacts with network resources through a deterministic *"contract"*, preventing common data-type mismatches.
```python
# 03-automation-and-agend-based-systems/examples/crewai_network_agent.py

from crewai.tools import BaseTool
from pydantic import BaseModel

class ValidateTool(BaseTool):
    name: str = "validate"
    description: str = "Validate a network change request."
    args_schema: type[BaseModel] = ValidateInput
```

- **Separation of Concerns (*Persona Definition*):** The logic is decomposed into specialized Agents - **Planner**, **Executor**, and **Verifier**. Each agent operates within a specific *"Backstory"* and *"Goal"*, allowing the LLM to focus on specialized sub-tasks rather than managing the entire workflow complexity at once.
```python
# 03-automation-and-agend-based-systems/examples/crewai_network_agent.py

from crewai import Agent

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
```

- **Sequential Context Management:** Instead of passing variables manually between functions, the system utilizes **Task Context**. This allows the output of a validation task to flow naturally into a configuration task, enabling the agents to *"understand"* the state of the network before proceeding.
```python
# 03-automation-and-agend-based-systems/examples/crewai_network_agent.py

from crewai import Task

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
```

- **Closed-Loop Reliability:** By providing the **Verifier** with a `RollbackTool` and specific instructional intent, the script moves from *"Fire-and-Forget"* automation to **Closed-Loop Automation**. The agent can interpret a *"down"* state as a trigger for a corrective action without additional hard-coded logic.
```python
# 03-automation-and-agend-based-systems/examples/crewai_network_agent.py

from crewai import Agent, Task

verifier_agent = Agent(
    llm=llm,
    role="Network Verifier",
    goal="Verify the state of network devices",
    backstory="Monitoring system ensuring network health",
    tools=[verify_tool, rollback_tool],
    verbose=True
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
```

The successful execution of this CrewAI implementation yields several critical outcomes for the field of network automation:

- **Resilience via Self-Correction:** One of the primary outcomes is the emergence of a self-healing loop. Unlike a traditional script that might crash upon a verification failure, the Agentic approach allows the LLM to perceive a failure as a new prompt, triggering the `RollbackTool` to restore the network to its last known-good state.

- **Input Determinism:** The use of structured schemas effectively eliminated the *"Multiple Values"* and *"List Index"* errors common in early iterations. This demonstrates that **Schema Enforcement** is a mandatory prerequisite for using LLMs in production-critical environments like networking.

- **Abstraction of Complexity:** The resulting code is more maintainable because the *"intelligence"* (*the instructions*) is separated from the *"capability"* (*the tools*). Adding a new vendor (*e.g., switching from Junos to Cisco*) would only require updating a tool’s logic, while the Agentic roles and task flow remain unchanged.

- **Semantic Verification:** The exercise proved that verification is not just a data-check but a semantic decision point. The Verifier Agent successfully translated the technical status `{"state": "down"}` into the operational decision to execute a rollback, mimicking the analytical judgment of a human network engineer.

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
---
Think of LangGraph like a **smart flowchart** for a Network Operations Center (NOC). Instead of a simple "*start-to-finish*" script, it builds a map of possibilities:
- **Entry Point:** The request arrives at the Planner Node.
- **The Loop:** The Executor Node attempts a configuration change.
- **The Decision:** The Verifier Node checks the result.
  - *If success:* Move to the **End Node**.
  - *If failure:* The **Conditional Edge** routes the flow back to the **Rollback Node**, and then potentially back to the **Planner** to try a different strategy.

```
Goal → Plan → Execute → Evaluate → Replan → Continue
```

LangGraph's *Closed-Loop Autonomous Agent* style can be used for unpredictable workflows with expected failures, where recovery is required with agent's autonomy.

> Source: https://langchain-ai.github.io/langgraph/ & https://blog.langchain.dev/langgraph/


### AutoGen
---
**AutoGen** is an open-source **Microsoft framework for building multi-agent AI applications**, allowing different AI agents (*powered by LLMs*) to converse, collaborate, and use tools to solve complex tasks, mimicking human teamwork with customizable roles and workflows, and offering low-code tools like *AutoGen Studio* for easier development and management. It simplifies creating systems that handle complex workflows, code execution, data analysis, and automation, reducing the need for extensive human intervention by coordinating specialized agents.

#### Key Features & Concepts
---
- **Multi-Agent Collaboration:** Define multiple agents (e.g., a planner, an engineer, a critic) that communicate and work together to achieve a goal.
- **Conversational Nature:** Agents interact through human-readable dialogue, making complex workflows more intuitive.
- **Customization & Flexibility:** Easily extend agents with new tools, define behaviors with code or natural language, and create modular workflows.
- **Tool Integration:** Agents can use external tools and APIs (like search engines, code interpreters).
- **Low-Code Development:** AutoGen Studio offers a visual, flow-chart-based interface for building and sharing workflows.
- **Observability:** Provides tracing and visibility into agent interactions and decisions, powered by OpenTelemetry.

#### How it Works
---
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
