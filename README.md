# VLA Robot Agent

Embodied AI demonstration using LLMs to control a simulated Franka Panda robot arm for tabletop manipulation tasks. Give instructions in plain English and watch the robot plan and execute them in a PyBullet physics simulation.

## Features

- **PyBullet Simulation** -- Realistic physics with a 7-DOF Franka Panda arm, parallel-jaw gripper, coloured blocks, and a bowl on a tabletop
- **LLM Planning** -- GPT-4 or Claude converts natural-language instructions into JSON action plans
- **Constraint-Based Grasping** -- Reliable pick-and-place with grasp constraints, automatic retry, and post-place verification
- **High-Level Skills** -- `pick`, `place`, `push`, `go_home`, and `execute_trajectory` primitives
- **Interactive CLI** -- Chat with the robot, inspect the scene, and reset the environment from the terminal
- **Modular Architecture** -- Swap LLM providers, add new skills, or extend the environment independently

## Requirements

- Python 3.8+
- PyBullet (with OpenGL for GUI mode)
- NumPy, SciPy, colorama
- **Optional**: `openai` and/or `anthropic` Python packages (only needed for LLM planning)

## Installation

```bash
# Clone the repository
git clone https://github.com/Ashritha-sai/VLA-Robot-Agent.git
cd VLA-Robot-Agent

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate    # Linux / macOS
venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

Set your LLM API key (only needed for LLM-guided demos):

```bash
# OpenAI (default)
export OPENAI_API_KEY="sk-..."

# Or Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
```

On Windows use `set` instead of `export`.

## Quick Start

### Interactive CLI

```bash
python main.py                          # OpenAI + GUI (default)
python main.py --llm anthropic          # Use Claude instead
python main.py --model gpt-4o           # Override model name
```

Inside the CLI:

```
You: Put the red block in the bowl
You: scene                               # inspect object positions
You: reset                               # randomise the scene
You: quit                                # exit
```

### Programmatic Usage

```python
from src.env import TableTopEnv
from src.skills import RobotSkills

env = TableTopEnv(gui=True)
env.reset()
skills = RobotSkills(env)

skills.pick("red_block")

bowl_pos = env.get_object_position("bowl")
skills.place(bowl_pos.tolist())

skills.go_home()
env.close()
```

### Using the LLM Agent

```python
from src.env import TableTopEnv
from src.agent import VLAAgent

env = TableTopEnv(gui=True)
env.reset()

agent = VLAAgent(env, llm_provider="openai")
agent.chat("Put all the blocks in the bowl")

env.close()
```

## Architecture

```
 User Instruction          "Put red block in the bowl"
        |
        v
 +--------------+
 |   VLAAgent   |   Builds prompt with scene state, queries LLM,
 |  (agent.py)  |   parses JSON plan, executes actions
 +--------------+
        |
        v
 +--------------+
 | RobotSkills  |   pick(), place(), push(), go_home()
 | (skills.py)  |   IK solver, joint-space motion, gripper control
 +--------------+
        |
        v
 +--------------+
 | TableTopEnv  |   PyBullet physics, Franka Panda URDF,
 |   (env.py)   |   object spawning, scene observation
 +--------------+
        |
        v
    PyBullet
```

For detailed technical documentation see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

For the full API reference see [`docs/API.md`](docs/API.md).

## Project Structure

```
VLA-Robot-Agent/
|-- src/
|   |-- __init__.py
|   |-- env.py             # PyBullet simulation environment
|   |-- skills.py          # Manipulation skill primitives
|   |-- agent.py           # LLM-based task planning agent
|   +-- utils.py           # Shared utilities
|-- tests/
|   |-- __init__.py
|   |-- test_env.py        # 38 tests for TableTopEnv
|   |-- test_skills.py     # 82 tests for RobotSkills
|   +-- test_agent.py      # 50 tests for VLAAgent
|-- examples/
|   |-- demo_basic.py      # Pick-and-place without LLM
|   |-- demo_llm.py        # LLM-guided task execution
|   +-- test_no_llm.py     # Hardcoded plan execution
|-- configs/
|   +-- robot_config.yaml  # Robot configuration
|-- docs/
|   |-- ARCHITECTURE.md    # Technical architecture deep-dive
|   +-- API.md             # Full API reference
|-- main.py                # Interactive CLI entry point
|-- requirements.txt
|-- CONTRIBUTING.md
|-- LICENSE
+-- README.md
```

## Examples

### No API Key Required

```bash
# Basic pick-and-place demo (uses skills directly)
python examples/demo_basic.py

# Execute a hardcoded plan (simulates LLM output)
python examples/test_no_llm.py
```

### Requires API Key

```bash
# LLM-guided demo with three sequential tasks
python examples/demo_llm.py
```

## Testing

```bash
# Run the full test suite (170 tests)
python -m pytest tests/ -v

# Run tests for a specific module
python -m pytest tests/test_env.py -v
python -m pytest tests/test_skills.py -v
python -m pytest tests/test_agent.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=term-missing
```

All tests run headless (no GUI window) and do not require API keys.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for code style, how to add new skills, and the pull request process.

## License

MIT

## Acknowledgments

- [PyBullet](https://pybullet.org/) for the physics simulation engine
- [Franka Emika](https://www.franka.de/) for the Panda robot design (URDF shipped with pybullet_data)
- [OpenAI](https://openai.com/) and [Anthropic](https://www.anthropic.com/) for LLM APIs
