# VLA Robot Agent -- Architecture

This document describes the internal architecture of the VLA Robot Agent system, covering each module, the data flow for a typical interaction, and how to extend the system.

## System Overview

```
+-------------------------------------------------------------------+
|                        main.py  (CLI)                             |
|  argparse | REPL loop | coloured output | command dispatch        |
+-------------------------------------------------------------------+
        |                       |
        v                       v
+----------------+     +----------------+
|   VLAAgent     |     |  TableTopEnv   |
|   agent.py     |---->|    env.py      |
+----------------+     +----------------+
        |                       |
        v                       v
+----------------+         PyBullet
|  RobotSkills   |         Physics
|   skills.py    |---------->Engine
+----------------+
```

**Three core modules** form a layered stack:

| Layer | Module | Responsibility |
|-------|--------|----------------|
| Planning | `VLAAgent` | Observe scene, build LLM prompt, parse JSON plan, execute actions |
| Skills | `RobotSkills` | High-level manipulation primitives (pick, place, push, go_home) |
| Environment | `TableTopEnv` | PyBullet physics, robot URDF, object spawning, state queries |

Each layer only depends on the one below it, making the system easy to test and extend.

---

## Module Descriptions

### Environment -- `src/env.py`

`TableTopEnv` manages the entire PyBullet simulation.

**Responsibilities:**

- Connect to PyBullet (GUI or headless DIRECT mode)
- Load ground plane, create table, load Franka Panda URDF
- Spawn coloured blocks at random non-overlapping positions and a static bowl at table centre
- Set physics parameters (gravity, time step, friction)
- Provide state queries: `get_scene_description()`, `get_object_position()`, `get_ee_position()`, `get_gripper_width()`, `get_joint_positions()`
- Step the simulation forward and handle cleanup

**Key design decisions:**

- Blocks have `lateralFriction=1.5` for reliable grasping
- Block positions are randomised with a minimum spacing of 0.10 m
- The bowl is static (`baseMass=0`) so it cannot be pushed
- `print_scene()` uses colorama for cross-platform coloured output, with a graceful fallback

**Constants:**

| Constant | Value | Description |
|----------|-------|-------------|
| `TABLE_SIZE` | 1.0 x 0.8 x 0.05 m | Table dimensions |
| `TABLE_POSITION` | [0.5, 0.0, 0.0] | Table centre |
| `BLOCK_HALF` | 0.02 m | Half-extent of each cube |
| `BOWL_RADIUS` | 0.1 m | Bowl cylinder radius |
| `HOME_JOINTS` | 7 angles | Safe upright arm posture |

---

### Skills -- `src/skills.py`

`RobotSkills` provides high-level manipulation primitives.

**Skill catalogue:**

| Skill | Signature | Description |
|-------|-----------|-------------|
| `go_home` | `(speed=0.3) -> bool` | Open gripper, drive to home joints, settle |
| `pick` | `(object_name) -> bool` | Open, pre-grasp, descend, close, lift, verify |
| `place` | `(target_position) -> bool` | Approach above, descend, release, retreat |
| `push` | `(object_name, direction, distance) -> bool` | Close gripper, approach behind, push through |
| `execute_trajectory` | `(waypoints) -> bool` | Sequence of Cartesian waypoints via IK |

**Motion pipeline:**

```
Cartesian target  -->  solve_ik()  -->  move_to_joint_positions()  -->  converge
  [x, y, z]         IK solver          position-control loop         check tol
```

1. `solve_ik()` calls `p.calculateInverseKinematics` with joint limits and rest poses
2. `move_to_joint_positions()` commands all 7 joints via `p.setJointMotorControlArray` each sim step
3. Convergence is checked per-joint against `CONVERGENCE_TOL` (0.01 rad)
4. A safety cap of `MAX_MOVE_STEPS` (720 steps = 3 s at 240 Hz) prevents infinite loops

**Constraint-based grasping:**

Pure friction grasping is unreliable in PyBullet for small objects. This system uses a two-part strategy:

1. **Grasp detection** -- After closing the gripper, two heuristics fire:
   - *Width check*: residual opening > `GRASP_WIDTH_THRESHOLD` (0.005 m) means something is between the fingers
   - *Contact check*: `p.getContactPoints` on both finger links; if both touch the same non-robot body, a grasp is confirmed

2. **Constraint attachment** -- When a grasp is detected, `_attach_object()` creates a `JOINT_FIXED` constraint between the end-effector link and the grasped body. This holds the object rigidly during transport.

3. **Safe release** -- `_detach_object()` gradually ramps the constraint force down (50 -> 20 -> 5 -> 0 N), zeroes the object velocity, removes the constraint, and lets physics settle. This prevents objects from being flung by residual solver forces.

**Tuning constants:**

| Constant | Value | Controls |
|----------|-------|----------|
| `GRIPPER_FORCE` | 100 N | Motor force for finger joints |
| `PRE_GRASP_HEIGHT` | 0.10 m | Approach height above object |
| `GRASP_OFFSET` | 0.02 m | Finger height relative to object centre |
| `LIFT_HEIGHT` | 0.15 m | How high to lift after grasping |
| `PLACE_APPROACH_HEIGHT` | 0.15 m | Approach height above place target |
| `PLACE_OFFSET` | 0.05 m | Release height above place target |

---

### Agent -- `src/agent.py`

`VLAAgent` is the planning layer that connects natural language to robot actions.

**Pipeline (the `chat()` method):**

```
1. PERCEIVE    env.get_scene_description()     ->  scene dict
2. PROMPT      _build_prompt(instruction, scene) ->  prompt string
3. QUERY       query_llm(prompt)                ->  raw JSON text
4. PARSE       parse_llm_response(text)         ->  plan dict
5. EXECUTE     execute_plan(plan)               ->  bool success
```

**Prompt structure:**

The prompt sent to the LLM contains:
- System role ("You are a robot task planner...")
- Current robot state (EE position, gripper width)
- Object list with positions
- Available actions with signatures
- User instruction
- Output format (strict JSON schema)
- Planning rules (pick before place, one object at a time, etc.)

**Plan format:**

```json
{
  "reasoning": "Pick the red block and place it in the bowl",
  "actions": [
    {"action": "pick", "target": "red_block"},
    {"action": "place", "target": "bowl"},
    {"action": "go_home"}
  ]
}
```

**LLM providers:**

| Provider | Default Model | Environment Variable |
|----------|---------------|---------------------|
| OpenAI | `gpt-4` | `OPENAI_API_KEY` |
| Anthropic | `claude-3-5-sonnet-20241022` | `ANTHROPIC_API_KEY` |

Both providers are optional imports -- the system gracefully handles missing packages.

---

## Data Flow -- Complete Interaction

```
User types: "Put the red block in the bowl"
    |
    v
main.py REPL -> agent.chat("Put the red block in the bowl")
    |
    |-- 1. env.get_scene_description()
    |       Returns: {robot_state: {ee_pos, gripper_width, joints},
    |                 objects: [{name, type, color, position, orientation, id}, ...]}
    |
    |-- 2. _build_prompt(instruction, scene)
    |       Returns: formatted prompt string with scene context
    |
    |-- 3. query_llm(prompt)
    |       Calls OpenAI/Anthropic API, returns raw JSON string
    |
    |-- 4. parse_llm_response(raw_json)
    |       Strips code fences, validates JSON, checks required fields
    |       Returns: {"reasoning": "...", "actions": [...]}
    |
    |-- 5. execute_plan(plan)
    |       For each action in plan:
    |           |
    |           |-- execute_action({"action": "pick", "target": "red_block"})
    |           |       -> skills.pick("red_block")
    |           |           -> env.get_object_position("red_block")
    |           |           -> open_gripper()
    |           |           -> move_to_position(pre_grasp)   [IK -> joint control]
    |           |           -> move_to_position(grasp)
    |           |           -> close_gripper()                [+ constraint attach]
    |           |           -> move_to_position(lift)
    |           |           -> verify _is_grasping()
    |           |
    |           |-- execute_action({"action": "place", "target": "bowl"})
    |           |       -> env.get_object_position("bowl")
    |           |       -> skills.place(bowl_position)
    |           |           -> move_to_position(above_target)
    |           |           -> move_to_position(place_height)
    |           |           -> open_gripper()                 [constraint detach]
    |           |           -> move_to_position(retreat)
    |           |
    |           +-- execute_action({"action": "go_home"})
    |                   -> skills.go_home()
    |
    +-- Returns True (all actions succeeded)
```

---

## Extension Points

### Adding a New Skill

1. Add a method to `RobotSkills` in `src/skills.py`:

```python
def stack(self, top_block: str, bottom_block: str) -> bool:
    """Stack top_block on top of bottom_block."""
    bottom_pos = self.env.get_object_position(bottom_block)
    if not self.pick(top_block):
        return False
    stack_target = [bottom_pos[0], bottom_pos[1], bottom_pos[2] + 0.05]
    return self.place(stack_target)
```

2. Register it in `VLAAgent.execute_action()` in `src/agent.py`:

```python
elif action == "stack":
    bottom = action_dict.get("bottom")
    return self.skills.stack(target, bottom)
```

3. Add the action description to `_build_prompt()` so the LLM knows about it.

4. Write tests in `tests/test_skills.py`.

### Adding New Objects

Edit `TableTopEnv` in `src/env.py`:

1. Add colour to `BLOCK_COLORS` (or create a new spawn method)
2. Add metadata to `OBJECT_META`
3. Add an icon to `_PRINT_ICONS`

### Adding a New LLM Provider

In `VLAAgent.__init__()`:

1. Add the provider to `DEFAULT_MODELS`
2. Add an `elif` branch to initialise the client
3. Add an `elif` branch in `query_llm()` to make the API call

### Running Headless

```python
env = TableTopEnv(gui=False)  # DIRECT mode, no window
```

All skills work identically in headless mode. This is used by the test suite.
