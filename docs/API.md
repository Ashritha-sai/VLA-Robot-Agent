# VLA Robot Agent -- API Reference

Complete reference for all public classes and methods.

---

## `TableTopEnv` -- `src/env.py`

PyBullet simulation environment for tabletop manipulation.

### Constructor

```python
TableTopEnv(gui: bool = True, time_step: float = 1/240)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gui` | `bool` | `True` | `True` for graphical window, `False` for headless |
| `time_step` | `float` | `1/240` | Physics time step in seconds |

### Methods

#### `reset() -> Dict`

Reset the environment to a fresh episode. Removes old objects, resets robot to home, spawns blocks at random positions and bowl at centre, lets physics settle.

**Returns:** Scene description dict (see `get_scene_description()`).

---

#### `get_scene_description() -> Dict`

Build a semantic description of the current scene.

**Returns:**

```python
{
    "robot_state": {
        "end_effector_position": [x, y, z],    # list of 3 floats
        "gripper_width": float,                 # metres
        "joint_positions": [j1, ..., j7]        # 7 joint angles (rad)
    },
    "objects": [
        {
            "name": "red_block",        # str
            "type": "block",            # "block" or "bowl"
            "color": "red",             # semantic colour name
            "position": [x, y, z],      # list of 3 floats
            "orientation": [qx,qy,qz,qw],  # quaternion
            "id": 5                     # PyBullet body id
        },
        ...
    ]
}
```

---

#### `get_object_position(name: str) -> np.ndarray`

Get the world-frame position of a tracked object.

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Object name: `"red_block"`, `"green_block"`, `"blue_block"`, or `"bowl"` |

**Returns:** `(3,)` numpy array `[x, y, z]`.

**Raises:** `ValueError` if the name is not found.

---

#### `get_object_by_name(name: str) -> int`

Look up a PyBullet body ID by semantic name.

**Returns:** Integer body ID.

**Raises:** `ValueError` if the name is not found.

---

#### `get_ee_position() -> np.ndarray`

Get the robot's end-effector position.

**Returns:** `(3,)` numpy array `[x, y, z]`.

---

#### `get_gripper_width() -> float`

Get the total opening width of the gripper (sum of both finger positions).

**Returns:** Width in metres (0.0 = fully closed, ~0.08 = fully open).

---

#### `get_joint_positions() -> np.ndarray`

Get the current positions of all 7 arm joints.

**Returns:** `(7,)` numpy array of joint angles in radians.

---

#### `step(num_steps: int = 1) -> None`

Advance the physics simulation by `num_steps` steps. In GUI mode, includes a real-time sleep per step.

---

#### `step_seconds(seconds: float) -> None`

Advance the simulation by a given duration (converted to step count internally).

---

#### `close() -> None`

Disconnect from the PyBullet server and release resources. Safe to call multiple times.

---

#### `print_scene() -> None`

Print a colour-coded scene summary to the terminal. Shows robot state (EE position, gripper width, joints) and all object positions.

---

### Class Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `HOME_JOINTS` | `[0, -0.785, 0, -2.356, 0, 1.571, 0.785]` | Home joint configuration |
| `ARM_JOINTS` | `[0, 1, 2, 3, 4, 5, 6]` | Arm joint indices |
| `GRIPPER_JOINTS` | `[9, 10]` | Gripper finger joint indices |
| `EE_LINK_INDEX` | `11` | End-effector link index |
| `TABLE_SIZE` | `[1.0, 0.8, 0.05]` | Table dimensions (m) |
| `TABLE_POSITION` | `[0.5, 0.0, 0.0]` | Table centre position |
| `BLOCK_HALF` | `0.02` | Block half-extent (m) |

---

## `RobotSkills` -- `src/skills.py`

High-level manipulation primitives for the Franka Panda robot.

### Constructor

```python
RobotSkills(env: TableTopEnv)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `env` | `TableTopEnv` | Environment instance (must already be connected) |

The constructor also sets lateral friction on gripper finger links to 1.5.

---

### High-Level Skills

#### `go_home(speed: float = 0.3) -> bool`

Return the arm to its safe home configuration.

**Sequence:** open gripper -> drive to home joints -> settle.

**Returns:** `True` if converged.

---

#### `pick(object_name: str) -> bool`

Pick up an object by name.

**Sequence:** query position -> open gripper -> pre-grasp -> descend -> close gripper (+ constraint attach) -> lift -> verify grasp.

| Parameter | Type | Description |
|-----------|------|-------------|
| `object_name` | `str` | e.g. `"red_block"`, `"green_block"`, `"blue_block"` |

**Returns:** `True` if all motions converged and the gripper is still holding the object after lifting.

**Raises:** `ValueError` (from env) if the object name is unknown.

---

#### `place(target_position: List[float]) -> bool`

Place the currently held object at a position.

**Sequence:** approach above -> descend -> open gripper (constraint detach) -> settle -> retreat.

| Parameter | Type | Description |
|-----------|------|-------------|
| `target_position` | `[x, y, z]` | World-frame placement coordinates |

**Returns:** `True` if all motions converged.

---

#### `push(object_name: str, direction: List[float], distance: float = 0.10) -> bool`

Push an object sideways.

| Parameter | Type | Description |
|-----------|------|-------------|
| `object_name` | `str` | Object to push |
| `direction` | `[dx, dy]` | Push direction (normalised internally) |
| `distance` | `float` | Push distance in metres |

**Returns:** `True` if all motions converged.

---

#### `execute_trajectory(waypoints: List[List[float]]) -> bool`

Execute a sequence of Cartesian waypoints. Aborts on the first failure.

| Parameter | Type | Description |
|-----------|------|-------------|
| `waypoints` | `[[x,y,z], ...]` | List of world-frame positions |

**Returns:** `True` if all waypoints were reached.

---

### Motion Primitives

#### `solve_ik(target_pos, target_orn=None) -> List[float]`

Compute arm joint angles for a target EE pose via inverse kinematics.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_pos` | `[x, y, z]` | required | Target position |
| `target_orn` | `[qx, qy, qz, qw]` | Gripper down | Target orientation quaternion |

**Returns:** List of 7 joint angles.

---

#### `move_to_joint_positions(target_joints, speed=0.3, max_steps=None) -> bool`

Drive all arm joints to target angles with position control.

**Returns:** `True` if converged within `CONVERGENCE_TOL`.

---

#### `move_to_position(target_pos, target_orn=None, speed=0.3) -> bool`

Move end-effector to a Cartesian position (IK + joint control).

**Returns:** `True` if converged.

---

### Gripper Control

#### `open_gripper() -> None`

Open the gripper to maximum width (~0.08 m). Detaches any constraint-held object first.

#### `close_gripper() -> bool`

Close the gripper. If an object is detected between the fingers, attaches it with a fixed constraint.

**Returns:** `True` if an object is being grasped.

#### `get_gripper_state() -> dict`

**Returns:** `{"width": float, "forces": [float, float], "is_grasping": bool}`

#### `get_ee_position() -> np.ndarray`

**Returns:** `(3,)` numpy array of end-effector world position.

---

### Class Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `GRIPPER_FORCE` | `100.0` | Newton grip force |
| `CONVERGENCE_TOL` | `0.01` | Joint convergence tolerance (rad) |
| `MAX_MOVE_STEPS` | `720` | Safety cap on motion steps |
| `PRE_GRASP_HEIGHT` | `0.10` | Approach height above object (m) |
| `GRASP_OFFSET` | `0.02` | Finger height above object centre (m) |
| `LIFT_HEIGHT` | `0.15` | Lift height after grasping (m) |
| `PLACE_APPROACH_HEIGHT` | `0.15` | Approach height for placing (m) |
| `PLACE_OFFSET` | `0.05` | Release height above target (m) |
| `GRASP_WIDTH_THRESHOLD` | `0.005` | Below this = empty gripper (m) |

---

## `VLAAgent` -- `src/agent.py`

Vision-Language-Action agent that uses LLMs for task planning.

### Constructor

```python
VLAAgent(env: TableTopEnv, llm_provider: str = "openai", model: str = None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `env` | `TableTopEnv` | required | Environment (must be reset) |
| `llm_provider` | `str` | `"openai"` | `"openai"` or `"anthropic"` |
| `model` | `str` | provider default | Model name override |

**Raises:** `ImportError` if the provider package is not installed. `ValueError` for unknown providers.

---

### Methods

#### `chat(user_instruction: str) -> bool`

Main interface. Perceives scene, queries LLM, parses plan, executes all actions.

| Parameter | Type | Description |
|-----------|------|-------------|
| `user_instruction` | `str` | Natural-language command |

**Returns:** `True` if all actions succeeded.

**Example:**

```python
agent.chat("Put the red block in the bowl")
```

---

#### `get_plan(user_instruction: str) -> Dict`

Same as `chat()` but returns the parsed plan without executing it.

**Returns:** Plan dict with `"reasoning"` and `"actions"` keys.

---

#### `query_llm(prompt: str) -> str`

Send a prompt to the configured LLM and return the raw response text.

**Raises:** `RuntimeError` if no API key is set.

---

#### `parse_llm_response(response_text: str) -> Dict`

Parse a JSON action plan from raw LLM output. Handles markdown code fences.

**Returns:** Plan dict with `"reasoning"` (str) and `"actions"` (list).

**Raises:** `ValueError` for invalid JSON or missing `"actions"` key.

---

#### `execute_action(action_dict: Dict) -> bool`

Execute a single action from a plan.

**Supported actions:**

| Action | Dict Format | Maps To |
|--------|-------------|---------|
| `pick` | `{"action": "pick", "target": "red_block"}` | `skills.pick("red_block")` |
| `place` | `{"action": "place", "target": [x,y,z]}` or `{"action": "place", "target": "bowl"}` | `skills.place(pos)` |
| `push` | `{"action": "push", "target": "red_block", "direction": [1,0], "distance": 0.1}` | `skills.push(...)` |
| `go_home` | `{"action": "go_home"}` | `skills.go_home()` |

**Raises:** `ValueError` for unknown action names.

---

#### `execute_plan(plan: Dict) -> bool`

Execute all actions in a plan sequentially, stopping on the first failure.

**Returns:** `True` if all actions succeeded.

---

### Class Constants

| Constant | Value |
|----------|-------|
| `DEFAULT_MODELS["openai"]` | `"gpt-4"` |
| `DEFAULT_MODELS["anthropic"]` | `"claude-3-5-sonnet-20241022"` |

---

## CLI -- `main.py`

Interactive command-line interface.

### Usage

```bash
python main.py [--gui] [--llm {openai,anthropic}] [--model MODEL]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--gui` | `True` | Launch with PyBullet GUI |
| `--llm` | `openai` | LLM provider |
| `--model` | provider default | Model name override |

### Commands

| Command | Description |
|---------|-------------|
| `scene` | Print current scene state |
| `reset` | Randomise object positions |
| `help` | Show available commands |
| `quit` / `exit` / `q` | Exit |
| *(any other text)* | Sent to `agent.chat()` as a natural-language instruction |
