# Contributing to VLA Robot Agent

## Getting Started

```bash
# Clone and install
git clone <repo-url>
cd vla_robot_agent
python -m venv venv
source venv/bin/activate          # Linux/macOS
venv\Scripts\activate             # Windows
pip install -r requirements.txt

# Verify everything works
python -m pytest tests/ -v
```

## Code Style

- **Python 3.8+** compatibility
- **Type hints** on all public method signatures
- **Docstrings** on all public classes and methods (Google/NumPy style)
- **4-space indentation**, no tabs
- **Snake_case** for functions/variables, **PascalCase** for classes, **UPPER_CASE** for module-level constants
- Keep lines under 100 characters where practical
- Use `logging` (module-level `logger`) for debug/info/warning messages
- Use `print()` only for user-facing output in skills and the CLI

## Project Layout

```
src/env.py      -- Simulation environment (PyBullet)
src/skills.py   -- Manipulation primitives (pick, place, push, ...)
src/agent.py    -- LLM-based planning agent
main.py         -- CLI entry point
tests/          -- pytest test suite (runs headless, no API keys)
examples/       -- Runnable demo scripts
docs/           -- Architecture and API documentation
```

## Adding a New Skill

### 1. Implement the skill in `src/skills.py`

Add a public method to `RobotSkills`. Follow the existing pattern:

```python
def stack(self, top_block: str, bottom_block: str) -> bool:
    """
    Stack top_block on bottom_block.

    Args:
        top_block:    Name of the block to pick up.
        bottom_block: Name of the block to stack on.

    Returns:
        True if the stack succeeded.
    """
    bottom_pos = self.env.get_object_position(bottom_block)
    if not self.pick(top_block):
        return False
    stack_target = [bottom_pos[0], bottom_pos[1], bottom_pos[2] + 0.05]
    return self.place(stack_target)
```

**Guidelines:**
- Return `bool` (True = success)
- Use `self.env.get_object_position()` to query objects
- Compose from existing primitives (`pick`, `place`, `move_to_position`) where possible
- Print user-facing status messages
- Use `logger.info()` / `logger.warning()` for detailed logging

### 2. Register the action in `src/agent.py`

Add an `elif` branch in `VLAAgent.execute_action()`:

```python
elif action == "stack":
    bottom = action_dict.get("bottom")
    return self.skills.stack(target, bottom)
```

### 3. Update the LLM prompt

In `VLAAgent._build_prompt()`, add the new action to the "Available Actions" section so the LLM knows it exists:

```python
"5. stack(top_block, bottom_block) - Stack one block on another\n"
```

### 4. Write tests

Add tests in `tests/test_skills.py`:

```python
class TestStack:
    def test_stack_returns_bool(self, skills, env):
        result = skills.stack("red_block", "blue_block")
        assert isinstance(result, bool)

    def test_stack_moves_block_above(self, skills, env):
        blue_pos = env.get_object_position("blue_block")
        skills.stack("red_block", "blue_block")
        red_pos = env.get_object_position("red_block")
        assert red_pos[2] > blue_pos[2]  # red is above blue
```

Run the full suite to check for regressions:

```bash
python -m pytest tests/ -v
```

## Adding New Objects

Edit `src/env.py`:

1. Add to `BLOCK_COLORS` or create a new spawn method
2. Add metadata to `OBJECT_META`
3. Add a display icon to `_PRINT_ICONS`
4. Add tests in `tests/test_env.py`

## Testing

### Running tests

```bash
# Full suite
python -m pytest tests/ -v

# Single module
python -m pytest tests/test_skills.py -v

# Single test class
python -m pytest tests/test_skills.py::TestPick -v

# With coverage
python -m pytest tests/ --cov=src --cov-report=term-missing
```

### Test conventions

- All tests run headless (`gui=False`) -- no window opens
- No API keys required -- LLM calls are mocked or skipped
- Tests use `pytest` fixtures defined in each test file
- Test class names: `TestFeatureName`
- Test method names: `test_specific_behaviour`
- Prefer assertion messages that explain what went wrong

### What to test

- **Return types** -- skills return `bool`, env queries return correct shapes
- **Success cases** -- motions converge, objects move to expected positions
- **Failure cases** -- unknown objects raise `ValueError`, empty gripper returns `False`
- **Edge cases** -- double close, zero-length push direction, empty plan
- **Print output** -- use `capsys` to check user-facing messages

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with tests
3. Run the full test suite: `python -m pytest tests/ -v`
4. Update documentation if adding public API
5. Write a clear commit message describing what and why
6. Open a pull request with a description of the changes

## Debugging Tips

- **Enable debug logging:**
  ```python
  import logging
  logging.basicConfig(level=logging.DEBUG)
  ```
- **Inspect physics state:** Use `env.print_scene()` or `skills.get_gripper_state()`
- **Slow down motion:** Lower the `speed` parameter (e.g. `skills.pick("red_block")` then tweak `POSITION_GAIN`)
- **Run with GUI:** Always test with `gui=True` when debugging physics issues
- **Check IK solutions:** `skills.solve_ik([x, y, z])` to see if a position is reachable
