#!/usr/bin/env python3
"""
VLA Robot Agent - Interactive Demo
"""

import argparse
import os
import sys
from src.env import TableTopEnv
from src.agent import VLAAgent
from colorama import init, Fore, Style

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Initialize colorama for colored terminal output
init(autoreset=True)

def print_banner():
    """Print welcome banner"""
    banner = f"""
{Fore.CYAN}{'='*70}
   🤖 Vision-Language-Action Robot Agent 🤖
   Embodied AI Demo with Franka Panda
{'='*70}{Style.RESET_ALL}
"""
    print(banner)

def print_help():
    """Print available commands"""
    help_text = f"""
{Fore.YELLOW}Available Commands:{Style.RESET_ALL}
  • Type any natural language instruction (e.g., "Put the red block in the bowl")
  • 'reset' - Reset environment with new random object positions
  • 'scene' - Display current scene description
  • 'help' - Show this help message
  • 'quit' or 'exit' - Exit the program

{Fore.YELLOW}Example Instructions:{Style.RESET_ALL}
  • "Put all the blocks in the bowl"
  • "Stack the red block on the green block"
  • "Move the blue block to the left side of the table"
  • "Put the green block in the bowl, then go home"
"""
    print(help_text)

def main():
    parser = argparse.ArgumentParser(description='VLA Robot Agent Demo')
    parser.add_argument('--gui', action='store_true', default=True,
                        help='Launch with PyBullet GUI (default: True)')
    parser.add_argument('--llm', type=str, default='openai',
                        choices=['openai', 'anthropic'],
                        help='LLM provider to use')
    parser.add_argument('--model', type=str, default=None,
                        help='Specific model name (optional)')

    args = parser.parse_args()

    print_banner()

    # Initialize environment
    print(f"{Fore.GREEN}Initializing PyBullet environment...{Style.RESET_ALL}")
    env = TableTopEnv(gui=args.gui)
    env.reset()

    # Initialize agent
    print(f"{Fore.GREEN}Initializing {args.llm} agent...{Style.RESET_ALL}")
    try:
        agent = VLAAgent(env, llm_provider=args.llm, model=args.model)
    except Exception as e:
        print(f"{Fore.RED}✗ Failed to initialize agent: {e}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Make sure you've set your API key:{Style.RESET_ALL}")
        print(f"  export OPENAI_API_KEY='your-key-here'")
        print(f"  or export ANTHROPIC_API_KEY='your-key-here'")
        sys.exit(1)

    print(f"{Fore.GREEN}✓ Ready!{Style.RESET_ALL}\n")
    print_help()

    # Main interaction loop
    while True:
        try:
            # Get user input
            user_input = input(f"\n{Fore.CYAN}You:{Style.RESET_ALL} ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print(f"{Fore.YELLOW}Shutting down...{Style.RESET_ALL}")
                break

            elif user_input.lower() == 'reset':
                print(f"{Fore.GREEN}Resetting environment...{Style.RESET_ALL}")
                env.reset()
                env.print_scene()

            elif user_input.lower() == 'scene':
                env.print_scene()

            elif user_input.lower() == 'help':
                print_help()

            else:
                # Execute user instruction
                agent.chat(user_input)

        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Interrupted. Type 'quit' to exit.{Style.RESET_ALL}")

        except Exception as e:
            print(f"{Fore.RED}✗ Error: {e}{Style.RESET_ALL}")
            import traceback
            traceback.print_exc()

    # Cleanup
    env.close()
    print(f"{Fore.GREEN}Goodbye!{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
