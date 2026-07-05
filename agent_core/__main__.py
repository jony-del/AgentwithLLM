from agent_core.cli import main

if __name__ == "__main__":
    # Propagate the CLI's return code so `python -m agent_core` and the `polaris`
    # console script (which setuptools wraps in sys.exit) are truly equivalent.
    raise SystemExit(main())
