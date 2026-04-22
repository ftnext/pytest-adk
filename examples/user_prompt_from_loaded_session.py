# Copyright 2026 pytest-adk contributors

import asyncio
from pathlib import Path

from pytest_adk import runner_from_exported_session

from fields_planner.agent import root_agent as fields_planner_agent


async def main() -> None:
    runner, session = await runner_from_exported_session(
        fields_planner_agent,
        Path(__file__).parent / "roll_die.json",
    )
    _ = await runner.run_debug(
        "What numbers did I get?",
        user_id=session.user_id,
        session_id=session.id,
    )


if __name__ == "__main__":
    asyncio.run(main())
