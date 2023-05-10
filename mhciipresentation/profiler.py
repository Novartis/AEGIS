"""Profiler to check if there are any bottlenecks in your code."""
import cProfile
import io
import logging
import pstats
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import pandas as pd
from pytorch_lightning.profilers.profiler import Profiler

log = logging.getLogger(__name__)


class CustomAdvancedProfiler(Profiler):
    """Custom profiler that spits out csv that can be processed."""

    def __init__(
        self,
        dirpath: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None,
        line_count_restriction: float = 1.0,
    ) -> None:
        super().__init__(dirpath=dirpath, filename=filename)
        self.profiled_actions: Dict[str, cProfile.Profile] = {}
        self.line_count_restriction = line_count_restriction

    def start(self, action_name: str) -> None:
        if action_name not in self.profiled_actions:
            self.profiled_actions[action_name] = cProfile.Profile()
        self.profiled_actions[action_name].enable()

    def stop(self, action_name: str) -> None:
        pr = self.profiled_actions.get(action_name)
        if pr is None:
            raise ValueError(
                f"Attempting to stop recording an action ({action_name}) which"
                " was never started."
            )
        pr.disable()

    def summary(self) -> str:
        recorded_stats = {}
        dfs = pd.DataFrame()
        for action_name, pr in self.profiled_actions.items():
            s = io.StringIO()
            ps = (
                pstats.Stats(pr, stream=s)
                .strip_dirs()
                .sort_stats("cumulative")
            )
            rows = []
            for func, (cc, nc, tt, ct, callers) in ps.stats.items():
                row = {
                    "filename": func[0],
                    "lineno": func[1],
                    "function_name": func[2],
                    "prim_calls": cc,
                    "total_calls": nc,
                    "total_time": tt,
                    "cumulative_time": ct,
                }
                rows.append(row)

            df = pd.DataFrame(rows)
            df["action_name"] = action_name
            dfs = pd.concat([dfs, df])
        return dfs

    def describe(self) -> None:
        return "CustomAdvancedProfilerEnd"

    def teardown(self, stage: Optional[str]) -> None:
        super().teardown(stage=stage)
        self.profiled_actions = {}

    def __reduce__(self) -> Tuple:
        # avoids `TypeError: cannot pickle 'cProfile.Profile' object`
        return (
            self.__class__,
            (),
            {
                "dirpath": self.dirpath,
                "filename": self.filename,
                "line_count_restriction": self.line_count_restriction,
            },
        )
