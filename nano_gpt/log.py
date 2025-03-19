"""Logging utilities."""

import logging
import pathlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class LogRecord(ABC):
    """Log record."""

    log_type: str
    data: dict[str, Any]

    @property
    def message(self) -> str:
        """String representation of the data."""
        return " | ".join(f"{key}: {value}" for key, value in self.data.items())

    def __str__(self) -> str:
        """String representation."""
        return f"{self.log_type}: {self.message}"


class TrainLog(ABC):
    """Train logger."""

    @abstractmethod
    def log(self, message: LogRecord) -> None:
        """Log a message."""


class LogFile(TrainLog):
    """Train logger."""

    def __init__(self, log_file: pathlib.Path) -> None:
        """Initialize the train logger."""
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.log_file.unlink(missing_ok=True)

    def log(self, message: LogRecord) -> None:
        """Log a message."""
        with self.log_file.open("a") as f:
            f.write(str(message))
            f.write("\n")
            f.flush()


class LogStdout(TrainLog):
    """Train logger."""

    def log(self, message: LogRecord) -> None:
        """Log a message."""
        print(str(message))


class LogMulti(TrainLog):
    """Train logger."""

    def __init__(self, logs: list[TrainLog]) -> None:
        """Initialize the train logger."""
        self.logs = logs

    def log(self, message: LogRecord) -> None:
        """Log a message."""
        for log in self.logs:
            log.log(message)


def create_log(
    log_file: pathlib.Path | None = None, log_stdout: bool = True
) -> TrainLog:
    """Create a train logger."""
    logs: list[TrainLog] = []
    if log_file:
        logs.append(LogFile(log_file))
    if log_stdout:
        logs.append(LogStdout())
    return LogMulti(logs)
