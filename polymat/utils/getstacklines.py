from abc import abstractmethod
import traceback

from dataclasses import dataclass
from dataclasses import fields


@dataclass(frozen=True)
class FrameSummary:
    filename: str
    lineno: int
    name: str
    line: str


def get_stack_lines(index: int = 3) -> tuple[FrameSummary]:
    def gen_stack_lines():
        for obj in traceback.extract_stack()[:-index]:
            yield FrameSummary(
                filename=obj.filename,
                lineno=obj.lineno,
                name=obj.name,
                line=obj.line,
            )

    return tuple(gen_stack_lines())


def to_operator_exception(
    message: str,
    stack: tuple[FrameSummary, ...],
) -> str:
    assert stack is not None

    exception_lines = (
        message,
        f"  Assertion traceback (most recent call last):",
        *(
            f'    File "{stack_line.filename}", line {stack_line.lineno}\n      {stack_line.line}'
            for stack_line in stack
        ),
    )

    return "\n".join(exception_lines)


class FrameSummaryMixin:
    @property
    @abstractmethod
    def stack(self) -> tuple[FrameSummary, ...]:
        ...

    # implement custom __repr__ method that returns a representation without the stack
    def __repr__(self):
        fields_str = ','.join(f'{field.name}={repr(getattr(self, field.name))}' for field in fields(self) if field.name != 'stack') # type: ignore

        return f"{self.__class__.__name__}({fields_str})"
