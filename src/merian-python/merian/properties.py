"""User interfaces for the properties of merian objects, implemented in Python."""

from ._core import ChildFlags, OptionsStyle


class Properties:
    """
    Records properties into an immediate-mode user interface like ImGui: each pass calls
    these methods in display order, and a config method returns the value the user set
    since the previous pass, else None.
    """

    ChildFlags = ChildFlags
    OptionsStyle = OptionsStyle

    def begin_child(self, name: str, label: str, flags: ChildFlags) -> bool:
        """Returns whether to record the child's content, which end_child closes."""
        return True

    def end_child(self) -> None:
        pass

    def separate(self, label: str) -> None:
        pass

    def no_space(self) -> None:
        """Keeps the next item with the previous one."""

    def output_text(self, text: str) -> None:
        pass

    def output_plot(
        self,
        label: str,
        samples: list[float],
        scale_min: float | None,
        scale_max: float | None,
    ) -> None:
        pass

    def config_float(
        self,
        name: str,
        value: float | tuple[float, ...],
        desc: str,
        sensitivity: float,
        minimum: float | None,
        maximum: float | None,
    ) -> float | tuple[float, ...] | None:
        """value and the result are tuples for vectors."""
        return None

    def config_int(
        self,
        name: str,
        value: int | tuple[int, ...],
        desc: str,
        minimum: int,
        maximum: int,
    ) -> int | tuple[int, ...] | None:
        """value and the result are tuples for vectors."""
        return None

    def config_bool(self, name: str, value: bool, desc: str) -> bool | None:
        return None

    def config_button(self, name: str, desc: str) -> bool:
        """Returns whether the button was pressed since the previous pass."""
        return False

    def config_split(
        self, name: str, labels: list[str], weights: tuple[float, ...], desc: str
    ) -> tuple[float, ...] | None:
        """Weights of the parts, relative to their sum."""
        return None

    def config_options(
        self,
        name: str,
        selected: int,
        options: list[str],
        style: OptionsStyle,
        desc: str,
    ) -> int | None:
        return None

    def config_color(
        self, name: str, color: tuple[float, ...], desc: str
    ) -> tuple[float, ...] | None:
        """RGB or RGBA, linear."""
        return None

    def config_text(
        self, name: str, text: str, multiline: bool, needs_submit: bool, desc: str
    ) -> str | None:
        """Returns the text at every change, or once submitted if needs_submit."""
        return None

    def config_path(self, name: str, path: str, desc: str) -> str | None:
        """Returns the path of a file to read."""
        return None
