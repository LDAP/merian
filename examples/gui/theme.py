"""merian's default ImGui theme (Mocha) as Qt colors."""

from PySide6 import QtCore, QtGui, QtWidgets

import merian.graph as mg

THEME = {
    "canvas": "#181825",
    "bg": "#1e1e2e",
    "raised": "#262637",
    "surface": "#313244",
    "surface_hi": "#45475a",
    "border": "#585b70",
    "overlay": "#6c7086",
    "text": "#cdd6f4",
    "text_dim": "#8f93ab",
    "accent": "#cba6f7",
    "accent_hi": "#b4befe",
    "blue": "#89b4fa",
    "green": "#a6e3a1",
    "yellow": "#f9e2af",
    "peach": "#fab387",
    "red": "#f38ba8",
    "teal": "#94e2d5",
}
NODE_STATE_COLORS = {
    mg.NodeState.OK: "green",
    mg.NodeState.DISABLED: "text_dim",
    mg.NodeState.BYPASSED: "teal",
    mg.NodeState.UNSUPPORTED: "yellow",
    mg.NodeState.FORCE_DISABLED: "peach",
    mg.NodeState.ERRORED: "red",
}
RIGHT_ALIGNED = (
    QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
)


def scrolled(widget: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
    area = QtWidgets.QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    area.setWidget(widget)
    return area


def theme_palette() -> QtGui.QPalette:
    roles = QtGui.QPalette.ColorRole
    palette = QtGui.QPalette()
    for role, color in (
        (roles.Window, THEME["bg"]),
        (roles.WindowText, THEME["text"]),
        (roles.Base, THEME["surface"]),
        (roles.AlternateBase, THEME["surface_hi"]),
        (roles.Text, THEME["text"]),
        (roles.PlaceholderText, THEME["text_dim"]),
        (roles.Button, THEME["surface_hi"]),
        (roles.ButtonText, THEME["text"]),
        (roles.Highlight, THEME["accent"]),
        (roles.HighlightedText, THEME["bg"]),
        (roles.ToolTipBase, THEME["surface"]),
        (roles.ToolTipText, THEME["text"]),
        (roles.Mid, THEME["border"]),
        (roles.Dark, THEME["bg"]),
        (roles.Light, THEME["surface_hi"]),
        (roles.Link, THEME["accent_hi"]),
    ):
        palette.setColor(role, QtGui.QColor(color))
    for role in (roles.Text, roles.ButtonText, roles.WindowText):
        palette.setColor(
            QtGui.QPalette.ColorGroup.Disabled, role, QtGui.QColor(THEME["text_dim"])
        )
    return palette


def theme_stylesheet() -> str:
    return """
    QMenuBar {{ background: {canvas}; padding: 3px 6px; }}
    QMenuBar::item {{ padding: 4px 10px; border-radius: 6px; background: transparent; }}
    QMenuBar::item:selected, QMenuBar::item:pressed {{ background: {surface}; }}
    QMenu {{
        background: {surface}; border: 1px solid {surface_hi};
        border-radius: 8px; padding: 5px;
    }}
    QMenu::item {{ padding: 5px 28px 5px 26px; border-radius: 5px; }}
    QMenu::item:selected {{ background: {surface_hi}; }}
    QMenu::item:disabled {{ color: {text_dim}; }}
    QMenu::separator {{ height: 1px; background: {surface_hi}; margin: 5px 8px; }}

    QStatusBar {{
        background: {canvas}; color: {text_dim};
        border-top: 1px solid {surface};
    }}
    QStatusBar::item {{ border: none; }}
    QStatusBar QLabel {{ color: {text_dim}; padding: 0 8px; }}
    QStatusBar QLabel#footerCaption {{ padding: 0 6px 0 0; }}
    QStatusBar QLabel#footerValue {{ color: {text}; padding: 0; }}

    QSplitter::handle {{ background: {canvas}; }}
    QSplitter::handle:horizontal {{ width: 3px; }}
    QSplitter::handle:vertical {{ height: 3px; }}
    QSplitter::handle:hover {{ background: {surface_hi}; }}

    QTabWidget::pane {{ border: none; border-top: 1px solid {surface}; }}
    QTabBar {{ qproperty-drawBase: 0; }}
    QTabBar::tab {{
        background: transparent; color: {text_dim}; padding: 6px 14px;
        border: none; border-bottom: 2px solid transparent;
    }}
    QTabBar::tab:hover {{ color: {text}; }}
    QTabBar::tab:selected {{ color: {text}; border-bottom: 2px solid {accent}; }}

    QToolButton {{ border: none; border-radius: 6px; padding: 4px; }}
    QToolButton:hover {{ background: {surface}; }}
    QToolButton:pressed, QToolButton:checked {{ background: {surface_hi}; }}

    QPushButton {{
        background: {surface_hi}; border: none; border-radius: 6px; padding: 5px 12px;
    }}
    QPushButton:hover {{ background: {border}; }}
    QPushButton:pressed {{ background: {accent}; color: {bg}; }}

    QComboBox, QLineEdit, QPlainTextEdit {{
        background: {surface}; border: 1px solid {surface_hi};
        border-radius: 6px; padding: 3px 8px;
        selection-background-color: {accent}; selection-color: {bg};
    }}
    QComboBox:hover, QLineEdit:hover, QPlainTextEdit:hover {{ border-color: {border}; }}
    QComboBox:focus, QLineEdit:focus, QPlainTextEdit:focus {{ border-color: {accent}; }}
    QComboBox QAbstractItemView {{
        background: {surface}; border: 1px solid {surface_hi}; outline: none;
        selection-background-color: {surface_hi}; selection-color: {text};
    }}
    QAbstractSpinBox QLineEdit {{
        border: none; background: transparent; padding: 0 0 0 4px;
    }}
    QAbstractSpinBox {{
        background: {surface}; border: 1px solid {surface_hi}; border-radius: 6px;
        padding: 2px 6px; selection-background-color: {accent}; selection-color: {bg};
    }}
    QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{
        width: 0; border: none;
    }}
    QAbstractSpinBox:hover {{ border-color: {border}; }}
    QAbstractSpinBox:focus {{ border-color: {accent}; }}

    QSlider::groove:horizontal {{ height: 4px; background: {surface_hi}; border-radius: 2px; }}
    QSlider::sub-page:horizontal {{ background: {accent}; border-radius: 2px; }}
    QSlider::handle:horizontal {{
        background: {text}; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px;
    }}
    QSlider::handle:horizontal:hover {{ background: {accent_hi}; }}

    QToolButton#segment {{
        background: {surface}; border: 1px solid {surface_hi}; border-radius: 0;
        padding: 3px 10px;
    }}
    QToolButton#segment[segment="first"] {{
        border-top-left-radius: 6px; border-bottom-left-radius: 6px;
    }}
    QToolButton#segment[segment="last"] {{
        border-top-right-radius: 6px; border-bottom-right-radius: 6px;
    }}
    QToolButton#segment[segment="only"] {{ border-radius: 6px; }}
    QToolButton#segment:hover {{ background: {surface_hi}; }}
    QToolButton#segment:checked {{
        background: {accent}; color: {bg}; border-color: {accent};
    }}

    QFrame#card {{ background: {surface}; border-radius: 8px; }}
    QLabel#cardTitle {{ color: {text_dim}; font-weight: bold; }}
    QLabel#cardValue {{ font-size: 15pt; font-weight: bold; }}
    QLabel#cardNote {{ color: {text_dim}; }}

    QWidget#propertyRow {{ border-radius: 6px; }}
    QWidget#propertyRow:hover {{ background: {raised}; }}
    QLabel#propertyLabel, QLabel#note {{ color: {text_dim}; }}
    QLabel#ruleLabel {{ color: {text_dim}; font-weight: bold; }}
    QFrame#rule {{ background: {surface_hi}; border: none; }}
    QToolButton#groupHeader {{ font-weight: bold; padding: 3px 8px 3px 2px; }}
    QToolButton#groupHeader:checked {{ background: transparent; }}
    QToolButton#groupHeader:hover {{ background: {raised}; }}

    QToolTip {{
        background: {surface}; color: {text};
        border: 1px solid {surface_hi}; border-radius: 6px; padding: 4px 8px;
    }}

    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
    QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
    QScrollBar::handle {{ background: {surface_hi}; border-radius: 3px; }}
    QScrollBar::handle:hover {{ background: {border}; }}
    QScrollBar::handle:vertical {{ min-height: 24px; }}
    QScrollBar::handle:horizontal {{ min-width: 24px; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}

    QHeaderView::section {{
        background: {bg}; color: {text_dim}; border: none; padding: 4px 6px;
    }}
    QTreeView, QTableView, QListView {{ border: none; }}
    """.format(**THEME)
