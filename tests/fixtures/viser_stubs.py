# SPDX-License-Identifier: Apache-2.0
"""Viser stand-ins: the studio's browser surface, without a browser.

The studio is library code and the interface this stack is used through, so its
panels, pickers, frustums and take-recorder are worth testing -- and none of that
needs a browser, a websocket or a human. These stubs stand in for viser's GUI and
scene handles: a widget is a value, a disabled flag and callbacks a test may
fire; a scene is a list of what was added to it.

Promoted out of a physics test module, where eight of them sat in the middle of two
and a half thousand lines of physics tests. Nothing about them is
physics-specific and nothing about them needs a GPU -- which is exactly why they
belong in the fixtures package: the studio tests that use them can then live
beside the other studio tests rather than beside the tape scene.

The pattern to preserve when extending these: a stub records what it was asked to
do and does nothing else. A stub that starts *behaving* -- laying out widgets,
computing what a real panel would compute -- is a second implementation of the
thing under test, and it will agree with the real one right up until it matters.
"""

from __future__ import annotations

import numpy as np


class _StubEvent:
    """What viser hands a callback: the handle on ``target``, and no browser behind it."""

    def __init__(self, target, phase: str = "update") -> None:
        self.target = target
        self.client = None
        self.client_id = None
        self.phase = phase


class _StubWidget:
    """A viser widget: a value, a disabled flag, and the callbacks viser would call.

    As viser 1.1.1 does (``_gui_handles.py``, the ``value`` setter): assigning a
    *changed* value calls every ``on_update`` callback, at once, on the thread
    that assigned it, and assigning the same value calls none -- except on a
    button, which is called every time. A panel that writes its own slider from
    ``drain`` therefore hears its own callback, here as in the browser, and a
    stub that only called back on :meth:`fire` hid exactly that.
    """

    def __init__(self, value=None, *, is_button: bool = False) -> None:
        self._value = value
        self.is_button = bool(is_button)
        self.disabled = False
        self.visible = True
        self.removed = False
        self.parent = None
        """The folder it was built in, which is how a test asks whether a hidden folder hides it."""
        self.callbacks: list = []

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value) -> None:
        value = _cast(self._value, value)
        if not self.is_button and _same(self._value, value):
            return
        self._value = value
        self._notify()

    def _notify(self) -> None:
        for fn in list(self.callbacks):
            fn(_StubEvent(self))

    def on_update(self, fn):
        self.callbacks.append(fn)
        return fn

    def on_click(self, fn):
        self.callbacks.append(fn)
        return fn

    def fire(self, value=None) -> None:
        """What the browser does: send a value, which viser takes only if it changed (a button, always).

        With no value it is a press: the callbacks are called whatever the value is.
        """
        if value is None:
            self._notify()
            return
        value = _cast(self._value, value)
        if not self.is_button and _same(self._value, value):
            return
        self._value = value
        self._notify()

    def remove(self) -> None:
        self.removed = True


def _cast(current, value):
    """``value`` in ``current``'s type, as viser casts an assigned value and one the browser sent.

    An array or a list becomes a tuple, each element in the type of the one it
    replaces where the lengths agree; anything else is ``type(current)(value)``.
    So ``(0.0, 0.0, 1.0)`` set as ``np.array([0, 0, 1])`` is the same value, and
    no change. A widget with no value yet takes whatever it is given.
    """
    if current is None or value is None:
        return value
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(current, tuple):
        if not isinstance(value, (list, tuple)):
            return value
        if len(value) == len(current):
            return tuple(type(c)(v) for c, v in zip(current, value, strict=True))
        return tuple(value)
    try:
        return type(current)(value)
    except (TypeError, ValueError):
        return value


def _same(a, b) -> bool:
    """``a == b`` as viser asks it, where an array compared to a tuple is not an error but a difference."""
    try:
        return bool(a == b)
    except (TypeError, ValueError):
        return False


class _StubImage(_StubWidget):
    def __init__(self, image, label) -> None:
        super().__init__()
        self.image = image
        self.label = label


class _StubMarkdown:
    """A markdown block: its content, which a panel rewrites on refresh, and the
    folder it was written into -- which is how a test asks what a section shows
    before anything is unrolled."""

    def __init__(self, content: str, parent=None) -> None:
        self.content = content
        self.parent = parent
        self.removed = False

    def remove(self) -> None:
        self.removed = True


class _StubFolder:
    """A container, where it was opened, and whether it is shown.

    ``parent`` is what makes nesting checkable: a page that unrolls twice to
    reach one control is a folder whose parent is a folder, and that is a claim
    a test can make without a browser. Entering pushes onto the gui's open
    stack, which is how the *next* folder learns who its parent is -- so the
    bookkeeping is on the type, where ``with`` will actually find it.
    ``visible`` is kept as viser keeps it, and a panel may reassign it.
    """

    def __init__(self, label=None, parent=None, gui=None, visible=True, expand_by_default=True) -> None:
        self.label = label
        self.parent = parent
        self.visible = bool(visible)
        self.expand_by_default = bool(expand_by_default)
        """Whether it opens unrolled: viser's default is True, and a panel that says False is making a claim."""
        self.removed = False
        self._gui = gui

    def __enter__(self):
        if self._gui is not None:
            self._gui._open.append(self)
        return self

    def __exit__(self, *exc) -> bool:
        if self._gui is not None and self._gui._open and self._gui._open[-1] is self:
            self._gui._open.pop()
        return False

    def remove(self) -> None:
        self.removed = True


class _StubGui:
    """Enough of ``server.gui`` for a panel to build and publish into."""

    def __init__(self) -> None:
        self.images: list[_StubImage] = []
        self.widgets: list[tuple[str, _StubWidget]] = []
        self.folders: list[_StubFolder] = []
        self.markdown: list[_StubMarkdown] = []
        self._open: list[_StubFolder] = []
        """The folders a build is currently inside, innermost last."""

    def _widget(self, label, value, *, disabled: bool = False, is_button: bool = False, **_ignored) -> _StubWidget:
        """One widget under ``label``. ``disabled`` is kept, as viser keeps it:
        a panel that builds a switch greyed out is making a claim a test may
        read, and a stub that dropped the keyword answered "enabled" for every
        widget whatever the panel had said."""
        handle = _StubWidget(value, is_button=is_button)
        handle.disabled = bool(disabled)
        handle.parent = self._open[-1] if self._open else None
        self.widgets.append((str(label), handle))
        return handle

    def find(self, label: str) -> _StubWidget:
        """The widget a panel built under this label, for a test to drive."""
        return next(w for name, w in self.widgets if name == label)

    def add_folder(self, label=None, *, visible=True, expand_by_default=True, **kwargs):
        folder = _StubFolder(
            label,
            self._open[-1] if self._open else None,
            gui=self,
            visible=visible,
            expand_by_default=expand_by_default,
        )
        self.folders.append(folder)
        return folder

    def add_markdown(self, content="", **kwargs):
        handle = _StubMarkdown(str(content), self._open[-1] if self._open else None)
        self.markdown.append(handle)
        return handle

    def add_checkbox(self, label, initial_value=False, **kwargs):
        return self._widget(label, bool(initial_value), **kwargs)

    def add_dropdown(self, label, options=(), initial_value=None, **kwargs):
        """A choice. Its ``options`` are kept, as viser keeps them, and a panel may reassign them."""
        handle = self._widget(label, initial_value if initial_value is not None else options[0], **kwargs)
        handle.options = tuple(options)
        return handle

    def add_button_group(self, label, options=(), **kwargs):
        """A row of buttons. ``fire("Close")`` is what pressing one of them does.

        Its handle carries the pressed option as ``value``, which is what the
        panel's callback reads -- so a test presses a button by naming it, the
        way the browser does, rather than by setting the state behind it.
        """
        return self._widget(label, options[0] if options else None, is_button=True, **kwargs)

    def add_number(self, label, initial_value=0, **kwargs):
        return self._widget(label, initial_value, **kwargs)

    def add_slider(self, label, min=0.0, max=1.0, step=0.1, initial_value=0.0, **kwargs):
        handle = self._widget(label, initial_value, **kwargs)
        handle.min, handle.max, handle.step = min, max, step
        return handle

    def add_vector3(self, label, initial_value=(0.0, 0.0, 0.0), **kwargs):
        return self._widget(label, tuple(initial_value), **kwargs)

    def add_rgb(self, label, initial_value=(0, 0, 0), **kwargs):
        return self._widget(label, tuple(initial_value), **kwargs)

    def add_text(self, label, initial_value="", **kwargs):
        return self._widget(label, initial_value, **kwargs)

    def add_button(self, label, **kwargs):
        return self._widget(label, label, is_button=True, **kwargs)

    def add_image(self, image, label=None, **kwargs):
        handle = _StubImage(image, label)
        self.images.append(handle)
        return handle


class _StubNode:
    """A scene node handle: a pose, a visibility, what it was added with, and the callbacks a drag calls.

    ``fire`` is the browser moving it: a gizmo dragged to ``position`` and
    ``quat`` (xyzw, the library's order; the handle keeps viser's wxyz), then
    every ``on_update`` callback called with an event whose ``target`` is this
    node and whose ``phase`` is the drag's -- ``"start"``, ``"update"`` or
    ``"end"`` -- which is what a transform control's callback is handed.
    """

    def __init__(self, name: str, **kwargs) -> None:
        self.name = name
        self.kwargs = dict(kwargs)
        """Everything it was added with, for a test to read back."""
        self.position = tuple(kwargs.get("position", (0.0, 0.0, 0.0)))
        self.wxyz = tuple(kwargs.get("wxyz", (1.0, 0.0, 0.0, 0.0)))
        self.visible = bool(kwargs.get("visible", True))
        self.color = kwargs.get("color")
        self.text = kwargs.get("text")
        self.removed = False
        self.callbacks: list = []
        """The ``on_update`` handlers: what a drag calls."""
        self.click_handlers: list = []
        """The ``on_click`` handlers: what a click calls, and a drag does not."""

    def on_update(self, fn):
        self.callbacks.append(fn)
        return fn

    def on_click(self, fn):
        self.click_handlers.append(fn)
        return fn

    def click(self) -> None:
        """What a click on it in the browser does: every ``on_click`` handler, and no ``on_update`` one."""
        for fn in list(self.click_handlers):
            fn(_StubEvent(self, "click"))

    def fire(self, position=None, quat=None, phase: str = "update") -> None:
        """What the browser does: move the node, then tell the server about it."""
        if position is not None:
            self.position = tuple(float(v) for v in position)
        if quat is not None:
            x, y, z, w = (float(v) for v in quat)
            self.wxyz = (w, x, y, z)
        for fn in list(self.callbacks):
            fn(_StubEvent(self, phase))

    def remove(self) -> None:
        self.removed = True


class _StubFrustum(_StubNode):
    """A camera frustum handle: a pose, an image, a colour and a click callback."""

    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.fov = kwargs.get("fov")
        self.aspect = kwargs.get("aspect")
        self.scale = kwargs.get("scale")
        self.color = kwargs.get("color")
        self.image = kwargs.get("image")


class _StubScene:
    """Enough of ``server.scene`` for a panel's gizmo, handles and frustums."""

    def __init__(self) -> None:
        self.nodes: list[_StubNode] = []
        # viser keeps one of these and it is the only way to ask what is in the
        # scene by name, which the viewport panel needs to hide the model and
        # keep the furniture. Stubbed under its real name on purpose: a rename
        # upstream should fail a test here rather than quietly draw nothing.
        self._handle_from_node_name: dict[str, _StubNode] = {}
        self.environment: dict = {}
        """What ``configure_environment_map`` was last told."""
        self.lights: dict[str, dict] = {}
        """Each light by name, with what it was added with: a light added again under its name replaces it."""

    def _node(self, name: str, visible: bool = False, **kwargs) -> _StubNode:
        return self._put(_StubNode(name, **{**kwargs, "visible": visible}))

    def _put(self, node: _StubNode) -> _StubNode:
        """Add ``node``, replacing one added before under its name, as viser's node table does."""
        before = self._handle_from_node_name.get(node.name)
        if before is not None:
            self.nodes = [n for n in self.nodes if n is not before]
        self.nodes.append(node)
        self._handle_from_node_name[node.name] = node
        return node

    def set_background_image(self, image, **kwargs):
        self.background = image

    def add_camera_frustum(self, name, **kwargs):
        return self._put(_StubFrustum(name, **kwargs))

    def add_transform_controls(self, name, **kwargs):
        return self._node(name, kwargs.pop("visible", True), **kwargs)

    def add_mesh_simple(self, name, **kwargs):
        return self._node(name, kwargs.pop("visible", False), **kwargs)

    def add_icosphere(self, name, **kwargs):
        return self._node(name, kwargs.pop("visible", True), **kwargs)

    def add_label(self, name, text="", **kwargs):
        return self._node(name, kwargs.pop("visible", True), text=text, **kwargs)

    def add_box(self, name, **kwargs):
        return self._node(name, kwargs.pop("visible", True), **kwargs)

    def add_frame(self, name, **kwargs):
        return self._node(name, kwargs.pop("visible", True), **kwargs)

    def add_point_cloud(self, name, points=(), colors=(), **kwargs):
        return self._node(name, kwargs.pop("visible", True), points=points, colors=colors, **kwargs)

    def add_line_segments(self, name, points=(), colors=(), **kwargs):
        return self._node(name, kwargs.pop("visible", True), points=points, colors=colors, **kwargs)

    def add_spline_catmull_rom(self, name, positions=(), **kwargs):
        return self._node(name, kwargs.pop("visible", True), positions=positions, **kwargs)

    def configure_environment_map(self, **kwargs) -> None:
        self.environment = dict(kwargs)

    def add_light_ambient(self, name, **kwargs):
        self.lights[name] = dict(kwargs)
        return self._node(name, kwargs.pop("visible", True), **kwargs)

    def add_light_directional(self, name, **kwargs):
        self.lights[name] = dict(kwargs)
        return self._node(name, kwargs.pop("visible", True), **kwargs)

    def add_glb(self, name, glb_data=b"", **kwargs):
        return self._node(name, kwargs.pop("visible", True), glb_data=glb_data, **kwargs)

    def on_click(self, *args, **kwargs):
        def register(fn):
            self.click_callbacks.append(fn)
            return fn

        return register

    def remove_click_callback(self, callback="all"):
        if callback == "all":
            self.click_callbacks.clear()
        else:
            self.click_callbacks[:] = [fn for fn in self.click_callbacks if fn != callback]

    @property
    def click_callbacks(self) -> list:
        """What ``on_click`` registered and ``remove_click_callback`` has not removed: viser keeps a list too."""
        if "_clicks" not in self.__dict__:
            self._clicks = []
        return self._clicks


class _StubSession:
    def __init__(self, env, gui) -> None:
        from robosimstudio.core.profile import NULL

        self.env = env
        # ``get_clients`` because a panel that renders the browser's own camera
        # has to ask who is looking; an empty dict is a page nobody has opened.
        self.server = type(
            "_Server", (), {"gui": gui, "scene": _StubScene(), "get_clients": staticmethod(lambda: {})}
        )()
        self.status = ""
        self.obs: dict = {}
        self.paused = False
        self.arms: dict = {}
        # The session's measurement surface, which panels now read: the
        # disabled profiler, and the draw cadence the frustum pass paces by.
        self.profile = NULL
        self.render_every = 4


__all__ = [
    "_StubEvent",
    "_StubFolder",
    "_StubFrustum",
    "_StubGui",
    "_StubImage",
    "_StubMarkdown",
    "_StubNode",
    "_StubScene",
    "_StubSession",
    "_StubWidget",
]
