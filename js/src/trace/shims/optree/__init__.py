"""Pure-Python stand-in for the `optree` C extension, covering exactly what
keras/src/tree/optree_impl.py calls (keras 3.15.1). optree has no
wasm/emscripten wheel, so under Pyodide this shim is imported instead.

Semantics mirrored from optree: dict children in SORTED key order (insertion
order if keys aren't mutually comparable), OrderedDict in insertion order,
namedtuple/tuple/list/deque by index, None is an empty node unless
none_is_leaf=True; custom nodes via register_pytree_node(_class) in a
namespace. tree_map flattens the extra trees "up to" the first tree's
structure (a rest's subtree at a leaf position is handed to func whole),
which keras' map_structure relies on for its same-structure check.

Anything outside that contract raises loudly rather than approximating.
"""

from collections import OrderedDict, defaultdict, deque

__version__ = "0.0.0-keras-tinygrad-shim"

_REGISTRY = {}  # (namespace, cls) -> (flatten_fn, unflatten_fn)


def _is_namedtuple(x):
    return isinstance(x, tuple) and hasattr(type(x), "_fields")


def _sorted_keys(d):
    keys = list(d.keys())
    try:
        return sorted(keys)
    except TypeError:
        return keys  # optree falls back to insertion order for unorderable keys


def _lookup(tree, namespace):
    for ns in (namespace, ""):
        entry = _REGISTRY.get((ns, type(tree)))
        if entry is not None:
            return entry
    return None


def _node_children(tree, none_is_leaf, namespace):
    """(kind, children, metadata, entries) for a node, or None for a leaf."""
    if tree is None:
        return None if none_is_leaf else ("none", [], None, [])
    t = type(tree)
    reg = _lookup(tree, namespace)
    if reg is not None:
        out = reg[0](tree)
        children, metadata = list(out[0]), out[1]
        entries = list(out[2]) if len(out) > 2 and out[2] is not None else list(range(len(children)))
        return ("custom", children, (t, metadata), entries)
    if _is_namedtuple(tree):
        return ("namedtuple", list(tree), t, list(range(len(tree))))
    if t is tuple:
        return ("tuple", list(tree), None, list(range(len(tree))))
    if t is list:
        return ("list", list(tree), None, list(range(len(tree))))
    if t is deque:
        return ("deque", list(tree), tree.maxlen, list(range(len(tree))))
    if t is OrderedDict:
        keys = list(tree.keys())
        return ("odict", [tree[k] for k in keys], keys, keys)
    if t is defaultdict:
        keys = _sorted_keys(tree)
        return ("ddict", [tree[k] for k in keys], (tree.default_factory, keys), keys)
    if t is dict:
        keys = _sorted_keys(tree)
        return ("dict", [tree[k] for k in keys], keys, keys)
    return None


def _rebuild(kind, metadata, children):
    if kind == "none":
        return None
    if kind == "custom":
        cls, meta = metadata
        return _REGISTRY_UNFLATTEN(cls)(meta, children)
    if kind == "namedtuple":
        return metadata(*children)
    if kind == "tuple":
        return tuple(children)
    if kind == "list":
        return list(children)
    if kind == "deque":
        return deque(children, maxlen=metadata)
    if kind == "odict":
        return OrderedDict(zip(metadata, children))
    if kind == "ddict":
        factory, keys = metadata
        return defaultdict(factory, zip(keys, children))
    if kind == "dict":
        return dict(zip(metadata, children))
    raise TypeError(f"optree shim: unknown node kind {kind!r}")


def _REGISTRY_UNFLATTEN(cls):
    for (ns, c), (_, unflatten) in _REGISTRY.items():
        if c is cls:
            return unflatten
    raise TypeError(f"optree shim: {cls!r} is not a registered pytree node")


class PyTreeSpec:
    """Structure without leaves. `kind is None` marks a leaf position."""

    __slots__ = ("kind", "metadata", "entries", "children")

    def __init__(self, kind, metadata=None, entries=(), children=()):
        self.kind, self.metadata, self.entries, self.children = kind, metadata, list(entries), list(children)

    @property
    def num_leaves(self):
        return 1 if self.kind is None else sum(c.num_leaves for c in self.children)

    @property
    def num_nodes(self):
        return 1 + (0 if self.kind is None else sum(c.num_nodes for c in self.children))

    def is_leaf(self):
        return self.kind is None

    def unflatten(self, leaves):
        leaves = list(leaves)
        if len(leaves) != self.num_leaves:
            raise ValueError(f"optree shim: treespec expects {self.num_leaves} leaves, got {len(leaves)}")
        it = iter(leaves)
        return self._unflatten(it)

    def _unflatten(self, it):
        if self.kind is None:
            return next(it)
        return _rebuild(self.kind, self.metadata, [c._unflatten(it) for c in self.children])

    def flatten_up_to(self, tree):
        """Leaves of `tree` taken at THIS spec's leaf positions (optree's
        treespec.flatten_up_to): subtrees below a leaf position stay whole."""
        out = []
        self._flatten_up_to(tree, out)
        return out

    def _flatten_up_to(self, tree, out):
        if self.kind is None:
            out.append(tree)
            return
        node = _node_children(tree, none_is_leaf=(self.kind != "none"), namespace=_NS_ANY)
        if node is None or node[0] != self.kind or len(node[1]) != len(self.children) or list(node[3]) != self.entries:
            raise ValueError(
                f"optree shim: structure mismatch — expected {self.kind} "
                f"with entries {self.entries}, got {type(tree).__name__}"
            )
        for child_spec, child in zip(self.children, node[1]):
            child_spec._flatten_up_to(child, out)

    def __eq__(self, other):
        return (
            isinstance(other, PyTreeSpec)
            and self.kind == other.kind
            and self.entries == other.entries
            and self.children == other.children
        )

    def __repr__(self):
        return f"PyTreeSpec({self.kind or 'leaf'}, leaves={self.num_leaves}, nodes={self.num_nodes})"


class _AnyNamespace:
    """flatten_up_to must match whatever namespace the spec was built in;
    registry lookup with this sentinel tries every namespace."""


_NS_ANY = _AnyNamespace()


def _lookup_any(tree):
    for (ns, c), entry in _REGISTRY.items():
        if c is type(tree):
            return entry
    return None


_orig_lookup = _lookup


def _lookup(tree, namespace):  # noqa: F811 — override to support _NS_ANY
    if namespace is _NS_ANY:
        return _lookup_any(tree)
    return _orig_lookup(tree, namespace)


def _flatten(tree, is_leaf, none_is_leaf, namespace, path, leaves, paths):
    if is_leaf is not None and is_leaf(tree):
        leaves.append(tree)
        paths.append(tuple(path))
        return PyTreeSpec(None)
    node = _node_children(tree, none_is_leaf, namespace)
    if node is None:
        leaves.append(tree)
        paths.append(tuple(path))
        return PyTreeSpec(None)
    kind, children, metadata, entries = node
    specs = [
        _flatten(child, is_leaf, none_is_leaf, namespace, path + [entry], leaves, paths)
        for child, entry in zip(children, entries)
    ]
    return PyTreeSpec(kind, metadata, entries, specs)


def tree_flatten(tree, is_leaf=None, *, none_is_leaf=False, namespace=""):
    leaves, paths = [], []
    spec = _flatten(tree, is_leaf, none_is_leaf, namespace, [], leaves, paths)
    return leaves, spec


def tree_flatten_with_path(tree, is_leaf=None, *, none_is_leaf=False, namespace=""):
    leaves, paths = [], []
    spec = _flatten(tree, is_leaf, none_is_leaf, namespace, [], leaves, paths)
    return paths, leaves, spec


def tree_paths(tree, is_leaf=None, *, none_is_leaf=False, namespace=""):
    return tree_flatten_with_path(tree, is_leaf, none_is_leaf=none_is_leaf, namespace=namespace)[0]


def tree_unflatten(treespec, leaves):
    return treespec.unflatten(leaves)


def tree_is_leaf(tree, is_leaf=None, *, none_is_leaf=False, namespace=""):
    if is_leaf is not None and is_leaf(tree):
        return True
    return _node_children(tree, none_is_leaf, namespace) is None


def tree_map(func, tree, *rests, is_leaf=None, none_is_leaf=False, namespace=""):
    leaves, spec = tree_flatten(tree, is_leaf, none_is_leaf=none_is_leaf, namespace=namespace)
    rest_leaves = [spec.flatten_up_to(r) for r in rests]
    return spec.unflatten(func(*args) for args in zip(leaves, *rest_leaves))


def register_pytree_node(cls, flatten_func, unflatten_func, *, namespace=""):
    key = (namespace, cls)
    if key in _REGISTRY:
        raise ValueError(f"PyTree type {cls!r} is already registered in namespace {namespace!r}.")
    _REGISTRY[key] = (flatten_func, unflatten_func)
    return cls


def register_pytree_node_class(cls=None, *, namespace=""):
    def deco(c):
        key = (namespace, c)
        if key not in _REGISTRY:  # decorator re-application is harmless
            _REGISTRY[key] = (lambda obj: obj.tree_flatten(), lambda meta, children: c.tree_unflatten(meta, children))
        return c

    return deco(cls) if cls is not None else deco


def unregister_pytree_node(cls, *, namespace=""):
    _REGISTRY.pop((namespace, cls), None)
