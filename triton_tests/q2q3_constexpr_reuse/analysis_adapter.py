"""Load the checkout's analyzer without importing Triton, torch or a driver.

Only JIT metadata and language object identities are adapted. Classification,
dependency propagation, operation constraints and tile matching are the real
Ascend implementation. Kernel files are parsed, never imported or executed.
"""

import ast
from contextlib import contextmanager
import hashlib
import importlib
import inspect
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace


class Symbol:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def __call__(self, *args, **kwargs):
        raise AssertionError("Analysis must not execute language operations")


class Constexpr:
    def __init__(self, value):
        self.value = value


class Dtype(Symbol):
    """Language dtype identity for source-only isinstance checks."""


class Pointer:
    """Type metadata only: no allocation, address or device access."""

    def __init__(self, dtype):
        self.dtype = dtype

    def data_ptr(self):
        raise AssertionError("Analysis must not read device addresses")


def jit_decorator(node):
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if ast.unparse(target) in ("triton.jit", "jit"):
            return decorator
    return None


def literal(node):
    if isinstance(node, ast.Call) and ast.unparse(node.func) in ("tl.constexpr", "core.constexpr"):
        return Constexpr(ast.literal_eval(node.args[0]))
    return ast.literal_eval(node)


class SourceJIT:
    """The source/signature subset used by Uses and match_grid_stride."""

    def __init__(self, node, source, path, scope):
        if node.args.posonlyargs or node.args.kwonlyargs or node.args.vararg or node.args.kwarg:
            raise ValueError(f"Unsupported signature in {path}:{node.name}")
        self.name = node.name
        self.path = Path(path)
        self.starting_line_number = node.lineno
        self.src = "".join(source.splitlines(keepends=True)[node.lineno - 1:node.end_lineno])
        self.scope = scope
        self.arg_names = [arg.arg for arg in node.args.args]
        self.cache_key = hashlib.sha256(self.src.encode()).hexdigest()
        decorator = jit_decorator(node)
        options = {kw.arg: literal(kw.value) for kw in decorator.keywords} if isinstance(decorator, ast.Call) else {}
        defaults = dict(zip(self.arg_names[len(self.arg_names) - len(node.args.defaults):], node.args.defaults))
        self.params = []
        signature = []
        canonical = {"int32": "i32", "int64": "i64", "float32": "fp32", "float16": "fp16", "bool": "u1"}
        for index, arg in enumerate(node.args.args):
            annotation = ast.unparse(arg.annotation) if arg.annotation else ""
            leaf = annotation.rsplit(".", 1)[-1]
            default = inspect.Parameter.empty
            if arg.arg in defaults:
                try:
                    default = literal(defaults[arg.arg])
                except (ValueError, TypeError):
                    # Nonliteral library defaults are opaque, never evaluated.
                    default = Symbol(ast.unparse(defaults[arg.arg]))
            signature.append(inspect.Parameter(arg.arg, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=default))
            self.params.append(SimpleNamespace(
                num=index, name=arg.arg, is_constexpr=leaf == "constexpr",
                annotation_type=canonical.get(leaf, ""),
                do_not_specialize=arg.arg in options.get("do_not_specialize", ()) or index in options.get("do_not_specialize", ()),
            ))
        self.signature = inspect.Signature(signature)
        self.constexprs = tuple(p.num for p in self.params if p.is_constexpr)

    def parse(self):
        return ast.parse(self.src)

    def get_capture_scope(self):
        return self.scope

    def __call__(self, *args, **kwargs):
        raise AssertionError("Analysis must not execute kernels or JIT helpers")


def module(name, path=None):
    result = ModuleType(name)
    if path is not None:
        result.__path__ = [str(path)]
    return result


class SourceLoader:
    def __init__(self, triton_root):
        self.triton_root = Path(triton_root)
        self.scopes = {}
        self.language = module("triton.language")
        self.triton = module("triton")
        self.triton.language = self.language
        self._language_scope()

    def _language_scope(self):
        # Public identities and aliases come from this checkout, not an invented
        # list of operations. Unmodeled builtins still have no analyzer summary.
        root = self.triton_root / "python/triton/language"
        namespaces = {}
        for name in ("core", "math", "standard", "extra", "random"):
            namespace = module("triton.language." + name)
            namespaces[name] = namespace
            setattr(self.language, name, namespace)
            path = root / (name + ".py")
            if path.exists():
                tree = ast.parse(path.read_text(encoding="utf-8-sig"))
                for node in tree.body:
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                        setattr(namespace, node.name, Symbol(node.name))
                    elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                        for target in targets:
                            if isinstance(target, ast.Name):
                                try:
                                    if (isinstance(node.value, ast.Call)
                                            and ast.unparse(node.value.func) == "dtype"):
                                        value = Dtype(ast.literal_eval(node.value.args[0]))
                                    else:
                                        value = literal(node.value)
                                except (ValueError, TypeError):
                                    value = Symbol(target.id)
                                setattr(namespace, target.id, value)
        namespaces["core"].constexpr = Constexpr
        namespaces["core"].dtype = Dtype
        standard = self.load(root / "standard.py", extra_scope=namespaces)
        vars(namespaces["standard"]).update(standard)
        for node in ast.parse((root / "__init__.py").read_text()).body:
            if isinstance(node, ast.ImportFrom) and node.module in namespaces:
                origin = namespaces[node.module]
                for alias in node.names:
                    if alias.name == "*":
                        raise ValueError("Language star exports need adapter support")
                    value = vars(origin).get(alias.name)
                    if value is None:
                        value = Symbol(alias.name)
                        setattr(origin, alias.name, value)
                    setattr(self.language, alias.asname or alias.name, value)

    def load(self, path, extra_scope=None):
        path = Path(path).resolve()
        if path in self.scopes:
            return self.scopes[path]
        source = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source)
        scope = dict(extra_scope or {})
        self.scopes[path] = scope
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "triton.language":
                        scope[alias.asname or "triton"] = self.language if alias.asname else self.triton
                    elif alias.name == "triton":
                        scope[alias.asname or "triton"] = self.triton
            elif isinstance(node, ast.ImportFrom) and node.module == "triton.language":
                for alias in node.names:
                    if alias.name in vars(self.language):
                        scope[alias.asname or alias.name] = vars(self.language)[alias.name]
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                try:
                    value = literal(node.value)
                except (ValueError, TypeError):
                    continue
                for target in targets:
                    if isinstance(target, ast.Name):
                        scope[target.id] = value
            elif isinstance(node, ast.FunctionDef) and jit_decorator(node) is not None:
                scope[node.name] = SourceJIT(node, source, path, scope)
        # Resolve repository-local helper imports only when referenced by a JIT.
        called = {n.func.id for fn in scope.values() if isinstance(fn, SourceJIT)
                  for n in ast.walk(fn.parse()) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        for node in tree.body:
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            aliases = [a for a in node.names if (a.asname or a.name) in called]
            if not aliases:
                continue
            if node.level:
                candidate = path.parents[node.level - 1].joinpath(*node.module.split(".")).with_suffix(".py")
            elif node.module.startswith("src."):
                base = next((p.parent for p in path.parents if p.name == "src"), None)
                candidate = base.joinpath(*node.module.split(".")).with_suffix(".py") if base else None
            else:
                candidate = None
            if candidate and candidate.is_file():
                imported = self.load(candidate)
                for alias in aliases:
                    if alias.name in imported:
                        scope[alias.asname or alias.name] = imported[alias.name]
        return scope

    def kernel(self, path, name):
        fn = self.load(path).get(name)
        if not isinstance(fn, SourceJIT):
            raise ValueError(f"Top-level JIT definition not found: {path}:{name}")
        return fn


@contextmanager
def analyzer(triton_root):
    """Temporarily adapt imports; restore even on failure or alongside NPU tests."""
    root = Path(triton_root).resolve()
    reuse = root / "third_party/ascend/backend/reuse"
    if not (reuse / "analysis.py").is_file():
        raise FileNotFoundError(f"Ascend reuse analyzer not found under {root}")
    loader = SourceLoader(root)
    runtime = module("triton.runtime")
    jit = module("triton.runtime.jit")
    jit.JITFunction = SourceJIT
    runtime.jit = jit
    loader.triton.runtime = runtime
    overrides = {"triton": loader.triton, "triton.language": loader.language,
                 "triton.runtime": runtime, "triton.runtime.jit": jit,
                 "_q2q3_ascend_reuse": module("_q2q3_ascend_reuse", reuse)}
    previous = dict(sys.modules)
    try:
        sys.modules.update(overrides)
        analysis = importlib.import_module("_q2q3_ascend_reuse.analysis")
        config = importlib.import_module("_q2q3_ascend_reuse.config")
        yield SimpleNamespace(analyze=analysis.analyze, config=config, loader=loader)
    finally:
        for name in list(sys.modules):
            if name in overrides or name.startswith("_q2q3_ascend_reuse."):
                if name in previous:
                    sys.modules[name] = previous[name]
                else:
                    del sys.modules[name]
