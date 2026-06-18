# /tmp/triton_param_recompile_probe.py
import hashlib
import json

import triton.compiler as tc
import triton.compiler.compiler as tcc
from triton.runtime.cache import get_cache_manager
from triton.runtime.driver import driver
from triton.runtime.jit import JITFunction

_INSTALLED = False
_MISSING = object()

_PREV = {}
_PENDING = {}

_ORIG_COMPILE = tcc.compile
_ORIG_CACHE_HOOK = JITFunction.cache_hook


def _kernel_name_from_jit_fn(jit_fn):
    return f"{jit_fn.__module__}.{jit_fn.__name__}"


def _kernel_name_from_astsrc(src):
    fn = src.fn
    return f"{fn.__module__}.{fn.__name__}"


def _param_names(jit_fn):
    return [p.name for p in jit_fn.params]


def _is_pointer_type(sig, name):
    ty = sig.get(name)
    return isinstance(ty, str) and ty.startswith("*")


def _fmt(v):
    if v is _MISSING:
        return "<absent>"
    return repr(v)


def install_triton_recompile_probe():
    global _INSTALLED
    if _INSTALLED:
        return

    def cache_hook(**kw):
        kernel = f"{kw['fn'].module}.{kw['fn'].name}"
        jit_fn = kw["fn"].jit_function
        names = _param_names(jit_fn)

        cur = json.loads(kw["compile"]["specialization_data"])
        cur_sig = dict(cur.get("signature", {}))
        cur_props = cur.get("attrs", {}).get("arg_properties", {})
        cur_div = set(cur_props.get("tt.divisibility", []))
        cur_eq1 = set(cur_props.get("tt.equal_to", []))
        cur_const = dict(cur.get("constants", {}))

        prev = _PREV.get(kernel)
        reasons = []

        if prev is not None:
            # 1. signature/type change
            for name in sorted(set(prev["sig"]) | set(cur_sig)):
                old = prev["sig"].get(name, _MISSING)
                new = cur_sig.get(name, _MISSING)
                if old != new:
                    reasons.append((name, f"signature {_fmt(old)} -> {_fmt(new)}"))

            # 2. tt.divisibility change
            for idx in sorted(prev["div"] | cur_div):
                before = idx in prev["div"]
                after = idx in cur_div
                if before == after:
                    continue
                pname = names[idx]
                label = (
                    "alignment"
                    if _is_pointer_type(cur_sig, pname) or _is_pointer_type(prev["sig"], pname)
                    else "divisibility_16"
                )
                reasons.append(
                    (pname, f"{label} {'D' if before else 'N'} -> {'D' if after else 'N'}")
                )

            # 3. tt.equal_to(=1) change
            eq1_changed = set()
            for idx in sorted(prev["eq1"] | cur_eq1):
                before = idx in prev["eq1"]
                after = idx in cur_eq1
                if before == after:
                    continue
                pname = names[idx]
                eq1_changed.add(pname)
                reasons.append(
                    (pname, f"equal_to_1 {'on' if before else 'off'} -> {'on' if after else 'off'}")
                )

            # 4. constexpr / constant value change
            for name in sorted(set(prev["const"]) | set(cur_const)):
                old = prev["const"].get(name, _MISSING)
                new = cur_const.get(name, _MISSING)
                if old != new and name not in eq1_changed:
                    reasons.append((name, f"constant_value {_fmt(old)} -> {_fmt(new)}"))

        _PENDING[kernel] = reasons
        _PREV[kernel] = {
            "sig": cur_sig,
            "div": cur_div,
            "eq1": cur_eq1,
            "const": cur_const,
        }

        if callable(_ORIG_CACHE_HOOK):
            return _ORIG_CACHE_HOOK(**kw)

    def wrapped_compile(src, target=None, options=None, _env_vars=None):
        tgt = driver.active.get_current_target() if target is None else target
        backend = tcc.make_backend(tgt)

        local_src = src if isinstance(src, tcc.ASTSource) else tcc.IRSource(src)
        backend_options = backend.parse_options(dict(options or {}, **local_src.parse_options()))
        env_vars = tcc.get_cache_invalidating_env_vars() if _env_vars is None else _env_vars

        cache_key = tcc.get_cache_key(local_src, backend, backend_options, env_vars)
        cache_hash = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()

        meta = f"{local_src.name[:150]}.json"
        disk_hit = get_cache_manager(cache_hash).get_group(meta) is not None

        out = _ORIG_COMPILE(src, target=target, options=options, _env_vars=_env_vars)

        if isinstance(local_src, tcc.ASTSource):
            kernel = _kernel_name_from_astsrc(local_src)
            reasons = _PENDING.pop(kernel, [])
            if (not disk_hit) and reasons:
                for param, reason in reasons:
                    print(f"[TRITON RECOMPILE] kernel={kernel} param={param} reason={reason}")

        return out

    JITFunction.cache_hook = cache_hook
    tcc.compile = wrapped_compile
    tc.compile = wrapped_compile
    _INSTALLED = True


def uninstall_triton_param_recompile_probe():
    global _INSTALLED
    if not _INSTALLED:
        return
    JITFunction.cache_hook = _ORIG_CACHE_HOOK
    tcc.compile = _ORIG_COMPILE
    tc.compile = _ORIG_COMPILE
    _INSTALLED = False
