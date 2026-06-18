# /tmp/triton_align_probe.py
import hashlib
import json

import triton.compiler as tc
import triton.compiler.compiler as tcc
from triton.runtime.cache import get_cache_manager
from triton.runtime.driver import driver
from triton.runtime.jit import JITFunction

_INSTALLED = False
_PREV = {}
_PENDING = {}

_ORIG_COMPILE = tcc.compile
_ORIG_CACHE_HOOK = JITFunction.cache_hook


def _kernel_name_from_astsrc(src):
    fn = src.fn
    return f"{fn.__module__}.{fn.__name__}"


def _arg_names(jit_fn):
    return [p.name for p in jit_fn.params]


def install_triton_align_probe():
    global _INSTALLED
    if _INSTALLED:
        return

    def cache_hook(**kw):
        kernel = f"{kw['fn'].module}.{kw['fn'].name}"
        jit_fn = kw["fn"].jit_function
        cur = json.loads(kw["compile"]["specialization_data"])

        cur_sig = cur["signature"]
        cur_div = set(cur["attrs"].get("arg_properties", {}).get("tt.divisibility", []))

        prev = _PREV.get(kernel)
        if prev is not None and prev["div"] != cur_div:
            names = _arg_names(jit_fn)
            changed = []
            for i in sorted(prev["div"] | cur_div):
                before = "D" if i in prev["div"] else "N"
                after = "D" if i in cur_div else "N"
                if before != after:
                    changed.append(f"{names[i]}:{before}->{after}")

            _PENDING[kernel] = {
                "params": changed,
                "sig_changed": int(prev["sig"] != cur_sig),
            }

        _PREV[kernel] = {
            "sig": cur_sig,
            "div": cur_div,
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
            info = _PENDING.pop(kernel, None)
            if info:
                print(
                    f"[TRITON ALIGN RECOMPILE] {kernel} "
                    f"disk_cache={'HIT' if disk_hit else 'MISS'} "
                    f"sig_changed={info['sig_changed']} "
                    f"params={','.join(info['params'])}"
                )

        return out

    JITFunction.cache_hook = cache_hook
    tcc.compile = wrapped_compile
    tc.compile = wrapped_compile
    _INSTALLED = True


def uninstall_triton_align_probe():
    global _INSTALLED
    if not _INSTALLED:
        return
    JITFunction.cache_hook = _ORIG_CACHE_HOOK
    tcc.compile = _ORIG_COMPILE
    tc.compile = _ORIG_COMPILE
    _INSTALLED = False
