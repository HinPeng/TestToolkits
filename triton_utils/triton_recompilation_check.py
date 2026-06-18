import hashlib
import json

import triton.compiler as tc
import triton.compiler.compiler as tcc
from triton.runtime.cache import get_cache_manager
from triton.runtime.driver import driver
from triton.runtime.jit import JITFunction

_prev = {}
_orig_compile = tcc.compile


def install_triton_probe():
    def _names(jit_fn, ids):
        xs = [p.name for p in jit_fn.params]
        return [xs[i] for i in ids]

    def cache_hook(**kw):
        kernel = f"{kw['fn'].module}.{kw['fn'].name}"
        cur = json.loads(kw["compile"]["specialization_data"])
        props = cur["attrs"].get("arg_properties", {})
        div = tuple(props.get("tt.divisibility", []))
        eq1 = tuple(props.get("tt.equal_to", []))

        prev = _prev.get(kernel)
        why = ["first"] if prev is None else []
        if prev and prev["sig"] != cur["signature"]:
            why.append("signature")
        if prev and prev["div"] != div:
            why.append(f"alignment:{_names(kw['fn'].jit_function, prev['div'])}->{_names(kw['fn'].jit_function, div)}")
        if prev and prev["eq1"] != eq1:
            why.append(f"equal_to_1:{_names(kw['fn'].jit_function, prev['eq1'])}->{_names(kw['fn'].jit_function, eq1)}")

        print(f"[TRITON JIT MISS] {kernel} why={'|'.join(why)}")
        _prev[kernel] = {"sig": cur["signature"], "div": div, "eq1": eq1}

    def wrapped_compile(src, target=None, options=None, _env_vars=None):
        tgt = driver.active.get_current_target() if target is None else target
        backend = tcc.make_backend(tgt)
        local_src = src if isinstance(src, tcc.ASTSource) else tcc.IRSource(src)
        backend_options = backend.parse_options(dict(options or {}, **local_src.parse_options()))
        env_vars = tcc.get_cache_invalidating_env_vars() if _env_vars is None else _env_vars

        cache_key = tcc.get_cache_key(local_src, backend, backend_options, env_vars)
        cache_hash = hashlib.sha256(cache_key.encode()).hexdigest()
        meta = f"{local_src.name[:150]}.json"
        disk_hit = get_cache_manager(cache_hash).get_group(meta) is not None

        out = _orig_compile(src, target=target, options=options, _env_vars=_env_vars)
        print(f"[TRITON COMPILE] {local_src.name} disk_cache={'HIT' if disk_hit else 'MISS'}")
        return out

    JITFunction.cache_hook = cache_hook
    tcc.compile = wrapped_compile
    tc.compile = wrapped_compile
