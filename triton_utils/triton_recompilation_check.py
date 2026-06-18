import hashlib
import json
import time

import triton.compiler as tc
import triton.compiler.compiler as tcc
from triton.runtime.cache import get_cache_manager
from triton.runtime.driver import driver
from triton.runtime.jit import JITFunction

_last_compile_req = {}
_orig_compile = tcc.compile


def install_triton_recompile_probe():
    def idxs_to_names(jit_fn, idxs):
        names = [p.name for p in jit_fn.params]
        return [names[i] if i < len(names) else f"arg{i}" for i in sorted(idxs)]

    def cache_hook(**kw):
        logical_name = f"{kw['fn'].module}.{kw['fn'].name}"
        cur = json.loads(kw["compile"]["specialization_data"])
        cur_attrs = cur["attrs"].get("arg_properties", {})
        prev = _last_compile_req.get(logical_name)

        reasons = []
        if prev is None:
            reasons.append("first specialization seen in this process")
        else:
            if prev["signature"] != cur["signature"]:
                reasons.append(f"signature changed: {prev['signature']} -> {cur['signature']}")

            prev_div = set(prev["attrs"].get("arg_properties", {}).get("tt.divisibility", []))
            cur_div = set(cur_attrs.get("tt.divisibility", []))
            if prev_div != cur_div:
                reasons.append(
                    "tt.divisibility changed: "
                    f"{idxs_to_names(kw['fn'].jit_function, prev_div)} -> "
                    f"{idxs_to_names(kw['fn'].jit_function, cur_div)}"
                )

            prev_eq1 = set(prev["attrs"].get("arg_properties", {}).get("tt.equal_to", []))
            cur_eq1 = set(cur_attrs.get("tt.equal_to", []))
            if prev_eq1 != cur_eq1:
                reasons.append(
                    "tt.equal_to(=1) changed: "
                    f"{idxs_to_names(kw['fn'].jit_function, prev_eq1)} -> "
                    f"{idxs_to_names(kw['fn'].jit_function, cur_eq1)}"
                )

            if prev["constants"] != cur["constants"]:
                reasons.append(f"constants changed: {prev['constants']} -> {cur['constants']}")

            keys = ("debug", "num_warps", "num_ctas", "num_stages",
                    "force_simt_only", "compile_mode", "multibuffer")
            opt_delta = {
                k: (prev["options"].get(k), cur["options"].get(k))
                for k in keys
                if prev["options"].get(k) != cur["options"].get(k)
            }
            if opt_delta:
                reasons.append(f"options changed: {opt_delta}")

        print(f"[TRITON JIT MISS] {logical_name}")
        print(f"  repr      = {kw['repr']}")
        print(f"  jit_key   = {cur['key']}")
        print(f"  signature = {cur['signature']}")
        print(f"  constants = {cur['constants']}")
        print(f"  attrs     = {cur_attrs}")
        print(f"  reason    = {' | '.join(reasons)}")

        _last_compile_req[logical_name] = cur

    def wrapped_compile(src, target=None, options=None, _env_vars=None):
        tgt = driver.active.get_current_target() if target is None else target
        backend = tcc.make_backend(tgt)
        local_src = src if isinstance(src, tcc.ASTSource) else tcc.IRSource(src)
        backend_options = backend.parse_options(dict(options or {}, **local_src.parse_options()))
        env_vars = tcc.get_cache_invalidating_env_vars() if _env_vars is None else _env_vars

        cache_key = tcc.get_cache_key(local_src, backend, backend_options, env_vars)
        cache_hash = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
        cache_mgr = get_cache_manager(cache_hash)

        meta = f"{local_src.name[:150]}.json"
        disk_hit = meta in (cache_mgr.get_group(meta) or {})

        t0 = time.time()
        out = _orig_compile(src, target=target, options=options, _env_vars=_env_vars)
        dt = time.time() - t0

        print(
            f"[TRITON COMPILE] {local_src.name} "
            f"disk_cache={'HIT' if disk_hit else 'MISS'} "
            f"elapsed={dt:.3f}s hash={cache_hash[:12]}"
        )
        return out

    JITFunction.cache_hook = cache_hook
    tcc.compile = wrapped_compile
    tc.compile = wrapped_compile