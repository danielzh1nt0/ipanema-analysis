"""Every top-level function (nested functions included with their parent) must only use names that are defined:
catches a missing import before it can crash a paid run (it caught 'np' in run_full on 21 Sep)."""
import ast, builtins, glob, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _undefined(path):
    tree = ast.parse(open(path).read()); bad = {}
    # module level = top-level statements only (an import inside another function must not count here)
    top = [n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    mod = {x.id for n in top for x in ast.walk(n) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store)}
    mod |= {a.asname or a.name.split(".")[0] for n in top for x in ast.walk(n) if isinstance(x, (ast.Import, ast.ImportFrom)) for a in x.names}
    mod |= {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    for fn in [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        d = {a.arg for n in ast.walk(fn) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) for a in n.args.args + n.args.kwonlyargs + ([n.args.vararg] if n.args.vararg else []) + ([n.args.kwarg] if n.args.kwarg else [])}
        d |= {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        d |= {a.asname or a.name.split(".")[0] for n in ast.walk(fn) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
        d |= {n.name for n in ast.walk(fn) if isinstance(n, ast.ExceptHandler) and n.name}
        d |= {n.name for n in ast.walk(fn) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
        used = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        miss = sorted(used - d - mod - set(dir(builtins)))
        if miss: bad[fn.name] = miss
    return bad

def test_no_undefined_names():
    files = [f"{ROOT}/modal_app.py", f"{ROOT}/scripts_watch.py"] + sorted(glob.glob(f"{ROOT}/ipanema/*.py"))
    problems = {os.path.basename(f): b for f in files if (b := _undefined(f))}
    assert not problems, problems
