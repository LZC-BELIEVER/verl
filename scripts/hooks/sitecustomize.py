"""Read-only experiment instrumentation; never part of the upstream patch."""
import os
if os.environ.get('QAT_AUDIT_DIR'):
    import importlib.abc
    import importlib.machinery
    import sys
    class AuditFinder(importlib.abc.MetaPathFinder):
        targets = {'verl.workers.engine.megatron.transformer_impl', 'verl.utils.vllm.vllm_quant_utils', 'verl.utils.debug.metrics'}
        def find_spec(self, fullname, path=None, target=None):
            if fullname not in self.targets:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec and spec.loader:
                execute = spec.loader.exec_module
                def audited_exec(module):
                    execute(module)
                    from qat_audit import install
                    install(module)
                spec.loader.exec_module = audited_exec
            return spec
    sys.meta_path.insert(0, AuditFinder())
