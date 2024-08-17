from . import llm_lambda

# This will only register if it hasn't been registered by the entry point
def register():
    from llm.plugins import pm
    if 'llm_lambda' not in pm.list_name_plugin():
        pm.register(llm_lambda)

# Don't call register() here, let it be called by the entry point or manually if needed
