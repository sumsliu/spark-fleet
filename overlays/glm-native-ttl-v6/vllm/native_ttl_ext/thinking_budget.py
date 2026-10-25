# SparkFleet reference overlay: modified deployment/latency behavior; see overlays/README.md and NOTICE.
# SPDX-License-Identifier: Apache-2.0
"""Local GLM policy for numeric enabled-thinking budgets.

Mutates only invalid numeric budgets. Never raises max_tokens, changes effort,
removes omitted signatures, or traverses tool arguments/history.
"""
def normalize_thinking_budget(payload):
    if not isinstance(payload, dict):
        return False
    thinking = payload.get('thinking')
    if not isinstance(thinking, dict) or thinking.get('type') != 'enabled':
        return False
    budget = thinking.get('budget_tokens')
    if type(budget) is not int:
        return False  # malformed values remain validation errors
    maximum = payload.get('max_tokens')
    if maximum is not None and (type(maximum) is not int or maximum <= 0):
        return False
    output = payload.get('output_config')
    task = output.get('task_budget') if isinstance(output, dict) else None
    if isinstance(task, dict):
        remaining = task.get('remaining', task.get('total'))
        if type(remaining) is int and remaining > 0 and maximum is not None:
            maximum = min(maximum, remaining)
    if maximum is not None and maximum <= 1024:
        # No legal enabled interval exists. GLM supports adaptive reasoning at
        # small output caps; keep that cap and the caller's display/effort.
        thinking['type'] = 'adaptive'
        del thinking['budget_tokens']
        return True
    repaired = max(1024, budget)
    if maximum is not None:
        repaired = min(repaired, maximum - 1)
    if repaired == budget:
        return False
    thinking['budget_tokens'] = repaired
    return True
