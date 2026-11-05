# Evaluation extension

Source: Spec 034 Q11–Q14; user authorized design followed by local implementation.

`model.eval_tasks` selects existing task heads for validation/test and overrides only their metrics and metric_label. Absent this key, task metrics retain existing behavior. Every metric may declare a Boolean filter evaluated on the same input namespace as its label. The graph passes label and filter tensors together to the accumulator. Counts, sums, and disk-backed AUC rows are isolated by task and metric. No effective rows produces null/empty; one-class AUC produces null/single_class. Loss weights never imply metric filtering. Pipeline must instantiate Metrics with params.eval_tasks; training losses remain based on params.tasks.
