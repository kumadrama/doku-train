# design

Use early_stopping_patience: null to disable stopping. Add selection: {mode: max, task: immersive_click, metric: auc}. Preserve existing weighted loss selection. Validate task and metric references. Undefined metrics do not select a checkpoint or substitute another metric; preserve unavailable status. Resume retains best score/state. Test evaluates the selected checkpoint. Rebuild the experiment image from this source revision.
