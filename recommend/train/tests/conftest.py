"""Test process bootstrap for native CPU runtimes."""

# XGBoost must initialize OpenMP before PyTorch on macOS. Test collection imports both.
import xgboost  # noqa: F401
