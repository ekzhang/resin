# CUTLASS 4.8.1 generates these wrappers dynamically. Resin uses scalar operands,
# for which the wrappers have the same signatures as the underlying operations.
from cutlass._mlir_helpers.math import exp2 as exp2
from cutlass._mlir_helpers.math import log as log
