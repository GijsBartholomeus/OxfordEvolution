#!/usr/bin/env python3
import ctypes

lib = ctypes.CDLL("/opt/rocm/lib/librocm_smi64.so")

# Init
lib.rsmi_init(0)

temp = ctypes.c_uint32()
usage = ctypes.c_uint32()

# Device index 0
RSMI_TEMP_TYPE_EDGE = 0

lib.rsmi_dev_temp_metric_get(0, RSMI_TEMP_TYPE_EDGE, ctypes.byref(temp))
lib.rsmi_dev_busy_percent_get(0, ctypes.byref(usage))

print(f"GPU Temp: {temp.value} °C")
print(f"GPU Load: {usage.value} %")

lib.rsmi_shut_down()
