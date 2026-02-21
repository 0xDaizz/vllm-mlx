# PR: Fix silent data corruption in JACCL by checking RDMA work completion status

## PR Title

```
[JACCL] Fix silent data corruption from unchecked RDMA work completion status
```

## PR Body

### Summary

The JACCL RDMA backend polls `ibv_wc` work completions to track in-flight RDMA operations but never checks `wc[i].status`. When an RDMA operation fails (e.g., due to memory pressure returning `ENOMEM`), the failure is silently ignored and the receive buffer—still containing stale or uninitialized data—is used as the result of the collective operation. This can cause **silent data corruption** in JACCL-based distributed workloads when RDMA operations fail. This PR adds `wc[i].status` validation to all 9 poll loops across `ring.cpp` and `mesh.cpp`, converting silent corruption into immediate, descriptive errors.

### Problem

All 9 completion-polling loops in the JACCL backend (5 in `ring.cpp`, 4 in `mesh.cpp`) follow the same pattern:

```cpp
ibv_wc wc[WC_NUM];
int n = poll(connections_, WC_NUM, wc);
for (int i = 0; i < n; i++) {
    // Only wr_id is examined to track in-flight count and buffer indices
    int work_type = wc[i].wr_id >> 16;
    int buff = (wc[i].wr_id >> 8) & 0xff;
    // ...
    in_flight--;
    // Proceed to use the receive buffer as if the operation succeeded
}
```

The `wc[i].status` field is never checked. Per the IBV specification, a polled completion with `status != IBV_WC_SUCCESS` indicates that the corresponding RDMA operation **failed**. The data in the associated receive buffer is undefined.

Additionally, the return value of `ibv_poll_cq` (wrapped by `poll()`) is not checked for negative values, which indicate a polling error.

**Affected functions:**
- `RingGroup::all_gather` (1 loop)
- `RingGroup::send` (1 loop)
- `RingGroup::recv` (1 loop)
- `RingGroup::all_reduce_impl` (2 loops: reduce-scatter phase + all-gather phase)
- `MeshGroup::all_gather` (1 loop)
- `MeshGroup::send` (1 loop)
- `MeshGroup::recv` (1 loop)
- `MeshGroup::all_reduce` (1 loop)

### Impact

- May affect JACCL-based distributed workloads where RDMA operations can fail
- Particularly relevant with large models that put memory pressure on RDMA buffer allocation
- Observed in practice: a 612GB MoE model (306GB per rank across 2 nodes via Thunderbolt 5 RDMA) produced corrupted output after approximately 22 tokens of autoregressive generation, as RDMA buffer allocation began to fail under memory pressure
- Smaller models that fit comfortably in RDMA buffer limits are unaffected because their RDMA operations always succeed — the bug is latent but real

### Fix

1. **Centralized helpers in `utils.h`** — instead of duplicating error-handling code in each file:
   - `wc_status_name()` — maps all `ibv_wc_status` enum values to human-readable strings (full coverage: SUCCESS, LOC_LEN_ERR, LOC_QP_OP_ERR, LOC_EEC_OP_ERR, LOC_PROT_ERR, WR_FLUSH_ERR, MW_BIND_ERR, BAD_RESP_ERR, LOC_ACCESS_ERR, REM_INV_REQ_ERR, REM_ACCESS_ERR, REM_OP_ERR, RETRY_EXC_ERR, RNR_RETRY_EXC_ERR, REM_ABORT_ERR, GENERAL_ERR, UNKNOWN)
   - `check_wc_status(const ibv_wc&)` — single-line status check that throws a descriptive `std::runtime_error`

2. **`poll()` helpers in `utils.h` fixed** to check `ibv_poll_cq` negative returns internally:
   - `Connection::poll()` — now throws on negative return
   - Free `poll(vector, ...)` — now throws on negative return from any CQ
   - This prevents negative returns from being masked when multiple CQ results are summed

3. **Error messages include `qp_num`** — valid on error per IBV spec, useful for identifying which connection failed in multi-peer topologies.

4. **Call sites simplified** — each of the 9 poll loops in `ring.cpp` and `mesh.cpp` now uses just `check_wc_status(wc[i])` instead of inline error handling.

Example error message:
```
[jaccl] RDMA work completion error: status=5 (WR_FLUSH_ERR) qp_num=42 vendor_err=0 wr_id=0x10001
```

### Affected Files

- `mlx/distributed/jaccl/utils.h` (new: centralized helpers + poll fixes)
- `mlx/distributed/jaccl/ring.cpp` (simplified: uses centralized helpers)
- `mlx/distributed/jaccl/mesh.cpp` (simplified: uses centralized helpers)

### Testing

- Tested with a 612GB MoE model using tensor parallelism across 2 nodes connected via Thunderbolt 5 RDMA (Apple Silicon, macOS)
- Verified with 256, 512, and 8490+512 token generation tests — all produce coherent output with zero RDMA errors
- Previously, the same configuration produced corrupted output after ~22 tokens
- Smaller models (< 10GB per rank) also tested to confirm no regression when RDMA operations succeed normally
- No performance impact: the status check is a single integer comparison on the hot path

### Diff

#### `mlx/distributed/jaccl/utils.h`

New includes and `Connection::poll()` fix:

```diff
 #include <infiniband/verbs.h>
+#include <sstream>
+#include <stdexcept>
+#include <string>

 #include <unordered_map>
 #include <vector>
```

```diff
   int poll(int num_completions, ibv_wc* work_completions) {
-    return ibv_poll_cq(completion_queue, num_completions, work_completions);
+    int n = ibv_poll_cq(completion_queue, num_completions, work_completions);
+    if (n < 0) {
+      throw std::runtime_error(
+          "[jaccl] ibv_poll_cq failed with error " + std::to_string(n));
+    }
+    return n;
   }
```

Free `poll()` fix:

```diff
     int n = ibv_poll_cq(
         c.completion_queue,
         num_completions - completions,
         work_completions + completions);
+    if (n < 0) {
+      throw std::runtime_error(
+          "[jaccl] ibv_poll_cq failed with error " + std::to_string(n));
+    }

     completions += n;
```

New centralized helpers (added inside `jaccl` namespace, after `poll` functions):

```diff
+inline const char* wc_status_name(int status) {
+  switch (status) {
+    case IBV_WC_SUCCESS: return "SUCCESS";
+    case IBV_WC_LOC_LEN_ERR: return "LOC_LEN_ERR";
+    case IBV_WC_LOC_QP_OP_ERR: return "LOC_QP_OP_ERR";
+    case IBV_WC_LOC_EEC_OP_ERR: return "LOC_EEC_OP_ERR";
+    case IBV_WC_LOC_PROT_ERR: return "LOC_PROT_ERR";
+    case IBV_WC_WR_FLUSH_ERR: return "WR_FLUSH_ERR";
+    case IBV_WC_MW_BIND_ERR: return "MW_BIND_ERR";
+    case IBV_WC_BAD_RESP_ERR: return "BAD_RESP_ERR";
+    case IBV_WC_LOC_ACCESS_ERR: return "LOC_ACCESS_ERR";
+    case IBV_WC_REM_INV_REQ_ERR: return "REM_INV_REQ_ERR";
+    case IBV_WC_REM_ACCESS_ERR: return "REM_ACCESS_ERR";
+    case IBV_WC_REM_OP_ERR: return "REM_OP_ERR";
+    case IBV_WC_RETRY_EXC_ERR: return "RETRY_EXC_ERR";
+    case IBV_WC_RNR_RETRY_EXC_ERR: return "RNR_RETRY_EXC_ERR";
+    case IBV_WC_REM_ABORT_ERR: return "REM_ABORT_ERR";
+    case IBV_WC_GENERAL_ERR: return "GENERAL_ERR";
+    default: return "UNKNOWN";
+  }
+}
+
+inline void check_wc_status(const ibv_wc& wc) {
+  if (wc.status != IBV_WC_SUCCESS) {
+    std::ostringstream msg;
+    msg << "[jaccl] RDMA work completion error: status=" << wc.status
+        << " (" << wc_status_name(wc.status) << ")"
+        << " qp_num=" << std::dec << wc.qp_num
+        << " vendor_err=" << wc.vendor_err
+        << " wr_id=0x" << std::hex << wc.wr_id;
+    throw std::runtime_error(msg.str());
+  }
+}
```

#### `mlx/distributed/jaccl/ring.cpp`

Removed duplicated local helpers and `<sstream>` include (now centralized in `utils.h`):

```diff
 #include "mlx/distributed/jaccl/ring.h"
-#include <sstream>
 #include "mlx/backend/cpu/encoder.h"
 #include "mlx/distributed/reduction_ops.h"
 #include "mlx/dtype_utils.h"
-
-
-namespace {
-const char* wc_status_name(int status) {
-  switch (status) {
-    case IBV_WC_SUCCESS: return "SUCCESS";
-    case IBV_WC_LOC_LEN_ERR: return "LOC_LEN_ERR";
-    case IBV_WC_LOC_QP_OP_ERR: return "LOC_QP_OP_ERR";
-    case IBV_WC_LOC_PROT_ERR: return "LOC_PROT_ERR";
-    case IBV_WC_WR_FLUSH_ERR: return "WR_FLUSH_ERR";
-    case IBV_WC_REM_ACCESS_ERR: return "REM_ACCESS_ERR";
-    case IBV_WC_GENERAL_ERR: return "GENERAL_ERR";
-    default: return "UNKNOWN";
-  }
-}
-} // namespace

  namespace mlx::core::distributed::jaccl {
```

Each of the 5 poll loops simplified. Example from `all_gather`:

```diff
       while (in_flight > 0) {
         ibv_wc wc[WC_NUM];
         int n = poll(left_, right_, WC_NUM, wc);
-        if (n < 0) {
-          throw std::runtime_error(
-              "[jaccl] ibv_poll_cq failed with error " + std::to_string(n));
-        }
         for (int i = 0; i < n; i++) {
           int work_type = wc[i].wr_id >> 16;
           int buff = (wc[i].wr_id >> 8) & 0xff;
@@ ... @@
           int lr = wire / MAX_CONNS;
           int lw = wire % MAX_CONNS;

-          if (wc[i].status != IBV_WC_SUCCESS) {
-            std::ostringstream msg;
-            msg << "[jaccl] RDMA work completion error: status=" << wc[i].status
-                << " (" << wc_status_name(wc[i].status) << ")"
-                << " vendor_err=" << wc[i].vendor_err
-                << " wr_id=0x" << std::hex << wc[i].wr_id;
-            throw std::runtime_error(msg.str());
-          }
+          check_wc_status(wc[i]);
           in_flight--;
```

The same pattern is applied to `send`, `recv`, and both phases of `all_reduce_impl`.

#### `mlx/distributed/jaccl/mesh.cpp`

Same simplification as `ring.cpp` — removed duplicated local helpers and replaced inline error handling with `check_wc_status(wc[i])` in all 4 poll loops.

```diff
 #include "mlx/distributed/jaccl/mesh.h"
-#include <sstream>
 #include "mlx/backend/cpu/encoder.h"
 #include "mlx/distributed/reduction_ops.h"
 #include "mlx/dtype_utils.h"
-
-
-namespace {
-const char* wc_status_name(int status) {
-  switch (status) {
-    case IBV_WC_SUCCESS: return "SUCCESS";
-    case IBV_WC_LOC_LEN_ERR: return "LOC_LEN_ERR";
-    case IBV_WC_LOC_QP_OP_ERR: return "LOC_QP_OP_ERR";
-    case IBV_WC_LOC_PROT_ERR: return "LOC_PROT_ERR";
-    case IBV_WC_WR_FLUSH_ERR: return "WR_FLUSH_ERR";
-    case IBV_WC_REM_ACCESS_ERR: return "REM_ACCESS_ERR";
-    case IBV_WC_GENERAL_ERR: return "GENERAL_ERR";
-    default: return "UNKNOWN";
-  }
-}
-} // namespace

  constexpr int MAX_PEERS = 8;
```

Example from `all_gather`:

```diff
     while (in_flight > 0) {
       ibv_wc wc[WC_NUM];
       int n = poll(connections_, WC_NUM, wc);
-      if (n < 0) {
-        throw std::runtime_error(
-            "[jaccl] ibv_poll_cq failed with error " + std::to_string(n));
-      }
       for (int i = 0; i < n; i++) {
         int work_type = wc[i].wr_id >> 16;
         int buff = (wc[i].wr_id >> 8) & 0xff;
         int rank = wc[i].wr_id & 0xff;

-        if (wc[i].status != IBV_WC_SUCCESS) {
-          std::ostringstream msg;
-          msg << "[jaccl] RDMA work completion error: status=" << wc[i].status
-              << " (" << wc_status_name(wc[i].status) << ")"
-              << " vendor_err=" << wc[i].vendor_err
-              << " wr_id=0x" << std::hex << wc[i].wr_id;
-          throw std::runtime_error(msg.str());
-        }
+        check_wc_status(wc[i]);
         in_flight--;
```

The same pattern is applied to `send`, `recv`, and `all_reduce`.

### Complete unified diff

<details>
<summary>Click to expand full diff</summary>

```diff
diff --git a/mlx/distributed/jaccl/utils.h b/mlx/distributed/jaccl/utils.h
index abc1234..def5678 100644
--- a/mlx/distributed/jaccl/utils.h
+++ b/mlx/distributed/jaccl/utils.h
@@ ... @@
 #include <infiniband/verbs.h>
+#include <sstream>
+#include <stdexcept>
+#include <string>

 #include <unordered_map>
 #include <vector>
@@ ... @@ struct Connection {
   int poll(int num_completions, ibv_wc* work_completions) {
-    return ibv_poll_cq(completion_queue, num_completions, work_completions);
+    int n = ibv_poll_cq(completion_queue, num_completions, work_completions);
+    if (n < 0) {
+      throw std::runtime_error(
+          "[jaccl] ibv_poll_cq failed with error " + std::to_string(n));
+    }
+    return n;
   }
@@ ... @@ inline int poll(
     int n = ibv_poll_cq(
         c.completion_queue,
         num_completions - completions,
         work_completions + completions);
+    if (n < 0) {
+      throw std::runtime_error(
+          "[jaccl] ibv_poll_cq failed with error " + std::to_string(n));
+    }

     completions += n;
@@ ... @@
+inline const char* wc_status_name(int status) {
+  switch (status) {
+    case IBV_WC_SUCCESS: return "SUCCESS";
+    case IBV_WC_LOC_LEN_ERR: return "LOC_LEN_ERR";
+    case IBV_WC_LOC_QP_OP_ERR: return "LOC_QP_OP_ERR";
+    case IBV_WC_LOC_EEC_OP_ERR: return "LOC_EEC_OP_ERR";
+    case IBV_WC_LOC_PROT_ERR: return "LOC_PROT_ERR";
+    case IBV_WC_WR_FLUSH_ERR: return "WR_FLUSH_ERR";
+    case IBV_WC_MW_BIND_ERR: return "MW_BIND_ERR";
+    case IBV_WC_BAD_RESP_ERR: return "BAD_RESP_ERR";
+    case IBV_WC_LOC_ACCESS_ERR: return "LOC_ACCESS_ERR";
+    case IBV_WC_REM_INV_REQ_ERR: return "REM_INV_REQ_ERR";
+    case IBV_WC_REM_ACCESS_ERR: return "REM_ACCESS_ERR";
+    case IBV_WC_REM_OP_ERR: return "REM_OP_ERR";
+    case IBV_WC_RETRY_EXC_ERR: return "RETRY_EXC_ERR";
+    case IBV_WC_RNR_RETRY_EXC_ERR: return "RNR_RETRY_EXC_ERR";
+    case IBV_WC_REM_ABORT_ERR: return "REM_ABORT_ERR";
+    case IBV_WC_GENERAL_ERR: return "GENERAL_ERR";
+    default: return "UNKNOWN";
+  }
+}
+
+inline void check_wc_status(const ibv_wc& wc) {
+  if (wc.status != IBV_WC_SUCCESS) {
+    std::ostringstream msg;
+    msg << "[jaccl] RDMA work completion error: status=" << wc.status
+        << " (" << wc_status_name(wc.status) << ")"
+        << " qp_num=" << std::dec << wc.qp_num
+        << " vendor_err=" << wc.vendor_err
+        << " wr_id=0x" << std::hex << wc.wr_id;
+    throw std::runtime_error(msg.str());
+  }
+}
+
diff --git a/mlx/distributed/jaccl/ring.cpp b/mlx/distributed/jaccl/ring.cpp
index abc1234..def5678 100644
--- a/mlx/distributed/jaccl/ring.cpp
+++ b/mlx/distributed/jaccl/ring.cpp
@@ -1,21 +1,7 @@
 // Copyright © 2026 Apple Inc.

 #include "mlx/distributed/jaccl/ring.h"
-#include <sstream>
 #include "mlx/backend/cpu/encoder.h"
 #include "mlx/distributed/reduction_ops.h"
 #include "mlx/dtype_utils.h"
-
-
-namespace {
-const char* wc_status_name(int status) {
-  switch (status) {
-    case IBV_WC_SUCCESS: return "SUCCESS";
-    case IBV_WC_LOC_LEN_ERR: return "LOC_LEN_ERR";
-    case IBV_WC_LOC_QP_OP_ERR: return "LOC_QP_OP_ERR";
-    case IBV_WC_LOC_PROT_ERR: return "LOC_PROT_ERR";
-    case IBV_WC_WR_FLUSH_ERR: return "WR_FLUSH_ERR";
-    case IBV_WC_REM_ACCESS_ERR: return "REM_ACCESS_ERR";
-    case IBV_WC_GENERAL_ERR: return "GENERAL_ERR";
-    default: return "UNKNOWN";
-  }
-}
-} // namespace

 namespace mlx::core::distributed::jaccl {
@@ ... @@ void RingGroup::all_gather(const array& input, array& output, Stream stream) {
       while (in_flight > 0) {
         ibv_wc wc[WC_NUM];
         int n = poll(left_, right_, WC_NUM, wc);
-        if (n < 0) {
-          throw std::runtime_error(
-              "[jaccl] ibv_poll_cq failed with error " + std::to_string(n));
-        }
         for (int i = 0; i < n; i++) {
           int work_type = wc[i].wr_id >> 16;
           int buff = (wc[i].wr_id >> 8) & 0xff;
           int wire = wc[i].wr_id & 0xff;
           int lr = wire / MAX_CONNS;
           int lw = wire % MAX_CONNS;

-          if (wc[i].status != IBV_WC_SUCCESS) {
-            std::ostringstream msg;
-            msg << "[jaccl] RDMA work completion error: status=" << wc[i].status
-                << " (" << wc_status_name(wc[i].status) << ")"
-                << " vendor_err=" << wc[i].vendor_err
-                << " wr_id=0x" << std::hex << wc[i].wr_id;
-            throw std::runtime_error(msg.str());
-          }
+          check_wc_status(wc[i]);
           in_flight--;

           if (work_type == SEND_WR && send_count[wire] < n_steps) {
@@ ... @@ void RingGroup::send(const array& input, int dst, Stream stream) {
       ibv_wc wc[WC_NUM];
       int n = poll(conns, WC_NUM, wc);
-      if (n < 0) {
-        throw std::runtime_error(
-            "[jaccl] ibv_poll_cq failed with error " + std::to_string(n));
-      }
       for (int i = 0; i < n; i++) {
         int buff = (wc[i].wr_id >> 8) & 0xff;
         int wire = wc[i].wr_id & 0xff;
         int lw = wire % MAX_CONNS;

-        if (wc[i].status != IBV_WC_SUCCESS) {
-          std::ostringstream msg;
-          msg << "[jaccl] RDMA work completion error: status=" << wc[i].status
-              << " (" << wc_status_name(wc[i].status) << ")"
-              << " vendor_err=" << wc[i].vendor_err
-              << " wr_id=0x" << std::hex << wc[i].wr_id;
-          throw std::runtime_error(msg.str());
-        }
+        check_wc_status(wc[i]);
         in_flight--;

         if (read_offset[lw] < limits[lw]) {
@@ ... @@ void RingGroup::recv(array& out, int src, Stream stream) {
       ibv_wc wc[WC_NUM];
       int n = poll(conns, WC_NUM, wc);
-      if (n < 0) {
-        throw std::runtime_error(
-            "[jaccl] ibv_poll_cq failed with error " + std::to_string(n));
-      }
       for (int i = 0; i < n; i++) {
         int buff = (wc[i].wr_id >> 8) & 0xff;
         int wire = wc[i].wr_id & 0xff;
         int lw = wire % MAX_CONNS;

-        if (wc[i].status != IBV_WC_SUCCESS) {
-          std::ostringstream msg;
-          msg << "[jaccl] RDMA work completion error: status=" << wc[i].status
-              << " (" << wc_status_name(wc[i].status) << ")"
-              << " vendor_err=" << wc[i].vendor_err
-              << " wr_id=0x" << std::hex << wc[i].wr_id;
-          throw std::runtime_error(msg.str());
-        }
+        check_wc_status(wc[i]);
         in_flight--;

         std::copy(
@@ ... @@ void RingGroup::all_reduce_impl( // reduce-scatter phase
     while (in_flight > 0) {
       ibv_wc wc[WC_NUM];
       int n = poll(left_, right_, WC_NUM, wc);
-      if (n < 0) {
-        throw std::runtime_error(
-            "[jaccl] ibv_poll_cq failed with error " + std::to_string(n));
-      }
       for (int i = 0; i < n; i++) {
         int work_type = wc[i].wr_id >> 16;
         int buff = (wc[i].wr_id >> 8) & 0xff;
         int wire = wc[i].wr_id & 0xff;
         int lr = wire / MAX_CONNS;
         int lw = wire % MAX_CONNS;

-        if (wc[i].status != IBV_WC_SUCCESS) {
-          std::ostringstream msg;
-          msg << "[jaccl] RDMA work completion error: status=" << wc[i].status
-              << " (" << wc_status_name(wc[i].status) << ")"
-              << " vendor_err=" << wc[i].vendor_err
-              << " wr_id=0x" << std::hex << wc[i].wr_id;
-          throw std::runtime_error(msg.str());
-        }
+        check_wc_status(wc[i]);
         in_flight--;

         if (work_type == SEND_WR && send_count[wire] < n_steps) {
@@ ... @@ void RingGroup::all_reduce_impl( // all-gather phase
     while (in_flight > 0) {
       ibv_wc wc[WC_NUM];
       int n = poll(left_, right_, WC_NUM, wc);
-      if (n < 0) {
-        throw std::runtime_error(
-            "[jaccl] ibv_poll_cq failed with error " + std::to_string(n));
-      }
       for (int i = 0; i < n; i++) {
         int work_type = wc[i].wr_id >> 16;
         int buff = (wc[i].wr_id >> 8) & 0xff;
         int wire = wc[i].wr_id & 0xff;
         int lr = wire / MAX_CONNS;
         int lw = wire % MAX_CONNS;

-        if (wc[i].status != IBV_WC_SUCCESS) {
-          std::ostringstream msg;
-          msg << "[jaccl] RDMA work completion error: status=" << wc[i].status
-              << " (" << wc_status_name(wc[i].status) << ")"
-              << " vendor_err=" << wc[i].vendor_err
-              << " wr_id=0x" << std::hex << wc[i].wr_id;
-          throw std::runtime_error(msg.str());
-        }
+        check_wc_status(wc[i]);
         in_flight--;

         if (work_type == SEND_WR && send_count[wire] < n_steps) {
diff --git a/mlx/distributed/jaccl/mesh.cpp b/mlx/distributed/jaccl/mesh.cpp
index abc1234..def5678 100644
--- a/mlx/distributed/jaccl/mesh.cpp
+++ b/mlx/distributed/jaccl/mesh.cpp
@@ -1,21 +1,7 @@
 // Copyright © 2026 Apple Inc.

 #include "mlx/distributed/jaccl/mesh.h"
-#include <sstream>
 #include "mlx/backend/cpu/encoder.h"
 #include "mlx/distributed/reduction_ops.h"
 #include "mlx/dtype_utils.h"
-
-
-namespace {
-const char* wc_status_name(int status) {
-  switch (status) {
-    case IBV_WC_SUCCESS: return "SUCCESS";
-    case IBV_WC_LOC_LEN_ERR: return "LOC_LEN_ERR";
-    case IBV_WC_LOC_QP_OP_ERR: return "LOC_QP_OP_ERR";
-    case IBV_WC_LOC_PROT_ERR: return "LOC_PROT_ERR";
-    case IBV_WC_WR_FLUSH_ERR: return "WR_FLUSH_ERR";
-    case IBV_WC_REM_ACCESS_ERR: return "REM_ACCESS_ERR";
-    case IBV_WC_GENERAL_ERR: return "GENERAL_ERR";
-    default: return "UNKNOWN";
-  }
-}
-} // namespace

 constexpr int MAX_PEERS = 8;
@@ ... @@ void MeshGroup::all_gather(const array& input, array& output, Stream stream) {
     while (in_flight > 0) {
       ibv_wc wc[WC_NUM];
       int n = poll(connections_, WC_NUM, wc);
-      if (n < 0) {
-        throw std::runtime_error(
-            "[jaccl] ibv_poll_cq failed with error " + std::to_string(n));
-      }
       for (int i = 0; i < n; i++) {
         int work_type = wc[i].wr_id >> 16;
         int buff = (wc[i].wr_id >> 8) & 0xff;
         int rank = wc[i].wr_id & 0xff;

-        if (wc[i].status != IBV_WC_SUCCESS) {
-          std::ostringstream msg;
-          msg << "[jaccl] RDMA work completion error: status=" << wc[i].status
-              << " (" << wc_status_name(wc[i].status) << ")"
-              << " vendor_err=" << wc[i].vendor_err
-              << " wr_id=0x" << std::hex << wc[i].wr_id;
-          throw std::runtime_error(msg.str());
-        }
+        check_wc_status(wc[i]);
         in_flight--;

         // Send completed. If all sends completed then send the next chunk.
@@ ... @@ void MeshGroup::send(const array& input, int dst, Stream stream) {
       ibv_wc wc[WC_NUM];
       int n = connections_[dst].poll(WC_NUM, wc);
-      if (n < 0) {
-        throw std::runtime_error(
-            "[jaccl] ibv_poll_cq failed with error " + std::to_string(n));
-      }
       for (int i = 0; i < n; i++) {
         int buff = (wc[i].wr_id >> 8) & 0xff;
         int rank = wc[i].wr_id & 0xff;

-        if (wc[i].status != IBV_WC_SUCCESS) {
-          std::ostringstream msg;
-          msg << "[jaccl] RDMA work completion error: status=" << wc[i].status
-              << " (" << wc_status_name(wc[i].status) << ")"
-              << " vendor_err=" << wc[i].vendor_err
-              << " wr_id=0x" << std::hex << wc[i].wr_id;
-          throw std::runtime_error(msg.str());
-        }
+        check_wc_status(wc[i]);
         in_flight--;

         if (read_offset < n_bytes) {
@@ ... @@ void MeshGroup::recv(array& out, int src, Stream stream) {
       ibv_wc wc[WC_NUM];
       int n = connections_[src].poll(WC_NUM, wc);
-      if (n < 0) {
-        throw std::runtime_error(
-            "[jaccl] ibv_poll_cq failed with error " + std::to_string(n));
-      }
       for (int i = 0; i < n; i++) {
         int buff = (wc[i].wr_id >> 8) & 0xff;
         int rank = wc[i].wr_id & 0xff;

-        if (wc[i].status != IBV_WC_SUCCESS) {
-          std::ostringstream msg;
-          msg << "[jaccl] RDMA work completion error: status=" << wc[i].status
-              << " (" << wc_status_name(wc[i].status) << ")"
-              << " vendor_err=" << wc[i].vendor_err
-              << " wr_id=0x" << std::hex << wc[i].wr_id;
-          throw std::runtime_error(msg.str());
-        }
+        check_wc_status(wc[i]);
         in_flight--;

         std::copy(
@@ ... @@ void MeshGroup::all_reduce(
       ibv_wc wc[WC_NUM];
       int n = poll(connections_, WC_NUM, wc);
-      if (n < 0) {
-        throw std::runtime_error(
-            "[jaccl] ibv_poll_cq failed with error " + std::to_string(n));
-      }
       for (int i = 0; i < n; i++) {
         int work_type = wc[i].wr_id >> 16;
         int buff = (wc[i].wr_id >> 8) & 0xff;
         int rank = wc[i].wr_id & 0xff;

-        if (wc[i].status != IBV_WC_SUCCESS) {
-          std::ostringstream msg;
-          msg << "[jaccl] RDMA work completion error: status=" << wc[i].status
-              << " (" << wc_status_name(wc[i].status) << ")"
-              << " vendor_err=" << wc[i].vendor_err
-              << " wr_id=0x" << std::hex << wc[i].wr_id;
-          throw std::runtime_error(msg.str());
-        }
+        check_wc_status(wc[i]);
         in_flight--;

         if (work_type == SEND_WR && read_offset < total) {
```

</details>
