// tools/v41/engine/ds41_dense.cpp - see ds41_dense.hpp.  Line references are to DeepSeek's inference/model.py.
#include "ds41_dense.hpp"

#include "ds41_engram.hpp"
#include "ds41_ops.hpp"

#include "strata/kernels/dense_f16_gemm.hpp"   // S16: the dense GEMM-shaped parts

#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#ifdef STRATA_ENABLE_CUDA
#include <cuda_runtime.h>
#endif
#ifdef DS41_HAVE_CUDA
#include "ggml-cuda.h"
#endif   // ggml_cgraph::uid (graph reuse)

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <type_traits>
#include <sys/stat.h>

#ifdef _WIN32
#include <string.h>
#define strcasecmp _stricmp
#else
#include <strings.h>
#endif

using namespace strata;
using namespace strata::v41;

namespace {

constexpr float NEG_INF = -INFINITY;

int64_t next_pow2(int64_t v) {
    int64_t p = 1;
    while (p < v) p <<= 1;
    return p;
}

bool env_on(const char * k, bool dflt) {
    const char * e = std::getenv(k);
    return e ? std::atoi(e) != 0 : dflt;
}

// ===================================================================================================================
// DS41_DPROF: per-call-site accounting of the dense path's backend traffic.  DIAGNOSTIC; inert when the variable is
// unset.  DS41_DPROF=1 makes every ggml_backend_* call in this file count itself and its wall time by call site
// (file:line), records each graph's build / alloc / compute split per (kind, layer) together with its node count, and
// prints the accumulated table per segment - Ds41Dense::dprof_checkpoint(tag) closes a segment ("prompt") and the
// rest is printed at exit.  Pure measurement: nothing here changes an op, a buffer or a number; with DS41_DPROF unset
// every entry point is one branch on a constant and no clock is read.
//
// Reading the numbers: ggml_backend_graph_compute is async+synchronize in this fork (src/ggml-backend.cpp:595), so a
// "compute" row is the whole submit-and-wait interval for the graph - it already contains every kernel and the final
// stream wait.  There is no graph_compute_async call site in this file (verified: the only graph submission in the
// dense path is Impl::compute), so encode-only time is not separable from the host side; see the report.
namespace dprof {

enum Cat {
    C_SYNC = 0, C_GET, C_GET_ASYNC, C_SET, C_COMPUTE, C_BUILD, C_ALLOC,           // leaf: exactly one backend call
    C_BEGIN, C_ATTN, C_FINISH, C_FINPRED, C_PREDICT, C_HEAD, C_SPAN_ATTN, C_SPAN_KV,   // top-level spans (dense API)
    C_UPLOAD, C_ENGRAM_ROWS, C_ENGRAM_WAIT,                                        // child spans
    C_NCAT
};

enum Kind { K_INIT = 0, K_ATTN, K_KV, K_FINISH, K_FINPRED, K_PREDICT, K_HEAD, K_PROBE, K_NKIND };

static const char * const kCatName[C_NCAT] = {
    "sync", "get", "get_async", "set", "compute", "build", "alloc",
    "span:begin", "span:attn_router", "span:finish", "span:finish_predict", "span:predict", "span:logits",
    "span:span_attn", "span:span_kv",
    "span:upload_inputs", "span:engram_rows", "span:engram_wait"
};
static const char * const kKindName[K_NKIND] = { "init", "attn", "kv", "finish", "finish_predict", "predict", "head", "probe" };
static const bool kTop[C_NCAT] = { false, false, false, false, false, false, false,
                                   true, true, true, true, true, true, true, true,
                                   false, false, false };

struct Leaf { long long n = 0; long long bytes = 0; double ms = 0; double max_us = 0; };
struct SpanAgg {
    long long n = 0;
    double wall_ms = 0, leaf_ms = 0;
    std::map<int, double> child_ms;   // wall ms per child span cat
};
struct GraphRow {
    int nodes = 0;
    long long builds = 0, computes = 0;
    double build_ms = 0, alloc_ms = 0, compute_ms = 0, compute_max_ms = 0;
    size_t buf_max = 0;      // the gallocr buffer's size (bytes) when this graph ran: what a cold run first-touches
};

struct State {
    long long n_in[C_NCAT] = {};                       // calls made INSIDE a span (the dense path itself)
    long long bytes_in[C_NCAT] = {};
    std::map<std::pair<int, int>, Leaf> leaf;          // (cat, line)
    std::map<std::pair<int, int>, GraphRow> graph;     // (kind, il); kind < 0 = a graph no build site registered
    std::map<const ggml_cgraph *, std::pair<int, int>> owner;
    SpanAgg span[C_NCAT];
    std::vector<SpanAgg *> open;                       // innermost last; the SpanGuard below owns the nodes
    long long begins = 0;
};

static const bool g_on = [] {
    const char * e = std::getenv("DS41_DPROF");
    return e && e[0] && e[0] != '0';
}();
static State * g_st = nullptr;
static void final_report();

static State & st() { if (!g_st) g_st = new State(); return *g_st; }   // leaked on purpose: printed from a static dtor
static inline double us_now() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// the size argument (always last) of a tensor_get / tensor_set call: gives the site table its byte totals
template <class... A> struct LastArg;
template <class A> struct LastArg<A> { static long long get(A a) { return (long long) a; } };
template <class A, class... R> struct LastArg<A, R...> {
    static long long get(A a, R... r) { (void) a; return LastArg<R...>::get(r...); }
};
template <class... A> static inline long long bytes_last(A... a) { return LastArg<A...>::get(a...); }

// one backend call: its site statistics, and the innermost open span's direct-call total
static void leaf_add(int cat, int line, double us, long long bytes) {
    if (!g_on) return;
    State & S = st();
    Leaf & r = S.leaf[std::make_pair(cat, line)];
    ++r.n; r.ms += us / 1000.0; r.bytes += bytes;
    if (us > r.max_us) r.max_us = us;
    if (!S.open.empty()) {
        S.open.back()->leaf_ms += us / 1000.0;
        ++S.n_in[cat]; S.bytes_in[cat] += bytes;
    }
}

template <class F>
static inline auto call(int cat, int line, long long bytes, F && f) -> decltype(f()) {
    if (!g_on) return f();
    const double t0 = us_now();
    if constexpr (std::is_void_v<decltype(f())>) {
        f();
        leaf_add(cat, line, us_now() - t0, bytes);
    } else {
        auto r = f();
        leaf_add(cat, line, us_now() - t0, bytes);
        return r;
    }
}

// an open span: wall time between construction and destruction, split into direct calls, child spans, and residue
struct SpanGuard {
    int cat = C_NCAT;
    double t0 = 0;
    bool active = false;
    SpanAgg node;
    SpanGuard(int c, int line) {
        (void) line;
        if (!g_on) return;
        active = true;
        cat = c;
        t0 = us_now();
        st().open.push_back(&node);
    }
    SpanGuard(const SpanGuard &) = delete;
    SpanGuard & operator=(const SpanGuard &) = delete;
    ~SpanGuard() {
        if (!active) return;
        const double us = us_now() - t0;
        State & S = st();
        auto it = std::find(S.open.begin(), S.open.end(), &node);
        if (it != S.open.end()) S.open.erase(it);
        SpanAgg & agg = S.span[cat];
        ++agg.n; agg.wall_ms += us / 1000.0; agg.leaf_ms += node.leaf_ms;
        for (const auto & kv : node.child_ms) agg.child_ms[kv.first] += kv.second;
        if (!S.open.empty()) S.open.back()->child_ms[cat] += us / 1000.0;
    }
};

// a graph build: its wall time, node count, and the (kind, layer) row that compute() charges its runs to
template <class F>
static inline ggml_cgraph * build(int kind, int il, int line, F && f) {
    if (!g_on) return f();
    const double t0 = us_now();
    ggml_cgraph * g = f();
    const double us = us_now() - t0;
    State & S = st();
    GraphRow & r = S.graph[std::make_pair(kind, il)];
    ++r.builds; r.build_ms += us / 1000.0;
    if (g) { r.nodes = ggml_graph_n_nodes(g); S.owner[g] = std::make_pair(kind, il); }
    leaf_add(C_BUILD, line, us, 0);
    return g;
}

// the device buffer a graph's nodes live in (the gallocr's planned buffer): a cold run first-touches these bytes
static size_t graph_buf_bytes(ggml_cgraph * g) {
    if (!g) return 0;
    std::set<ggml_backend_buffer_t> seen;
    size_t mx = 0;
    for (int i = 0; i < ggml_graph_n_nodes(g); ++i) {
        ggml_tensor * t = ggml_graph_node(g, i);
        if (!t || !t->buffer) continue;
        if (!seen.insert(t->buffer).second) continue;
        const size_t bs = ggml_backend_buffer_get_size(t->buffer);
        if (bs > mx) mx = bs;
    }
    return mx;
}

// a graph run: alloc + compute charged to the row its build registered
static void note_graph(ggml_cgraph * g, double alloc_us, double compute_us, size_t buf_bytes) {
    if (!g_on) return;
    State & S = st();
    auto it = S.owner.find(g);
    if (it == S.owner.end()) {   // a graph whose build site is not registered above (none expected)
        it = S.owner.emplace(g, std::make_pair(-1, -1)).first;
        S.graph[it->second].nodes = g ? ggml_graph_n_nodes(g) : 0;
    }
    GraphRow & r = S.graph[it->second];
    ++r.computes; r.alloc_ms += alloc_us / 1000.0; r.compute_ms += compute_us / 1000.0;
    if (compute_us / 1000.0 > r.compute_max_ms) r.compute_max_ms = compute_us / 1000.0;
    if (buf_bytes > r.buf_max) r.buf_max = buf_bytes;
}

// ------------------------------------------------------------------------------------------------------------------
static void print_seg(const State & S, const char * tag) {
    bool any = S.begins != 0;
    for (int c = 0; c < C_NCAT && !any; ++c) if (S.span[c].n != 0) any = true;
    if (!any) return;

    double top_wall = 0, top_leaf = 0, top_child = 0;
    for (int c = 0; c < C_NCAT; ++c) {
        if (!kTop[c]) continue;
        top_wall += S.span[c].wall_ms;
        top_leaf += S.span[c].leaf_ms;
        for (const auto & kv : S.span[c].child_ms) top_child += kv.second;
    }

    std::fprintf(stderr, "\n=== DS41_DPROF segment \"%s\": %lld begin() pass(es), dense top-level wall %.2f ms ===\n", tag,
                 S.begins, top_wall);

    std::fprintf(stderr, "[1] spans (wall ms; a top-level span is one dense API call, a child span is nested in it)\n"
                         "    %-22s %8s %12s %12s %12s %12s  %s\n", "span", "count", "wall", "mean", "direct_calls", "residue", "children");
    for (int pass = 0; pass < 2; ++pass) {
        if (pass == 1) std::fprintf(stderr, "    ---- child spans ----\n");
        for (int c = 0; c < C_NCAT; ++c) {
            if (kTop[c] != (pass == 0) || S.span[c].n == 0) continue;
            double child = 0;
            std::string kids;
            for (const auto & kv : S.span[c].child_ms) {
                char b[64];
                std::snprintf(b, sizeof(b), "%s %.1f", kCatName[kv.first], kv.second);
                if (!kids.empty()) kids += ", ";
                kids += b;
                child += kv.second;
            }
            const SpanAgg & a = S.span[c];
            std::fprintf(stderr, "    %-22s %8lld %12.2f %12.2f %12.2f %12.2f  %s\n", kCatName[c], a.n, a.wall_ms,
                         a.wall_ms / (double) a.n, a.leaf_ms, a.wall_ms - a.leaf_ms - child, kids.c_str());
        }
    }

    std::vector<std::pair<std::pair<int, int>, Leaf>> lv(S.leaf.begin(), S.leaf.end());
    std::sort(lv.begin(), lv.end(), [](const auto & a, const auto & b) { return a.second.ms > b.second.ms; });
    std::fprintf(stderr, "[2] backend call sites, sorted by total (leaf = one backend call, timer around it)\n"
                         "    %-10s %-26s %8s %12s %10s %10s %12s\n", "cat", "site", "count", "total_ms", "mean_us", "max_us", "KiB");
    for (const auto & kv : lv) {
        const Leaf & r = kv.second;
        const std::string site = std::string("ds41_dense.cpp:") + std::to_string(kv.first.second);
        std::fprintf(stderr, "    %-10s %-26s %8lld %12.3f %10.1f %10.1f %12.1f\n", kCatName[kv.first.first],
                     site.c_str(), r.n, r.ms, r.ms * 1000.0 / (double) r.n, r.max_us, r.bytes / 1024.0);
    }

    const double tok = S.begins > 0 ? (double) S.begins : 1.0;
    std::fprintf(stderr,
                 "[3] per token (inside the dense spans, divided by the %lld begin() passes): graph_compute %.2f, sync %.2f, "
                 "get %.2f, get_async %.2f, set %.2f, bytes down (get) %.1f KiB, bytes up (set) %.1f KiB\n",
                 S.begins, S.n_in[C_COMPUTE] / tok, S.n_in[C_SYNC] / tok, S.n_in[C_GET] / tok, S.n_in[C_GET_ASYNC] / tok,
                 S.n_in[C_SET] / tok, (S.bytes_in[C_GET] + S.bytes_in[C_GET_ASYNC]) / 1024.0 / tok,
                 S.bytes_in[C_SET] / 1024.0 / tok);

    std::vector<std::pair<std::pair<int, int>, GraphRow>> gv(S.graph.begin(), S.graph.end());
    std::sort(gv.begin(), gv.end(), [](const auto & a, const auto & b) {
        if (a.first.first != b.first.first) return a.first.first < b.first.first;
        return a.first.second < b.first.second;
    });
    std::fprintf(stderr, "[4] graphs: build vs alloc vs compute, per (kind, layer)\n"
                         "    %-15s %6s %7s %8s %10s %8s %10s %9s %12s %12s %12s %10s %10s\n", "kind", "layer", "nodes",
                 "builds", "build_ms", "runs", "alloc_ms", "computes", "compute_ms", "compute_max_ms", "us/compute", "us/node",
                 "buf_MiB");
    for (const auto & kv : gv) {
        const GraphRow & r = kv.second;
        std::fprintf(stderr, "    %-15s %6d %7d %8lld %10.3f %8lld %10.3f %9lld %12.3f %12.3f %12.1f %10.2f %10.1f\n",
                     kv.first.first >= 0 ? kKindName[kv.first.first] : "?", kv.first.second, r.nodes, r.builds, r.build_ms,
                     r.computes, r.alloc_ms, r.computes, r.compute_ms, r.compute_max_ms,
                     r.computes ? r.compute_ms * 1000.0 / (double) r.computes : 0.0,
                     (r.computes && r.nodes) ? r.compute_ms * 1000.0 / (double) r.computes / (double) r.nodes : 0.0,
                     r.buf_max / 1048576.0);
    }

    std::fprintf(stderr, "[5] reconciled: top-level spans %.2f ms = direct backend calls %.2f + child spans %.2f + host residue %.2f\n",
                 top_wall, top_leaf, top_child, top_wall - top_leaf - top_child);
}

static void checkpoint(const char * tag) {
    if (!g_on) return;
    State & S = st();
    print_seg(S, tag);
    S.leaf.clear();
    S.graph.clear();
    S.owner.clear();
    for (int c = 0; c < C_NCAT; ++c) { S.span[c] = SpanAgg(); S.n_in[c] = 0; S.bytes_in[c] = 0; }
    S.begins = 0;
}

static void final_report() {
    if (!g_on || !g_st) return;
    print_seg(*g_st, "rest of run (after the last checkpoint)");
}

struct AtExit { ~AtExit(); };
static AtExit g_atexit;
AtExit::~AtExit() { final_report(); }

}  // namespace dprof

// ================================================================== DS41_DEC_PROF: the decode step's dense side
// DS41_DEC_PROF=1 (diagnostic; inert when unset - one branch on a constant per graph submission and no clock read).
// The decode path submits ONE ggml graph per dense API call (82-83 a token: 1 init + 40 attn_router + 40 finish or
// finish_predict + 1 head) and every one of them is submit+Synchronize, so the two things that decide a decode token's
// dense wall are (a) how many kernels the graph holds - one ggml node is one kernel launch - and (b) how much of the
// wall is device time against host submit/sync.  This measures both per graph kind:
//
//   * wall      = host clock around the submission (what strate DEC prof's stage buckets already see)
//   * dev       = an event PAIR on the same in-order queue (the shim's null stream IS ggml's adopted queue, so the
//                 pair brackets exactly this graph's kernels), read only AFTER the submission's own sync returned -
//                 i.e. when both events have completed.  The old prompt-path bug was reading a pair that was still
//                 pending: cudaEventElapsedTime returns cudaErrorNotReady then, and the caller left the accumulator
//                 at 0, so a 50.86 us/entry kernel read as 3.09 us.  A pending or failed read here is counted as
//                 `missed` and printed; it is never added as 0 ms.
//   * nodes     = gf->n_nodes, the kernel-launch count (the attn node probe's own per-node table is the same count)
//   * dev/launch = dev / nodes, the mean device time of one kernel of that kind
namespace decprof {

enum Kind { G_INIT = 0, G_ATTN, G_FINISH, G_FINPRED, G_PREDICT, G_HEAD, G_NK };
static const char * const kName[G_NK] = { "init/embed", "attn_router+shexp", "finish", "finish_predict", "predict",
                                          "head/logits" };
enum { S_INIT = 0, S_ATTN, S_FIN, S_FINPRED, S_PRED, S_HEAD, S_SAMPLE, S_ENGRAM, S_NS };
static const char * const kSName[S_NS] = { "begin(init graph)", "attn_router", "finish", "finish_predict", "predict",
                                           "head_logits", "sample", "engram+upload" };

struct Row { long long calls = 0, nodes = 0, missed = 0; double wall_ms = 0, dev_ms = 0; };

static const bool g_on = [] {
    const char * e = std::getenv("DS41_DEC_PROF");
    return e && e[0] && e[0] != '0';
}();
static Row g_row[G_NK];
static int g_kind = -1;                       // the kind the next compute() charges; -1 = not a decode graph
static int g_layers = 0;                      // layer count (from attn_router): tokens = attn calls / this
static cudaEvent_t g_e0 = nullptr, g_e1 = nullptr;
static long long g_pairs = 0;
static double g_upload_ms = 0;
static long long g_uploads = 0;

static inline double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

/// Marks the next compute() as graph kind `k` and restores "not a decode graph" on scope exit (a failure before the
/// submission must not leave the kind set for the next graph).
struct KindGuard {
    explicit KindGuard(int k) { if (g_on) g_kind = k; }
    ~KindGuard() { if (g_on) g_kind = -1; }
};

/// Just before the submission: enqueue the opening barrier (both events are created once).
static void open_pair() {
    if (!g_on || g_kind < 0) return;
    if (!g_e0) {
        if (cudaEventCreate(&g_e0) != cudaSuccess || cudaEventCreate(&g_e1) != cudaSuccess) {
            cudaGetLastError();
            g_e0 = g_e1 = nullptr;
            return;
        }
    }
    if (cudaEventRecord(g_e0, nullptr) != cudaSuccess) { cudaGetLastError(); g_e0 = nullptr; }
}

/// Right after ggml_backend_graph_compute returned (it synchronizes), so the closing barrier has completed.
static void histogram(ggml_cgraph * gf, int kind);   // DS41_NODE_HIST (defined below, used per submission)

static void close_pair(ggml_cgraph * gf, double wall0) {
    const long long nodes = gf ? gf->n_nodes : 0;
    if (!g_on) return;
    const int k = g_kind;
    if (k < 0 || k >= G_NK) return;
    histogram(gf, k);
    Row & r = g_row[k];
    ++r.calls;
    r.nodes += nodes;
    r.wall_ms += now_ms() - wall0;
    if (!g_e0) { ++r.missed; return; }
    cudaEventRecord(g_e1, nullptr);
    cudaEventSynchronize(g_e1);              // the submission's sync already did this: the pair is complete
    float ms = 0;
    if (cudaEventElapsedTime(&ms, g_e0, g_e1) == cudaSuccess) { r.dev_ms += ms; ++g_pairs; }
    else { ++r.missed; cudaGetLastError(); } // reported, not silently counted as 0
}

static void add_upload(double ms) { if (!g_on) return; g_upload_ms += ms; ++g_uploads; }

// DS41_NODE_HIST=1 (with DS41_DEC_PROF): one histogram of the op kinds in the first graph of each kind.  This is the
// exact composition behind "200.6 nodes/call" - one ggml node is one kernel launch - and it is what says which fusions
// are worth anything (how many of the launches are elementwise/cast/copy glue against real GEMMs).
static const bool g_hist = std::getenv("DS41_NODE_HIST") != nullptr;
static int g_hist_done[G_NK] = {};
static void histogram(ggml_cgraph * gf, int kind) {
    if (!g_hist || gf == nullptr || kind < 0 || kind >= G_NK || g_hist_done[kind] >= 4) return;
    ++g_hist_done[kind];
    std::map<std::string, int> m;
    for (int i = 0; i < gf->n_nodes; ++i) {
        char b[96];
        const ggml_tensor * t = gf->nodes[i];
        std::snprintf(b, sizeof(b), "%s %s", ggml_op_name(t->op), ggml_type_name(t->type));
        m[b] += 1;
    }
    std::vector<std::pair<std::string, int>> v(m.begin(), m.end());
    std::sort(v.begin(), v.end(), [](const auto & a, const auto & c) { return a.second > c.second; });
    int launch = 0;   // nodes that become a kernel launch: the backend skips views/reshapes/permutes (ggml_sycl_is_view_or_noop)
    for (int i = 0; i < gf->n_nodes; ++i) {
        const ggml_op op = gf->nodes[i]->op;
        if (op == GGML_OP_RESHAPE || op == GGML_OP_TRANSPOSE || op == GGML_OP_VIEW || op == GGML_OP_PERMUTE ||
            op == GGML_OP_NONE) continue;
        ++launch;
    }
    std::fprintf(stderr, "\nDS41_NODE_HIST %s: %d nodes, %d launchable (backend fusion not counted):", kName[kind],
                 gf->n_nodes, launch);
    for (const auto & kv : v) std::fprintf(stderr, " [%s x%d]", kv.first.c_str(), kv.second);
    std::fprintf(stderr, "\n");
}


}  // namespace decprof

// The call-site wrappers.  Each keeps the original call's argument list verbatim (a pure callee rename), so the diff
// of this file is the instrumentation and nothing else: cat, bytes and __LINE__ (the call site) are the only additions.
#define DPROF_SET(...)       dprof::call(dprof::C_SET,       __LINE__, dprof::bytes_last(__VA_ARGS__), [&] { ggml_backend_tensor_set(__VA_ARGS__); })
#define DPROF_GET(...)       dprof::call(dprof::C_GET,       __LINE__, dprof::bytes_last(__VA_ARGS__), [&] { ggml_backend_tensor_get(__VA_ARGS__); })
#define DPROF_GET_ASYNC(...) dprof::call(dprof::C_GET_ASYNC, __LINE__, dprof::bytes_last(__VA_ARGS__), [&] { ggml_backend_tensor_get_async(__VA_ARGS__); })
#define DPROF_SYNC(...)      dprof::call(dprof::C_SYNC,      __LINE__, 0, [&] { ggml_backend_synchronize(__VA_ARGS__); })
#define DPROF_BUILD(kind, il, expr) dprof::build(kind, il, __LINE__, [&] { return (expr); })
#define DPROF_SPAN(cat)      dprof::SpanGuard dprof_span_(cat, __LINE__)

// The decode graphs are built once and reused; stamping a uid lets ggml-cuda reuse its captured CUDA graph.  One
// allocator serves every cached graph (they run one at a time); when it has to grow, every graph allocated before
// is re-allocated on its next use (the epoch), as tools/ds4/ds4_dense.cpp alloc_graph does.
struct Allo {
    ggml_gallocr_t a = nullptr;
    uint64_t epoch = 0;
    bool failed = false;   // the last alloc failed: the gallocr holds a plan without a buffer (see alloc)
    std::map<const ggml_cgraph *, uint64_t> graph_epoch;
    std::set<const void *> compute_bufs;
    ~Allo() { if (a) ggml_gallocr_free(a); }
    bool alloc(ggml_cgraph * g, bool reuse) {
        auto it = graph_epoch.find(g);
        if (reuse && g->uid != 0 && it != graph_epoch.end() && it->second == epoch) return true;
        if (it != graph_epoch.end() && it->second != epoch) {
            auto computed = [&](const ggml_tensor * t) { return t && t->buffer && compute_bufs.count(t->buffer); };
            auto unbind = [&](ggml_tensor * t) {
                if (computed(t) || (t->view_src && computed(t->view_src))) { t->data = nullptr; t->buffer = nullptr; }
            };
            for (int i = 0; i < ggml_graph_n_nodes(g); ++i) {
                ggml_tensor * t = ggml_graph_node(g, i);
                for (int j = 0; j < GGML_MAX_SRC; ++j) if (t->src[j]) unbind(t->src[j]);
                unbind(t);
            }
            for (int i = 0; i < g->n_leafs; ++i) unbind(g->leafs[i]);
        }
        const size_t before = ggml_gallocr_get_buffer_size(a, 0);
        // A failed allocation leaves the gallocr PLANNED for that graph with NO buffer (ggml_gallocr_reserve_n frees
        // the old buffer, then the larger one does not fit).  The next alloc_graph of the same graph then finds the plan
        // current, skips the reserve and places the tensors in the null buffer - a segfault (2026-10-20: the oom_hook
        // retry on a 768K prompt).  So after a failure re-plan explicitly; the freed buffer also makes every graph
        // allocated before it stale (the epoch).
        if (failed && !ggml_gallocr_reserve(a, g)) { ++epoch; return false; }
        if (!ggml_gallocr_alloc_graph(a, g)) { failed = true; ++epoch; return false; }
        failed = false;
        if (ggml_gallocr_get_buffer_size(a, 0) != before && before != 0) ++epoch;
        auto note = [&](const ggml_tensor * t) {
            if (!t || !t->buffer) return;
            if (ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) compute_bufs.insert(t->buffer);
            else compute_bufs.erase(t->buffer);
        };
        for (int i = 0; i < ggml_graph_n_nodes(g); ++i) {
            const ggml_tensor * t = ggml_graph_node(g, i);
            note(t);
            for (int j = 0; j < GGML_MAX_SRC; ++j) note(t->src[j]);
        }
        for (int i = 0; i < g->n_leafs; ++i) note(g->leafs[i]);
        graph_epoch[g] = epoch;
        if (reuse) g->uid = ggml_graph_next_uid();
        return true;
    }
};

struct WStore {
    ggml_context * ctx = nullptr;
    std::vector<ggml_backend_buffer_t> bufs;
    std::map<std::string, ggml_tensor *> t;
    ~WStore() {
        for (auto b : bufs) ggml_backend_buffer_free(b);
        if (ctx) ggml_free(ctx);
    }
    ggml_tensor * get(const std::string & n) const {
        auto it = t.find(n);
        if (it == t.end()) { std::fprintf(stderr, "[ds41_dense] missing weight %s\n", n.c_str()); std::abort(); }
        return it->second;
    }
    bool has(const std::string & n) const { return t.count(n) != 0; }
};

bool is_routed(const std::string & n) {
    return n.find("_exps.") != std::string::npos || n.find("ffn_exps_") != std::string::npos;
}

// The smart router (STRATA_RB_SMART=1, or STRATA_RB_PROTECT / STRATA_RB_POOL set): the protect / pool ranks come from
// the rknob tensor, so a request can change them (set_route_select) without a graph rebuild, and the router also
// writes the exact top-k for the route meter.  kRK = the deepest rank it can select.
constexpr int kRK = 32;
int env_rank(const char * k) {
    const char * e = std::getenv(k);
    return e ? std::max(0, std::min(kRK, std::atoi(e))) : 0;
}
bool smart_router() {
    static const bool on = env_on("STRATA_RB_SMART", false) || env_rank("STRATA_RB_PROTECT") > 0 || env_rank("STRATA_RB_POOL") > 0;
    return on;
}

}  // namespace

// ================================================================== Impl

struct Ds41Dense::Impl {
    std::function<bool()> oom_hook;       // set_oom_hook: frees device memory when a graph buffer does not fit
    Ds41DenseConfig cfg;
    Ds41Geometry g;
    std::unique_ptr<GgufModel> model;
    std::vector<int64_t> layers;          // blocks in the file (0..nl-1)
    int64_t nl = 0;
    int64_t layer_lo = 0, layer_hi = 0;   // [layer_lo, layer_hi) this instance runs and holds KV for
    ggml_backend_t backend = nullptr;
    bool own_backend = false, cpu = true;
    WStore w;
    Ds41Engram engram;
    std::string err;

    int64_t D = 0, HC = 0, DH = 0, DHR = 0, NH = 0, OG = 0, OL = 0, OGD = 0, RQ = 0, SWA = 0, RAW = 0;
    int64_t IH = 0, ID = 0, ITOP = 0, NE = 0, NU = 0, NFF = 0, NC = 0, CB = 0, CBS = 0, EC = 0, ED = 0;
    int64_t cmax[3] = { 0, 0, 0 };        // compressed rows per ratio (index 1 / 2)

    // ---- persistent state (sctx / sbuf)
    ggml_context * sctx = nullptr;
    ggml_backend_buffer_t sbuf = nullptr;
    ggml_tensor * x_state = nullptr;      // [D, HC, NC]
    ggml_tensor * pre_state = nullptr;    // [HC, NC]   the next attention's pre-mix
    ggml_tensor * hap = nullptr;          // [D, HC, NC] the FFN residual (hc_post of the attention)
    ggml_tensor * post_f = nullptr;       // [HC, NC]
    ggml_tensor * comb_f = nullptr;       // [HC, HC, NC]
    ggml_tensor * shexp = nullptr;        // [D, NC]
    ggml_tensor * o_fn = nullptr;         // [D, NC]    ffn_norm activation (the routed experts' input)
    ggml_tensor * o_ids = nullptr;        // I32 [NU, NC]
    ggml_tensor * o_wts = nullptr;        // [NU, NC]
    ggml_tensor * o_xids = nullptr;       // I32 [NU, NC]  route meter (smart router): the exact top-NU
    ggml_tensor * o_xwts = nullptr;       // [NU, NC]      ... and their normalized weights
    ggml_tensor * rknob = nullptr;        // F32 [kRK + 1, 2] smart router: protect / pool rank selectors (set_route_select)
    ggml_tensor * routed = nullptr;       // [D, NC]
    ggml_tensor * cand = nullptr;         // I32 [CB, NC]  candidate blocks of cand_layer
    struct Layer {
        int64_t ratio = 0;
        bool kv_src = false, idx_src = false, cand_src = false, uses_cand = false;
        int64_t kvl = -1, idxl = -1;      // the layers whose compressed KV / top-k this one reads
        int64_t eslot = -1;               // Engram slot
        ggml_tensor * raw = nullptr;      // [DH, RAW]       window K (= V), post-RoPE
        ggml_tensor * comp = nullptr;     // [DH, cmax + 1]  compressed KV (kv sources); row cmax = spare
        ggml_tensor * idxk = nullptr;     // [ID, cmax + 1]  indexer keys (kv sources)
        ggml_tensor * st_kv = nullptr;    // [DH, 2]         ratio-2 compressor state (values)
        ggml_tensor * st_sc = nullptr;    // [DH, 2]         ... (scores)
        ggml_tensor * sel = nullptr;      // I32 [ITOP, NC]  top-k rows (index sources)
        ggml_tensor * selm = nullptr;     // F32 [ITOP, NC]  0 / -inf (row visible to the query)
        ggml_tensor * raw_snap = nullptr, * st_kv_snap = nullptr, * st_sc_snap = nullptr;   // snapshot()
        // batch slots with a slab (Impl::slab): every slot's ring / compressed KV / indexer keys / compressor state of this
        // layer in ONE tensor, the slots' own tensors views of their region (a batch pass then writes and gathers every
        // row's in one op); null otherwise
        ggml_tensor * raw_all = nullptr, * comp_all = nullptr, * idxk_all = nullptr, * st_kv_all = nullptr, * st_sc_all = nullptr;
        ggml_tensor * rbias = nullptr;    // F32 [NE]        selection-only routing bias (set_route_bias), 0 = model routing
    };
    int snap_pos = -1;
    struct ParkedRows { ggml_tensor * t; size_t off; std::vector<uint8_t> b; };
    std::vector<ParkedRows> parked;     // park(): host copies of the snapshot rings + the cache rows an unrelated request overwrites
    int parked_pos = -1;
    std::vector<Layer> ly;
    // ---- batch slots: every slot's own conversation state.  The BOUND slot's lives in the members above (Layer::raw ..
    // idxk, snap_pos, parked, parked_pos) - which is what every single-sequence path reads, unchanged - and the other
    // slots' are parked here; bind_slot() exchanges them (pointers only).  slots[cur_slot] is stale while bound.
    struct SlotLayer {
        ggml_tensor * raw = nullptr, * raw_snap = nullptr, * st_kv = nullptr, * st_sc = nullptr, * st_kv_snap = nullptr,
                    * st_sc_snap = nullptr, * comp = nullptr, * idxk = nullptr;
    };
    struct Slot {
        std::vector<SlotLayer> ly;
        int snap_pos = -1, parked_pos = -1;
        std::vector<ParkedRows> parked;
    };
    std::vector<Slot> slots;
    int cur_slot = 0;
    bool slab = false;                  // the slot tensors are views of per-layer slabs (n_slots > 1, STRATA_SLOT_SLAB != 0)
    // a slot's region in the slabs (its tensors are views at region * their size; swap_slots moves the views, not data)
    int slot_region(int s) const {
        const SlotLayer T = slot_layer(s, 0);
        return T.raw && T.raw->view_src ? (int) (T.raw->view_offs / ggml_nbytes(T.raw)) : 0;
    }
    void stash_slot();                  // the bound members -> slots[cur_slot]
    void load_slot(int s);              // slots[s] -> the bound members; cur_slot = s
    void reset_graphs();                // drop every cached graph (their tensors' addresses changed) - set_columns too
    bool gctx_room() const;             // the cached-graph context can take another layer graph
    // a slot's layer tensors whether bound or not (the batch pass reads several slots at once)
    SlotLayer slot_layer(int s, size_t l) const {
        if (s == cur_slot) {
            const Layer & L = ly[l];
            return SlotLayer { L.raw, L.raw_snap, L.st_kv, L.st_sc, L.st_kv_snap, L.st_sc_snap, L.comp, L.idxk };
        }
        return slots[(size_t) s].ly[l];
    }

    std::vector<int64_t> shadow_layers;

    bool owns_layer(int64_t l) const {
        return l >= layer_lo && l < layer_hi;
    }
    bool active_layer(int64_t l) const {
        if (owns_layer(l)) return true;
        for (int64_t s : shadow_layers) if (s == l) return true;
        return false;
    }

    // ---- per-pass inputs (ictx / ibuf), uploaded by begin()
    ggml_context * ictx = nullptr;
    ggml_backend_buffer_t ibuf = nullptr;
    // ---- the NC-column tensors (state columns + their per-pass inputs) in their own context / buffer, so
    // set_columns() can resize them: --serve keeps S columns while it decodes and grows to the prompt batch only for a
    // layer-major batch (~0.31 MiB a column: 8192 columns = 2.6 GiB that decode gives to expert slots instead)
    ggml_context * nctx = nullptr;
    ggml_backend_buffer_t nbuf = nullptr;
    int64_t NCmax = 0;
    ggml_tensor * i_emb = nullptr;        // [D, NC]
    ggml_tensor * i_pos = nullptr;        // I32 [NC]
    ggml_tensor * i_widx = nullptr;       // I32 [SWA * NC]   per query: rows of [ring | pass rows]
    ggml_tensor * i_wmask = nullptr;      // F32 [SWA, NC]
    ggml_tensor * i_wslot = nullptr;      // I32 [min(NC, RAW)]  ring slots of the pass's last rows
    ggml_tensor * i_nvis[3] = {};         // F32 [NC]   visible compressed rows per query (ratio 1 / 2)
    ggml_tensor * i_cslot[3] = {};        // I32 [NC]   compressed row each token writes (spare if its group is open)
    ggml_tensor * i_cpos[3] = {};         // I32 [NC]   RoPE position of that row (group start)
    ggml_tensor * i_sidx = nullptr;       // I32 [2 * NC] ratio-2 state rows per token, of [state ring | pass states]
    ggml_tensor * i_sslot = nullptr;      // I32 [min(NC, 2)]
    // a batch pass on slabs: the ratio-2 state rows it writes (their pass columns; i_sslot = their slab rows), and per
    // column the slab row its slot's compressed rows start at (F32: added to the top-k rows before the one gather)
    ggml_tensor * i_sscol = nullptr;      // I32 [2 * min(NC, kMaxRows)]
    ggml_tensor * i_cbase[3] = {};        // F32 [NC]
    // upload_span's last (pos0, col0, n): its four index tensors depend only on these, not on the layer, so the 39
    // layers after the first skip identical re-uploads.  Those tiny H2D copies queued behind the chunk prestage's
    // ~1.5 GiB DMA of the next layer's experts and cost ~65 ms a layer (~2.5 s a turn).  Any other writer of the four
    // tensors (upload_inputs) or their reallocation (alloc_state) resets it.
    int64_t span_memo_pos0 = -1;
    int span_memo_col0 = -1, span_memo_n = -1;
    ggml_tensor * i_lastblk = nullptr;    // F32 [NC]   (nvis - 1) / block size: the pinned candidate block (ratio 1)
    ggml_tensor * i_eng[2] = {};          // F32 [EC * ED, NC]  Engram rows
    ggml_tensor * neg_ramp = nullptr;     // F32 [cmax1]  -i     (constant)
    ggml_tensor * blk_ramp = nullptr;     // F32 [cmax1 / CBS + 1]  i (constant)
    ggml_tensor * zpad16 = nullptr;       // F16 [DH, 1] zeros: the flash-attn key padding (STRATA_ATTN_F16_GATHER)
    ggml_tensor * zidx = nullptr;         // I32 [kZidx] zeros: gathers that zero row once per pad slot
    // STRATA_ATTN_F16_GATHER=1 (CUDA): the attention's key rows are gathered straight into F16 and concatenated /
    // padded in F16, instead of gathered F32 -> concat F32 -> pad F32 -> cast F16.  Same values bit for bit (the
    // gather dequantizes in float and rounds with the same __float2half as the cast; concat and pad only move bits),
    // but the 512-query sub-batch's key tensors peak at ~0.74 GB instead of ~1.48 GB (the "1424 MiB" pass-graph
    // buffer of V41_START_HERE run20/21).
    bool f16_gather = false;

    // ---- the pass
    int pos0 = 0, n = 0;
    int64_t pass_cap[3] = { 0, 0, 0 };
    // ---- a batch pass (begin_rows): one group per slot - its columns [c, c + k), first position pos0, visible caps
    struct RowGroup { int slot; int64_t c, k, pos0; int64_t cap[3]; int reg = 0; };   // reg: the slot's slab region
    std::vector<RowGroup> rgroups;
    int64_t rows_nss = 0;                // a batch pass on slabs: the ratio-2 state rows it writes (i_sscol / i_sslot)
    bool rows_mode = false;
    // STRATA_ROWS_GRAPH_CACHE (default on, =0 off): a batch pass's graphs are cached by its LAYOUT (the groups' slot,
    // columns and visible caps): consecutive decode windows of the same slots reuse them (and ggml-cuda its CUDA graph
    // captures) instead of building / allocating 40 layer graphs a window.  rows_sig = the layout's key (0 = uncached).
    int64_t rows_sig = 0;
    std::map<int64_t, std::vector<int64_t>> rows_sigs;   // key -> the layout it stands for (a hash collision is not cached)
    std::vector<int> rows_slot;          // per row: its slot / position (the Engram reads)
    std::vector<int64_t> rows_pos;
    bool upload_rows(const Ds41Dense::Row * rows, int n);
    // ---- the layer-major prompt batch (begin_batch): S = the dense sub-batch size, batch = B, span_cap per sub-batch
    int64_t S = 1, batch = 0, col0 = 0;
    int batch_pos0 = 0;                             // the batch's first position
    std::vector<std::array<int64_t, 3>> span_cap;   // [sub-batch][ratio] pass_cap
    // A sub-batch whose graph does not fit is run as narrower spans (attn_router_span): span_max is the width that
    // fitted, kept for the rest of the prompt (release_pass resets it); 0 = the full sub-batch.  alloc_failed: the last
    // compute() failed in the graph allocation (not in the compute) - the only failure a narrower span can cure.
    int64_t span_max = 0;
    bool alloc_failed = false;
    // The indexer's selection over a long context (build_attn): its [visible compressed rows x queries] score / mask
    // tensors are built for at most idx_tile elements at a time (query chunks); 0 = one piece.  Default 2^28 = a 512K
    // context's 512-query sub-batch, so every context up to 512K builds the very same graph as before.
    int64_t idx_tile = 0;
    std::map<std::array<int64_t, 3>, size_t> graph_bytes_memo;   // span_graph_bytes: (cap 1, cap 2, n) -> bytes
    std::vector<float> host_fn, host_wts, host_logits, host_emb;
    std::vector<int32_t> host_ids;
    std::vector<int32_t> host_xids;       // route meter: o_xids / o_xwts readbacks, and the counters since route_meter(true)
    std::vector<float> host_xwts;
    long long rm_slots = 0, rm_swaps = 0, rm_rows = 0;
    double rm_kept = 0;
    const uint8_t * embd = nullptr;       // token_embd F16 rows on the mmap
    // ---- vision (DeepSeek-V4.1 image input): the request's image span rows, by absolute position.  When enabled,
    // set_vision() swaps in exp_probs_b_vl on those positions and
    // marks them DEAD for Engram.  Empty (n_vis = 0) for a text request.
    bool vision = false;                  // opt-in; requires per-layer exp_probs_b_vl.bias
    ggml_tensor * i_img = nullptr;        // F32 [NC]  1.0 at an image position, 0.0 elsewhere (the VL bias / gate mask)
    const float * vis_rows = nullptr;     // n_vis * D floats, already in prompt order
    const int64_t * vis_pos = nullptr;    // n_vis ascending absolute prompt positions
    int n_vis = 0;
    std::vector<std::vector<float>> eng_rows;
    // The pass's Engram reads, started (async) by upload_inputs and waited at the first Engram layer's span /
    // attn_router: the whole pass's rows are read from disk while the embeddings, the init graph and the layers below
    // the first Engram layer (1) run.  Null once uploaded (or when Engram is off) - see engram_ready().
    std::shared_ptr<Ds41Engram::Batch> eng_batch;

    // ---- graphs
    ggml_context * gctx = nullptr;        // node structs of the cached (n = 1) graphs
    std::vector<uint8_t> pf_meta;         // node structs of a pass graph with n > 1 (built per pass)
    Allo allo_dec, allo_pass;
    // slot: the batch slot whose state tensors the graph reads (the attention graphs; 0 for the others)
    struct GKey { int il, kind; int64_t cap, n; int slot = 0; bool operator<(const GKey & o) const {
        return std::tie(il, kind, cap, n, slot) < std::tie(o.il, o.kind, o.cap, o.n, o.slot); } };
    std::map<GKey, ggml_cgraph *> graphs;
    // pass graphs with 1 < n <= kPassGraphN (the DSpark verify widths: t + depth-4 drafts + a re-fed row), cached
    // like the n = 1 ones instead of built + allocated per layer per pass (DS41_PASS_GRAPHS=0: per pass, as before).
    // Their own node context, dropped wholesale when it runs low (the KV cap buckets move as the context grows).
    ggml_context * gctx_m = nullptr;
    std::map<GKey, ggml_cgraph *> mgraphs;
    long mgraph_resets = 0;
    std::map<int, ggml_tensor *> logits_out;   // decode head graph output, by nrows
    std::map<int, ggml_tensor *> predict_out;  // predict graph output, by layer
    std::map<std::pair<int, int>, ggml_tensor *> predict_out_n;   // predict_multi output, by (layer, pass width)
    std::map<int, ggml_tensor *> fpredict_out; // finish_predict graph output, by the predicted layer
    std::map<std::pair<int, int>, std::vector<ggml_tensor *>> fpredict_ahead; // finish_predict_ahead extra outputs

    ~Impl() {
        if (nbuf) ggml_backend_buffer_free(nbuf);
        if (nctx) ggml_free(nctx);
        if (sbuf) ggml_backend_buffer_free(sbuf);
        if (ibuf) ggml_backend_buffer_free(ibuf);
        if (sctx) ggml_free(sctx);
        if (ictx) ggml_free(ictx);
        if (gctx) ggml_free(gctx);
        if (gctx_m) ggml_free(gctx_m);
        if (own_backend && backend) ggml_backend_free(backend);
    }
    ggml_tensor * W(const std::string & n) const { return w.get(n); }
    ggml_tensor * BL(int64_t il, const std::string & s) const { return w.get("blk." + std::to_string(il) + "." + s); }
    // the router as the prediction graphs read it: the STRATA_PRED_ROUTER copy when the loader made one, else the F32
    ggml_tensor * BLpred(int64_t il) const {
        const auto it = w.t.find("blk." + std::to_string(il) + ".ffn_gate_inp_pred.weight");
        return it != w.t.end() ? it->second : BL(il, "ffn_gate_inp.weight");
    }
    bool compute(ggml_cgraph * gf, bool cached);
    bool upload_inputs(const int * tids, int n, int pos0);
};

// ================================================================== weights

static bool load_weights(Ds41Dense::Impl & im, std::string & err) {
    WStore & w = im.w;
    const GgufModel & model = *im.model;
    ggml_init_params ip = { 256ull * 1024 * 1024, nullptr, true };
    w.ctx = ggml_init(ip);
    auto make = [&](const TensorInfo & ti, int type = -1) {   // type: the in-memory type (default the file's)
        int64_t ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
        for (size_t d = 0; d < ti.shape.size() && d < GGML_MAX_DIMS; ++d) ne[d] = (int64_t) ti.shape[d];
        ggml_tensor * t = ggml_new_tensor(w.ctx, (ggml_type) (type >= 0 ? type : (int) ti.type), (int) ti.shape.size(), ne);
        ggml_set_name(t, ti.name.c_str());
        return t;
    };
    // routed experts never reach ggml (PTQ1_0 = 143 is past GGML_TYPE_COUNT); the I32 maps are read by the geometry
    auto blk_layer = [](const std::string & n) -> int64_t {
        if (n.rfind("blk.", 0) != 0) return -1;
        const size_t dot = n.find('.', 4);
        if (dot == std::string::npos) return -1;
        return std::atoll(n.substr(4, dot - 4).c_str());
    };
    auto in_dense_range = [&](const std::string & n) {
        if (im.cfg.head_only) return n == "output.weight" || n == "output_norm.weight";
        if (n == "token_embd.weight") return im.layer_lo == 0;
        if (n == "output.weight" || n == "output_norm.weight") return im.layer_hi == im.nl && !im.cfg.omit_output_head;
        const int64_t l = blk_layer(n);
        if (l < 0) return true;
        return l >= im.layer_lo && l < im.layer_hi;
    };
    auto skip = [&](const std::string & n) { return is_routed(n) || n == "token_embd.weight" || !in_dense_range(n); };
    if (im.cpu) {
        for (size_t s = 0; s < model.size(); ++s) {
            const GgufFile & sh = model.shard(s);
            const auto & ts = sh.tensors();
            if (ts.empty()) continue;
            uint8_t * base = const_cast<uint8_t *>(sh.tensor_data(ts[0])) - sh.data_start() - ts[0].offset;
            ggml_backend_buffer_t buf = ggml_backend_cpu_buffer_from_ptr(base, (size_t) sh.file_size());
            w.bufs.push_back(buf);
            for (const auto & ti : ts) {
                if (skip(ti.name)) continue;
                ggml_tensor * t = make(ti);
                if (ggml_backend_tensor_alloc(buf, t, const_cast<uint8_t *>(sh.tensor_data(ti))) != GGML_STATUS_SUCCESS) {
                    err = "cannot bind " + ti.name;
                    return false;
                }
                w.t[ti.name] = t;
            }
        }
    } else {
        // STRATA_DENSE_REQUANT=<ggml type name> (q6_K, q5_K, q4_K, ...; quality experiment, OFF by default): every
        // Q8_0 dense matrix (attention, shared expert; output.weight unless STRATA_DENSE_REQUANT_HEAD=0) is
        // re-quantized in memory at load - rows dequantized, quantized to that type.  The file is not touched.
        ggml_type rq = GGML_TYPE_COUNT;
        if (const char * e = std::getenv("STRATA_DENSE_REQUANT"); e && *e)
            for (int t = 0; t < GGML_TYPE_COUNT; ++t)
                if (ggml_get_type_traits((ggml_type) t)->type_name && strcasecmp(ggml_type_name((ggml_type) t), e) == 0) rq = (ggml_type) t;
        const bool rq_head = !(std::getenv("STRATA_DENSE_REQUANT_HEAD") && std::getenv("STRATA_DENSE_REQUANT_HEAD")[0] == '0');
        auto requant = [&](const TensorInfo & ti) {
            return rq != GGML_TYPE_COUNT && (ggml_type) ti.type == GGML_TYPE_Q8_0 && ti.shape.size() == 2 &&
                   (int64_t) ti.shape[0] % ggml_blck_size(rq) == 0 && (rq_head || ti.name != "output.weight");
        };
        // STRATA_PRED_ROUTER=<ggml type> (q8_0, f16; OFF by default): the PREDICTION graphs (the prefetch top-16 and the
        // speculative reads' rows) read a copy of each layer's F32 router in that type (q8_0: 7.9 -> 2.1 MB a read); the
        // routing itself (build_attn) keeps the file's F32.  Only which experts are prefetched can change.
        ggml_type pq = GGML_TYPE_COUNT;
        if (const char * e = std::getenv("STRATA_PRED_ROUTER"); e && *e)
            for (int t = 0; t < GGML_TYPE_COUNT; ++t)
                if (ggml_get_type_traits((ggml_type) t)->type_name && strcasecmp(ggml_type_name((ggml_type) t), e) == 0) pq = (ggml_type) t;
        auto pred_name = [&](const TensorInfo & ti) -> std::string {   // "" = no prediction copy for this tensor
            static const std::string suf = "ffn_gate_inp.weight";
            const std::string & n = ti.name;
            if (pq == GGML_TYPE_COUNT || (ggml_type) ti.type != GGML_TYPE_F32 || ti.shape.size() != 2 ||
                (int64_t) ti.shape[0] % ggml_blck_size(pq) != 0 || n.size() <= suf.size() ||
                n.compare(n.size() - suf.size(), suf.size(), suf) != 0) return std::string();
            return n.substr(0, n.size() - suf.size()) + "ffn_gate_inp_pred.weight";
        };
        size_t rq_n = 0, rq_from = 0, rq_to = 0, pq_n = 0;
        for (size_t s = 0; s < model.size(); ++s)
            for (const auto & ti : model.shard(s).tensors()) {
                if (skip(ti.name)) continue;
                w.t[ti.name] = make(ti, requant(ti) ? (int) rq : -1);
                if (const std::string pn = pred_name(ti); !pn.empty()) {
                    ggml_tensor * tp = make(ti, (int) pq);
                    ggml_set_name(tp, pn.c_str());
                    w.t[pn] = tp;
                }
            }
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(w.ctx, ggml_backend_get_default_buffer_type(im.backend));
        if (!buf) { err = "cannot allocate the dense weights on the backend"; return false; }
        w.bufs.push_back(buf);
        // DIAG (env-gated, DS41_WRESID=1): how many dense-weight bytes actually live on the device?
        // Hypothesis under test: weights in a HOST buffer would be streamed over PCIe by every matmul.
        if (std::getenv("DS41_WRESID")) {
            size_t nb = 0, db = 0, hb = 0, cnt = 0, hcnt = 0;
            for (ggml_tensor * t = ggml_get_first_tensor(w.ctx); t; t = ggml_get_next_tensor(w.ctx, t)) {
                if (!t->buffer) continue;
                const size_t n = ggml_nbytes(t);
                nb += n; ++cnt;
                if (ggml_backend_buffer_is_host(t->buffer)) { hb += n; ++hcnt; } else db += n;
            }
            std::fprintf(stderr,
                "DIAG weights: total %.3f GiB in %zu tensors | device %.3f GiB | host %.3f GiB (%zu host tensors)\n",
                nb / 1073741824.0, cnt, db / 1073741824.0, hb / 1073741824.0, hcnt);
            for (auto * b : w.bufs)
                std::fprintf(stderr, "DIAG weight buffer: %.3f GiB is_host=%d\n",
                             ggml_backend_buffer_get_size(b) / 1073741824.0,
                             (int) ggml_backend_buffer_is_host(b));
        }
        // STRATA_DENSE_REQUANT_CACHE=<dir> (llm-server): the conversions below cost ~25 s a load (q6_K's reference
        // quantizer is CPU-bound; the reads are not the cost - measured with the source in the page cache).  Their
        // output is written once to <dir>/<model>.<key>.dense-rq, in the order this loop converts, and read back on
        // later loads.  The key: the model file's size and mtime, the requant / prediction types and the head switch;
        // a cache of the wrong total size is ignored (and rewritten).
        std::FILE * rq_in = nullptr;
        std::FILE * rq_out = nullptr;
        std::string rq_path, rq_tmp;
        if (const char * cd = std::getenv("STRATA_DENSE_REQUANT_CACHE"); cd && *cd && (rq != GGML_TYPE_COUNT || pq != GGML_TYPE_COUNT)) {
            size_t want = 0;
            for (size_t s = 0; s < model.size(); ++s)
                for (const auto & ti : model.shard(s).tensors()) {
                    if (skip(ti.name)) continue;
                    if (const std::string pn = pred_name(ti); !pn.empty()) want += ggml_nbytes(w.t[pn]);
                    if ((int) w.t[ti.name]->type != (int) ti.type) want += ggml_nbytes(w.t[ti.name]);
                }
            struct stat st {};
            const std::string mp = model.shard(0).path();
            if (want > 0 && ::stat(mp.c_str(), &st) == 0) {
                const size_t sl = mp.find_last_of('/');
                char key[160];
                std::snprintf(key, sizeof key, ".%llx.%llx.%s.%s.h%d.dense-rq", (unsigned long long) st.st_size,
                              (unsigned long long) st.st_mtime, rq != GGML_TYPE_COUNT ? ggml_type_name(rq) : "none",
                              pq != GGML_TYPE_COUNT ? ggml_type_name(pq) : "none", rq_head ? 1 : 0);
                rq_path = std::string(cd) + "/" + (sl == std::string::npos ? mp : mp.substr(sl + 1)) + key;
                struct stat cs {};
                if (::stat(rq_path.c_str(), &cs) == 0 && (size_t) cs.st_size == want) {
                    rq_in = std::fopen(rq_path.c_str(), "rb");
                } else {
                    rq_tmp = rq_path + ".tmp";
                    rq_out = std::fopen(rq_tmp.c_str(), "wb");
                    if (!rq_out) std::fprintf(stderr, "dense requant cache: cannot write %s\n", rq_tmp.c_str());
                }
            }
        }
        // rows of the file's tensor `ti` (bytes `src`) converted to t->type and uploaded; returns the bytes uploaded
        auto convert_upload = [&](ggml_tensor * t, const TensorInfo & ti, const uint8_t * src) -> size_t {
            const int64_t n0 = t->ne[0], nr = ggml_nrows(t);
            const size_t src_row = ggml_row_size((ggml_type) ti.type, n0), dst_row = ggml_row_size(t->type, n0);
            std::vector<uint8_t> dst((size_t) nr * dst_row);
            if (rq_in) {
                if (std::fread(dst.data(), 1, dst.size(), rq_in) == dst.size()) {
                    DPROF_SET(t, dst.data(), 0, dst.size());
                    return dst.size();
                }
                std::fprintf(stderr, "dense requant cache: short read in %s - converting\n", rq_path.c_str());
                std::fclose(rq_in);
                rq_in = nullptr;
                // (a later tensor of this load is converted too: nothing below reads rq_in again)
            }
            const int nth = (int) std::max(1u, std::thread::hardware_concurrency());
            std::vector<std::thread> th;
            for (int k = 0; k < nth; ++k)
                th.emplace_back([&, k] {
                    std::vector<float> f((size_t) n0);
                    for (int64_t r = k; r < nr; r += nth) {
                        if ((ggml_type) ti.type == GGML_TYPE_F32) std::memcpy(f.data(), src + (size_t) r * src_row, (size_t) n0 * 4);
                        else ggml_get_type_traits((ggml_type) ti.type)->to_float(src + (size_t) r * src_row, f.data(), n0);
                        ggml_quantize_chunk(t->type, f.data(), dst.data() + (size_t) r * dst_row, 0, 1, n0, nullptr);
                    }
                });
            for (auto & x : th) x.join();
            if (rq_out && std::fwrite(dst.data(), 1, dst.size(), rq_out) != dst.size()) {
                std::fclose(rq_out);
                rq_out = nullptr;
                std::remove(rq_tmp.c_str());
            }
            DPROF_SET(t, dst.data(), 0, dst.size());
            return dst.size();
        };
        for (size_t s = 0; s < model.size(); ++s)
            for (const auto & ti : model.shard(s).tensors()) {
                if (skip(ti.name)) continue;
                ggml_tensor * t = w.t[ti.name];
                const uint8_t * src = model.shard(s).tensor_data(ti);
                if (const std::string pn = pred_name(ti); !pn.empty()) {
                    convert_upload(w.t[pn], ti, src);
                    ++pq_n;
                }
                if ((int) t->type == (int) ti.type) {
                    DPROF_SET(t, src, 0, ggml_nbytes(t));
                    continue;
                }
                rq_to += convert_upload(t, ti, src);
                ++rq_n;
                rq_from += (size_t) ggml_nrows(t) * ggml_row_size((ggml_type) ti.type, t->ne[0]);
            }
        if (rq_n) std::fprintf(stderr, "dense requant: %zu Q8_0 matrices -> %s, %.2f -> %.2f GB%s\n", rq_n, ggml_type_name(rq),
                               rq_from / 1e9, rq_to / 1e9, rq_in ? " (from the cache)" : rq_out ? " (cache written)" : "");
        if (rq_in) std::fclose(rq_in);
        if (rq_out) {
            const bool ok = std::fclose(rq_out) == 0 && std::rename(rq_tmp.c_str(), rq_path.c_str()) == 0;
            if (!ok) std::remove(rq_tmp.c_str());
        }
        if (pq_n) std::fprintf(stderr, "prediction routers: %zu F32 copies as %s\n", pq_n, ggml_type_name(pq));
    }
    size_t si = 0;
    const TensorInfo * te = model.find("token_embd.weight", &si);
    im.embd = model.shard(si).tensor_data(*te);
    return true;
}

// ================================================================== graph helpers

namespace {

struct B {
    Ds41Dense::Impl * im;
    ggml_context * c;
    const Ds41Geometry * g;

    ggml_tensor * v1(ggml_tensor * a, int64_t n0, size_t off) const { return ggml_view_1d(c, a, n0, off); }
    ggml_tensor * cols2(ggml_tensor * t, int64_t n0, int64_t n1, int64_t c0 = 0) const {   // columns c0.. of a [n0, *] tensor
        return ggml_view_2d(c, t, n0, n1, t->nb[1], (size_t) c0 * t->nb[1]);
    }
    ggml_tensor * cols3(ggml_tensor * t, int64_t n0, int64_t n1, int64_t n2, int64_t c0 = 0) const {
        return ggml_view_3d(c, t, n0, n1, n2, t->nb[1], t->nb[2], (size_t) c0 * t->nb[2]);
    }
    ggml_tensor * flat2(ggml_tensor * t, int64_t ne0, int64_t ne1) const {
        return ggml_is_contiguous(t) ? ggml_reshape_2d(c, t, ne0, ne1) : ggml_cont_2d(c, t, ne0, ne1);
    }
    ggml_tensor * rms_w(ggml_tensor * x, ggml_tensor * wt) const {
        return ggml_mul(c, ggml_rms_norm(c, x, (float) g->rms_eps), wt);
    }
    // hc_mixes (model.py Block.hc_mixes + kernel hc_split_sinkhorn): x [D, HC, n] -> pre [HC, n], post [HC, n],
    // comb [HC(dst), HC(src), n]
    void hc_mixes(ggml_tensor * x, ggml_tensor * fn, ggml_tensor * sc, ggml_tensor * bs,
                  ggml_tensor ** pre, ggml_tensor ** post, ggml_tensor ** comb) const {
        const int64_t HC = g->hc, nt = x->ne[2], D = g->n_embd;
        ggml_tensor * flat = ggml_rms_norm(c, ggml_reshape_2d(c, x, HC * D, nt), (float) g->rms_eps);
        ggml_tensor * mixes = ggml_mul_mat(c, fn, flat);                                  // [24, n]
        ggml_mul_mat_set_prec(mixes, GGML_PREC_F32);
#if defined(STRATA_DSV41_HC_MIX)
        // -DSTRATA_DSV41_HC_MIX=ON on a llama.cpp tree patched with tools/v41/ggml/dsv41-fuse.patch: the two mixes as
        // ONE op (GGML_OP_DSV41_HC_MIXES), the same mul/add/sigmoid/scale steps bit for bit (test_ds41_hcmix).  It
        // replaces 8 of the layer graph's ~82 tiny kernels per call.  STRATA_HC_MIX_COMPOSITE=1: the composite.
        static const bool composite = std::getenv("STRATA_HC_MIX_COMPOSITE") != nullptr;
        if (!composite) {
            ggml_tensor * mix = ggml_dsv41_hc_mixes(c, mixes, v1(sc, 2, 0), v1(bs, 2 * HC, 0), (float) g->hc_eps);
            *pre  = ggml_view_2d(c, mix, HC, nt, mix->nb[1], 0);
            *post = ggml_view_2d(c, mix, HC, nt, mix->nb[1], (size_t) nt * mix->nb[1]);
            *comb = ggml_dsv4_hc_comb(c, mixes, sc, bs, (float) g->hc_eps, (int32_t) g->hc_iters);
            return;
        }
#endif
        ggml_tensor * m_pre = ggml_view_2d(c, mixes, HC, nt, mixes->nb[1], 0);
        ggml_tensor * m_post = ggml_view_2d(c, mixes, HC, nt, mixes->nb[1], (size_t) HC * mixes->nb[0]);
        *pre = ggml_add(c, ggml_mul(c, m_pre, v1(sc, 1, 0)), v1(bs, HC, 0));
        *pre = ggml_scale_bias(c, ggml_sigmoid(c, *pre), 1.0f, (float) g->hc_eps);
        *post = ggml_add(c, ggml_mul(c, m_post, v1(sc, 1, sizeof(float))), v1(bs, HC, (size_t) HC * sizeof(float)));
        *post = ggml_scale(c, ggml_sigmoid(c, *post), 2.0f);
        *comb = ggml_dsv4_hc_comb(c, mixes, sc, bs, (float) g->hc_eps, (int32_t) g->hc_iters);
    }
    // RoPE over the last `nd` dims of each `ne0` row (model.py apply_rotary_emb: adjacent pairs = NORMAL mode).
    // ratio > 0: compress_rope_theta with YaRN; ratio 0: rope_theta, no YaRN (model.py Attention.__init__)
    ggml_tensor * rope(ggml_tensor * a, ggml_tensor * pos, int64_t ratio, bool back = false) const {
        const bool yarn = ratio != 0;
        const float base = (float) (yarn ? g->compress_rope_base : g->rope_base);
        const float fscale = yarn ? (float) (1.0 / g->rope_factor) : 1.0f;
        const float ext = yarn ? 1.0f : 0.0f;
        // ggml multiplies by mscale = attn_factor * (1 + 0.1 ln(1/fscale)) under YaRN; model.py has no such scale
        const float attn = yarn ? 1.0f / (1.0f + 0.1f * std::log(1.0f / fscale)) : 1.0f;
        const float bf = yarn ? (float) g->beta_fast : 0.0f, bsl = yarn ? (float) g->beta_slow : 0.0f;
        const int n_ctx = yarn ? (int) g->orig_ctx : 0;
        const int nd = (int) g->d_rope;
        ggml_tensor * r = back ? ggml_rope_ext_back(c, a, pos, nullptr, nd, GGML_ROPE_TYPE_NORMAL, n_ctx, base, fscale, ext, attn, bf, bsl)
                               : ggml_rope_ext(c, a, pos, nullptr, nd, GGML_ROPE_TYPE_NORMAL, n_ctx, base, fscale, ext, attn, bf, bsl);
        return ggml_rope_set_offset(r, (int) (a->ne[0] - nd));
    }
};

}  // namespace

// STRATA_DENSE_PROF=1 (diagnostic, off by default): where a layer-major batch's dense time goes - graph build, gallocr
// allocation, GPU compute, Engram wait, readback - summed per prompt and printed by prof_report()
static const bool g_dprof_on = std::getenv("STRATA_DENSE_PROF") != nullptr;
static double g_dprof[8] = {};   // build, alloc, compute, engram, readback, finish build, upload, calls
static double dprof_now() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static int g_dprof_slot = -1;    // which of build/alloc/compute buckets compute() charges: set by the span callers

// ================================================================== the GEMM-shaped dense parts, through oneMKL
// The dense half's q/kv/out projections, its shared expert and its router are one shape of work - a quantized weight
// times a full-prompt activation block - and they ran as ggml-sycl's quantized matmul kernels at 0.8-12 TFLOPS
// against the 60-70 TFLOPS this card measures on oneMKL FP16 GEMM.  The expert path already runs its two matmuls that
// way (native_expert_f16.dp.cpp: dequantize, then one oneMKL FP16 GEMM with FP32 accumulation); this routes the dense
// ones through the same code (strata::kernels::dense_gemm_f16).
//
// HOW, WITHOUT REBUILDING THE GRAPH.  Mark the node as it is built (note() right after the ggml_mul_mat), then at the
// single submission point (partprof::run below) cut the graph just before each marked node, SKIP the node, and run the
// oneMKL GEMM into the node's own buffer instead.  A committed graph's node order is a topological order, so the cut
// changes nothing about the arithmetic or about buffer lifetimes: the allocator still sees the node, so its dst buffer
// (size, position, liveness) is exactly as planned, and every consumer still reads the same tensor.  The cost is one
// submit+sync per marked node - ggml_backend_graph_compute synchronizes (src/ggml-backend.cpp:595) and the GEMM is
// submitted on the same in-order queue in between.  DS41_PART_PROF's probe already submits the same graph in node
// ranges for the same reason and sums to 27.0 s against the 27.7 s dense wall, so the cuts are not expensive.
//
// WHAT IS NOT ROUTED: the attention core (16.5 s at 1.53 TFLOPS).  Its content is the per-query K gather + cast, the
// flash attention over 128+512 keys of width 512 and the indexer's block scores/top-k over every visible row - a new
// kernel, not a GEMM.  It stays on its kernels, and nothing here is ever marked inside it.
namespace densegemm {

/// STRATA_DENSE_GEMM=0: off (the ggml ops, as before).  STRATA_DENSE_GEMM_PARTS=<letters>: a subset of
/// q (q_a/q_b) k (kv) o (attn out LoRA) s (shared expert) r (router), the bisect switch for a part whose numerics
/// look off.  The default is "qkos": the router is F32, so it takes an all-FP32 GEMM, and oneMKL's sgemm at N = 384
/// measured SLOWER than ggml's own op (161 vs 131 ms) - 'r' is available, not a win.
static bool all_on() {
    static const bool on = [] {
        const char * e = std::getenv("STRATA_DENSE_GEMM");
        return !(e != nullptr && e[0] == '0');
    }();
    return on;
}
static bool part_on(char part) {
    static const std::string parts = [] {
        const char * e = std::getenv("STRATA_DENSE_GEMM_PARTS");
        return std::string(e != nullptr && e[0] ? e : "qkos");
    }();
    return parts.find(part) != std::string::npos;
}

/// Decode calls run these matmuls with M = 1: a GEMM's launch and the per-call dequant would be a regression there,
/// and the decode path is not what this is for.  The prompt's chunks are >= 1024 rows.
static constexpr int64_t kMinRows = 256;

/// The marked nodes, per graph: a graph built once and computed again (the pass cache) keeps its marks, and a mark
/// can never be read by a graph it was not recorded for.  The pointers are tensors of that graph's context, and
/// exec() re-checks the op, both sources and every shape, so a context that reuses an address cannot turn a mark into
/// a wrong GEMM (it would be the same GEMM, or a refusal).
static std::map<const ggml_cgraph *, std::set<ggml_tensor *>> g_by_graph;
static ggml_cgraph * g_building = nullptr;

static void begin(ggml_cgraph * gf) {
    g_building = gf;
    if (gf != nullptr) g_by_graph[gf].clear();
}
static void note(ggml_tensor * t, char part) {
    if (t == nullptr || g_building == nullptr || !all_on() || !part_on(part)) return;
    g_by_graph[g_building].insert(t);
}
static const std::set<ggml_tensor *> * all(const ggml_cgraph * gf) {
    auto it = g_by_graph.find(gf);
    return it == g_by_graph.end() ? nullptr : &it->second;
}
static bool is_ours(const ggml_cgraph * gf, const ggml_tensor * t) {
    const std::set<ggml_tensor *> * s = all(gf);
    return s != nullptr && s->find(const_cast<ggml_tensor *>(t)) != s->end();
}

static bool dbg_on() {
    static const bool on = std::getenv("DS41_DENSE_GEMM_DBG") != nullptr;
    return on;
}

/// Whether this graph has a marked node wide enough to be worth a oneMKL GEMM (false = run the graph as one submit,
/// e.g. every decode step).
static bool active(ggml_cgraph * gf) {
    const std::set<ggml_tensor *> * s = all(gf);
    if (s == nullptr || s->empty()) return false;
    for (int i = 0; i < gf->n_nodes; ++i) {
        if (!is_ours(gf, gf->nodes[i])) continue;
        const ggml_tensor * x = gf->nodes[i]->src[1];
        if (x != nullptr && x->ne[1] >= kMinRows) return true;
    }
    if (dbg_on()) std::fprintf(stderr, "densegemm: active(%p) = false over %d nodes (%zu marked)\n", (void*) gf, gf->n_nodes, s->size());
    return false;
}

/// Runs one marked node through dense_gemm_f16, writing the node's own buffer.  False = not ours or not supported:
/// nothing has been written, so the caller runs the node on ggml exactly as it would have.
static bool exec(ggml_cgraph * gf, ggml_tensor * t) {
#if defined(DS41_SYCL)
    if (dbg_on())
        std::fprintf(stderr, "densegemm: exec %s op=%s type=%d ours=%d\n", t->name, ggml_op_name(t->op), (int) t->type,
                     (int) is_ours(gf, t));
    if (!is_ours(gf, t) || t->op != GGML_OP_MUL_MAT || t->type != GGML_TYPE_F32) return false;
    const ggml_tensor * w = t->src[0];
    const ggml_tensor * x = t->src[1];
    auto no = [&](const char * why) {
        if (dbg_on())
            std::fprintf(stderr, "densegemm:   refused (%s): w %s type %d ne %lldx%lldx%lld cont %d nb %zu/%zu | x type %d nb0 %zu ne %lldx%lldx%lld | dst %lldx%lldx%lld\n",
                         why, w ? w->name : "-", w ? (int) w->type : -1, w ? (long long) w->ne[0] : 0,
                         w ? (long long) w->ne[1] : 0, w ? (long long) w->ne[2] : 0, w ? (int) ggml_is_contiguous(w) : 0,
                         w ? w->nb[0] : 0, w ? w->nb[1] : 0, x ? (int) x->type : -1, x ? x->nb[0] : 0,
                         x ? (long long) x->ne[0] : 0, x ? (long long) x->ne[1] : 0, x ? (long long) x->ne[2] : 0,
                         (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2]);
        return false;
    };
    if (w == nullptr || x == nullptr || w->data == nullptr || x->data == nullptr || t->data == nullptr) return no("null");
    if (x->type != GGML_TYPE_F32 || x->nb[0] != ggml_type_size(x->type)) return no("activation kind");   // K contiguous
    if (!ggml_is_contiguous(w)) return no("weight not contiguous");                                      // the layout the GEMM reads
    if (w->type != GGML_TYPE_Q8_0 && w->type != GGML_TYPE_F16 && w->type != GGML_TYPE_F32) return no("weight type");
    if (t->ne[0] != w->ne[1] || t->ne[1] != x->ne[1] || x->ne[0] != w->ne[0]) return no("shape");
    if (t->ne[1] < kMinRows) return no("narrow");
    const int64_t groups = x->ne[2] > 1 ? x->ne[2] : 1;
    if (groups > 1 && (w->ne[2] != groups || t->ne[2] != groups)) return no("groups");
    strata::kernels::DenseGemmArgs a;
    a.key = w;   // the tensor, not just its data: the GEMM's dequantized-weight cache is keyed by it
    a.w = w->data;
    a.wtype = (int) w->type;
    a.x = x->data;
    a.xtype = (int) x->type;
    a.y = t->data;
    a.K = w->ne[0];
    a.N = w->ne[1];
    a.M = x->ne[1];
    a.groups = groups;
    a.x_row = x->nb[1] / 4;
    a.x_group = groups > 1 ? x->nb[2] / 4 : 0;
    a.y_row = t->nb[1] / 4;
    a.y_group = groups > 1 ? t->nb[2] / 4 : 0;
    a.w_group = (int64_t) (w->ne[0] * w->ne[1]);
    // The default stream IS ggml's in-order queue (ds41_sycl_adopt_ggml_queue at startup), so the GEMM lands between
    // the two graph_compute calls that surround it and needs no event of its own.
    void * q = ds41_sycl_default_queue();
    if (q == nullptr) return no("no default queue");
    static bool said = false;
    if (!said) {
        said = true;
        std::fprintf(stderr,
                     "strata: dense GEMM-shaped parts on dequantize + oneMKL FP16 GEMM (parts %s, FP32 accumulate; the "
                     "attention core stays on its kernels): %.1f MiB of FP16 scratch\n",
                     std::getenv("STRATA_DENSE_GEMM_PARTS") ? std::getenv("STRATA_DENSE_GEMM_PARTS") : "qkos",
                     (double) strata::kernels::dense_gemm_f16_scratch_bytes() / 1048576.0);
    }
    return strata::kernels::dense_gemm_f16(q, a);
#else
    (void) gf;
    (void) t;
    return false;
#endif
}

}  // namespace densegemm

// ================================================================== per-op time of the attention core
// DS41_ATTN_PROF=1 (diagnostic, off by default): the attention core is ONE ggml submission - a node range of the layer
// graph - so the part table cannot say which of its ops costs the 16.5 s.  This probe submits that range ONE NODE PER
// SUBMISSION: the same nodes, in the same order, on the same in-order queue, so the arithmetic is bit-identical and
// only the submission granularity changes (it is a probe, not a fast path: the per-node launch cost is measured too and
// reported).  Each node is filed under a bucket and under its op/type/shape, and the slowest nodes are kept, so the
// numbers say which of gather / cast / QK / softmax / AV / indexer / selection / glue the time is in.  Inert when the
// variable is unset.  It also turns the part probe on, because the node ranges are that probe's marks.
namespace attnprof {

enum {
    B_GATHER = 0, B_CAST, B_FA, B_IDX, B_SEL, B_WRITE, B_PROJ, B_MASK, B_OTHER, B_NB
};
static const char * const kName[B_NB] = {
    "a  K gather      get_rows of a DH-wide row set",
    "a  K cast        f32 -> f16 cpy of a DH-wide row set",
    "b+c+d  flash-attn  QK^T + softmax/mask + AV (fused, one kernel)",
    "e  indexer scan   lightning_indexer over every visible row",
    "f  selection      top_k / argsort / 1-wide row gathers / sel copies",
    "g  state writes   set_rows (ring, compressed, sel, state)",
    "g  projections    mul_mat inside the core (compressor, indexer q/k)",
    "g  mask build     vis/m1 chain, gate soft_max, fills",
    "g  other          concat / pad / cont / elementwise glue",
};

static const bool on = std::getenv("DS41_ATTN_PROF") != nullptr;
static int g_dh = 512;

struct Acc { double ms = 0.0; long long n = 0; };
struct Slow { double ms = 0.0; int op = 0; int64_t ne[4] = { 0, 0, 0, 0 }; };
struct Note { int il = -1; int64_t n = 0, col0 = 0; };
struct Rec {
    int il = -1;
    int64_t col0 = 0, n = 0;
    long long nodes = 0;
    double part = 0.0;              // the range's wall time (one submission), for the launch-cost residue
    double b[B_NB] = {};
    long long bn[B_NB] = {};
    std::map<std::string, Acc> ops;
    std::vector<Slow> slow;
};

static std::map<const ggml_cgraph *, Note> notes;   // filled by build_attn: which layer / columns this graph is
static std::vector<Rec> recs;                      // one record per attention-core range
// the whole layer graph, by part: the core's 16.5 s is one cut interval, but the indexer's scan and the compressor's
// projections land in the neighbouring intervals, so the probe has to see every node to place them.
static const char * const kPartName[12] = { "0 hc_pre+attn_norm", "1 q_a/q_b proj + rope", "2 kv proj + rope",
                                            "3 attention core", "4 attn out LoRA + hc_post", "5 router",
                                            "6 shared expert", "7 finish", "8 head", "9 writes", "10 init" };
static double pms[12];
static long long pn[12];
static std::map<int, std::map<std::string, Acc>> pops;

static double now_ms() {
    return (double) std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count() / 1e6;
}
static void set_dh(int64_t dh) { g_dh = (int) dh; }
static void note(const ggml_cgraph * gf, int il, int64_t n, int64_t col0) {
    if (!on || gf == nullptr) return;
    notes[gf] = Note{ il, n, col0 };
}

static std::string opkey(const ggml_tensor * t) {
    char buf[128];
    std::string k = ggml_op_name(t->op);
    if (t->op == GGML_OP_UNARY) { k += " "; k += ggml_unary_op_name(ggml_get_unary_op(t)); }
    if (t->op == GGML_OP_CPY && t->src[0] != nullptr && t->src[0]->type != t->type) {
        k += " "; k += ggml_type_name(t->src[0]->type); k += "->"; k += ggml_type_name(t->type);
    } else {
        k += " "; k += ggml_type_name(t->type);
    }
    const int64_t w = t->src[0] != nullptr ? t->src[0]->ne[0] : 0;   // the row width a gather/cast reads
    std::snprintf(buf, sizeof(buf), " d%lldx%lldx%lldx%lld s%lld", (long long) t->ne[0], (long long) t->ne[1],
                  (long long) t->ne[2], (long long) t->ne[3], (long long) w);
    return k + buf;
}

static int classify(const ggml_tensor * t) {
    switch (t->op) {
        case GGML_OP_FLASH_ATTN_EXT:    return B_FA;
        case GGML_OP_LIGHTNING_INDEXER: return B_IDX;
        case GGML_OP_TOP_K:
        case GGML_OP_ARGSORT:           return B_SEL;
        case GGML_OP_MUL_MAT:           return B_PROJ;
        case GGML_OP_SOFT_MAX:
        case GGML_OP_LOG:
        case GGML_OP_FILL:              return B_MASK;
        case GGML_OP_UNARY: {   // the vis mask's step
            const enum ggml_unary_op u = ggml_get_unary_op(t);
            return u == GGML_UNARY_OP_STEP ? B_MASK : B_OTHER;
        }
        case GGML_OP_GET_ROWS: {
            const ggml_tensor * s = t->src[0];
            if (s != nullptr && s->ne[0] > 1) return B_GATHER;   // a K / state row set (DH or ID wide)
            return B_SEL;                                       // the sel reorder or the vis mask gather
        }
        case GGML_OP_SET_ROWS:          return B_WRITE;
        case GGML_OP_CPY: {
            const ggml_tensor * s = t->src[0];
            if (s != nullptr && s->type != t->type && s->ne[0] > 1) {
                return s->ne[0] == g_dh ? B_CAST : B_MASK;       // the K cast, or the mask's f32 -> f16 cast
            }
            return B_OTHER;   // cont / a same-type copy: glue, not a state write
        }
        default:                        return B_OTHER;
    }
}

/// Times EVERY node of gf, one submission each, and files it under its cut's part and under its op.  The attention
/// core's interval (part 3) also gets the per-bucket record.
static void run_all(ggml_backend_t be, ggml_cgraph * gf, const std::vector<std::pair<int, int>> & cut) {
    if (!on) return;
    Rec r;
    auto it = notes.find(gf);
    if (it != notes.end()) { r.il = it->second.il; r.n = it->second.n; r.col0 = it->second.col0; }
    size_t ci = 0;
    const double tot0 = now_ms();
    for (int i = 0; i < gf->n_nodes; ++i) {
        ggml_tensor * t = gf->nodes[i];
        while (ci < cut.size() && cut[ci].first < i) ++ci;
        const int part = ci < cut.size() ? cut[ci].second : 9;   // 9 = writes, the tail part
        ggml_cgraph sub = *gf;
        sub.nodes = gf->nodes + i; sub.n_nodes = 1;
        sub.leafs = nullptr; sub.n_leafs = 0; sub.grads = nullptr; sub.grad_accs = nullptr;
        const double t0 = now_ms();
        ggml_backend_graph_compute(be, &sub);
        const double dt = now_ms() - t0;
        if (part >= 0 && part < 12) { pms[part] += dt; pn[part]++; }
        const std::string k = opkey(t);
        pops[part][k].ms += dt; pops[part][k].n++;
        if (part == 3) {   // the attention core
            const int bk = classify(t);
            r.b[bk] += dt; r.bn[bk]++;
            r.ops[k].ms += dt; r.ops[k].n++;
            ++r.nodes;
            Slow sl;
            sl.ms = dt; sl.op = (int) t->op;
            for (int d = 0; d < 4; ++d) sl.ne[d] = t->ne[d];
            r.slow.push_back(sl);
        }
    }
    r.part = now_ms() - tot0;
    if (r.nodes > 0 && r.n == 0) {
        // a graph whose core interval is empty still needs a phase: fall back to the graph's widest tensor
        for (int i = 0; i < gf->n_nodes; ++i) r.n = std::max<int64_t>(r.n, gf->nodes[i]->ne[3]);
    }
    recs.push_back(std::move(r));
}

/// Times the nodes [i0, i1) of gf one submission each and files them per graph.  Returns their summed wall (ms).
static double run_range(ggml_backend_t be, ggml_cgraph * gf, int i0, int i1) {
    if (!on || i1 <= i0) return 0.0;
    Rec r;
    auto it = notes.find(gf);
    if (it != notes.end()) { r.il = it->second.il; r.n = it->second.n; r.col0 = it->second.col0; }
    for (int i = i0; i < i1; ++i) {
        if (r.n == 0) {
            const ggml_tensor * t = gf->nodes[i];
            if (t->op == GGML_OP_FLASH_ATTN_EXT && t->src[0] != nullptr) r.n = t->src[0]->ne[3];   // Q = [DH, 1, NH, rows]
        }
    }
    const double tot0 = now_ms();
    for (int i = i0; i < i1; ++i) {
        ggml_tensor * t = gf->nodes[i];
        ggml_cgraph sub = *gf;
        sub.nodes = gf->nodes + i; sub.n_nodes = 1;
        sub.leafs = nullptr; sub.n_leafs = 0; sub.grads = nullptr; sub.grad_accs = nullptr;
        const double t0 = now_ms();
        ggml_backend_graph_compute(be, &sub);
        const double dt = now_ms() - t0;
        const int bk = classify(t);
        r.b[bk] += dt; r.bn[bk]++;
        const std::string k = opkey(t);
        r.ops[k].ms += dt; r.ops[k].n++;
        ++r.nodes;
        Slow s;
        s.ms = dt; s.op = (int) t->op;
        for (int d = 0; d < 4; ++d) s.ne[d] = t->ne[d];
        r.slow.push_back(s);
    }
    r.part = now_ms() - tot0;
    recs.push_back(std::move(r));
    return recs.back().part;
}

static void report(const char * tag) {
    if (!on) return;
    if (recs.empty()) return;
    // the enc-skip phases: the first B - replay columns (the wide phase) run layers 0..19 only; the replay window (the
    // narrow phase) runs every layer, so the boundary is the smallest column of a graph that belongs to a layer >= 20
    int64_t cut = (int64_t) 1 << 62;
    for (const Rec & r : recs) if (r.il >= 20 && r.col0 < cut) cut = r.col0;
    if (cut >= ((int64_t) 1 << 62)) cut = 0;
    auto iswide = [&](const Rec & r) { return r.col0 < cut; };
    double all = 0.0;
    long long allnodes = 0;
    for (const Rec & r : recs) { all += r.part; allnodes += r.nodes; }
    std::fprintf(stderr, "\nattention core node prof (%s): %zu core ranges, %lld nodes, %.1f ms; enc-skip boundary at column %lld\n",
                 tag, recs.size(), allnodes, all, (long long) cut);
    for (int ph = 0; ph < 2; ++ph) {
        const char * pn = ph == 0 ? "wide  (first B-replay columns, layers 0-19)" : "narrow (replay window, layers 0-39)";
        double tot = 0.0, ppart = 0.0;
        long long nodes = 0, g = 0, rows = 0;
        double b[B_NB] = {};
        long long bn[B_NB] = {};
        std::map<std::string, Acc> ops;
        std::vector<Slow> slow;
        for (const Rec & r : recs) {
            if (iswide(r) != (ph == 0)) continue;
            ++g; nodes += r.nodes; rows += r.n; ppart += r.part;
            for (int i = 0; i < B_NB; ++i) { b[i] += r.b[i]; bn[i] += r.bn[i]; }
            for (const auto & kv : r.ops) { ops[kv.first].ms += kv.second.ms; ops[kv.first].n += kv.second.n; }
            slow.insert(slow.end(), r.slow.begin(), r.slow.end());
            tot += r.b[0] + r.b[1] + r.b[2] + r.b[3] + r.b[4] + r.b[5] + r.b[6] + r.b[7] + r.b[8];
        }
        if (g == 0) continue;
        std::fprintf(stderr, "\n  -- %s: %lld ranges, %lld rows, %.1f ms (%.1f %% of the core), %.1f nodes/range\n",
                     pn, g, rows, tot, all > 0 ? 100.0 * tot / all : 0.0, (double) nodes / (double) g);
        std::fprintf(stderr, "     node time %.1f ms vs range-submission time %.1f ms -> %.1f ms of submission/launch cost (%.2f %%)\n",
                     tot, ppart, ppart - tot, ppart > 0 ? 100.0 * (ppart - tot) / ppart : 0.0);
        std::fprintf(stderr, "     %-56s %9s %6s %7s\n", "bucket", "ms", "%", "nodes");
        for (int i = 0; i < B_NB; ++i) {
            if (bn[i] == 0) continue;
            std::fprintf(stderr, "     %-56s %9.1f %6.1f %7lld\n", kName[i], b[i], 100.0 * b[i] / tot, bn[i]);
        }
        std::vector<std::pair<std::string, Acc>> v(ops.begin(), ops.end());
        std::sort(v.begin(), v.end(), [](const std::pair<std::string, Acc> & a, const std::pair<std::string, Acc> & c) {
            return a.second.ms > c.second.ms;
        });
        std::fprintf(stderr, "     ops (top 20):\n");
        for (size_t i = 0; i < v.size() && i < 20; ++i) {
            std::fprintf(stderr, "       %-72s %9.1f %6.1f %7lld\n", v[i].first.c_str(), v[i].second.ms,
                         100.0 * v[i].second.ms / tot, v[i].second.n);
        }
        std::sort(slow.begin(), slow.end(), [](const Slow & a, const Slow & c) { return a.ms > c.ms; });
        std::fprintf(stderr, "     slowest nodes:\n");
        for (size_t i = 0; i < slow.size() && i < 8; ++i)
            std::fprintf(stderr, "       %9.3f ms  %-18s [%lld, %lld, %lld, %lld]\n", slow[i].ms,
                         ggml_op_name((ggml_op) slow[i].op), (long long) slow[i].ne[0], (long long) slow[i].ne[1],
                         (long long) slow[i].ne[2], (long long) slow[i].ne[3]);
    }
    // the whole layer graph by part: this is where the indexer's scan and the compressor's projections really live
    std::fprintf(stderr, "\n  -- every node of the layer graph, by part (%.1f ms, one submission per node):\n",
                 [&] { double v = 0; for (int i = 0; i < 12; ++i) v += pms[i]; return v; }());
    for (int p = 0; p < 12; ++p) {
        if (pn[p] == 0) continue;
        std::fprintf(stderr, "     %-28s %9.1f ms  %7lld nodes\n", kPartName[p], pms[p], pn[p]);
        std::vector<std::pair<std::string, Acc>> v(pops[p].begin(), pops[p].end());
        std::sort(v.begin(), v.end(), [](const std::pair<std::string, Acc> & a, const std::pair<std::string, Acc> & c) {
            return a.second.ms > c.second.ms;
        });
        for (size_t i = 0; i < v.size() && i < 6; ++i)
            std::fprintf(stderr, "        %-64s %9.1f %6.1f %7lld\n", v[i].first.c_str(), v[i].second.ms,
                         100.0 * v[i].second.ms / pms[p], v[i].second.n);
    }
    // one CSV line per range so the table can be re-cut by layer / column / span without another run
    std::fprintf(stderr, "\nattention core per-range CSV (%s): il col0 n nodes part_ms gather_ms cast_ms fa_ms idx_ms sel_ms write_ms proj_ms mask_ms other_ms\n", tag);
    for (const Rec & r : recs)
        std::fprintf(stderr, "ACSV %d %lld %lld %lld %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n",
                     r.il, (long long) r.col0, (long long) r.n, r.nodes, r.part, r.b[B_GATHER], r.b[B_CAST], r.b[B_FA],
                     r.b[B_IDX], r.b[B_SEL], r.b[B_WRITE], r.b[B_PROJ], r.b[B_MASK], r.b[B_OTHER]);
    for (int p = 0; p < 12; ++p) { pms[p] = 0; pn[p] = 0; }
    pops.clear();
    recs.clear();
    notes.clear();
}

}  // namespace attnprof

// ================================================================== per-part GPU time of the dense half
// DS41_PART_PROF=1 (diagnostic, off by default): the dense half is ONE ggml graph per layer (attention + router + shared
// expert) submitted as a single call, so the only way to say where its 27 s go is to submit the SAME graph in consecutive
// node RANGES and time each range.  The ranges are bounded by tensors the builders record (partprof::mark), and the node
// order of a committed graph is a topological order, so a cut at tensor T's node separates T's ancestors from the rest:
// every node of a part still runs after its inputs and before its consumers, in the same order, on the same in-order
// queue.  The arithmetic is identical; the cost is one submit+sync per part (this is a probe, not a fast path) and the
// loss of an op fusion that would have straddled a cut.
namespace partprof {

constexpr int kParts = 12;
enum {
    P_NORM = 0, P_QPROJ, P_KVPROJ, P_ATTN, P_OUTPROJ, P_ROUTER, P_SHEXP, P_FINISH, P_HEAD, P_WRITES, P_INIT, P_N
};
static const char * kNames[kParts] = { "hc_pre+attn_norm", "q_a/q_b proj + rope", "kv proj + rope",
                                       "attention core (window+compressed+indexer+FA)", "attn out LoRA + hc_post",
                                       "router", "shared expert", "finish (routed add+hc_post+state cpy)",
                                       "head (LM head + logits)", "writes (cpys)", "init (embedding)" };

// the node probe (DS41_ATTN_PROF) needs the part marks to find its range, so it turns this table on too
static const bool on = std::getenv("DS41_PART_PROF") != nullptr || attnprof::on;
static double ms[kParts] = {};
static int64_t calls[kParts] = {};
static int64_t graphs = 0;
struct Bound { const ggml_tensor * t; int part; };
static std::vector<Bound> bounds;
static int tail_part = P_WRITES;

static void begin(int tail) {
    if (!on) return;
    bounds.clear();
    tail_part = tail;
}
static void mark(ggml_tensor * t, int part) {
    if (!on || t == nullptr) return;
    bounds.push_back(Bound{ t, part });
}

/// Submits gf in the recorded ranges, timing each with the host clock.  ggml_backend_graph_compute is
/// async + synchronize in this fork, so a range's wall time is its device time plus one submit.  A node marked for
/// densegemm is not submitted at all: the range ends before it, its oneMKL GEMM runs into its own buffer, and the next
/// range starts after it (S16).  Inside such a range the two match, so the table still adds up.
static ggml_status run(ggml_backend_t be, ggml_cgraph * gf) {
    auto ms_now = [] { return (double) std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch()).count() / 1e6; };
    std::map<const ggml_tensor *, int> idx;
    for (int i = 0; i < gf->n_nodes; ++i) idx[gf->nodes[i]] = i;
    // the nodes of THIS graph that densegemm takes over (and that are wide enough to be worth it)
    std::vector<char> gemm_node((size_t) gf->n_nodes, 0);
    bool any_gemm = false;
    if (densegemm::active(gf)) {
        for (int i = 0; i < gf->n_nodes; ++i) {
            ggml_tensor * nd = gf->nodes[i];
            if (!densegemm::is_ours(gf, nd) || nd->op != GGML_OP_MUL_MAT) continue;
            const ggml_tensor * x = nd->src[1];
            if (x == nullptr || x->ne[1] < densegemm::kMinRows) continue;
            gemm_node[(size_t) i] = 1;
            any_gemm = true;
        }
    }
    if (!on && !any_gemm) return ggml_backend_graph_compute(be, gf);
    // A committed graph's node order is a topological order, not the builders' order: an output expanded earlier
    // (e.g. the shared expert's cpy, pushed before the router's o_wts) pulls its whole branch in first.  Sort the
    // bounds by node index so each interval is the part that ENDS at it.
    std::vector<std::pair<int, int>> cut;   // (node index of the bound, part)
    for (const Bound & bd : bounds) {
        auto it = idx.find(bd.t);
        if (it != idx.end()) cut.emplace_back(it->second, bd.part);
    }
    std::stable_sort(cut.begin(), cut.end());
    if (attnprof::on) {   // DS41_ATTN_PROF: every node, one submission each, filed by part and by op
        attnprof::run_all(be, gf, cut);
        if (on) ++graphs;
        return GGML_STATUS_SUCCESS;
    }
    ggml_status st = GGML_STATUS_SUCCESS;
    // The probe times each range, so it must wait for it.  The production path does NOT: every range, every densegemm
    // GEMM and every kernel inside them is submitted to the same in-order queue, so the cut costs a submission rather
    // than a submit + drain (~0.2 ms each, 18 per layer = the difference between a win and a wash).  One synchronize
    // at the end keeps the caller's contract (compute() returns with the outputs ready).
    const bool sync_each = on;
    auto submit = [&](int i0, int i1, int part) {
        if (i1 <= i0) return;
        struct ggml_cgraph sub = *gf;
        sub.nodes = gf->nodes + i0;
        sub.n_nodes = i1 - i0;
        sub.leafs = nullptr;
        sub.n_leafs = 0;
        sub.grads = nullptr;
        sub.grad_accs = nullptr;
        const double t0 = sync_each ? ms_now() : 0.0;
        const ggml_status r = sync_each ? ggml_backend_graph_compute(be, &sub)
                                        : ggml_backend_graph_compute_async(be, &sub);
        if (on && part >= 0 && part < kParts) {
            ms[part] += ms_now() - t0;
            calls[part]++;
        }
        if (r != GGML_STATUS_SUCCESS) st = r;
    };
    // one part's interval [i0, i1), cut again at each densegemm node so that node can run on its own
    auto emit = [&](int i0, int i1, int part) {
        int s = i0;
        for (int i = i0; i < i1; ++i) {
            if (!gemm_node[(size_t) i]) continue;
            submit(s, i, part);
            const double t0 = on ? ms_now() : 0.0;
            const bool ran = densegemm::exec(gf, gf->nodes[i]);
            if (on && part >= 0 && part < kParts) { ms[part] += ms_now() - t0; calls[part]++; }
            if (!ran) submit(i, i + 1, part);   // refused: the node runs on ggml after all
            s = i + 1;
        }
        submit(s, i1, part);
    };
    int prev = 0;
    for (const auto & c : cut) {
        if (c.first < prev) continue;
        emit(prev, c.first + 1, c.second);
        prev = c.first + 1;
    }
    emit(prev, gf->n_nodes, tail_part);
    if (!sync_each) ggml_backend_synchronize(be);
    if (on) ++graphs;
    return st;
}

static void report(const char * tag) {
    if (!on || attnprof::on) return;   // the node probe prints its own per-part table instead
    double tot = 0;
    for (int i = 0; i < kParts; ++i) tot += ms[i];
    std::fprintf(stderr, "\ndense part prof (%s): %lld graphs, %.1f ms of submitted GPU time\n", tag, (long long) graphs, tot);
    for (int i = 0; i < kParts; ++i) {
        if (ms[i] <= 0) continue;
        std::fprintf(stderr, "  %-46s %8.1f ms  %5.1f %%  (%lld submits)\n", kNames[i], ms[i],
                     tot > 0 ? 100.0 * ms[i] / tot : 0.0, (long long) calls[i]);
    }
    for (int i = 0; i < kParts; ++i) { ms[i] = 0; calls[i] = 0; }
    graphs = 0;
}

}  // namespace partprof

// DS41_NODEDUMP=<dir> (diagnostic, off by default): after each graph compute, copy EVERY node's tensor to the host and
// append it to <dir>/nodes.bin (raw records) + <dir>/nodes.txt (a readable manifest).  DS41_NODEDUMP_MAXCALLS caps how
// many graphs are dumped (default 4: the embedding graph + the first layer's attn/router and finish graphs).  Nothing is
// written when the variable is unset, and the copy is read-only, so the numbers are unchanged.
static void nodedump_graph(ggml_backend_t be, ggml_cgraph * gf) {
    static const char * dir = std::getenv("DS41_NODEDUMP");
    if (!dir || !dir[0]) return;
    static long long call = 0;
    static const long long maxcalls = std::getenv("DS41_NODEDUMP_MAXCALLS")
                                          ? std::atoll(std::getenv("DS41_NODEDUMP_MAXCALLS")) : 4;
    static FILE * fp = nullptr, * mp = nullptr;
    static std::vector<uint8_t> buf;
    const long long ci = call++;
    if (ci >= maxcalls) return;
    if (!fp) {
        char p[4096];
        std::snprintf(p, sizeof(p), "%s/nodes.bin", dir);
        fp = std::fopen(p, "wb");
        std::snprintf(p, sizeof(p), "%s/nodes.txt", dir);
        mp = std::fopen(p, "w");
        if (!fp || !mp) { std::fprintf(stderr, "nodedump: cannot write to %s\n", dir); return; }
    }
    DPROF_SYNC(be);
    std::map<const ggml_tensor *, int> idx;
    for (int i = 0; i < gf->n_nodes; ++i) idx[gf->nodes[i]] = i;
    // the addresses, so the reader can prove that no two nodes share a buffer (see nodedump_pin below)
    {
        char p[4096];
        std::snprintf(p, sizeof(p), "%s/addrs-%lld.txt", dir, ci);
        FILE * af = std::fopen(p, "w");
        if (af) {
            for (int i = 0; i < gf->n_nodes; ++i)
                std::fprintf(af, "%d %llu %zu\n", i, (unsigned long long) (uintptr_t) gf->nodes[i]->data,
                             ggml_nbytes(gf->nodes[i]));
            std::fclose(af);
        }
    }
    std::fprintf(mp, "# call %lld nodes %d\n", ci, gf->n_nodes);
    for (int i = 0; i < gf->n_nodes; ++i) {
        ggml_tensor * t = gf->nodes[i];
        const size_t nb = ggml_nbytes(t);
        std::fprintf(mp, "%4d %-16s %-20s %-6s ne=[%3lld,%3lld,%3lld,%3lld] src=", i, ggml_op_name(t->op), t->name,
                     ggml_type_name(t->type), (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2],
                     (long long) t->ne[3]);
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (!t->src[s]) break;
            auto it = idx.find(t->src[s]);
            std::fprintf(mp, "%d(%s) ", it == idx.end() ? -1 : it->second, ggml_op_name(t->src[s]->op));
        }
        std::fprintf(mp, "\n");
        const bool plain = t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_BF16 ||
                           t->type == GGML_TYPE_I32 || t->type == GGML_TYPE_I16;
        const bool dumpable = plain && nb <= ((size_t) 32 << 20) && t->buffer != nullptr;
        const int32_t name_len = (int32_t) std::strlen(t->name);
        const int32_t call32 = (int32_t) ci;
        const int64_t ne[4] = { t->ne[0], t->ne[1], t->ne[2], t->ne[3] };
        const int64_t nbytes = dumpable ? (int64_t) nb : -(int64_t) nb;   // negative = header only, no data
        std::fwrite(&call32, 4, 1, fp);
        std::fwrite(&i, 4, 1, fp);
        std::fwrite(&t->op, 4, 1, fp);
        std::fwrite(&t->type, 4, 1, fp);
        std::fwrite(ne, 8, 4, fp);
        std::fwrite(&name_len, 4, 1, fp);
        if (name_len > 0) std::fwrite(t->name, 1, (size_t) name_len, fp);
        std::fwrite(&nbytes, 8, 1, fp);
        if (dumpable) {
            buf.resize(nb);
            DPROF_GET(t, buf.data(), 0, nb);
            std::fwrite(buf.data(), 1, nb, fp);
        }
    }
    std::fflush(fp);
    std::fflush(mp);
}

// DS41_NODEDUMP: mark every node an output before allocation, so the gallocr neither reuses a parent's buffer in place
// nor hands a dead tensor's region to a later node - otherwise a post-compute copy of an early node reads whatever
// aliased it later (nodes 3-6 of layer 0 showed exactly that).  Costs memory: every node gets its own buffer.  The
// arithmetic of every node is unchanged, but disabling in-place reuse changes the allocator's buffer layout, so an
// END-TO-END result may differ from an unpinned run (measured: token 303 vs 223); use node dumps to localise a first
// divergence (sub-ULP, before aliasing can matter), not to characterise late-layer propagation.
static void nodedump_pin(ggml_cgraph * gf) {
    static const char * dir = std::getenv("DS41_NODEDUMP");
    if (!dir || !dir[0]) return;
    for (int i = 0; i < gf->n_nodes; ++i) gf->nodes[i]->flags |= GGML_TENSOR_FLAG_OUTPUT;
}

// DS41_PERTURB_STATE=<rel> (diagnostic, off by default): after the FIRST graph of a pass (the embedding graph) multiply
// every x_state element by 1 + rel * u, u uniform in [-1,1] with a fixed seed, and report the applied max|d|.  This
// measures how much THIS model amplifies a seed-level relative perturbation of the size a backend difference injects -
// i.e. whether a token flip is evidence of a port bug or of the model's own conditioning.
static void perturb_state_if_asked(ggml_tensor * x_state) {
    static const char * rels = std::getenv("DS41_PERTURB_STATE");
    if (!rels || !x_state) return;
    static bool done = false;
    if (done) return;
    done = true;
    const float rel = (float) std::atof(rels);
    std::vector<float> buf((size_t) ggml_nelements(x_state));
    DPROF_GET(x_state, buf.data(), 0, buf.size() * 4);
    std::mt19937 rng(12345u);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    float amax = 0.0f, dmax = 0.0f;
    for (auto & v : buf) {
        const float d = rel * u(rng) * v;
        v += d;
        amax = std::max(amax, std::fabs(v));
        dmax = std::max(dmax, std::fabs(d));
    }
    DPROF_SET(x_state, buf.data(), 0, buf.size() * 4);
    std::fprintf(stderr, "nodedump: perturbed x_state by rel %.3g (max|d| %.4g, max|v| %.4g)\n", (double) rel, (double) dmax,
                 (double) amax);
}

bool Ds41Dense::Impl::compute(ggml_cgraph * gf, bool cached) {
    Allo & a = cached ? allo_dec : allo_pass;
    nodedump_pin(gf);
    const double tq0 = g_dprof_on ? dprof_now() : 0;
    const bool dpo = dprof::g_on;
    const double d0 = dpo ? dprof::us_now() : 0;
    bool ok_alloc = a.alloc(gf, cached && env_on("DS41_GRAPH_REUSE", true));
    const double tq1 = g_dprof_on ? dprof_now() : 0;
    if (!ok_alloc && oom_hook && oom_hook()) {   // something gave VRAM back: once more
        std::fprintf(stderr, "ds41_dense: graph buffer did not fit - freed the tier's prestage region, retrying\n");
        ok_alloc = a.alloc(gf, false);
    }
    const double d1 = dpo ? dprof::us_now() : 0;
    alloc_failed = !ok_alloc;
    if (!ok_alloc) {
        if (dpo) dprof::leaf_add(dprof::C_ALLOC, __LINE__, d1 - d0, 0);
        err = "graph allocation failed";
        return false;
    }
    // DS41_DPROF: the dense path's ONLY graph submission.  In this fork ggml_backend_graph_compute = async +
    // synchronize (src/ggml-backend.cpp:595), so this interval is submit + every kernel + the final stream wait.
    const double d2 = dpo ? dprof::us_now() : 0;
    // DS41_DEC_PROF: the device/launch pair.  The opening barrier goes on the in-order queue right before the
    // submission; the closing one is read after the submission's sync, i.e. when both have completed (see decprof).
    decprof::open_pair();
    const double dw0 = decprof::g_on ? decprof::now_ms() : 0.0;
    // partprof::run is the path when either the part probe or the dense GEMM routing has something to do
    const ggml_status gst = (partprof::on || densegemm::active(gf)) ? partprof::run(backend, gf)
                                                                    : ggml_backend_graph_compute(backend, gf);
    if (decprof::g_on) decprof::close_pair(gf, dw0);
    if (dpo) {
        const double d3 = dprof::us_now();
        dprof::leaf_add(dprof::C_ALLOC, __LINE__, d1 - d0, 0);
        dprof::leaf_add(dprof::C_COMPUTE, __LINE__, d3 - d2, 0);
        dprof::note_graph(gf, d1 - d0, d3 - d2, dprof::graph_buf_bytes(gf));
    }
    if (gst != GGML_STATUS_SUCCESS) { err = "graph compute failed"; return false; }
    nodedump_graph(backend, gf);
    perturb_state_if_asked(x_state);
    if (g_dprof_on && g_dprof_slot >= 0) { g_dprof[1] += tq1 - tq0; g_dprof[2] += dprof_now() - tq1; }
    return true;
}

// ================================================================== the attention graph (+ router + shared expert)

// the model's activation rounding: ds41_ops.hpp's stock-op composite (~45 ops per call), or - built with
// -DSTRATA_DSV41_FQ_OP=ON on a llama.cpp tree patched with tools/v41/ggml/dsv41-fake-quant.patch - the fused
// GGML_OP_DSV41_FAKE_QUANT, one kernel per call (bit-identical on CUDA and CPU: test_ds41_fq; ~3,000 fewer tiny
// kernels per decode token).  STRATA_FQ_COMPOSITE=1 in such a build: the composite.
// get_rows whose result is F16 (f16) or the usual F32.  ggml_get_rows always declares F32; the CUDA kernel writes any
// of F32 / F16 / BF16, dequantizing in float and rounding with the same conversion a later cast would use, so retyping
// the (not yet allocated) result is the gather + cast in one step without the F32 intermediate.  ggml-sycl now writes
// an F16 result too (getrows.cpp, k_get_rows_f16x8: 8 elements per work-item, the same float -> f16 rounding the cpy
// used), so both backends take the one-step form; the tensor is retyped before the graph is allocated, so the F32
// result of the same rows does not exist and neither does the cpy over it.
// DS41_ROWS_F32=1: the pre-S16 form (gather F32 then cpy to F16), for bisecting the F16 get_rows result.
static bool rows_as_f32_path() {
    static const bool v = std::getenv("DS41_ROWS_F32") != nullptr;
    return v;
}

static ggml_tensor * rows_as(ggml_context * gc, ggml_tensor * a, ggml_tensor * idx, bool f16) {
    ggml_tensor * t = ggml_get_rows(gc, a, idx);
    if (f16 && t->type == GGML_TYPE_F32) {
        if (rows_as_f32_path()) return ggml_cast(gc, t, GGML_TYPE_F16);
        t->type = GGML_TYPE_F16;
        t->nb[0] = ggml_type_size(GGML_TYPE_F16);
        for (int i = 1; i < GGML_MAX_DIMS; ++i) t->nb[i] = t->nb[i - 1] * t->ne[i - 1];
    }
    return t;
}

// DS41_ATTN_K16=0: the pre-S16 attention structure (separate window / compressed gathers into F32-concat keys, pad).
static bool attn_k16_enabled() {
    static const bool v = [] {
        const char * e = std::getenv("DS41_ATTN_K16");
        return !(e && e[0] == '0');
    }();
    return v;
}

static ggml_tensor * fake_quant(ggml_context * gc, ggml_tensor * x, Ds41Quant mode) {
#if defined(STRATA_DSV41_FQ_OP)
    static const bool composite = std::getenv("STRATA_FQ_COMPOSITE") != nullptr;
    return composite ? ds41_fake_quant(gc, x, mode) : ggml_dsv41_fake_quant(gc, x, (int) mode);
#else
    return ds41_fake_quant(gc, x, mode);
#endif
}

// DS41_FIX_TOPK_NEG=1 (diagnostic, off by default; S3b TEST 2's causal probe - SUPERSEDED by the kernel fix in
// tools/xolotl/ggml/xolotl-sycl-topk.patch, kept as a cross-check: it must now be a bit-exact no-op on SYCL too).
// ggml-sycl's scan/merge top_k (used for k <= SYCL_TOP_K_SCAN_MERGE_MAX_K == 8, topk-radix.hpp:13; on the short-context
// repro gcap = next_pow2(max(tokens/ratio, 8)) = 8 and k = min(gcap, index_topk=512) = 8) rejected with `val > kth`
// starting at kth == -FLT_MAX, so a -inf score was NEVER inserted; the k <= ncols slots it did not fill kept the
// kernel's `li[...] = -1` initialiser (ggml-sycl.cpp:2723-2727).  ggml-cpu's top_k and ggml-cuda's CUB DeviceTopK both
// return valid row ids for the -inf entries, so -1 is not an index.  The -1s then reached two get_rows: the
// compressed-row gather (Kc, below) and the mask gather (m1 = vis at sel).  With the mask gather reading element (-1)
// the -1 slots did NOT get the -inf they need.  This helper restores the mask the caller asked for (vis at a negative
// id = -inf) without touching the selection, so it is a NO-OP on any backend whose top_k returns valid ids (CPU: sel
// has no -1 -> adds -0.0).  See docs/xolotl-sycl-s3b-topk-report.md.
static ggml_tensor * mask_neg_sel(ggml_context * gc, ggml_tensor * m, ggml_tensor * sel) {
    static const bool on = std::getenv("DS41_FIX_TOPK_NEG") != nullptr;
    if (!on) return m;
    ggml_tensor * s = ggml_cast(gc, sel, GGML_TYPE_F32);
    ggml_tensor * neg = ggml_step(gc, ggml_scale(gc, s, -1.0f));   // -1 -> 1, >= 0 -> 0  (0 * -1e30 == -0.0, no NaN)
    return ggml_add(gc, m, ggml_scale(gc, neg, -1.0e30f));
}

// A batch pass's attention core on the slabs (Impl::slab): every row's window K/V goes into its slot's ring region and
// every query's SWA rows are gathered back in ONE op each; the compressor runs over every row (its state rows and
// compressed / indexer rows written by slab row); the indexer ranks per slot (its own rows and visible count), and the
// chosen rows of every query - its group's top-k, the rest of the longest group's padded with -inf - are gathered in one
// op and attended in ONE flash-attn.  The values are a solo pass's: the ring holds the rows the solo pass concatenates
// (the window's FP8 grid is exact in the ring type), and the -inf padding takes no part in the attention.
static ggml_tensor * attn_core_slab(Ds41Dense::Impl & im, ggml_context * gc, const B & b, int il, int64_t n, ggml_tensor * xn,
                                    ggml_tensor * qr, ggml_tensor * q, ggml_tensor * kv2, ggml_tensor * pos,
                                    std::vector<ggml_tensor *> & outs, std::vector<ggml_tensor *> & late) {
    using L_t = Ds41Dense::Impl::Layer;
    L_t & L = im.ly[(size_t) il];
    const int64_t DH = im.DH, NH = im.NH, SWA = im.SWA;
    const int64_t r = L.ratio;
    const bool f16 = im.f16_gather && 255 * n <= im.zidx->ne[0];
    ggml_tensor * ring = ggml_set_rows(gc, L.raw_all, kv2, b.v1(im.i_wslot, n, 0));
    ggml_tensor * Kw = ggml_reshape_3d(gc, rows_as(gc, ring, b.v1(im.i_widx, SWA * n, 0), f16), DH, SWA, n);
    ggml_tensor * Mw = b.cols2(im.i_wmask, SWA, n, 0);
    ggml_tensor * Kc = nullptr, * Mc = nullptr;
    if (r) {
        ggml_tensor * comp_w = nullptr, * idxk_w = nullptr;
        if (L.kv_src) {
            ggml_tensor * latent = nullptr;
            if (r == 1) {
                latent = ggml_mul_mat(gc, im.BL(il, "attn_compressor_kv.weight"), xn);
            } else {
                ggml_tensor * skv = ggml_mul_mat(gc, im.BL(il, "attn_compressor_kv.weight"), xn);
                ggml_tensor * ssc = ggml_mul_mat(gc, im.BL(il, "attn_compressor_gate.weight"), xn);
                ggml_mul_mat_set_prec(skv, GGML_PREC_F32);
                ggml_mul_mat_set_prec(ssc, GGML_PREC_F32);
                ggml_tensor * src_kv = ggml_concat(gc, L.st_kv_all, skv, 1);                     // [DH, 2 NS + n]
                ggml_tensor * src_sc = ggml_concat(gc, L.st_sc_all, ssc, 1);
                ggml_tensor * gi = b.v1(im.i_sidx, 2 * n, 0);
                ggml_tensor * vk = ggml_reshape_3d(gc, ggml_get_rows(gc, src_kv, gi), DH, 2, n);
                ggml_tensor * vs = ggml_reshape_3d(gc, ggml_get_rows(gc, src_sc, gi), DH, 2, n);
                vk = ggml_cont(gc, ggml_permute(gc, vk, 1, 0, 2, 3));
                vs = ggml_cont(gc, ggml_permute(gc, vs, 1, 0, 2, 3));
                latent = ggml_reshape_2d(gc, ggml_sum_rows(gc, ggml_mul(gc, vk, ggml_soft_max(gc, vs))), DH, n);
                // each slot's state ring: its group's last min(k, 2) rows
                ggml_tensor * col = b.v1(im.i_sscol, im.rows_nss, 0), * dst = b.v1(im.i_sslot, im.rows_nss, 0);
                late.push_back(ggml_set_rows(gc, L.st_kv_all, ggml_get_rows(gc, skv, col), dst));
                late.push_back(ggml_set_rows(gc, L.st_sc_all, ggml_get_rows(gc, ssc, col), dst));
            }
            latent = b.rms_w(latent, im.BL(il, "attn_compressor_norm.weight"));
            ggml_tensor * cpos = b.v1(im.i_cpos[r], n, 0);
            ggml_tensor * cslot = b.v1(im.i_cslot[r], n, 0);   // slab rows
            ggml_tensor * ik = b.rms_w(ggml_mul_mat(gc, im.BL(il, "indexer.attn_k.weight"), latent), im.BL(il, "indexer.k_norm.weight"));
            ik = ggml_reshape_2d(gc, b.rope(ggml_reshape_3d(gc, ik, im.ID, 1, n), cpos, r), im.ID, n);
            if (im.cfg.act_quant) ik = fake_quant(gc, ggml_cont(gc, ik), Ds41Quant::FP4_POW2_32);
            idxk_w = ggml_set_rows(gc, L.idxk_all, ik, cslot);
            outs.push_back(idxk_w);
            ggml_tensor * lr = ggml_reshape_2d(gc, b.rope(ggml_reshape_3d(gc, latent, DH, 1, n), cpos, r), DH, n);
            if (im.cfg.act_quant) lr = fake_quant(gc, ggml_cont(gc, lr), Ds41Quant::FP4_E4M3_16);
            comp_w = ggml_set_rows(gc, L.comp_all, lr, cslot);
            outs.push_back(comp_w);
        }
        int64_t Kmax = 0;
        for (const auto & G : im.rgroups) if (G.cap[r] > 0) Kmax = std::max<int64_t>(Kmax, std::min<int64_t>(G.cap[r], im.ITOP));
        if (Kmax > 0) {
            const L_t & KS = im.ly[(size_t) L.kvl];
            ggml_tensor * comp = comp_w ? comp_w : KS.comp_all;
            ggml_tensor * idxk = idxk_w ? idxk_w : KS.idxk_all;
            const L_t & IS = L.idx_src ? L : im.ly[(size_t) L.idxl];
            if (L.idx_src) {
                ggml_tensor * x_iq = ggml_mul_mat(gc, im.BL(il, "indexer.attn_q_b.weight"), qr);
                x_iq = b.rope(ggml_reshape_3d(gc, x_iq, im.ID, im.IH, n), pos, r);
                if (im.cfg.act_quant) x_iq = fake_quant(gc, ggml_cont(gc, x_iq), Ds41Quant::FP4_POW2_32);
                ggml_tensor * x_iw = ggml_mul_mat(gc, im.BL(il, "indexer.proj.weight"), xn);
                x_iw = ggml_scale(gc, x_iw, 1.0f / std::sqrt((float) (im.ID * im.IH)));
                // every column's selection mask is -inf past its own group's top-k (graph order: before the groups write)
                outs.push_back(ggml_cpy(gc, ggml_fill(gc, ggml_new_tensor_2d(gc, GGML_TYPE_F32, Kmax, n), NEG_INF),
                                        ggml_view_2d(gc, L.selm, Kmax, n, L.selm->nb[1], 0)));
                for (const auto & G : im.rgroups) {
                    const int64_t gcap = G.cap[r], gk = G.k, gc0 = G.c;
                    if (gcap <= 0) continue;
                    const int64_t K = std::min<int64_t>(gcap, im.ITOP);
                    const size_t base = (size_t) G.reg * (size_t) (im.cmax[r] + 1);
                    ggml_tensor * iq = ggml_cont(gc, ggml_view_3d(gc, x_iq, im.ID, im.IH, gk, x_iq->nb[1], x_iq->nb[2], (size_t) gc0 * x_iq->nb[2]));
                    ggml_tensor * iw = ggml_cont(gc, ggml_view_2d(gc, x_iw, x_iw->ne[0], gk, x_iw->nb[1], (size_t) gc0 * x_iw->nb[1]));
                    ggml_tensor * ksrc = ggml_view_3d(gc, idxk, im.ID, 1, gcap, idxk->nb[1], idxk->nb[1], base * idxk->nb[1]);
                    const int64_t qc = im.idx_tile > 0 && gcap * gk > im.idx_tile ? std::max<int64_t>(1, im.idx_tile / gcap) : gk;
                    for (int64_t q0 = 0; q0 < gk; q0 += qc) {   // the query chunks of c8c7076a (one chunk up to a 512K context)
                        const int64_t nq = std::min(qc, gk - q0);
                        const bool wq = q0 == 0 && nq == gk;
                        const size_t c0q = (size_t) (gc0 + q0);
                        ggml_tensor * iq_c = wq ? iq : ggml_view_3d(gc, iq, im.ID, im.IH, nq, iq->nb[1], iq->nb[2], (size_t) q0 * iq->nb[2]);
                        ggml_tensor * iw_c = wq ? iw : ggml_view_2d(gc, iw, im.IH, nq, iw->nb[1], (size_t) q0 * iw->nb[1]);
                        ggml_tensor * nvs = ggml_reshape_2d(gc, b.v1(im.i_nvis[r], nq, c0q * 4), 1, nq);
                        ggml_tensor * ramp = ggml_view_2d(gc, im.neg_ramp, gcap, 1, (size_t) gcap * 4, 0);
                        ggml_tensor * vis = ggml_log(gc, ggml_step(gc, ggml_add(gc, ggml_repeat(gc, ramp, ggml_new_tensor_2d(gc, GGML_TYPE_F32, gcap, nq)), nvs)));
                        ggml_tensor * sc = ggml_lightning_indexer(gc, iq_c, ksrc, iw_c, ggml_cast(gc, vis, GGML_TYPE_F16));
                        sc = ggml_reshape_2d(gc, sc, gcap, nq);
                        if (L.uses_cand) {
                            const int64_t nblk = (gcap + im.CBS - 1) / im.CBS, ncb = std::min<int64_t>(im.CB, nblk);
                            ggml_tensor * m = ggml_fill(gc, ggml_new_tensor_3d(gc, GGML_TYPE_F32, im.CBS, nblk, nq), NEG_INF);
                            ggml_tensor * z = ggml_fill(gc, ggml_new_tensor_3d(gc, GGML_TYPE_F32, im.CBS, ncb, nq), 0.0f);
                            m = ggml_set_rows(gc, m, z, ggml_cont(gc, ggml_view_2d(gc, im.cand, ncb, nq, im.cand->nb[1], c0q * im.cand->nb[1])));
                            m = ggml_view_2d(gc, m, gcap, nq, m->nb[2], 0);
                            sc = ggml_add(gc, sc, m);
                        }
                        if (L.cand_src) {
                            const int64_t nblk = (gcap + im.CBS - 1) / im.CBS, ncb = std::min<int64_t>(im.CB, nblk);
                            GGML_ASSERT(gcap % im.CBS == 0);
                            ggml_tensor * bm = ggml_pool_2d(gc, sc, GGML_OP_POOL_MAX, (int) im.CBS, 1, (int) im.CBS, 1, 0, 0);
                            ggml_tensor * br = ggml_view_2d(gc, im.blk_ramp, nblk, 1, (size_t) nblk * 4, 0);
                            ggml_tensor * lb = ggml_reshape_2d(gc, b.v1(im.i_lastblk, nq, c0q * 4), 1, nq);
                            ggml_tensor * eq = ggml_step(gc, ggml_scale_bias(gc, ggml_abs(gc, ggml_sub(gc, ggml_repeat(gc, br, bm), lb)), -1.0f, 0.5f));
                            bm = ggml_add(gc, bm, ggml_scale(gc, eq, 1e30f));
                            ggml_tensor * cb = ggml_cont(gc, ggml_top_k(gc, bm, (int) ncb));
                            outs.push_back(ggml_cpy(gc, cb, ggml_view_2d(gc, im.cand, ncb, nq, im.cand->nb[1], c0q * im.cand->nb[1])));
                        }
                        ggml_tensor * s1 = ggml_cont(gc, ggml_top_k(gc, sc, (int) K));
                        s1 = ggml_reshape_2d(gc, ggml_get_rows(gc, ggml_reshape_3d(gc, s1, 1, K, nq),
                                                               ggml_argsort(gc, ggml_cast(gc, s1, GGML_TYPE_F32),
                                                                            GGML_SORT_ORDER_ASC)), K, nq);   // canonical order (build_attn)
                        ggml_tensor * m1 = ggml_reshape_2d(gc, ggml_get_rows(gc, ggml_reshape_3d(gc, vis, 1, gcap, nq), s1), K, nq);
                        m1 = mask_neg_sel(gc, m1, s1);   // DS41_FIX_TOPK_NEG probe: see its definition
                        outs.push_back(ggml_cpy(gc, s1, ggml_view_2d(gc, L.sel, K, nq, L.sel->nb[1], c0q * L.sel->nb[1])));
                        outs.push_back(ggml_cpy(gc, m1, ggml_view_2d(gc, L.selm, K, nq, L.selm->nb[1], c0q * L.selm->nb[1])));
                    }
                }
            }
            // every query's rows: its group's top-k (relative to its slot's region) + the region's first slab row, then
            // ONE gather from the slab (the rows past a group's own top-k are masked)
            ggml_tensor * selv = ggml_cont(gc, ggml_view_2d(gc, IS.sel, Kmax, n, IS.sel->nb[1], 0));
            ggml_tensor * self = ggml_add(gc, ggml_cast(gc, selv, GGML_TYPE_F32), ggml_reshape_2d(gc, b.v1(im.i_cbase[r], n, 0), 1, n));
            ggml_tensor * sela = ggml_cast(gc, self, GGML_TYPE_I32);
            Kc = ggml_reshape_3d(gc, rows_as(gc, comp, ggml_reshape_1d(gc, sela, Kmax * n), f16), DH, Kmax, n);
            Mc = ggml_cont(gc, ggml_view_2d(gc, IS.selm, Kmax, n, IS.selm->nb[1], 0));
        }
    }
    ggml_tensor * Kall = Kc ? ggml_concat(gc, Kw, Kc, 1) : Kw;                                     // [DH, M, n]
    ggml_tensor * Mall = Mc ? ggml_concat(gc, Mw, Mc, 0) : Mw;
    int64_t M = Kall->ne[1];
    if (!im.cpu && M % 256) {
        const int64_t pad = 256 - M % 256;
        Kall = f16 ? ggml_concat(gc, Kall, ggml_reshape_3d(gc, rows_as(gc, im.zpad16, ggml_view_1d(gc, im.zidx, pad * n, 0), true),
                                                           DH, pad, n), 1)
             : ggml_pad(gc, Kall, 0, (int) pad, 0, 0);
        Mall = ggml_concat(gc, Mall, ggml_fill(gc, ggml_new_tensor_2d(gc, GGML_TYPE_F32, pad, n), NEG_INF), 0);
        M += pad;
    }
    ggml_tensor * k4 = Kall->type == GGML_TYPE_F16 ? ggml_reshape_4d(gc, Kall, DH, M, 1, n)
                                                   : ggml_cast(gc, ggml_reshape_4d(gc, Kall, DH, M, 1, n), GGML_TYPE_F16);
    ggml_tensor * q4 = ggml_permute(gc, ggml_reshape_4d(gc, q, DH, NH, 1, n), 0, 2, 1, 3);
    ggml_tensor * m4 = ggml_cast(gc, ggml_reshape_4d(gc, Mall, M, 1, 1, n), GGML_TYPE_F16);
    ggml_tensor * o = ggml_flash_attn_ext(gc, q4, k4, k4, m4, 1.0f / std::sqrt((float) DH), 0.0f, 0.0f);
    ggml_flash_attn_ext_add_sinks(o, im.BL(il, "attn_sinks.weight"));
    return ggml_reshape_3d(gc, o, DH, NH, n);
}

static ggml_cgraph * build_attn(Ds41Dense::Impl & im, ggml_context * gc, int il, int64_t n, int64_t col0 = 0) {
    using L_t = Ds41Dense::Impl::Layer;
    L_t & L = im.ly[(size_t) il];
    const Ds41Geometry & G = im.g;
    B b { &im, gc, &G };
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 8192, false);
    densegemm::begin(gf);   // this graph's GEMM-shaped nodes are marked as they are built (densegemm::note)
    attnprof::set_dh(im.DH);   // DS41_ATTN_PROF: classify a cpy as the K cast (DH-wide) or the mask cast (inert)
    attnprof::note(gf, il, n, col0);   // DS41_ATTN_PROF: which layer / columns this graph's core belongs to (inert)
    partprof::begin(partprof::P_WRITES);
    const int64_t D = im.D, HC = im.HC, DH = im.DH, NH = im.NH, SWA = im.SWA, RAW = im.RAW;
    const int64_t r = L.ratio;
    const int64_t cap = r ? im.pass_cap[r] : 0;
    std::vector<ggml_tensor *> outs, late;    // late: ring / state write-backs, after every read
    ggml_tensor * pos = b.v1(im.i_pos, n, (size_t) col0 * 4);

    // ---- Engram (model.py Engram.forward), before the block, on the hc state
    ggml_tensor * x = b.cols3(im.x_state, D, HC, n, col0);
    if (L.eslot >= 0 && im.engram.enabled()) {
        ggml_tensor * rows = b.cols2(im.i_eng[L.eslot], im.EC * im.ED, n, col0);
        ggml_tensor * kv = ggml_mul_mat(gc, im.BL(il, "engram_kv.weight"), rows);          // [(HC+1) D, n]
        ggml_tensor * key = ggml_view_3d(gc, kv, D, HC, n, (size_t) D * 4, kv->nb[1], 0);
        ggml_tensor * val = ggml_view_3d(gc, kv, D, 1, n, (size_t) D * 4, kv->nb[1], (size_t) HC * D * 4);
        ggml_tensor * wqk = ggml_mul(gc, im.BL(il, "engram_q_norm.weight"), im.BL(il, "engram_k_norm.weight"));   // [D, HC]
        ggml_tensor * hn = ggml_rms_norm(gc, x, (float) G.rms_eps);
        ggml_tensor * kn = ggml_rms_norm(gc, ggml_cont(gc, key), (float) G.rms_eps);
        ggml_tensor * dot = ggml_sum_rows(gc, ggml_mul(gc, ggml_mul(gc, hn, wqk), kn));    // [1, HC, n]
        dot = ggml_scale(gc, dot, 1.0f / std::sqrt((float) D));
        // gate = sigmoid(sign(dot) * sqrt(max(|dot|, 1e-6)))  (signed sqrt, as the training kernel)
        ggml_tensor * mag = ggml_sqrt(gc, ggml_clamp(gc, ggml_abs(gc, dot), 1e-6f, INFINITY));
        ggml_tensor * gate = ggml_sigmoid(gc, ggml_mul(gc, ggml_sgn(gc, dot), mag));
        if (im.vision && im.i_img) {   // image positions take no Engram contribution (model.py: gate masked_fill ~token_mask)
            ggml_tensor * m = ggml_reshape_3d(gc, b.v1(im.i_img, n, (size_t) col0 * 4), 1, 1, n);
            gate = ggml_mul(gc, gate, ggml_scale_bias(gc, m, -1.0f, 1.0f));   // gate * (1 - image)
        }
        ggml_tensor * gv = ggml_mul(gc, ggml_repeat(gc, ggml_cont(gc, val), x), gate);
        x = ggml_add(gc, x, gv);
        outs.push_back(ggml_cpy(gc, x, b.cols3(im.x_state, D, HC, n, col0)));   // the block input (residual) = state + Engram
    }

    // ---- attention hc: mixes of the block input; its pre-mix is the previous FFN's (pre_state)
    ggml_tensor * a_pre = nullptr, * a_post = nullptr, * a_comb = nullptr;
    b.hc_mixes(x, im.BL(il, "hc_attn_fn.weight"), im.BL(il, "hc_attn_scale.weight"), im.BL(il, "hc_attn_base.weight"),
               &a_pre, &a_post, &a_comb);
    ggml_tensor * xin = ggml_dsv4_hc_pre(gc, x, b.cols2(im.pre_state, HC, n, col0));             // [D, n]
    ggml_tensor * xn = b.rms_w(xin, im.BL(il, "attn_norm.weight"));

    partprof::mark(xn, partprof::P_NORM);
    // ---- q (no per-head norm in V4.1)
    ggml_tensor * q_a = ggml_mul_mat(gc, im.BL(il, "attn_q_a.weight"), xn);
    densegemm::note(q_a, 'q');
    ggml_tensor * qr = b.rms_w(q_a, im.BL(il, "attn_q_a_norm.weight"));
    ggml_tensor * q_b = ggml_mul_mat(gc, im.BL(il, "attn_q_b.weight"), qr);
    densegemm::note(q_b, 'q');
    ggml_tensor * q = ggml_reshape_3d(gc, q_b, DH, NH, n);
    q = b.rope(q, pos, r);
    partprof::mark(q, partprof::P_QPROJ);

    // ---- the attention core runs per GROUP: the pass's columns of one conversation (one batch slot) on that slot's own
    // state - its window ring, its compressor state, its compressed KV + indexer keys and its visible-row count.  A
    // single-sequence pass (begin / begin_batch spans) is ONE group of every column on the bound slot and builds the very
    // graph it always did (no views of the projections: `whole`); a batch pass (begin_rows) has one group per slot,
    // the projections / router / shared expert batched over all rows around it.
    struct Grp { int slot; int64_t c, k, cap; int64_t wslot_off, sslot_off; };
    std::vector<Grp> grps;
    if (im.rows_mode) {
        for (const auto & rg : im.rgroups) grps.push_back(Grp { rg.slot, rg.c, rg.k, r ? rg.cap[r] : 0, rg.c, 2 * rg.c });
    } else {
        grps.push_back(Grp { im.cur_slot, col0, n, cap, 0, 0 });
    }

    // ---- window K (= V) projection (every row); per group: [ring | the group's rows] -> each query's SWA rows
    ggml_tensor * kv_p = ggml_mul_mat(gc, im.BL(il, "attn_kv.weight"), xn);
    densegemm::note(kv_p, 'k');
    ggml_tensor * kv = b.rms_w(kv_p, im.BL(il, "attn_kv_a_norm.weight"));
    kv = b.rope(ggml_reshape_3d(gc, kv, DH, 1, n), pos, r);
    ggml_tensor * kv2 = ggml_reshape_2d(gc, kv, DH, n);
    if (im.cfg.act_quant) kv2 = fake_quant(gc, ggml_cont(gc, kv2), Ds41Quant::FP8_POW2_32);   // model.py _window_kv
    partprof::mark(kv2, partprof::P_KVPROJ);   // (the compressor's projections below land in the attention core)
    // the compressor's projections (kv source layers), every row: ratio 1 = the latent itself, ratio 2 = the state rows
    ggml_tensor * c_lat = nullptr, * c_skv = nullptr, * c_ssc = nullptr;
    // the indexer's query / weights (index source layers), every row
    ggml_tensor * x_iq = nullptr, * x_iw = nullptr;
    std::vector<ggml_tensor *> o_parts;
    if (im.rows_mode && im.slab) {   // the slabs: one write / gather / flash-attn for every row (attn_core_slab)
        o_parts.push_back(attn_core_slab(im, gc, b, il, n, xn, qr, q, kv2, pos, outs, late));
        grps.clear();
    }
    // a batch pass with several groups: their key sets (window + top-k rows, each padded to the longest) go into ONE
    // flash-attn over every row - per query its own keys either way, so only the -inf padding is added (STRATA_ROWS_FA=0:
    // one flash-attn per group)
    static const bool rows_fa = !(std::getenv("STRATA_ROWS_FA") && std::getenv("STRATA_ROWS_FA")[0] == '0');
    const bool merge_fa = rows_fa && im.rows_mode && grps.size() > 1;
    struct KPart { ggml_tensor * K, * M; int64_t k; bool f16; };
    std::vector<KPart> k_parts;
    for (const Grp & gp : grps) {
        const bool whole = grps.size() == 1 && gp.k == n;   // the single-sequence graph, exactly as before
        const int64_t gk = gp.k, gc0 = gp.c, rel = gp.c - col0;   // rel: the group's first column among this graph's n
        const Ds41Dense::Impl::SlotLayer SL = im.slot_layer(gp.slot, (size_t) il);
        auto colv = [&](ggml_tensor * t2) {   // the group's columns of a [ne0, n] tensor of this graph
            return whole ? t2 : ggml_cont(gc, ggml_view_2d(gc, t2, t2->ne[0], gk, t2->nb[1], (size_t) rel * t2->nb[1]));
        };
        ggml_tensor * gkv2 = colv(kv2);
        // F16 key path for this graph only when the zero index covers its padding (255 pad rows x k queries at most)
        const bool f16 = im.f16_gather && 255 * gk <= im.zidx->ne[0];
        // k16: the one-source key path.  Every row a query can attend - its window rows (the slot's ring and this pass's
        // own rows) and its selected compressed rows - is gathered by ONE get_rows from ONE F16 image of the sources,
        // straight into the [DH, M, k] F16 key tensor the flash-attn reads.  So the F32 gather result, the f32->f16 cpy
        // over it and the two concatenations of the key rows (window+compressed, then the pad) do not exist.  The
        // values are the old path's: an F16 image of the same source rounds each value exactly once, which is what the
        // old gather(f32) + cpy(f32->f16) did (bf16 -> float is exact, an f32 value rounds once), and the pad is fully
        // masked (see the assembly below).
        const bool k16 = f16 && !merge_fa && attn_k16_enabled();
        ggml_tensor * Mw = b.cols2(im.i_wmask, SWA, gk, gc0);
        ggml_tensor * ksrc = nullptr, * kiidx = nullptr;   // k16: the merged F16 source and its row indices
        ggml_tensor * Kw = nullptr;                        // the F32 path's window keys
        if (k16) {
            ksrc = ggml_concat(gc, ggml_cast(gc, SL.raw, GGML_TYPE_F16), ggml_cast(gc, gkv2, GGML_TYPE_F16), 1);   // [DH, RAW + k]
            // The window rows' indices, [SWA, k], column t at t*SWA - the slab's own layout (span_attn_one fills
            // i_widx as widx[t*SWA + j] and uploads it at gc0*SWA*4).  im.i_widx is 1-D, so its nb[1] is the WHOLE
            // slab (SWA*NC*4); the view must take the packed row stride SWA*4, which is what the 1-D view the old
            // path uses (b.v1 below) is.  reshaped 2-D only so the compressed/pad rows can be concatenated per column.
            kiidx = ggml_reshape_2d(gc, b.v1(im.i_widx, SWA * gk, (size_t) gc0 * SWA * 4), SWA, gk);               // [SWA, k]
        } else {
            ggml_tensor * wsrc = ggml_concat(gc, SL.raw->type == GGML_TYPE_F32 ? SL.raw : ggml_cast(gc, SL.raw, GGML_TYPE_F32), gkv2, 1);     // [DH, RAW + k]
            Kw = ggml_reshape_3d(gc, rows_as(gc, wsrc, b.v1(im.i_widx, SWA * gk, (size_t) gc0 * SWA * 4), f16), DH, SWA, gk);
        }
        {
            const int64_t k = std::min<int64_t>(gk, RAW);
            ggml_tensor * last = ggml_view_2d(gc, gkv2, DH, k, gkv2->nb[1], (size_t) (gk - k) * gkv2->nb[1]);
            late.push_back(ggml_set_rows(gc, SL.raw, last, b.v1(im.i_wslot, k, (size_t) gp.wslot_off * 4)));
        }

        // ---- compressed KV (kv sources write it; every ratio > 0 layer reads its source's)
        ggml_tensor * Kc = nullptr, * Mc = nullptr;
        int64_t Ksel = 0;   // k16: how many selected compressed rows the merged index carries
        ggml_tensor * comp_w = nullptr, * idxk_w = nullptr;   // this group's writes (later reads depend on them)
        const int64_t gcap = gp.cap;
        if (r) {
            if (L.kv_src) {
                ggml_tensor * latent = nullptr;                                                  // [DH, k]
                if (r == 1) {   // model.py Compressor.forward: ratio 1 = a plain projection, no gate
                    if (!c_lat) c_lat = ggml_mul_mat(gc, im.BL(il, "attn_compressor_kv.weight"), xn);
                    latent = colv(c_lat);
                } else {        // ratio 2: softmax-gated pooling over the group's 2 states, per dim (fp32)
                    if (!c_skv) {
                        c_skv = ggml_mul_mat(gc, im.BL(il, "attn_compressor_kv.weight"), xn);
                        c_ssc = ggml_mul_mat(gc, im.BL(il, "attn_compressor_gate.weight"), xn);
                        ggml_mul_mat_set_prec(c_skv, GGML_PREC_F32);
                        ggml_mul_mat_set_prec(c_ssc, GGML_PREC_F32);
                    }
                    ggml_tensor * skv = colv(c_skv), * ssc = colv(c_ssc);
                    ggml_tensor * src_kv = ggml_concat(gc, SL.st_kv, skv, 1);                     // [DH, 2 + k]
                    ggml_tensor * src_sc = ggml_concat(gc, SL.st_sc, ssc, 1);
                    ggml_tensor * gi = b.v1(im.i_sidx, 2 * gk, (size_t) gc0 * 2 * 4);
                    ggml_tensor * vk = ggml_reshape_3d(gc, ggml_get_rows(gc, src_kv, gi), DH, 2, gk);
                    ggml_tensor * vs = ggml_reshape_3d(gc, ggml_get_rows(gc, src_sc, gi), DH, 2, gk);
                    vk = ggml_cont(gc, ggml_permute(gc, vk, 1, 0, 2, 3));                       // [2, DH, k]
                    vs = ggml_cont(gc, ggml_permute(gc, vs, 1, 0, 2, 3));
                    latent = ggml_reshape_2d(gc, ggml_sum_rows(gc, ggml_mul(gc, vk, ggml_soft_max(gc, vs))), DH, gk);
                    const int64_t k = std::min<int64_t>(gk, 2);
                    late.push_back(ggml_set_rows(gc, SL.st_kv, ggml_view_2d(gc, skv, DH, k, skv->nb[1], (size_t) (gk - k) * skv->nb[1]),
                                                 b.v1(im.i_sslot, k, (size_t) gp.sslot_off * 4)));
                    late.push_back(ggml_set_rows(gc, SL.st_sc, ggml_view_2d(gc, ssc, DH, k, ssc->nb[1], (size_t) (gk - k) * ssc->nb[1]),
                                                 b.v1(im.i_sslot, k, (size_t) gp.sslot_off * 4)));
                }
                latent = b.rms_w(latent, im.BL(il, "attn_compressor_norm.weight"));
                ggml_tensor * cpos = b.v1(im.i_cpos[r], gk, (size_t) gc0 * 4);
                ggml_tensor * cslot = b.v1(im.i_cslot[r], gk, (size_t) gc0 * 4);
                // indexer keys from the UNROTATED latent (model.py Indexer.forward, owns_k)
                ggml_tensor * ik = b.rms_w(ggml_mul_mat(gc, im.BL(il, "indexer.attn_k.weight"), latent), im.BL(il, "indexer.k_norm.weight"));
                ik = ggml_reshape_2d(gc, b.rope(ggml_reshape_3d(gc, ik, im.ID, 1, gk), cpos, r), im.ID, gk);
                if (im.cfg.act_quant) ik = fake_quant(gc, ggml_cont(gc, ik), Ds41Quant::FP4_POW2_32);
                idxk_w = ggml_set_rows(gc, SL.idxk, ik, cslot);
                outs.push_back(idxk_w);
                ggml_tensor * lr = ggml_reshape_2d(gc, b.rope(ggml_reshape_3d(gc, latent, DH, 1, gk), cpos, r), DH, gk);
                if (im.cfg.act_quant) lr = fake_quant(gc, ggml_cont(gc, lr), Ds41Quant::FP4_E4M3_16);
                comp_w = ggml_set_rows(gc, SL.comp, lr, cslot);
                outs.push_back(comp_w);
            }
            if (gcap > 0) {
                const Ds41Dense::Impl::SlotLayer KS = im.slot_layer(gp.slot, (size_t) L.kvl);
                const int64_t K = std::min<int64_t>(gcap, im.ITOP);
                ggml_tensor * comp = comp_w ? comp_w : KS.comp;
                ggml_tensor * idxk = idxk_w ? idxk_w : KS.idxk;
                ggml_tensor * sel = nullptr, * selm = nullptr;
                if (L.idx_src) {
                    // ---- indexer (model.py Indexer.forward): scores over the visible rows of the shared index keys
                    if (!x_iq) {
                        x_iq = ggml_mul_mat(gc, im.BL(il, "indexer.attn_q_b.weight"), qr);
                        x_iq = b.rope(ggml_reshape_3d(gc, x_iq, im.ID, im.IH, n), pos, r);
                        if (im.cfg.act_quant) x_iq = fake_quant(gc, ggml_cont(gc, x_iq), Ds41Quant::FP4_POW2_32);
                        x_iw = ggml_mul_mat(gc, im.BL(il, "indexer.proj.weight"), xn);
                        x_iw = ggml_scale(gc, x_iw, 1.0f / std::sqrt((float) (im.ID * im.IH)));         // softmax_scale * heads^-0.5
                    }
                    ggml_tensor * iq = whole ? x_iq : ggml_cont(gc, ggml_view_3d(gc, x_iq, im.ID, im.IH, gk, x_iq->nb[1], x_iq->nb[2], (size_t) rel * x_iq->nb[2]));
                    ggml_tensor * iw = colv(x_iw);
                    ggml_tensor * ksrc = ggml_view_3d(gc, idxk, im.ID, 1, gcap, idxk->nb[1], idxk->nb[1], 0);
                    // The selection of the group's queries [q0, q0 + nq).  Its score / mask tensors are [cap, nq]: with
                    // cap = the visible compressed rows (a power of two) they grow with the context - a 1M context's
                    // ratio-1 layers hold ~18 bytes x 2^20 rows x 512 queries = a 9.3 GB graph buffer (2026-10-20: the
                    // 768K / 1M prompts' last batch, the first to run layers 21-39 past 2^19 rows, OOM'd on it).  Every
                    // op here is per query (the indexer kernel's grid is [kv blocks, queries], top-k / argsort / get_rows
                    // are per column), so the span's queries are taken in chunks of at most idx_tile score elements, each
                    // chunk's tensors dying before the next is built.  Up to idx_tile (default: a 512K context) the span
                    // is ONE chunk and the graph is the very one it always was.
                    auto select = [&](int64_t q0, int64_t nq, ggml_tensor ** sel_out, ggml_tensor ** selm_out) {
                        const bool wq = q0 == 0 && nq == gk;
                        const size_t c0q = (size_t) (gc0 + q0);
                        ggml_tensor * iq_c = wq ? iq : ggml_view_3d(gc, iq, im.ID, im.IH, nq, iq->nb[1], iq->nb[2], (size_t) q0 * iq->nb[2]);
                        ggml_tensor * iw_c = wq ? iw : ggml_view_2d(gc, iw, im.IH, nq, iw->nb[1], (size_t) q0 * iw->nb[1]);
                        // visibility: row i visible to query t iff i < nvis_t  -> 0 / -inf
                        ggml_tensor * nvs = ggml_reshape_2d(gc, b.v1(im.i_nvis[r], nq, c0q * 4), 1, nq);
                        ggml_tensor * ramp = ggml_view_2d(gc, im.neg_ramp, gcap, 1, (size_t) gcap * 4, 0);
                        ggml_tensor * vis = ggml_log(gc, ggml_step(gc, ggml_add(gc, ggml_repeat(gc, ramp, ggml_new_tensor_2d(gc, GGML_TYPE_F32, gcap, nq)), nvs)));
                        ggml_tensor * sc = ggml_lightning_indexer(gc, iq_c, ksrc, iw_c, ggml_cast(gc, vis, GGML_TYPE_F16));   // [cap, nq]
                        sc = ggml_reshape_2d(gc, sc, gcap, nq);
                        if (L.uses_cand) {   // rank only inside cand_layer's candidate blocks
                            const int64_t nblk = (gcap + im.CBS - 1) / im.CBS, ncb = std::min<int64_t>(im.CB, nblk);
                            ggml_tensor * m = ggml_fill(gc, ggml_new_tensor_3d(gc, GGML_TYPE_F32, im.CBS, nblk, nq), NEG_INF);
                            ggml_tensor * z = ggml_fill(gc, ggml_new_tensor_3d(gc, GGML_TYPE_F32, im.CBS, ncb, nq), 0.0f);
                            m = ggml_set_rows(gc, m, z, ggml_cont(gc, ggml_view_2d(gc, im.cand, ncb, nq, im.cand->nb[1], c0q * im.cand->nb[1])));
                            m = ggml_view_2d(gc, m, gcap, nq, m->nb[2], 0);
                            sc = ggml_add(gc, sc, m);
                        }
                        if (L.cand_src) {   // model.py select_candidate_blocks: block score = max, newest block pinned
                            const int64_t nblk = (gcap + im.CBS - 1) / im.CBS, ncb = std::min<int64_t>(im.CB, nblk);
                            GGML_ASSERT(gcap % im.CBS == 0);
                            ggml_tensor * bm = ggml_pool_2d(gc, sc, GGML_OP_POOL_MAX, (int) im.CBS, 1, (int) im.CBS, 1, 0, 0);   // [nblk, nq]
                            // pin: + 1e30 at block (nvis - 1) / CBS of each query
                            ggml_tensor * br = ggml_view_2d(gc, im.blk_ramp, nblk, 1, (size_t) nblk * 4, 0);
                            ggml_tensor * lb = ggml_reshape_2d(gc, b.v1(im.i_lastblk, nq, c0q * 4), 1, nq);
                            ggml_tensor * eq = ggml_step(gc, ggml_scale_bias(gc, ggml_abs(gc, ggml_sub(gc, ggml_repeat(gc, br, bm), lb)), -1.0f, 0.5f));
                            bm = ggml_add(gc, bm, ggml_scale(gc, eq, 1e30f));
                            ggml_tensor * cb = ggml_cont(gc, ggml_top_k(gc, bm, (int) ncb));                     // I32 [ncb, nq]
                            outs.push_back(ggml_cpy(gc, cb, ggml_view_2d(gc, im.cand, ncb, nq, im.cand->nb[1], c0q * im.cand->nb[1])));
                        }
                        ggml_tensor * s1 = ggml_cont(gc, ggml_top_k(gc, sc, (int) K));                          // I32 [K, nq]
                        // Canonicalise the top-k order.  ggml_cuda_op_top_k asks CUB's DeviceTopK for
                        // output_ordering::unsorted, and CUB hands the k ids back in an order its atomically assigned
                        // output slots produce - run-to-run unstable, while the SET is stable (measured: order-only on
                        // every differing row).  The ids drive the gather of the compressed rows, so that order is the
                        // flash-attn's accumulation order; left alone the whole run stops being reproducible from the
                        // first query whose compressed rows are attended (the SWA window's 128).  Sort ascending.
                        s1 = ggml_reshape_2d(gc, ggml_get_rows(gc, ggml_reshape_3d(gc, s1, 1, K, nq),
                                                               ggml_argsort(gc, ggml_cast(gc, s1, GGML_TYPE_F32),
                                                                            GGML_SORT_ORDER_ASC)), K, nq);
                        ggml_tensor * m1 = ggml_reshape_2d(gc, ggml_get_rows(gc, ggml_reshape_3d(gc, vis, 1, gcap, nq), s1), K, nq);   // vis at sel
                        m1 = mask_neg_sel(gc, m1, s1);   // DS41_FIX_TOPK_NEG probe: see its definition
                        // the chunk's writes go to outs now, so the graph runs (and frees) this chunk before the next
                        outs.push_back(ggml_cpy(gc, s1, ggml_view_2d(gc, L.sel, K, nq, L.sel->nb[1], c0q * L.sel->nb[1])));
                        outs.push_back(ggml_cpy(gc, m1, ggml_view_2d(gc, L.selm, K, nq, L.selm->nb[1], c0q * L.selm->nb[1])));
                        *sel_out = s1;
                        *selm_out = m1;
                    };
                    const int64_t qc = im.idx_tile > 0 && gcap * gk > im.idx_tile ? std::max<int64_t>(1, im.idx_tile / gcap) : gk;
                    for (int64_t q0 = 0; q0 < gk; q0 += qc) {
                        ggml_tensor * s1 = nullptr, * m1 = nullptr;
                        select(q0, std::min(qc, gk - q0), &s1, &m1);
                        sel = sel ? ggml_concat(gc, sel, s1, 1) : s1;      // [K, k] for the gather below
                        selm = selm ? ggml_concat(gc, selm, m1, 1) : m1;
                    }
                } else {
                    const L_t & IS = im.ly[(size_t) L.idxl];
                    sel = ggml_view_2d(gc, IS.sel, K, gk, IS.sel->nb[1], (size_t) gc0 * IS.sel->nb[1]);
                    selm = ggml_view_2d(gc, IS.selm, K, gk, IS.selm->nb[1], (size_t) gc0 * IS.selm->nb[1]);
                }
                if (k16) {   // the compressed rows join the same F16 source (the old path gathered them separately)
                    // `sel` indexes the compressed pool from ITS row 0 (that is what the separate gather it replaces
                    // read); in the merged source the pool starts after the window rows, so the selection carries that
                    // base.  F32 holds these row ids exactly (a context's row count is far under 2^24) and the round
                    // trip through F32 is the pattern attn_core_slab uses for the same job.
                    const float base = (float) ksrc->ne[1];   // = the rows of [ring | this pass's rows]
                    ggml_tensor * selo = ggml_cast(gc, ggml_scale_bias(gc, ggml_cast(gc, ggml_cont(gc, sel), GGML_TYPE_F32),
                                                                      1.0f, base), GGML_TYPE_I32);
                    ksrc = ggml_concat(gc, ksrc, ggml_cast(gc, comp, GGML_TYPE_F16), 1);
                    kiidx = ggml_concat(gc, kiidx, selo, 0);   // (sel may be a strided view of a sel slab)
                    Ksel = K;
                } else {
                    Kc = ggml_reshape_3d(gc, rows_as(gc, comp, ggml_cont(gc, ggml_reshape_1d(gc, ggml_cont(gc, sel), K * gk)), f16), DH, K, gk);
                }
                Mc = ggml_cont(gc, selm);
                // the source's rows must be read AFTER this pass wrote its own (a kv source attends to its new row)
            }
        }

        // ---- attention: per query its window rows + its top-k compressed rows, one flash-attn with the query as batch
        ggml_tensor * Mall = Mc ? ggml_concat(gc, Mw, Mc, 0) : Mw;                                 // [M, k]
        int64_t M = Mw->ne[0] + (Mc ? Ksel : 0);
        ggml_tensor * k4 = nullptr;
        if (k16) {
            // the key count is completed to a multiple of 256 (the flash-attn's KV_max scan works in 256-key blocks).
            // The pad rows come from the same gather, indexed at the source's row 0, and their mask is -inf: the
            // flash-attn's pad terms are exactly exp(-inf) = 0, so what those keys hold cannot change the output (the
            // md5 gate below is the check) - they are not gathered, cast and concatenated as their own 128x512 rows.
            const int64_t pad = M % 256 ? 256 - M % 256 : 0;
            if (pad > 0) {
                kiidx = ggml_concat(gc, kiidx, ggml_view_2d(gc, im.zidx, pad, gk, (size_t) pad * 4, 0), 0);   // = 0
                Mall = ggml_concat(gc, Mall, ggml_fill(gc, ggml_new_tensor_2d(gc, GGML_TYPE_F32, pad, gk), NEG_INF), 0);
                M += pad;
            }
            ggml_tensor * kf = rows_as(gc, ksrc, ggml_reshape_1d(gc, kiidx, M * gk), true);             // [DH, M * k]
            k4 = ggml_reshape_4d(gc, kf, DH, M, 1, gk);
        } else {
            ggml_tensor * Kall = Kc ? ggml_concat(gc, Kw, Kc, 1) : Kw;                             // [DH, M, k]
            if (merge_fa) { k_parts.push_back(KPart { Kall, Mall, gk, f16 }); continue; }
            M = Kall->ne[1];
            if (!im.cpu && M % 256) {   // ggml-cuda's flash-attn for d 512 wants the key count a multiple of 256
                const int64_t pad = 256 - M % 256;
                // F16 has no CUDA pad: concat zeros gathered from the one zero row (F16 -> F16, exact) - the same zeros
                Kall = f16 ? ggml_concat(gc, Kall, ggml_reshape_3d(gc, rows_as(gc, im.zpad16, ggml_view_1d(gc, im.zidx, pad * gk, 0), true),
                                                                   DH, pad, gk), 1)
                     : ggml_pad(gc, Kall, 0, (int) pad, 0, 0);
                Mall = ggml_concat(gc, Mall, ggml_fill(gc, ggml_new_tensor_2d(gc, GGML_TYPE_F32, pad, gk), NEG_INF), 0);
                M += pad;
            }
            k4 = Kall->type == GGML_TYPE_F16 ? ggml_reshape_4d(gc, Kall, DH, M, 1, gk)
                                             : ggml_cast(gc, ggml_reshape_4d(gc, Kall, DH, M, 1, gk), GGML_TYPE_F16);
        }
        if (std::getenv("DS41_FA_TRACE")) {   // bisect aid: the exact flash-attn inputs of the first few cores
            static int n_tr = 0;
            if (n_tr < 8) {
                ++n_tr;
                std::fprintf(stderr, "[FA-TRACE] #%d il=%d k16=%d gk=%lld M=%lld Ksel=%lld gcap=%lld SWA=%lld RAW=%lld "
                        "Kne=%lld,%lld,%lld,%lld Knb=%zu,%zu,%zu,%zu Ktype=%s Knb=%.1fMB maskne=%lld,%lld,%lld,%lld "
                        "ksrcrows=%lld\n",
                        n_tr, il, (int) k16, (long long) gk, (long long) M, (long long) Ksel, (long long) gcap,
                        (long long) SWA, (long long) RAW,
                        (long long) k4->ne[0], (long long) k4->ne[1], (long long) k4->ne[2], (long long) k4->ne[3],
                        k4->nb[0], k4->nb[1], k4->nb[2], k4->nb[3], ggml_type_name(k4->type), ggml_nbytes(k4) / 1048576.0,
                        (long long) Mall->ne[0], (long long) Mall->ne[1], (long long) Mall->ne[2], (long long) Mall->ne[3],
                        (long long) (ksrc ? ksrc->ne[1] : 0));
            }
        }
        ggml_tensor * gq = whole ? q : ggml_cont(gc, ggml_view_3d(gc, q, DH, NH, gk, q->nb[1], q->nb[2], (size_t) rel * q->nb[2]));
        ggml_tensor * q4 = ggml_permute(gc, ggml_reshape_4d(gc, gq, DH, NH, 1, gk), 0, 2, 1, 3);    // [DH, 1, NH, k]
        ggml_tensor * m4 = ggml_cast(gc, ggml_reshape_4d(gc, Mall, M, 1, 1, gk), GGML_TYPE_F16);
        ggml_tensor * og = ggml_flash_attn_ext(gc, q4, k4, k4, m4, 1.0f / std::sqrt((float) DH), 0.0f, 0.0f);
        ggml_flash_attn_ext_add_sinks(og, im.BL(il, "attn_sinks.weight"));
        o_parts.push_back(ggml_reshape_3d(gc, og, DH, NH, gk));
    }
    if (merge_fa) {
        int64_t Mx = 0;
        for (const KPart & kp : k_parts) Mx = std::max<int64_t>(Mx, kp.K->ne[1]);
        if (!im.cpu && Mx % 256) Mx += 256 - Mx % 256;   // ggml-cuda's flash-attn for d 512: a multiple of 256 keys
        ggml_tensor * Kx = nullptr, * Mxk = nullptr;
        for (const KPart & kp : k_parts) {
            ggml_tensor * K = kp.K, * Mk = kp.M;
            const int64_t pad = Mx - K->ne[1];
            if (pad > 0) {
                K = kp.f16 ? ggml_concat(gc, K, ggml_reshape_3d(gc, rows_as(gc, im.zpad16, ggml_view_1d(gc, im.zidx, pad * kp.k, 0), true),
                                                                DH, pad, kp.k), 1)
                           : ggml_pad(gc, K, 0, (int) pad, 0, 0);
                Mk = ggml_concat(gc, Mk, ggml_fill(gc, ggml_new_tensor_2d(gc, GGML_TYPE_F32, pad, kp.k), NEG_INF), 0);
            }
            if (Kx && K->type != Kx->type) K = ggml_cast(gc, K, Kx->type);
            Kx = Kx ? ggml_concat(gc, Kx, K, 2) : K;                                               // [DH, Mx, n]
            Mxk = Mxk ? ggml_concat(gc, Mxk, Mk, 1) : Mk;                                          // [Mx, n]
        }
        ggml_tensor * k4 = Kx->type == GGML_TYPE_F16 ? ggml_reshape_4d(gc, Kx, DH, Mx, 1, n)
                                                     : ggml_cast(gc, ggml_reshape_4d(gc, Kx, DH, Mx, 1, n), GGML_TYPE_F16);
        ggml_tensor * q4 = ggml_permute(gc, ggml_reshape_4d(gc, q, DH, NH, 1, n), 0, 2, 1, 3);      // [DH, 1, NH, n]
        ggml_tensor * m4 = ggml_cast(gc, ggml_reshape_4d(gc, Mxk, Mx, 1, 1, n), GGML_TYPE_F16);
        ggml_tensor * og = ggml_flash_attn_ext(gc, q4, k4, k4, m4, 1.0f / std::sqrt((float) DH), 0.0f, 0.0f);
        ggml_flash_attn_ext_add_sinks(og, im.BL(il, "attn_sinks.weight"));
        o_parts.push_back(ggml_reshape_3d(gc, og, DH, NH, n));
    }
    ggml_tensor * o = o_parts[0];
    for (size_t i = 1; i < o_parts.size(); ++i) o = ggml_concat(gc, o, o_parts[i], 2);         // [DH, NH, n]
    partprof::mark(o, partprof::P_ATTN);   // the attention core's output, before the de-RoPE / output LoRA
    o = b.rope(o, pos, r, true);                                                               // de-RoPE the output
    // grouped output LoRA (model.py: einsum over groups) then wo_b
    o = ggml_permute(gc, ggml_reshape_3d(gc, o, im.OGD, im.OG, n), 0, 2, 1, 3);               // [OGD, n, OG]
    ggml_tensor * wo_a = ggml_reshape_3d(gc, im.BL(il, "attn_output_a.weight"), im.OGD, im.OL, im.OG);
    ggml_tensor * oa_m = ggml_mul_mat(gc, wo_a, o);                                         // [OL, n, OG]
    densegemm::note(oa_m, 'o');
    ggml_tensor * oa = ggml_permute(gc, oa_m, 0, 2, 1, 3);                                   // [OL, OG, n]
    ggml_tensor * attn = ggml_mul_mat(gc, im.BL(il, "attn_output_b.weight"), b.flat2(oa, im.OL * im.OG, n));   // [D, n]
    densegemm::note(attn, 'o');

    // ---- hc_post(attention) -> FFN hc: its mixes give the next layer's pre-mix; its input uses the attention's
    ggml_tensor * h = ggml_dsv4_hc_post(gc, attn, x, a_post, a_comb);                       // [D, HC, n]
    partprof::mark(h, partprof::P_OUTPROJ);
    ggml_tensor * f_pre = nullptr, * f_post = nullptr, * f_comb = nullptr;
    b.hc_mixes(h, im.BL(il, "hc_ffn_fn.weight"), im.BL(il, "hc_ffn_scale.weight"), im.BL(il, "hc_ffn_base.weight"),
               &f_pre, &f_post, &f_comb);
    ggml_tensor * fn = b.rms_w(ggml_dsv4_hc_pre(gc, h, a_pre), im.BL(il, "ffn_norm.weight"));   // [D, n]

    // ---- router (model.py Gate): sqrt(softplus), top-k on score + bias, weights from the raw scores
    ggml_tensor * rl = ggml_mul_mat(gc, im.BL(il, "ffn_gate_inp.weight"), fn);
    ggml_mul_mat_set_prec(rl, GGML_PREC_F32);
    densegemm::note(rl, 'r');
    ggml_tensor * sc = ggml_sqrt(gc, ggml_softplus(gc, rl));                                  // [NE, n]
    // selection = score + exp_probs_b + the selection-only rbias (soft prune / cache bonus); the weights below come
    // from the unbiased scores, as model.py's Gate - all-zero rbias (the default) = the model's routing bit for bit
    // vision: image positions select with exp_probs_b_vl.bias instead (model.py Gate: torch.where(image_mask, bias_vl, bias))
    ggml_tensor * bias_sel = im.BL(il, "exp_probs_b.bias");
    if (im.vision && im.i_img) {
        ggml_tensor * bt = ggml_reshape_2d(gc, im.BL(il, "exp_probs_b.bias"), im.NE, 1);
        ggml_tensor * bv = ggml_reshape_2d(gc, im.BL(il, "exp_probs_b_vl.bias"), im.NE, 1);
        ggml_tensor * m = ggml_reshape_2d(gc, b.v1(im.i_img, n, (size_t) col0 * 4), 1, n);
        bias_sel = ggml_add(gc, ggml_mul(gc, ggml_repeat(gc, ggml_sub(gc, bv, bt), sc), m), bt);
    }
    ggml_tensor * s0 = ggml_add(gc, sc, bias_sel);                                             // the model's own selection score
    ggml_tensor * rb = ggml_reshape_2d(gc, im.ly[(size_t) il].rbias, im.NE, 1);
    ggml_tensor * sel = nullptr;
    // SMART route bias (opt-in; the flat bias changes ~every choice it can: -14 pt API top-1, repetition loops):
    //   STRATA_RB_PROTECT=P  the exact top-P experts are always kept (+1e4); the bias only competes for the other slots
    //   STRATA_RB_MARGIN=d   the bias only applies to experts whose exact score is within d of the cutoff (midpoint of the
    //                        exact NU-th / NU+1-th scores); experts confidently above it (> cut + d) are always kept -
    //                        only near-ties can flip
    //   STRATA_RB_POOL=M     the bias only moves selections inside the exact top-M; every pick stays a top-M expert
    //                        (colibri's CACHE_ROUTE max-rank window, arXiv:2412.00099: PROTECT=2 POOL=12 there)
    // Both may be set.  Weights still come from the unbiased scores.  Unset = the flat bias, bit for bit as before.
    // PROTECT / POOL (and STRATA_RB_SMART=1) build them from rknob: P / M then change per request (set_route_select),
    // 0 = off, and P = M = 0 selects exactly as the flat bias.
    static const float rb_margin = std::getenv("STRATA_RB_MARGIN") ? (float) std::atof(std::getenv("STRATA_RB_MARGIN")) : -1.0f;
    ggml_tensor * xids = nullptr;   // the exact top-NU (route meter)
    if (smart_router() || rb_margin >= 0.0f) {
        const int kt = smart_router() ? kRK : (int) im.NU + 1;
        ggml_tensor * ti = ggml_argsort_top_k(gc, s0, kt);                                         // [kt, n]
        ggml_tensor * tv = ggml_reshape_2d(gc, ggml_get_rows(gc, ggml_reshape_3d(gc, s0, 1, im.NE, n), ti), kt, n);   // [kt, n] desc
        auto row = [&](int r) { return ggml_view_2d(gc, tv, 1, n, tv->nb[1], (size_t) r * tv->nb[0]); };   // [1, n]
        ggml_tensor * like = ggml_new_tensor_2d(gc, GGML_TYPE_F32, im.NE, n);
        ggml_tensor * protect = nullptr, * pool = nullptr;
        if (smart_router()) {
            xids = ggml_cont(gc, ggml_view_2d(gc, ti, im.NU, n, ti->nb[1], 0));
            // rknob column c = a one-hot over the kRK exact ranks + an offset: the P-th / M-th exact score, or +-1e30 for
            // rank 0 (protect nothing / pool everything).  mul + sum_rows, not mul_mat: one nonzero term, exact in F32.
            auto thr = [&](int c) {
                ggml_tensor * oh = ggml_view_2d(gc, im.rknob, kRK, 1, im.rknob->nb[1], (size_t) c * im.rknob->nb[1]);
                ggml_tensor * off = ggml_view_2d(gc, im.rknob, 1, 1, im.rknob->nb[1], (size_t) c * im.rknob->nb[1] + (size_t) kRK * 4);
                return ggml_repeat(gc, ggml_add(gc, ggml_sum_rows(gc, ggml_mul(gc, tv, oh)), off), like);
            };
            protect = ggml_step(gc, ggml_scale_bias(gc, ggml_sub(gc, s0, thr(0)), 1.0f, 1e-6f));   // s0 >= the P-th exact score
            pool = ggml_step(gc, ggml_scale_bias(gc, ggml_sub(gc, s0, thr(1)), 1.0f, 1e-6f));      // s0 >= the M-th
        }
        ggml_tensor * gate = nullptr;
        if (rb_margin >= 0.0f) {
            ggml_tensor * cut = ggml_repeat(gc, ggml_scale(gc, ggml_add(gc, ggml_cont(gc, row((int) im.NU - 1)), ggml_cont(gc, row((int) im.NU))), 0.5f), like);
            ggml_tensor * dv = ggml_sub(gc, s0, cut);                                              // [NE, n]
            ggml_tensor * above = ggml_step(gc, ggml_scale_bias(gc, dv, 1.0f, -rb_margin));          // dv > d
            ggml_tensor * below = ggml_step(gc, ggml_scale_bias(gc, dv, -1.0f, -rb_margin));         // dv < -d
            protect = protect ? ggml_clamp(gc, ggml_add(gc, protect, above), 0.0f, 1.0f) : above;
            gate = ggml_scale_bias(gc, ggml_clamp(gc, ggml_add(gc, above, below), 0.0f, 1.0f), -1.0f, 1.0f);   // in the band
        }
        ggml_tensor * unprot = ggml_scale_bias(gc, protect, -1.0f, 1.0f);
        if (gate) unprot = ggml_mul(gc, unprot, gate);
        if (pool) unprot = ggml_mul(gc, unprot, pool);
        sel = ggml_add(gc, ggml_add(gc, s0, ggml_mul(gc, ggml_repeat(gc, rb, like), unprot)), ggml_scale(gc, protect, 1e4f));
        if (pool) sel = ggml_add(gc, sel, ggml_scale_bias(gc, pool, 1e4f, -1e4f));   // outside the pool: -1e4 (pool = 1: exactly 0)
    } else {
        sel = ggml_add(gc, s0, rb);
    }
    ggml_tensor * ids = ggml_argsort_top_k(gc, sel, (int) im.NU);   // [NU, n]
    ggml_tensor * wts = ggml_reshape_2d(gc, ggml_get_rows(gc, ggml_reshape_3d(gc, sc, 1, im.NE, n), ids), im.NU, n);
    if (G.norm_topk) wts = ggml_div(gc, wts, ggml_scale_bias(gc, ggml_sum_rows(gc, wts), 1.0f, 1e-20f));
    wts = ggml_scale(gc, wts, (float) G.weights_scale);
    partprof::mark(wts, partprof::P_ROUTER);

    // ---- shared expert (model.py Expert, clamped SwiGLU)
    const float lim = (float) G.swiglu_limit;
    ggml_tensor * up = ggml_mul_mat(gc, im.BL(il, "ffn_up_shexp.weight"), fn);
    densegemm::note(up, 's');
    ggml_tensor * gt = ggml_mul_mat(gc, im.BL(il, "ffn_gate_shexp.weight"), fn);
    densegemm::note(gt, 's');
    if (lim > 0) {
        up = ggml_clamp(gc, up, -lim, lim);
        gt = ggml_clamp(gc, gt, NEG_INF, lim);
    }
    ggml_tensor * se = ggml_mul_mat(gc, im.BL(il, "ffn_down_shexp.weight"), ggml_swiglu_split(gc, gt, up));
    densegemm::note(se, 's');
    partprof::mark(se, partprof::P_SHEXP);

    outs.push_back(ggml_cpy(gc, h, b.cols3(im.hap, D, HC, n, col0)));
    outs.push_back(ggml_cpy(gc, f_post, b.cols2(im.post_f, HC, n, col0)));
    outs.push_back(ggml_cpy(gc, ggml_reshape_3d(gc, f_comb, HC, HC, n), b.cols3(im.comb_f, HC, HC, n, col0)));
    outs.push_back(ggml_cpy(gc, se, b.cols2(im.shexp, D, n, col0)));
    outs.push_back(ggml_cpy(gc, fn, b.cols2(im.o_fn, D, n, col0)));
    outs.push_back(ggml_cpy(gc, ids, b.cols2(im.o_ids, im.NU, n, col0)));
    outs.push_back(ggml_cpy(gc, wts, b.cols2(im.o_wts, im.NU, n, col0)));
    if (xids) {   // route meter: the exact top-NU and their normalized weights
        ggml_tensor * xw = ggml_reshape_2d(gc, ggml_get_rows(gc, ggml_reshape_3d(gc, sc, 1, im.NE, n), xids), im.NU, n);
        xw = ggml_div(gc, xw, ggml_scale_bias(gc, ggml_sum_rows(gc, xw), 1.0f, 1e-20f));
        outs.push_back(ggml_cpy(gc, xids, b.cols2(im.o_xids, im.NU, n, col0)));
        outs.push_back(ggml_cpy(gc, xw, b.cols2(im.o_xwts, im.NU, n, col0)));
    }
    outs.push_back(ggml_cpy(gc, f_pre, b.cols2(im.pre_state, HC, n, col0)));   // reads of pre_state happen above
    for (ggml_tensor * t : outs) ggml_build_forward_expand(gf, t);
    for (ggml_tensor * t : late) ggml_build_forward_expand(gf, t);
    return gf;
}

// enc-skip: the persistent-KV half of build_attn for one span.  Runs hc_pre / attn_norm, writes the raw window K/V
// ring, and on a kv source layer writes the compressed-KV row + indexer-key row (+ the ratio-2 compressor state).
// Everything else build_attn does (q, the attention + output LoRA, hc_post, the router, the shared expert, the indexer
// selection / candidate blocks, and the pre_state / hap / shexp / o_* writes) only ever flows into THIS layer's own
// output columns, so a prompt token whose later layers are never read does not need them.  The state it does write is
// exactly what build_attn writes for those positions, so a later query's window / compressed reads see the same
// values as the full path (verified by the enc-skip gate).
static ggml_cgraph * build_attn_kv_only(Ds41Dense::Impl & im, ggml_context * gc, int il, int64_t n, int64_t col0 = 0) {
    using L_t = Ds41Dense::Impl::Layer;
    L_t & L = im.ly[(size_t) il];
    const Ds41Geometry & G = im.g;
    partprof::begin(partprof::P_INIT);
    B b { &im, gc, &G };
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 8192, false);
    const int64_t D = im.D, HC = im.HC, DH = im.DH;
    const int64_t r = L.ratio;
    std::vector<ggml_tensor *> outs, late;   // late: the ring / state write-backs, after every read
    ggml_tensor * pos = b.v1(im.i_pos, n, (size_t) col0 * 4);

    ggml_tensor * x = b.cols3(im.x_state, D, HC, n, col0);
    if (L.eslot >= 0 && im.engram.enabled()) {   // the Engram add still feeds the K/V + compressor projections
        ggml_tensor * rows = b.cols2(im.i_eng[L.eslot], im.EC * im.ED, n, col0);
        ggml_tensor * kv = ggml_mul_mat(gc, im.BL(il, "engram_kv.weight"), rows);
        ggml_tensor * key = ggml_view_3d(gc, kv, D, HC, n, (size_t) D * 4, kv->nb[1], 0);
        ggml_tensor * val = ggml_view_3d(gc, kv, D, 1, n, (size_t) D * 4, kv->nb[1], (size_t) HC * D * 4);
        ggml_tensor * wqk = ggml_mul(gc, im.BL(il, "engram_q_norm.weight"), im.BL(il, "engram_k_norm.weight"));
        ggml_tensor * hn = ggml_rms_norm(gc, x, (float) G.rms_eps);
        ggml_tensor * kn = ggml_rms_norm(gc, ggml_cont(gc, key), (float) G.rms_eps);
        ggml_tensor * dot = ggml_scale(gc, ggml_sum_rows(gc, ggml_mul(gc, ggml_mul(gc, hn, wqk), kn)), 1.0f / std::sqrt((float) D));
        ggml_tensor * mag = ggml_sqrt(gc, ggml_clamp(gc, ggml_abs(gc, dot), 1e-6f, INFINITY));
        ggml_tensor * gate = ggml_sigmoid(gc, ggml_mul(gc, ggml_sgn(gc, dot), mag));
        if (im.vision && im.i_img) {   // image positions take no Engram contribution (model.py: gate masked_fill ~token_mask)
            ggml_tensor * m = ggml_reshape_3d(gc, b.v1(im.i_img, n, (size_t) col0 * 4), 1, 1, n);
            gate = ggml_mul(gc, gate, ggml_scale_bias(gc, m, -1.0f, 1.0f));
        }
        x = ggml_add(gc, x, ggml_mul(gc, ggml_repeat(gc, ggml_cont(gc, val), x), gate));
        outs.push_back(ggml_cpy(gc, x, b.cols3(im.x_state, D, HC, n, col0)));
    }
    ggml_tensor * xin = ggml_dsv4_hc_pre(gc, x, b.cols2(im.pre_state, HC, n, col0));
    ggml_tensor * xn = b.rms_w(xin, im.BL(il, "attn_norm.weight"));

    // ---- window K (= V): the same post-RoPE raw row build_attn writes into the ring
    ggml_tensor * kv = b.rms_w(ggml_mul_mat(gc, im.BL(il, "attn_kv.weight"), xn), im.BL(il, "attn_kv_a_norm.weight"));
    kv = b.rope(ggml_reshape_3d(gc, kv, DH, 1, n), pos, r);
    ggml_tensor * kv2 = ggml_reshape_2d(gc, kv, DH, n);
    if (im.cfg.act_quant) kv2 = fake_quant(gc, ggml_cont(gc, kv2), Ds41Quant::FP8_POW2_32);
    {
        const int64_t k = std::min<int64_t>(n, im.RAW);
        ggml_tensor * last = ggml_view_2d(gc, kv2, DH, k, kv2->nb[1], (size_t) (n - k) * kv2->nb[1]);
        late.push_back(ggml_set_rows(gc, L.raw, last, b.v1(im.i_wslot, k, 0)));
    }

    // ---- compressed KV + indexer keys (kv source layers; the only other persistent state a later query reads)
    if (r && L.kv_src) {
        ggml_tensor * latent = nullptr;
        if (r == 1) {
            latent = ggml_mul_mat(gc, im.BL(il, "attn_compressor_kv.weight"), xn);
        } else {
            ggml_tensor * skv = ggml_mul_mat(gc, im.BL(il, "attn_compressor_kv.weight"), xn);
            ggml_tensor * ssc = ggml_mul_mat(gc, im.BL(il, "attn_compressor_gate.weight"), xn);
            ggml_mul_mat_set_prec(skv, GGML_PREC_F32);
            ggml_mul_mat_set_prec(ssc, GGML_PREC_F32);
            ggml_tensor * src_kv = ggml_concat(gc, L.st_kv, skv, 1);
            ggml_tensor * src_sc = ggml_concat(gc, L.st_sc, ssc, 1);
            ggml_tensor * gi = b.v1(im.i_sidx, 2 * n, (size_t) col0 * 2 * 4);
            ggml_tensor * vk = ggml_reshape_3d(gc, ggml_get_rows(gc, src_kv, gi), DH, 2, n);
            ggml_tensor * vs = ggml_reshape_3d(gc, ggml_get_rows(gc, src_sc, gi), DH, 2, n);
            vk = ggml_cont(gc, ggml_permute(gc, vk, 1, 0, 2, 3));
            vs = ggml_cont(gc, ggml_permute(gc, vs, 1, 0, 2, 3));
            latent = ggml_reshape_2d(gc, ggml_sum_rows(gc, ggml_mul(gc, vk, ggml_soft_max(gc, vs))), DH, n);
            const int64_t k = std::min<int64_t>(n, 2);
            late.push_back(ggml_set_rows(gc, L.st_kv, ggml_view_2d(gc, skv, DH, k, skv->nb[1], (size_t) (n - k) * skv->nb[1]),
                                         b.v1(im.i_sslot, k, 0)));
            late.push_back(ggml_set_rows(gc, L.st_sc, ggml_view_2d(gc, ssc, DH, k, ssc->nb[1], (size_t) (n - k) * ssc->nb[1]),
                                         b.v1(im.i_sslot, k, 0)));
        }
        latent = b.rms_w(latent, im.BL(il, "attn_compressor_norm.weight"));
        ggml_tensor * cpos = b.v1(im.i_cpos[r], n, (size_t) col0 * 4);
        ggml_tensor * cslot = b.v1(im.i_cslot[r], n, (size_t) col0 * 4);
        ggml_tensor * ik = b.rms_w(ggml_mul_mat(gc, im.BL(il, "indexer.attn_k.weight"), latent), im.BL(il, "indexer.k_norm.weight"));
        ik = ggml_reshape_2d(gc, b.rope(ggml_reshape_3d(gc, ik, im.ID, 1, n), cpos, r), im.ID, n);
        if (im.cfg.act_quant) ik = fake_quant(gc, ggml_cont(gc, ik), Ds41Quant::FP4_POW2_32);
        outs.push_back(ggml_set_rows(gc, L.idxk, ik, cslot));
        ggml_tensor * lr = ggml_reshape_2d(gc, b.rope(ggml_reshape_3d(gc, latent, DH, 1, n), cpos, r), DH, n);
        if (im.cfg.act_quant) lr = fake_quant(gc, ggml_cont(gc, lr), Ds41Quant::FP4_E4M3_16);
        outs.push_back(ggml_set_rows(gc, L.comp, lr, cslot));
    }
    for (ggml_tensor * t : outs) ggml_build_forward_expand(gf, t);
    for (ggml_tensor * t : late) ggml_build_forward_expand(gf, t);
    return gf;
}

// predict: router(rms_norm(hc_pre(state, pre)) * ffn_norm) + bias, top 16 of token 0
static ggml_cgraph * build_predict(Ds41Dense::Impl & im, ggml_context * gc, int il, ggml_tensor ** out, int64_t n = 1) {
    partprof::begin(partprof::P_INIT);
    B b { &im, gc, &im.g };
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 64, false);
    const int64_t D = im.D, HC = im.HC;
    ggml_tensor * x = ggml_dsv4_hc_pre(gc, b.cols3(im.x_state, D, HC, n), b.cols2(im.pre_state, HC, n));
    ggml_tensor * rl = ggml_mul_mat(gc, im.BLpred(il), b.rms_w(x, im.BL(il, "ffn_norm.weight")));
    ggml_mul_mat_set_prec(rl, GGML_PREC_F32);
    ggml_tensor * sc = ggml_add(gc, ggml_add(gc, ggml_sqrt(gc, ggml_softplus(gc, rl)), im.BL(il, "exp_probs_b.bias")),
                                ggml_reshape_2d(gc, im.ly[(size_t) il].rbias, im.NE, 1));
    *out = ggml_argsort_top_k(gc, sc, 16);
    ggml_set_output(*out);
    ggml_build_forward_expand(gf, *out);
    return gf;
}

// finish: x_state = hc_post(shexp + routed, hap, post_f, comb_f).  col0: the batch column (0 on the single-pass paths).
static ggml_cgraph * build_finish(Ds41Dense::Impl & im, ggml_context * gc, int64_t n, int64_t col0 = 0) {
    partprof::begin(partprof::P_FINISH);
    B b { &im, gc, &im.g };
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 64, false);
    const int64_t D = im.D, HC = im.HC;
    ggml_tensor * out = ggml_add(gc, b.cols2(im.shexp, D, n, col0), b.cols2(im.routed, D, n, col0));
    ggml_tensor * x = ggml_dsv4_hc_post(gc, out, b.cols3(im.hap, D, HC, n, col0), b.cols2(im.post_f, HC, n, col0),
                                        b.cols3(im.comb_f, HC, HC, n, col0));
    ggml_build_forward_expand(gf, ggml_cpy(gc, x, b.cols3(im.x_state, D, HC, n, col0)));
    return gf;
}

// finish (n = 1) + the next layer's routing prediction on the state it writes (build_predict's ops on the cpy result)
static ggml_cgraph * build_finish_predict(Ds41Dense::Impl & im, ggml_context * gc, int next_il, ggml_tensor ** pout,
                                          int ahead = 0, std::vector<ggml_tensor *> * aout = nullptr) {
    partprof::begin(partprof::P_FINISH);
    B b { &im, gc, &im.g };
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 128, false);
    const int64_t D = im.D, HC = im.HC;
    ggml_tensor * out = ggml_add(gc, b.cols2(im.shexp, D, 1, 0), b.cols2(im.routed, D, 1, 0));
    ggml_tensor * x = ggml_dsv4_hc_post(gc, out, b.cols3(im.hap, D, HC, 1, 0), b.cols2(im.post_f, HC, 1, 0),
                                        b.cols3(im.comb_f, HC, HC, 1, 0));
    ggml_tensor * st = ggml_cpy(gc, x, b.cols3(im.x_state, D, HC, 1, 0));
    ggml_build_forward_expand(gf, st);
    ggml_tensor * xp = ggml_dsv4_hc_pre(gc, st, b.cols2(im.pre_state, HC, 1));
    ggml_tensor * rl = ggml_mul_mat(gc, im.BLpred(next_il), b.rms_w(xp, im.BL(next_il, "ffn_norm.weight")));
    ggml_mul_mat_set_prec(rl, GGML_PREC_F32);
    ggml_tensor * sc = ggml_add(gc, ggml_add(gc, ggml_sqrt(gc, ggml_softplus(gc, rl)), im.BL(next_il, "exp_probs_b.bias")),
                                ggml_reshape_2d(gc, im.ly[(size_t) next_il].rbias, im.NE, 1));
    // the layers after next_il on the same state (speculative reads): their scores are stacked under next_il's so ONE
    // argsort gives all the rows and ONE readback fetches them ([16, 1 + ahead] I32)
    for (int a = 1; a <= ahead && aout; ++a) {
        const int il = next_il + a;
        if (il >= (int) im.ly.size()) break;
        ggml_tensor * ra = ggml_mul_mat(gc, im.BLpred(il), b.rms_w(xp, im.BL(il, "ffn_norm.weight")));
        ggml_mul_mat_set_prec(ra, GGML_PREC_F32);
        ggml_tensor * sa = ggml_add(gc, ggml_add(gc, ggml_sqrt(gc, ggml_softplus(gc, ra)), im.BL(il, "exp_probs_b.bias")),
                                    ggml_reshape_2d(gc, im.ly[(size_t) il].rbias, im.NE, 1));
        sc = ggml_concat(gc, sc, sa, 1);
        aout->push_back(nullptr);   // a row of *pout
    }
    *pout = ggml_argsort_top_k(gc, sc, 16);
    ggml_set_output(*pout);
    ggml_build_forward_expand(gf, *pout);
    return gf;
}

// head: hc_pre(state, pre_state) -> output_norm -> output, rows row0..row0+nr-1
static ggml_cgraph * build_head(Ds41Dense::Impl & im, ggml_context * gc, int64_t row0, int64_t nr, ggml_tensor ** lg) {
    partprof::begin(partprof::P_HEAD);
    B b { &im, gc, &im.g };
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 64, false);
    const int64_t D = im.D, HC = im.HC;
    ggml_tensor * x = b.cols3(im.x_state, D, HC, nr, row0);
    ggml_tensor * p = b.cols2(im.pre_state, HC, nr, row0);
    ggml_tensor * h = b.rms_w(ggml_dsv4_hc_pre(gc, x, p), im.W("output_norm.weight"));
    *lg = ggml_mul_mat(gc, im.W("output.weight"), h);
    ggml_set_output(*lg);
    ggml_build_forward_expand(gf, *lg);
    return gf;
}

// begin: x_state = repeat(embd), pre_state = one-hot (model.py make_identity_pre_mix).  col0: the batch column the
// pass's n tokens live at (0 on the single-pass paths).
static ggml_cgraph * build_init(Ds41Dense::Impl & im, ggml_context * gc, int64_t n, int64_t col0 = 0) {
    partprof::begin(partprof::P_INIT);
    B b { &im, gc, &im.g };
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 64, false);
    const int64_t D = im.D, HC = im.HC;
    ggml_tensor * e = ggml_reshape_3d(gc, b.cols2(im.i_emb, D, n, col0), D, 1, n);
    ggml_build_forward_expand(gf, ggml_cpy(gc, ggml_repeat_4d(gc, e, D, HC, n, 1), b.cols3(im.x_state, D, HC, n, col0)));
    ggml_tensor * one = ggml_fill(gc, ggml_new_tensor_2d(gc, GGML_TYPE_F32, 1, n), 1.0f);
    ggml_tensor * oh = ggml_pad(gc, one, (int) (HC - 1), 0, 0, 0);                            // [1, 0, 0, 0] per token
    ggml_build_forward_expand(gf, ggml_cpy(gc, oh, b.cols2(im.pre_state, HC, n, col0)));
    return gf;
}

// ================================================================== cached pass graphs (1 < n <= kPassGraphN)

static constexpr int64_t kPassGraphN = 8;
static constexpr size_t kPassGraphCtx = (size_t) 256 << 20;

static bool pass_graphs_on(int64_t n) {
    static const bool on = env_on("DS41_PASS_GRAPHS", true);
    return on && n > 1 && n <= kPassGraphN;
}

template <class F>
static ggml_cgraph * pass_graph(Ds41Dense::Impl & im, const Ds41Dense::Impl::GKey & key, F build) {
    auto it = im.mgraphs.find(key);
    if (it != im.mgraphs.end()) return it->second;
    // worst case of one graph: 8192 nodes' tensors + the graph struct
    const size_t need = ggml_tensor_overhead() * 8192 * 2 + ggml_graph_overhead_custom(8192, false);
    if (!im.gctx_m || ggml_get_mem_size(im.gctx_m) - ggml_used_mem(im.gctx_m) < need) {
        if (im.gctx_m) {
            for (const auto & kv : im.mgraphs) im.allo_dec.graph_epoch.erase(kv.second);
            std::fprintf(stderr, "ds41_dense: pass graph cache full (%zu graphs) - dropped\n", im.mgraphs.size());
            im.mgraphs.clear();
            ggml_free(im.gctx_m);
            ++im.mgraph_resets;
        }
        ggml_init_params gp = { kPassGraphCtx, nullptr, true };
        im.gctx_m = ggml_init(gp);
    }
    return im.mgraphs.emplace(key, build(im.gctx_m)).first->second;
}

// ================================================================== public API

Ds41Dense::Ds41Dense() : p_(new Impl()) {}
Ds41Dense::~Ds41Dense() = default;
const Ds41Geometry & Ds41Dense::geom() const { return p_->g; }
ggml_tensor * Ds41Dense::weight(const std::string & name) const { return p_->w.get(name); }
int64_t Ds41Dense::n_layer() const { return p_->nl; }
ggml_backend_t Ds41Dense::backend() const { return p_->backend; }
const std::string & Ds41Dense::last_error() const { return p_->err; }
bool Ds41Dense::vision() const { return p_->vision; }

void Ds41Dense::set_vision(const float * rows, const int64_t * pos, int n_rows) {
    Impl & im = *p_;
    im.vis_rows = rows;
    im.vis_pos = pos;
    im.n_vis = n_rows;
}

void Ds41Dense::clear_vision() {
    Impl & im = *p_;
    im.vis_rows = nullptr;
    im.vis_pos = nullptr;
    im.n_vis = 0;
}

// The NC-column tensors (see Impl::nctx): (re)created for `NC` columns, zeroed.  Every one is per-pass scratch (a pass
// writes the columns it reads), so a resize between passes loses nothing.
static bool alloc_cols(Ds41Dense::Impl & im, int64_t NC, std::string & err) {
    if (im.nbuf) { ggml_backend_buffer_free(im.nbuf); im.nbuf = nullptr; }
    if (im.nctx) { ggml_free(im.nctx); im.nctx = nullptr; }
    im.NC = NC;
    ggml_init_params cp = { ggml_tensor_overhead() * 256, nullptr, true };
    im.nctx = ggml_init(cp);
    ggml_context * c = im.nctx;
    const int64_t D = im.D, HC = im.HC;
    im.x_state = ggml_new_tensor_3d(c, GGML_TYPE_F32, D, HC, NC);
    im.pre_state = ggml_new_tensor_2d(c, GGML_TYPE_F32, HC, NC);
    im.hap = ggml_new_tensor_3d(c, GGML_TYPE_F32, D, HC, NC);
    im.post_f = ggml_new_tensor_2d(c, GGML_TYPE_F32, HC, NC);
    im.comb_f = ggml_new_tensor_3d(c, GGML_TYPE_F32, HC, HC, NC);
    im.shexp = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, NC);
    im.o_fn = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, NC);
    im.o_ids = ggml_new_tensor_2d(c, GGML_TYPE_I32, im.NU, NC);
    im.o_wts = ggml_new_tensor_2d(c, GGML_TYPE_F32, im.NU, NC);
    if (smart_router()) {
        im.o_xids = ggml_new_tensor_2d(c, GGML_TYPE_I32, im.NU, NC);
        im.o_xwts = ggml_new_tensor_2d(c, GGML_TYPE_F32, im.NU, NC);
    }
    im.routed = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, NC);
    im.cand = ggml_new_tensor_2d(c, GGML_TYPE_I32, im.CB, NC);
    for (int64_t l = 0; l < im.nl; ++l) {
        auto & L = im.ly[(size_t) l];
        if (!L.idx_src) continue;
        L.sel = ggml_new_tensor_2d(c, GGML_TYPE_I32, im.ITOP, NC);
        L.selm = ggml_new_tensor_2d(c, GGML_TYPE_F32, im.ITOP, NC);
    }
    im.i_emb = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, NC);
    im.i_pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, NC);
    im.i_img = im.vision ? ggml_new_tensor_1d(c, GGML_TYPE_F32, NC) : nullptr;   // vision: the image-position mask
    im.i_widx = ggml_new_tensor_1d(c, GGML_TYPE_I32, im.SWA * NC);
    im.span_memo_pos0 = -1;   // new tensors: nothing uploaded yet
    im.i_wmask = ggml_new_tensor_2d(c, GGML_TYPE_F32, im.SWA, NC);
    im.i_wslot = ggml_new_tensor_1d(c, GGML_TYPE_I32, std::min<int64_t>(NC, im.RAW));
    for (int r = 1; r <= 2; ++r) {
        im.i_nvis[r] = ggml_new_tensor_1d(c, GGML_TYPE_F32, NC);
        im.i_cslot[r] = ggml_new_tensor_1d(c, GGML_TYPE_I32, NC);
        im.i_cpos[r] = ggml_new_tensor_1d(c, GGML_TYPE_I32, NC);
    }
    im.i_sidx = ggml_new_tensor_1d(c, GGML_TYPE_I32, 2 * NC);
    // a batch pass (begin_rows) keeps each group's 2 state slots at 2 x its first column
    im.i_sslot = ggml_new_tensor_1d(c, GGML_TYPE_I32, std::max<int64_t>(std::min<int64_t>(NC, 2), 2 * std::min<int64_t>(NC, Ds41Dense::kMaxRows)));
    im.i_sscol = ggml_new_tensor_1d(c, GGML_TYPE_I32, 2 * std::min<int64_t>(NC, Ds41Dense::kMaxRows));
    for (int r = 1; r <= 2; ++r) im.i_cbase[r] = ggml_new_tensor_1d(c, GGML_TYPE_F32, NC);
    im.i_lastblk = ggml_new_tensor_1d(c, GGML_TYPE_F32, NC);
    for (int e = 0; e < 2; ++e) im.i_eng[e] = ggml_new_tensor_2d(c, GGML_TYPE_F32, im.EC * im.ED, NC);
    // STRATA_VRAM_TRACE: WHICH state tensor owns the columns' VRAM (s13c: a batch's state alloc was 2.45-2.76 GiB and
    // a 2193.81 MiB allocation later failed with 6.83 GiB "free" - name the owner before sizing anything).
    if (std::getenv("STRATA_VRAM_TRACE")) {
        double tot = 0;
        for (ggml_tensor * t = ggml_get_first_tensor(c); t; t = ggml_get_next_tensor(c, t)) tot += (double) ggml_nbytes(t);
        std::fprintf(stderr, "vram trace: state NC=%d wants %.2f GiB (i_eng %lld x %d = %.2f GiB each, i_wmask %.1f MiB, "
                             "i_widx %.1f MiB)\n", NC, tot / 1073741824.0, (long long) im.EC, (int) im.ED,
                     (double) ggml_nbytes(im.i_eng[0]) / 1073741824.0, (double) ggml_nbytes(im.i_wmask) / 1048576.0,
                     (double) ggml_nbytes(im.i_widx) / 1048576.0);
    }
    im.nbuf = ggml_backend_alloc_ctx_tensors(c, im.backend);
    if (!im.nbuf) { err = "cannot allocate " + std::to_string(NC) + " state columns"; return false; }
    ggml_backend_buffer_clear(im.nbuf, 0);
    return true;
}

bool Ds41Dense::set_columns(int nc) {
    Impl & im = *p_;
    const int64_t want = std::max<int64_t>(im.S, std::min<int64_t>(nc, im.NCmax));
    if (want == im.NC) return true;
    DPROF_SYNC(im.backend);
    // every cached graph holds views of the old columns (their addresses are fixed at the first allocation), and the
    // allocators remember graphs by address - all of it goes, and is rebuilt lazily for the new tensors
    im.reset_graphs();
    return alloc_cols(im, want, im.err);
}

void Ds41Dense::Impl::reset_graphs() {
    Impl & im = *this;
    im.graphs.clear();
    for (const auto & kv : im.mgraphs) im.allo_dec.graph_epoch.erase(kv.second);
    im.mgraphs.clear();
    if (im.gctx_m) { ggml_free(im.gctx_m); im.gctx_m = nullptr; }
    im.logits_out.clear();
    im.predict_out.clear();
    im.predict_out_n.clear();
    im.fpredict_out.clear();
    im.fpredict_ahead.clear();
    for (Allo * al : { &im.allo_dec, &im.allo_pass }) {
        if (al->a) ggml_gallocr_free(al->a);
        al->a = ggml_gallocr_new(ggml_backend_get_default_buffer_type(im.backend));
        al->graph_epoch.clear();
        al->compute_bufs.clear();
        al->failed = false;
        ++al->epoch;
    }
    if (im.gctx) ggml_free(im.gctx);
    ggml_init_params gp = { ggml_tensor_overhead() * 200000 + ggml_graph_overhead_custom(8192, false) * 400, nullptr, true };
    im.gctx = ggml_init(gp);
}

// Room for one more cached layer graph (a layer's attention graph is ~1-2K tensors; 8192 is the build_attn graph's
// node cap): with several slots each slot caches its own decode graphs, so the context can fill where one slot never
// did - then every cached graph is dropped and rebuilt lazily (as set_columns does) instead of overflowing it.
bool Ds41Dense::Impl::gctx_room() const {
    if (!gctx) return false;
    const size_t need = ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false) * 2;
    return ggml_used_mem(gctx) + need <= ggml_get_mem_size(gctx);
}

void Ds41Dense::Impl::stash_slot() {
    Slot & S = slots[(size_t) cur_slot];
    S.ly.resize(ly.size());
    for (size_t l = 0; l < ly.size(); ++l) {
        const Layer & L = ly[l];
        S.ly[l] = SlotLayer { L.raw, L.raw_snap, L.st_kv, L.st_sc, L.st_kv_snap, L.st_sc_snap, L.comp, L.idxk };
    }
    S.snap_pos = snap_pos;
    S.parked_pos = parked_pos;
    S.parked = std::move(parked);
    parked.clear();
}

void Ds41Dense::Impl::load_slot(int s) {
    Slot & S = slots[(size_t) s];
    for (size_t l = 0; l < ly.size(); ++l) {
        Layer & L = ly[l];
        const SlotLayer & T = S.ly[l];
        L.raw = T.raw; L.raw_snap = T.raw_snap; L.st_kv = T.st_kv; L.st_sc = T.st_sc;
        L.st_kv_snap = T.st_kv_snap; L.st_sc_snap = T.st_sc_snap; L.comp = T.comp; L.idxk = T.idxk;
    }
    snap_pos = S.snap_pos;
    parked_pos = S.parked_pos;
    parked = std::move(S.parked);
    S.parked.clear();
    cur_slot = s;
    engram.select_history(s);
}

int Ds41Dense::n_slots() const { return (int) p_->slots.size(); }
int Ds41Dense::slot() const { return p_->cur_slot; }

bool Ds41Dense::bind_slot(int s) {
    Impl & im = *p_;
    if (s < 0 || s >= (int) im.slots.size()) { im.err = "bind_slot: no slot " + std::to_string(s); return false; }
    if (s == im.cur_slot) return true;
    if (im.eng_batch) { im.err = "bind_slot inside a pass"; return false; }
    im.stash_slot();
    im.load_slot(s);
    return true;
}

bool Ds41Dense::swap_slots(int a, int b) {
    Impl & im = *p_;
    const int n = (int) im.slots.size();
    if (a < 0 || b < 0 || a >= n || b >= n) { im.err = "swap_slots: bad slot"; return false; }
    if (a == b) return true;
    if (im.eng_batch) { im.err = "swap_slots inside a pass"; return false; }
    DPROF_SYNC(im.backend);
    const int cur = im.cur_slot;
    im.stash_slot();
    std::swap(im.slots[(size_t) a], im.slots[(size_t) b]);
    im.engram.swap_histories(a, b);   // the selected history (cur's) is swapped in place too
    im.load_slot(cur);
    im.reset_graphs();   // the cached attention graphs of slots a / b hold the other state's tensors now
    return true;
}

size_t Ds41Dense::slot_bytes() const {
    const Impl & im = *p_;
    size_t b = 0;
    for (size_t l = 0; l < im.ly.size(); ++l) {
        const Impl::SlotLayer T = im.slot_layer(im.cur_slot, l);
        for (ggml_tensor * t : { T.raw, T.raw_snap, T.st_kv, T.st_sc, T.st_kv_snap, T.st_sc_snap, T.comp, T.idxk })
            if (t) b += ggml_nbytes(t);
    }
    return b;
}

int Ds41Dense::columns() const { return (int) p_->NC; }

bool Ds41Dense::init(const std::string & path, const Ds41DenseConfig & cfg, std::string & err) {
    Impl & im = *p_;
    im.cfg = cfg;
    try {
        im.model = std::make_unique<GgufModel>(GgufModel::open(path));
    } catch (const std::exception & e) {
        err = e.what();
        return false;
    }
    if (!(err = ds41_read_geometry(*im.model, im.g)).empty()) return false;
    im.layers = ds41_file_layers(*im.model, im.g);
    if (!(err = ds41_check_tensors(*im.model, im.g, im.layers)).empty()) return false;
    for (size_t i = 0; i < im.layers.size(); ++i)
        if (im.layers[i] != (int64_t) i) { err = "the file's blocks must be 0..k"; return false; }
    im.nl = (int64_t) im.layers.size();
    // gate-only: run the first n layers (a faster CPU gate of the same code; the logits are not the model's then)
    if (const char * e = std::getenv("DS41_TEST_LAYERS"); e && std::atoi(e) > 0) im.nl = std::min<int64_t>(im.nl, std::atoi(e));
    if (cfg.head_only) {
        im.layer_lo = im.nl;
        im.layer_hi = im.nl;
    } else {
        im.layer_lo = std::max<int64_t>(0, cfg.layer_lo);
        im.layer_hi = cfg.layer_hi < 0 ? im.nl : std::min<int64_t>(cfg.layer_hi, im.nl);
        if (im.layer_lo >= im.layer_hi) { err = "invalid layer range"; return false; }
    }
    const Ds41Geometry & G = im.g;
    im.D = G.n_embd; im.HC = G.hc; im.DH = G.d_head; im.DHR = G.d_rope; im.NH = G.n_head; im.OG = G.o_groups;
    im.OL = G.o_lora; im.OGD = G.n_head * G.d_head / G.o_groups; im.RQ = G.r_q; im.SWA = G.n_swa;
    im.IH = G.idx_heads; im.ID = G.idx_dim; im.ITOP = G.idx_topk; im.NE = G.n_expert; im.NU = G.n_used; im.NFF = G.n_ff;
    im.CB = G.cand_blocks; im.CBS = G.cand_bsize; im.EC = (G.engram_ngram - 1) * G.engram_heads; im.ED = G.engram_dim;
    // gate-only overrides (the golden takes the same): small top-k / candidate counts make the selection logic bite
    // on a short prompt
    if (const char * e = std::getenv("DS41_TEST_TOPK")) im.ITOP = std::atoi(e);
    if (const char * e = std::getenv("DS41_TEST_CAND_BLOCKS")) im.CB = std::atoi(e);
    im.S = std::max(1, cfg.max_pass);                       // the dense sub-batch size of the layer-major walk
    // STRATA_INDEXER_TILE=<elements>: the indexer selection's per-graph piece (see build_attn); 0 = never split
    im.idx_tile = std::getenv("STRATA_INDEXER_TILE") ? std::max<long long>(0, std::atoll(std::getenv("STRATA_INDEXER_TILE")))
                                                     : (int64_t) 1 << 28;
    im.NCmax = std::max<int64_t>(im.S, std::max(1, cfg.prompt_batch));   // state columns: the whole prompt batch, or the pass
    im.NC = cfg.elastic_cols ? im.S : im.NCmax;                           // elastic: S until a batch asks for more
    // the window ring: every row is gathered BY POSITION (slot = pos % RAW), so its size never changes results.  DSpark
    // verify passes write up to k rejected draft positions; a rejected q evicts q - RAW, which a later query q' (>= the
    // last accepted position, re-fed) needs iff q - RAW >= q' - SWA + 1  ->  safe for k <= RAW - SWA = 16 drafts.
    im.RAW = im.SWA + 16;
    im.cmax[1] = cfg.ctx_max; im.cmax[2] = cfg.ctx_max / 2;
    if (im.ITOP > im.cmax[2] || im.CBS <= 0 || cfg.ctx_max % 256) { err = "ctx_max must be a multiple of 256 and >= 1024"; return false; }

    if (cfg.backend) { im.backend = cfg.backend; }
    else {
        im.backend = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(im.backend, cfg.n_threads);
        im.own_backend = true;
    }
    im.cpu = ggml_backend_is_cpu(im.backend);
    if (!load_weights(im, err)) return false;
    im.engram.set_hot_path(cfg.engram_hot);
    im.vision = cfg.vision;
    if (im.vision) {
        for (int64_t l = 0; l < im.nl; ++l) {
            if (!im.w.has("blk." + std::to_string(l) + ".exp_probs_b_vl.bias")) {
                err = "--vision requires exp_probs_b_vl.bias in every layer";
                return false;
            }
        }
    }
    im.engram.set_hot_path(cfg.engram_hot);
    if (!cfg.head_only && !cfg.engram_dirs.empty() && !im.engram.open(G, cfg.engram_dirs, cfg.engram_threads, err)) return false;

    im.shadow_layers = cfg.shadow_layers;
    // layers
    im.ly.resize(cfg.head_only ? 0 : (size_t) im.nl);
    for (int64_t l = 0; !cfg.head_only && l < im.nl; ++l) {
        auto & L = im.ly[(size_t) l];
        L.ratio = G.ratio[(size_t) l];
        if (!im.active_layer(l)) continue;
        L.kv_src = G.is_kv_source(l);
        L.idx_src = G.is_index_source(l);
        L.cand_src = l == G.cand_layer;
        L.uses_cand = G.cand_layer >= 0 && l > G.cand_layer && L.idx_src;
        L.kvl = G.kv_src(l);
        L.idxl = G.idx_src(l);
        L.eslot = G.engram_slot(l);
    }
    // persistent state.  A layer split (buster): a stage holds state only for the layers it runs or shadows
    // (active_layer); a head_only instance (output head + DSpark on its own card) holds none.
    if (!cfg.head_only && cfg.kv_fp4 && (!cfg.act_quant || im.cpu || im.DH % 64 || im.ID % 32)) {
        err = "--kv-fp4 needs act_quant (the FP4 grids), the CUDA backend, d_head % 64 == 0 and idx_dim % 32 == 0";
        return false;
    }
    const ggml_type kvt = cfg.kv_type >= 0 ? (ggml_type) cfg.kv_type : cfg.act_quant ? GGML_TYPE_BF16 : GGML_TYPE_F32;
    constexpr int kSlotsMax = 64;
    int NS = cfg.head_only ? 1 : cfg.n_slots > 0 ? std::min(cfg.n_slots, kSlotsMax) : 8;   // <= 0: auto (below), at most 8
    ggml_init_params sp = { ggml_tensor_overhead() * (size_t) (512 + 8 * im.nl * NS), nullptr, true };
    im.sctx = ggml_init(sp);
    auto T = [&](ggml_type t, std::initializer_list<int64_t> ne) {
        const int64_t * d = ne.begin();
        switch (ne.size()) {
            case 1: return ggml_new_tensor_1d(im.sctx, t, d[0]);
            case 2: return ggml_new_tensor_2d(im.sctx, t, d[0], d[1]);
            default: return ggml_new_tensor_3d(im.sctx, t, d[0], d[1], d[2]);
        }
    };
    const int64_t D = im.D, NCM = im.NCmax;
    // STRATA_KV_FP4_ONLY=comp|idx (bisect tap): pack only one of the two caches
    static const char * only = std::getenv("STRATA_KV_FP4_ONLY");
    const ggml_type comp_t = cfg.kv_fp4 && !(only && std::string(only) == "idx") ? GGML_TYPE_NVFP4 : kvt;
    const ggml_type idxk_t = cfg.kv_fp4 && !(only && std::string(only) == "comp") ? GGML_TYPE_MXFP4 : kvt;
    // one slot's state: the window ring + its snapshot, and on kv sources the compressed KV + indexer keys (+ the ratio-2
    // compressor state and its snapshot)
    size_t slot0 = 0;
    for (int64_t l = 0; l < im.nl; ++l) {
        const auto & L = im.ly[(size_t) l];
        if (cfg.head_only || !im.active_layer(l)) continue;
        slot0 += 2 * ggml_row_size(kvt, im.DH) * (size_t) im.RAW;
        if (!L.kv_src) continue;
        slot0 += (ggml_row_size(comp_t, im.DH) + ggml_row_size(idxk_t, im.ID)) * (size_t) (im.cmax[L.ratio] + 1);
        if (L.ratio == 2) slot0 += 4 * ggml_row_size(GGML_TYPE_F32, im.DH) * 2;
    }
    if (cfg.n_slots <= 0 && !cfg.head_only) {   // auto: as many as STRATA_BATCH_SLOTS_GIB (default 2 GiB) holds, 2..8
        const double gib = std::getenv("STRATA_BATCH_SLOTS_GIB") ? std::atof(std::getenv("STRATA_BATCH_SLOTS_GIB")) : 2.0;
        NS = (int) std::max<int64_t>(2, std::min<int64_t>(8, (int64_t) (gib * 1073741824.0 / (double) std::max<size_t>(1, slot0))));
    }
    // the slab: a batch pass gathers compressed rows by slab row = region * (cmax + 1) + row, computed in F32 - exact
    // below 2^24 rows (8 slots of 1M)
    im.slab = NS > 1 && !(std::getenv("STRATA_SLOT_SLAB") && std::getenv("STRATA_SLOT_SLAB")[0] == '0') &&
              (int64_t) NS * (im.cmax[1] + 1) < (int64_t) 1 << 24;
    if (NS > 1)
        std::fprintf(stderr, "batch slots: %d x %.1f MiB of conversation state (ctx %lld each) = %.2f GiB%s\n", NS,
                     slot0 / 1048576.0, (long long) cfg.ctx_max, (double) slot0 * NS / 1073741824.0, im.slab ? ", slabs" : "");
    im.slots.assign((size_t) NS, Impl::Slot());
    im.cur_slot = 0;
    for (int sl = 1; sl < NS; ++sl) im.slots[(size_t) sl].ly.resize((size_t) im.nl);
    for (int64_t l = 0; l < im.nl; ++l) {
        auto & L = im.ly[(size_t) l];
        if (cfg.head_only || !im.active_layer(l)) continue;   // another stage's layer: no state here
        if (!im.slab) {
            L.raw = T(kvt, { im.DH, im.RAW });
            L.raw_snap = T(kvt, { im.DH, im.RAW });
            L.rbias = T(GGML_TYPE_F32, { im.NE });   // zeroed with the buffer
            if (L.kv_src) {
                L.comp = T(comp_t, { im.DH, im.cmax[L.ratio] + 1 });
                L.idxk = T(idxk_t, { im.ID, im.cmax[L.ratio] + 1 });
                if (L.ratio == 2) {
                    L.st_kv = T(GGML_TYPE_F32, { im.DH, 2 });
                    L.st_sc = T(GGML_TYPE_F32, { im.DH, 2 });
                    L.st_kv_snap = T(GGML_TYPE_F32, { im.DH, 2 });
                    L.st_sc_snap = T(GGML_TYPE_F32, { im.DH, 2 });
                }
            }
            continue;
        }
        L.rbias = T(GGML_TYPE_F32, { im.NE });
        const int64_t CR = im.cmax[L.ratio] + 1;
        L.raw_all = T(kvt, { im.DH, im.RAW * NS });
        if (L.kv_src) {
            L.comp_all = T(comp_t, { im.DH, CR * NS });
            L.idxk_all = T(idxk_t, { im.ID, CR * NS });
            if (L.ratio == 2) { L.st_kv_all = T(GGML_TYPE_F32, { im.DH, 2 * NS }); L.st_sc_all = T(GGML_TYPE_F32, { im.DH, 2 * NS }); }
        }
        auto rows = [&](ggml_tensor * all, int64_t per, int sl) {   // slot sl's region of a slab
            return all ? ggml_view_2d(im.sctx, all, all->ne[0], per, all->nb[1], (size_t) sl * (size_t) per * all->nb[1]) : nullptr;
        };
        for (int sl = 0; sl < NS; ++sl) {
            Impl::SlotLayer X;
            X.raw = rows(L.raw_all, im.RAW, sl);
            X.raw_snap = T(kvt, { im.DH, im.RAW });
            X.comp = rows(L.comp_all, CR, sl);
            X.idxk = rows(L.idxk_all, CR, sl);
            X.st_kv = rows(L.st_kv_all, 2, sl);
            X.st_sc = rows(L.st_sc_all, 2, sl);
            if (L.st_kv_all) { X.st_kv_snap = T(GGML_TYPE_F32, { im.DH, 2 }); X.st_sc_snap = T(GGML_TYPE_F32, { im.DH, 2 }); }
            if (sl == 0) {
                L.raw = X.raw; L.raw_snap = X.raw_snap; L.comp = X.comp; L.idxk = X.idxk;
                L.st_kv = X.st_kv; L.st_sc = X.st_sc; L.st_kv_snap = X.st_kv_snap; L.st_sc_snap = X.st_sc_snap;
            } else {
                im.slots[(size_t) sl].ly[(size_t) l] = X;
            }
        }
    }
    if (!cfg.head_only && smart_router()) im.rknob = T(GGML_TYPE_F32, { kRK + 1, 2 });
    // batch slots 1..NS-1 without a slab: the same per-conversation tensors again (slot 0 = the ones above)
    for (int sl = 1; sl < NS && !im.slab; ++sl) {
        auto & S = im.slots[(size_t) sl];
        for (int64_t l = 0; l < im.nl; ++l) {
            const auto & L = im.ly[(size_t) l];
            auto & X = S.ly[(size_t) l];
            auto like = [&](ggml_tensor * t) { return t ? T(t->type, { t->ne[0], t->ne[1] }) : nullptr; };
            X.raw = like(L.raw); X.raw_snap = like(L.raw_snap);
            X.comp = like(L.comp); X.idxk = like(L.idxk);
            X.st_kv = like(L.st_kv); X.st_sc = like(L.st_sc); X.st_kv_snap = like(L.st_kv_snap); X.st_sc_snap = like(L.st_sc_snap);
        }
    }
    if (!cfg.head_only) {   // (a head_only instance has no state tensors: an empty context allocates nothing)
        im.sbuf = ggml_backend_alloc_ctx_tensors(im.sctx, im.backend);
        if (!im.sbuf) { err = "cannot allocate the decode state (" + std::to_string(NS) + " slots)"; return false; }
        ggml_backend_buffer_clear(im.sbuf, 0);
    }
    // the ratio-2 state scores start at -1e30 (an unwritten state gets softmax weight 0); sel = 0 (a valid row)
    for (auto & L : im.ly)
        if (L.st_sc) { std::vector<float> v((size_t) (im.DH * 2), -1.0e30f); DPROF_SET(L.st_sc, v.data(), 0, v.size() * 4); }
    for (int sl = 1; sl < NS; ++sl)
        for (auto & X : im.slots[(size_t) sl].ly)
            if (X.st_sc) { std::vector<float> v((size_t) (im.DH * 2), -1.0e30f); DPROF_SET(X.st_sc, v.data(), 0, v.size() * 4); }
    im.stash_slot();   // slots[0] = slot 0's tensors too (slot_layer / slot_bytes read every slot the same way)
    if (im.rknob) set_route_select(env_rank("STRATA_RB_PROTECT"), env_rank("STRATA_RB_POOL"));   // all-zero knobs would protect everything
    // inputs (skipped on head_only: only the pass columns hold x / pre for the logits)
    if (!cfg.head_only) {
        ggml_init_params ipp = { ggml_tensor_overhead() * 64, nullptr, true };
        im.ictx = ggml_init(ipp);
        auto I = [&](ggml_type t, int64_t a, int64_t b2 = 1) { return b2 == 1 ? ggml_new_tensor_1d(im.ictx, t, a) : ggml_new_tensor_2d(im.ictx, t, a, b2); };
        im.neg_ramp = I(GGML_TYPE_F32, im.cmax[1] + 1);
        im.blk_ramp = I(GGML_TYPE_F32, im.cmax[1] / im.CBS + 2);
        // default ON (CUDA) since 2026-10-18: 1017 8K prompt 239.8 -> 255.2 tok/s, +1.06 GiB min free VRAM, logits + tokens
        // identical (verify/s7/vram-f16.sh GATE A).  STRATA_ATTN_F16_GATHER=0 restores the F32 gather.
        im.f16_gather = !im.cpu && !(std::getenv("STRATA_ATTN_F16_GATHER") && std::getenv("STRATA_ATTN_F16_GATHER")[0] == '0');
        im.zidx = I(GGML_TYPE_I32, im.f16_gather ? 255 * 2048 : 1);   // ibuf is cleared to 0 below: all zeros
        if (im.f16_gather) im.zpad16 = I(GGML_TYPE_F16, im.DH, 1);
        im.ibuf = ggml_backend_alloc_ctx_tensors(im.ictx, im.backend);
        if (!im.ibuf) { err = "cannot allocate the inputs"; return false; }
        ggml_backend_buffer_clear(im.ibuf, 0);
        {
            std::vector<float> v((size_t) im.neg_ramp->ne[0]);
            for (size_t i = 0; i < v.size(); ++i) v[i] = -(float) i;
            DPROF_SET(im.neg_ramp, v.data(), 0, v.size() * 4);
            v.assign((size_t) im.blk_ramp->ne[0], 0);
            for (size_t i = 0; i < v.size(); ++i) v[i] = (float) i;
            DPROF_SET(im.blk_ramp, v.data(), 0, v.size() * 4);
        }
    }
    ggml_init_params gp = { ggml_tensor_overhead() * 200000 + ggml_graph_overhead_custom(8192, false) * 400, nullptr, true };
    im.gctx = ggml_init(gp);
    im.allo_dec.a = ggml_gallocr_new(ggml_backend_get_default_buffer_type(im.backend));
    im.allo_pass.a = ggml_gallocr_new(ggml_backend_get_default_buffer_type(im.backend));
    if (!alloc_cols(im, im.NC, err)) return false;
    if (!cfg.head_only) {
        const int64_t D = im.D, NCM = im.NCmax;
        im.host_fn.resize((size_t) (D * NCM));
// S2: ggml-sycl has no ggml_backend_sycl_register_host_buffer (commented out in ggml-sycl.h): these buffers stay
// pageable and the shim stages their copies.  Guarded out rather than stubbed - a fake success would tell the engine
// the memory is DMA-able when it is not.
#if defined(DS41_HAVE_CUDA) && !defined(DS41_SYCL)
        if (!im.cpu && !im.host_fn.empty()) ggml_backend_cuda_register_host_buffer(im.host_fn.data(), im.host_fn.size() * 4);
#endif
        im.host_ids.resize((size_t) (im.NU * NCM));
        im.host_wts.resize((size_t) (im.NU * NCM));
        if (smart_router()) {
            im.host_xids.resize((size_t) (im.NU * NCM));
            im.host_xwts.resize((size_t) (im.NU * NCM));
        }
#if defined(DS41_HAVE_CUDA) && !defined(DS41_SYCL)
        // pinned too: attn_router reads fn / ids / wts back with three async copies and one sync (truly async only from
        // pinned memory)
        if (!im.cpu) {
            ggml_backend_cuda_register_host_buffer(im.host_ids.data(), im.host_ids.size() * 4);
            ggml_backend_cuda_register_host_buffer(im.host_wts.data(), im.host_wts.size() * 4);
            if (!im.host_xids.empty()) {
                ggml_backend_cuda_register_host_buffer(im.host_xids.data(), im.host_xids.size() * 4);
                ggml_backend_cuda_register_host_buffer(im.host_xwts.data(), im.host_xwts.size() * 4);
            }
        }
#endif
    }
    return true;
}

void Ds41Dense::set_route_bias(int il, const float * bias) {
    Impl & im = *p_;
    if (il < 0 || il >= (int) im.ly.size() || !im.ly[(size_t) il].rbias) return;
    DPROF_SET(im.ly[(size_t) il].rbias, bias, 0, (size_t) im.NE * 4);
}

bool Ds41Dense::set_route_select(int protect, int pool) {
    Impl & im = *p_;
    if (!im.rknob) return false;
    protect = std::max(0, std::min(protect, (int) im.NU));
    pool = pool <= 0 ? 0 : std::max((int) im.NU, std::min(pool, kRK));   // a pool smaller than top-k would force outsiders in
    std::vector<float> k((size_t) (kRK + 1) * 2, 0.0f);
    if (protect > 0) k[(size_t) protect - 1] = 1.0f; else k[kRK] = 1e30f;                          // column 0: protect
    if (pool > 0) k[(size_t) (kRK + 1 + pool - 1)] = 1.0f; else k[(size_t) (2 * kRK + 1)] = -1e30f;  // column 1: pool
    DPROF_SET(im.rknob, k.data(), 0, k.size() * 4);
    return true;
}

Ds41Dense::RouteMeter Ds41Dense::route_meter(bool reset) {
    Impl & im = *p_;
    RouteMeter m;
    m.on = !im.host_xids.empty();
    m.slots = im.rm_slots;
    m.swaps = im.rm_swaps;
    m.weight_kept = im.rm_rows > 0 ? im.rm_kept / (double) im.rm_rows : 1.0;
    if (reset) { im.rm_slots = im.rm_swaps = im.rm_rows = 0; im.rm_kept = 0; }
    return m;
}

void Ds41Dense::set_oom_hook(std::function<bool()> hook) { p_->oom_hook = std::move(hook); }

void Ds41Dense::reset_span_width() { p_->span_max = 0; }

void Ds41Dense::release_pass() {
    Impl & im = *p_;
    if (!im.allo_pass.a) return;
    ggml_gallocr_free(im.allo_pass.a);
    im.allo_pass.a = ggml_gallocr_new(ggml_backend_get_default_buffer_type(im.backend));
    im.allo_pass.graph_epoch.clear();
    im.allo_pass.compute_bufs.clear();
    im.allo_pass.failed = false;
    ++im.allo_pass.epoch;
    im.span_max = 0;   // the next prompt starts at full sub-batch width again (see attn_router_span)
}

// pass_cap (per ratio) of a pass of n tokens ending at pos0+n: the visible compressed rows it ranks over.
static void span_pass_caps(const Ds41Dense::Impl & im, int pos0, int n, int64_t out[3]) {
    out[0] = 0;
    for (int r = 1; r <= 2; ++r) {
        const int64_t nvis_last = (pos0 + n) / r;
        out[r] = nvis_last == 0 ? 0 : std::min<int64_t>(next_pow2(std::max<int64_t>(nvis_last, im.CBS)), im.cmax[r]);
    }
}

size_t Ds41Dense::span_graph_bytes(int pos_end, int n) {
    Impl & im = *p_;
    n = (int) std::max<int64_t>(1, std::min<int64_t>(n, im.S));
    pos_end = (int) std::max<int64_t>(n, std::min<int64_t>(pos_end, im.cfg.ctx_max));
    int64_t caps[3];
    span_pass_caps(im, pos_end - n, n, caps);
    const std::array<int64_t, 3> key { caps[1], caps[2], n };
    auto it = im.graph_bytes_memo.find(key);
    if (it != im.graph_bytes_memo.end()) return it->second;
    // build every layer KIND's span graph (they differ in ratio, kv / index / candidate source, Engram) and let a
    // throwaway allocator plan it: ggml_gallocr_reserve_n_size sizes the buffer without allocating anything
    int64_t saved[3] = { im.pass_cap[0], im.pass_cap[1], im.pass_cap[2] };
    for (int r = 0; r < 3; ++r) im.pass_cap[r] = caps[r];
    const bool saved_rows = im.rows_mode;   // a single-sequence span's graph (a batch pass before it left the flag set)
    im.rows_mode = false;
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(im.backend));
    std::vector<uint8_t> meta(ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false));
    std::set<std::array<int64_t, 6>> kinds;
    size_t mx = 0;
    for (int64_t l = 0; l < im.nl; ++l) {
        const Impl::Layer & L = im.ly[(size_t) l];
        if (!kinds.insert({ L.ratio, L.kv_src, L.idx_src, L.cand_src, L.uses_cand, L.eslot >= 0 }).second) continue;
        ggml_init_params p = { meta.size(), meta.data(), true };
        ggml_context * gc = ggml_init(p);
        ggml_cgraph * gf = DPROF_BUILD(dprof::K_PROBE, (int) l, build_attn(im, gc, (int) l, n, 0));
        size_t sz = 0;
        ggml_gallocr_reserve_n_size(ga, gf, nullptr, nullptr, &sz);
        mx = std::max(mx, sz);
        ggml_free(gc);
    }
    ggml_gallocr_free(ga);
    for (int r = 0; r < 3; ++r) im.pass_cap[r] = saved[r];
    im.rows_mode = saved_rows;
    im.graph_bytes_memo[key] = mx;
    return mx;
}

// TASK 2: a prompt's FIRST attention span pays costs that never come back.  DS41_ATTN_PROF (2026-10-25) billed that
// span (layer 0, col0 0) at 1,825 ms where every later span's same nodes cost ~1.6 ms: the two DH-wide f32 -> f16 K
// casts measured 850 ms EACH (2.6 warm) and the merged F16 gather 94 ms (1.6 warm).  The ratios say what it is - the
// small casts are hit 327x while the 384 MiB gather is hit 59x, so it is per-KERNEL one-shot cost (first launch), not
// per-byte - and either way the cure is the same: run the identical op / type / shape signatures once here, at setup,
// on scratch tensors.  No model state is read or written and the scratch graph and its buffer are freed before the
// prompt.  The scratch destination is the real [DH, M, 1, n] F16 key tensor, so its 384 MiB are first-touched here too.
bool Ds41Dense::warm_attn(int span) {
    Impl & im = *p_;
    if (im.cpu || !im.backend || im.DH < 1 || im.SWA < 1) return true;
    auto now = [] {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
    };
    const double t0 = now();
    const int64_t DH = im.DH, SWA = im.SWA, RAW = im.RAW;
    const int64_t n = std::max<int64_t>(1, std::min<int64_t>((int64_t) span, im.S > 0 ? im.S : (int64_t) span));
    const int64_t K = 512;                     // the top-k an index-sourced layer selects (what the flash-attn pads to)
    int64_t M = SWA + K;
    if (M % 256) M += 256 - M % 256;           // the key count the flash-attn's 256-key block scan wants
    std::vector<uint8_t> meta(ggml_tensor_overhead() * 256 + ggml_graph_overhead_custom(256, false));
    ggml_init_params ip = { meta.size(), meta.data(), true };
    ggml_context * gc = ggml_init(ip);
    if (!gc) return false;
    bool ok = false;
    ggml_gallocr_t ga = nullptr;
    do {
        // the merged F16 key source of one span: the ring rows and this pass's rows (both cast f32 -> f16 at DH width,
        // the two 850 ms nodes), the compressed pool, and the ONE gather that drives the merged index into the
        // [DH, M, 1, n] F16 key tensor (the 94 ms node).  The mask cast (M x n f32 -> f16) rides along.
        ggml_tensor * ring = ggml_new_tensor_2d(gc, GGML_TYPE_F32, DH, RAW);
        ggml_tensor * kv2  = ggml_new_tensor_2d(gc, GGML_TYPE_F32, DH, n);
        ggml_tensor * comp = ggml_new_tensor_2d(gc, GGML_TYPE_F16, DH, K);
        ggml_tensor * src16 = ggml_concat(gc, ggml_cast(gc, ring, GGML_TYPE_F16), ggml_cast(gc, kv2, GGML_TYPE_F16), 1);
        src16 = ggml_concat(gc, src16, comp, 1);
        ggml_tensor * idx = ggml_new_tensor_1d(gc, GGML_TYPE_I32, M * n);
        ggml_tensor * k16 = ggml_reshape_4d(gc, rows_as(gc, src16, idx, true), DH, M, 1, n);
        ggml_tensor * mask = ggml_cast(gc, ggml_new_tensor_2d(gc, GGML_TYPE_F32, M, n), GGML_TYPE_F16);
        ggml_cgraph * gf = ggml_new_graph_custom(gc, 256, false);
        ggml_build_forward_expand(gf, k16);
        ggml_build_forward_expand(gf, mask);
        ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(im.backend));
        if (!ga || !ggml_gallocr_alloc_graph(ga, gf)) break;
        // the gather reads its index: all zeros = the source's row 0 (in bounds) - garbage ids would read past the source
        if (idx->buffer) {
            std::vector<int32_t> z((size_t) (M * n), 0);
            ggml_backend_tensor_set(idx, z.data(), 0, z.size() * 4);
        }
        if (ggml_backend_graph_compute(im.backend, gf) != GGML_STATUS_SUCCESS) break;
        ggml_backend_synchronize(im.backend);
        ok = true;
    } while (false);
    if (ga) ggml_gallocr_free(ga);
    ggml_free(gc);
    std::fprintf(stderr, "attn warm: the core's one-shot copy/gather kernels paid at setup (n %lld, M %lld, %.0f ms%s)\n",
                 (long long) n, (long long) M, now() - t0, ok ? "" : ", SCRATCH GRAPH FAILED");
    return ok;
}

// Wait for the pass's Engram reads (started by upload_inputs) and push the rows into the state tensor's columns.  A
// no-op once done (eng_batch is null), so it is safe to call at every layer - but the caller gates it on L.eslot >= 0
// so the layers BELOW the first Engram layer run while the disk is still reading.  The whole pass's rows (both Engram
// slots, all n columns) go up in one shot, so the second Engram layer (14) is a plain read of the same tensor.
static bool engram_ready(Ds41Dense::Impl & im) {
    DPROF_SPAN(dprof::C_ENGRAM_WAIT);
    if (!im.eng_batch) return true;
    if (!im.engram.wait(im.eng_batch, im.err)) { im.eng_batch.reset(); return false; }
    im.eng_batch.reset();
    for (size_t e = 0; e < im.eng_rows.size() && e < 2; ++e)
        DPROF_SET(im.i_eng[e], im.eng_rows[e].data(), 0, im.eng_rows[e].size() * 4);
    return true;
}

// Per-token inputs of n tokens at positions pos0.. (embeddings, positions, the window / compressed / indexer index
// maps, the Engram rows and the per-pass ring slots).  Shared by begin() and begin_batch().
bool Ds41Dense::Impl::upload_inputs(const int * tids, int n, int pos0) {
    Impl & im = *this;
    DPROF_SPAN(dprof::C_UPLOAD);
    im.span_memo_pos0 = -1;   // this writes i_widx / i_wslot / i_sidx / i_sslot: upload_span's memo is stale
    const int64_t D = im.D, SWA = im.SWA, RAW = im.RAW;
    // embeddings (F16 rows on the mmap)
    im.host_emb.resize((size_t) (D * n));
    for (int t = 0; t < n; ++t) {
        if (tids[t] < 0 || tids[t] >= im.g.vocab) { im.err = "token id out of range"; return false; }
        const uint16_t * row = reinterpret_cast<const uint16_t *>(im.embd) + (size_t) tids[t] * (size_t) D;
        for (int64_t i = 0; i < D; ++i) im.host_emb[(size_t) (t * D + i)] = ggml_fp16_to_fp32(row[i]);
    }
    // vision: overwrite the image span positions with the encoder's rows (model.py merge_image_embeddings) and mark
    // them, so the router picks exp_probs_b_vl there and Engram skips them.  vis_pos is ascending, the pass contiguous.
    std::vector<float> img;
    if (im.vision) {
        img.assign((size_t) n, 0.0f);
        for (int v = 0; v < im.n_vis; ++v) {
            const int64_t p = im.vis_pos[v];
            if (p < pos0 || p >= (int64_t) pos0 + n) continue;
            const int t = (int) (p - pos0);
            std::memcpy(im.host_emb.data() + (size_t) t * D, im.vis_rows + (size_t) v * D, (size_t) D * 4);
            img[(size_t) t] = 1.0f;
        }
        DPROF_SET(im.i_img, img.data(), 0, (size_t) n * 4);
    }
    DPROF_SET(im.i_emb, im.host_emb.data(), 0, im.host_emb.size() * 4);
    std::vector<int32_t> pos(n);
    for (int t = 0; t < n; ++t) pos[t] = pos0 + t;
    DPROF_SET(im.i_pos, pos.data(), 0, (size_t) n * 4);
    // window: per query the SWA positions [p - SWA + 1, p] as rows of [ring (RAW) | pass rows]
    std::vector<int32_t> widx((size_t) (SWA * n));
    std::vector<float> wm((size_t) (SWA * n));
    for (int t = 0; t < n; ++t) {
        const int64_t p = pos0 + t;
        for (int64_t j = 0; j < SWA; ++j) {
            const int64_t q = p - SWA + 1 + j;
            int32_t row = 0;
            float m = 0.0f;
            if (q < 0) m = NEG_INF;
            else if (q >= pos0) row = (int32_t) (RAW + (q - pos0));
            else row = (int32_t) (q % RAW);
            widx[(size_t) (t * SWA + j)] = row;
            wm[(size_t) (t * SWA + j)] = m;
        }
    }
    DPROF_SET(im.i_widx, widx.data(), 0, widx.size() * 4);
    DPROF_SET(im.i_wmask, wm.data(), 0, wm.size() * 4);
    {
        const int64_t k = std::min<int64_t>(n, RAW);
        std::vector<int32_t> ws((size_t) k);
        for (int64_t j = 0; j < k; ++j) ws[(size_t) j] = (int32_t) ((pos0 + n - k + j) % RAW);
        DPROF_SET(im.i_wslot, ws.data(), 0, ws.size() * 4);
    }
    // compressed rows per ratio
    for (int r = 1; r <= 2; ++r) {
        std::vector<float> nv(n);
        std::vector<int32_t> cs(n), cp(n);
        for (int t = 0; t < n; ++t) {
            const int64_t p = pos0 + t;
            nv[t] = (float) ((p + 1) / r);
            const bool done = (p + 1) % r == 0;
            cs[t] = (int32_t) (done ? p / r : im.cmax[r]);   // an open group writes the spare row
            cp[t] = (int32_t) ((p / r) * r);
        }
        DPROF_SET(im.i_nvis[r], nv.data(), 0, (size_t) n * 4);
        DPROF_SET(im.i_cslot[r], cs.data(), 0, (size_t) n * 4);
        DPROF_SET(im.i_cpos[r], cp.data(), 0, (size_t) n * 4);
        if (r == 1) {
            std::vector<float> lb(n);
            for (int t = 0; t < n; ++t) lb[t] = (float) ((nv[t] - 1) >= 0 ? (int64_t) (nv[t] - 1) / im.CBS : -1);
            DPROF_SET(im.i_lastblk, lb.data(), 0, (size_t) n * 4);
        }
    }
    {   // ratio-2 state: rows of [ring (2) | pass states] for each token's group (2g, 2g+1); open groups repeat 2g
        std::vector<int32_t> si((size_t) (2 * n)), ss;
        auto row = [&](int64_t q) { return (int32_t) (q >= pos0 ? 2 + (q - pos0) : q % 2); };
        for (int t = 0; t < n; ++t) {
            const int64_t p = pos0 + t, g0 = (p / 2) * 2;
            si[(size_t) (2 * t)] = row(g0);
            si[(size_t) (2 * t + 1)] = row(p >= g0 + 1 ? g0 + 1 : g0);
        }
        const int64_t k = std::min<int64_t>(n, 2);
        for (int64_t j = 0; j < k; ++j) ss.push_back((int32_t) ((pos0 + n - k + j) % 2));
        DPROF_SET(im.i_sidx, si.data(), 0, si.size() * 4);
        DPROF_SET(im.i_sslot, ss.data(), 0, ss.size() * 4);
    }
    // Engram rows: hash the whole pass now and START the (dedup/sort/coalesce) reads; they are waited - and the rows
    // moved to the state tensor - by the first Engram layer's span (engram_ready), so the disk read overlaps the
    // embeddings upload, the init graph and the layers below it.  A tiny batch (decode) still runs on the caller.
    if (im.engram.enabled()) {
        if (!engram_ready(im)) return false;    // flush a still-pending batch from a previous pass first
        for (int t = 0; t < n; ++t) {
            if (!img.empty() && img[(size_t) t] > 0.0f) im.engram.set_dead(pos0 + t);   // an image span: n-grams never span it
            else im.engram.set_token(pos0 + t, tids[t]);
        }
        if (n <= 64) {   // decode / verify: the rows are needed at layer 1 - read them now (synchronously)
            static const bool eprof = std::getenv("STRATA_ENG_PROF") != nullptr;
            static double eng_ms = 0; static long eng_n = 0;
            const auto te = std::chrono::steady_clock::now();
            bool eng_ok;
            { DPROF_SPAN(dprof::C_ENGRAM_ROWS); eng_ok = im.engram.rows(pos0, n, im.eng_rows, im.err); }
            if (!eng_ok) return false;
            if (eprof) {
                eng_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - te).count();
                if (++eng_n % 64 == 0) std::fprintf(stderr, "engram decode reads: %.3f ms/call over %ld calls\n", eng_ms / eng_n, eng_n);
            }
            for (size_t e = 0; e < im.eng_rows.size() && e < 2; ++e)
                DPROF_SET(im.i_eng[e], im.eng_rows[e].data(), 0, im.eng_rows[e].size() * 4);
        } else {
            im.eng_batch = im.engram.start(pos0, n, im.eng_rows, im.err);
            if (!im.eng_batch) return false;
        }
    }
    return true;
}

bool Ds41Dense::begin(const int * tids, int n, int pos0) {
    Impl & im = *p_;
    DPROF_SPAN(dprof::C_BEGIN);
    if (dprof::g_on) ++dprof::st().begins;
    decprof::KindGuard dec_kind(decprof::G_INIT);   // DS41_DEC_PROF: charge this graph to init (inert when unset)
    if (n < 1 || n > im.NC || pos0 < 0 || pos0 + n > im.cfg.ctx_max) { im.err = "bad pass"; return false; }
    im.pos0 = pos0;
    im.n = n;
    im.rows_mode = false;
    const double du0 = decprof::g_on ? decprof::now_ms() : 0.0;
    if (!im.upload_inputs(tids, n, pos0)) return false;
    if (decprof::g_on) decprof::add_upload(decprof::now_ms() - du0);   // token/position uploads + the Engram row reads
    span_pass_caps(im, pos0, n, im.pass_cap);
    // embeddings -> state
    ggml_context * gc = nullptr;
    ggml_cgraph * gf = nullptr;
    const bool cached = n == 1;
    const bool mcached = pass_graphs_on(n);
    std::vector<uint8_t> meta;
    if (cached) {
        auto it = im.graphs.find({ -1, 0, 0, 1 });
        if (it == im.graphs.end()) it = im.graphs.emplace(Impl::GKey { -1, 0, 0, 1 }, DPROF_BUILD(dprof::K_INIT, -1, build_init(im, im.gctx, 1))).first;
        gf = it->second;
    } else if (mcached) {
        gf = pass_graph(im, Impl::GKey { -1, 0, 0, n }, [&](ggml_context * c) { return DPROF_BUILD(dprof::K_INIT, -1, build_init(im, c, n)); });
    } else {
        meta.resize(ggml_tensor_overhead() * 256 + ggml_graph_overhead_custom(64, false));
        ggml_init_params p = { meta.size(), meta.data(), true };
        gc = ggml_init(p);
        gf = DPROF_BUILD(dprof::K_INIT, -1, build_init(im, gc, n));
    }
    const bool ok = im.compute(gf, cached || mcached);
    if (gc) ggml_free(gc);
    return ok;
}

// ================================================================== batch slots: a pass over rows of several slots

// The per-row inputs of a batch pass: as upload_inputs for each slot's group of rows (its own ring rows, its own
// [state ring | pass states], its own Engram history), the per-group ring / state slots at the group's first column.
bool Ds41Dense::Impl::upload_rows(const Ds41Dense::Row * rows, int n) {
    Impl & im = *this;
    im.span_memo_pos0 = -1;   // i_widx / i_wslot / i_sidx / i_sslot are rewritten
    const int64_t D = im.D, SWA = im.SWA, RAW = im.RAW;
    im.host_emb.resize((size_t) (D * n));
    for (int t = 0; t < n; ++t) {
        if (rows[t].tid < 0 || rows[t].tid >= im.g.vocab) { im.err = "token id out of range"; return false; }
        const uint16_t * row = reinterpret_cast<const uint16_t *>(im.embd) + (size_t) rows[t].tid * (size_t) D;
        for (int64_t i = 0; i < D; ++i) im.host_emb[(size_t) (t * D + i)] = ggml_fp16_to_fp32(row[i]);
    }
    DPROF_SET(im.i_emb, im.host_emb.data(), 0, im.host_emb.size() * 4);
    if (im.vision && im.i_img) {   // text rows only
        std::vector<float> z((size_t) n, 0.0f);
        DPROF_SET(im.i_img, z.data(), 0, (size_t) n * 4);
    }
    std::vector<int32_t> pos(n), widx((size_t) (SWA * n)), ws, si((size_t) (2 * n)), ss((size_t) (2 * n), 0), scol;
    std::vector<float> wm((size_t) (SWA * n));
    ws.assign((size_t) n, 0);
    // slabs (attn_core_slab): every index is a slab row - the ring region of the row's slot (its own rows are written to
    // the ring before the gather), the state region (or the pass's own states past 2 x n_slots), the compressed region
    const bool slab = im.slab;
    const int64_t NSS = (int64_t) im.slots.size();
    im.rows_nss = 0;
    for (const RowGroup & G : im.rgroups) {
        const int64_t p0 = G.pos0, k = G.k, rb = (int64_t) G.reg * RAW;
        for (int64_t i = 0; i < k; ++i) {
            const int64_t t = G.c + i, p = p0 + i;
            pos[(size_t) t] = (int32_t) p;
            for (int64_t j = 0; j < SWA; ++j) {   // per query the SWA positions [p - SWA + 1, p]: [its slot's ring | its rows]
                const int64_t q = p - SWA + 1 + j;
                int32_t row = 0;
                float m = 0.0f;
                if (q < 0) m = NEG_INF;
                else if (slab) row = (int32_t) (rb + q % RAW);
                else if (q >= p0) row = (int32_t) (RAW + (q - p0));
                else row = (int32_t) (q % RAW);
                widx[(size_t) (t * SWA + j)] = row;
                wm[(size_t) (t * SWA + j)] = m;
            }
            const int64_t g0 = (p / 2) * 2;   // ratio-2 state rows of [its slot's state ring (2) | its pass states]
            auto srow = [&](int64_t q) {
                if (slab) return (int32_t) (q >= p0 ? 2 * NSS + G.c + (q - p0) : 2 * G.reg + q % 2);
                return (int32_t) (q >= p0 ? 2 + (q - p0) : q % 2);
            };
            si[(size_t) (2 * t)] = srow(g0);
            si[(size_t) (2 * t + 1)] = srow(p >= g0 + 1 ? g0 + 1 : g0);
            if (slab) ws[(size_t) t] = (int32_t) (rb + p % RAW);
        }
        const int64_t kw = std::min<int64_t>(k, RAW), ks = std::min<int64_t>(k, 2);
        if (slab) {
            for (int64_t j = 0; j < ks; ++j) {
                scol.push_back((int32_t) (G.c + k - ks + j));
                ss[(size_t) im.rows_nss++] = (int32_t) (2 * G.reg + (p0 + k - ks + j) % 2);
            }
            continue;
        }
        for (int64_t j = 0; j < kw; ++j) ws[(size_t) (G.c + j)] = (int32_t) ((p0 + k - kw + j) % RAW);
        for (int64_t j = 0; j < ks; ++j) ss[(size_t) (2 * G.c + j)] = (int32_t) ((p0 + k - ks + j) % 2);
    }
    if (slab && !scol.empty()) DPROF_SET(im.i_sscol, scol.data(), 0, scol.size() * 4);
    DPROF_SET(im.i_pos, pos.data(), 0, (size_t) n * 4);
    DPROF_SET(im.i_widx, widx.data(), 0, widx.size() * 4);
    DPROF_SET(im.i_wmask, wm.data(), 0, wm.size() * 4);
    DPROF_SET(im.i_wslot, ws.data(), 0, (size_t) std::min<int64_t>(n, im.i_wslot->ne[0]) * 4);
    DPROF_SET(im.i_sidx, si.data(), 0, si.size() * 4);
    DPROF_SET(im.i_sslot, ss.data(), 0, (size_t) std::min<int64_t>(2 * n, im.i_sslot->ne[0]) * 4);
    for (int r = 1; r <= 2; ++r) {   // compressed rows per ratio: per position, as upload_inputs
        std::vector<float> nv(n), lb(n);
        std::vector<int32_t> cs(n), cp(n);
        std::vector<float> cb(slab ? n : 0);
        for (const RowGroup & G : im.rgroups)
            for (int64_t i = 0; i < G.k; ++i) {
                const int64_t t = G.c + i, p = pos[(size_t) t];
                const int64_t base = slab ? (int64_t) G.reg * (im.cmax[r] + 1) : 0;   // the slot's compressed region
                nv[t] = (float) ((p + 1) / r);
                const bool done = (p + 1) % r == 0;
                cs[t] = (int32_t) (base + (done ? p / r : im.cmax[r]));   // an open group writes the spare row
                cp[t] = (int32_t) ((p / r) * r);
                lb[t] = (float) ((nv[t] - 1) >= 0 ? (int64_t) (nv[t] - 1) / im.CBS : -1);
                if (slab) cb[(size_t) t] = (float) base;
            }
        if (slab) DPROF_SET(im.i_cbase[r], cb.data(), 0, (size_t) n * 4);
        DPROF_SET(im.i_nvis[r], nv.data(), 0, (size_t) n * 4);
        DPROF_SET(im.i_cslot[r], cs.data(), 0, (size_t) n * 4);
        DPROF_SET(im.i_cpos[r], cp.data(), 0, (size_t) n * 4);
        if (r == 1) DPROF_SET(im.i_lastblk, lb.data(), 0, (size_t) n * 4);
    }
    if (im.engram.enabled()) {   // each row hashes its own slot's history (written here first, as a solo pass does)
        if (!engram_ready(im)) return false;
        im.rows_slot.resize((size_t) n);
        im.rows_pos.resize((size_t) n);
        for (int t = 0; t < n; ++t) {
            im.engram.set_token_in(rows[t].slot, rows[t].pos, rows[t].tid);
            im.rows_slot[(size_t) t] = rows[t].slot;
            im.rows_pos[(size_t) t] = rows[t].pos;
        }
        if (!im.engram.rows_at(im.rows_slot.data(), im.rows_pos.data(), n, im.eng_rows, im.err)) return false;
        for (size_t e = 0; e < im.eng_rows.size() && e < 2; ++e)
            DPROF_SET(im.i_eng[e], im.eng_rows[e].data(), 0, im.eng_rows[e].size() * 4);
    }
    return true;
}

bool Ds41Dense::begin_rows(const Row * rows, int n) {
    Impl & im = *p_;
    if (n < 1 || n > kMaxRows || n > im.NC) { im.err = "begin_rows: 1.." + std::to_string(std::min<int64_t>(kMaxRows, im.NC)) + " rows"; return false; }
    if (im.eng_batch) { im.err = "begin_rows: a pass's Engram reads are pending"; return false; }
    im.rgroups.clear();
    for (int i = 0; i < n; ++i) {
        const Row & R = rows[i];
        if (R.slot < 0 || R.slot >= (int) im.slots.size() || R.pos < 0 || R.pos >= im.cfg.ctx_max) { im.err = "begin_rows: bad slot / position"; return false; }
        if (i > 0 && R.slot == rows[i - 1].slot) {
            if (R.pos != rows[i - 1].pos + 1) { im.err = "begin_rows: a slot's rows must be consecutive positions"; return false; }
            ++im.rgroups.back().k;
            continue;
        }
        for (const auto & G : im.rgroups)
            if (G.slot == R.slot) { im.err = "begin_rows: a slot's rows must be contiguous"; return false; }
        im.rgroups.push_back(Impl::RowGroup { R.slot, i, 1, R.pos, { 0, 0, 0 }, im.slot_region(R.slot) });
    }
    for (auto & G : im.rgroups) span_pass_caps(im, (int) G.pos0, (int) G.k, G.cap);
    im.rows_mode = true;
    im.rows_sig = 0;
    static const bool rows_cache = !(std::getenv("STRATA_ROWS_GRAPH_CACHE") && std::getenv("STRATA_ROWS_GRAPH_CACHE")[0] == '0');
    if (rows_cache) {
        std::vector<int64_t> sig;
        for (const auto & G : im.rgroups) { sig.push_back(G.slot); sig.push_back(G.c); sig.push_back(G.k); sig.push_back(G.cap[1]); sig.push_back(G.cap[2]); sig.push_back(G.reg); }
        uint64_t h = 1469598103934665603ull;
        for (int64_t v : sig) { h ^= (uint64_t) v; h *= 1099511628211ull; }
        int64_t key = (int64_t) (h & 0x3fffffffffffffffull) | 1;
        auto it = im.rows_sigs.find(key);
        if (it == im.rows_sigs.end()) im.rows_sigs.emplace(key, sig);
        else if (it->second != sig) key = 0;   // a collision: this layout runs uncached
        im.rows_sig = key;
    }
    im.pos0 = 0;
    im.n = n;
    im.pass_cap[0] = im.pass_cap[1] = im.pass_cap[2] = 0;   // per group (read_sel has nothing to show)
    if (!im.upload_rows(rows, n)) { im.rows_mode = false; return false; }
    // embeddings -> state, as begin()
    if (im.rows_sig) {
        const Impl::GKey key { -1, 10, 0, n, -1 };
        auto it = im.graphs.find(key);
        if (it == im.graphs.end()) {
            if (!im.gctx_room()) im.reset_graphs();
            it = im.graphs.emplace(key, DPROF_BUILD(dprof::K_INIT, -1, build_init(im, im.gctx, n))).first;
        }
        return im.compute(it->second, true);
    }
    std::vector<uint8_t> meta(ggml_tensor_overhead() * 256 + ggml_graph_overhead_custom(64, false));
    ggml_init_params p = { meta.size(), meta.data(), true };
    ggml_context * gc = ggml_init(p);
    ggml_cgraph * gf = DPROF_BUILD(dprof::K_INIT, -1, build_init(im, gc, n));
    const bool ok = im.compute(gf, false);
    ggml_free(gc);
    return ok;
}

// ================================================================== the layer-major prompt batch

bool Ds41Dense::begin_batch(const int * tids, int B, int pos0) {
    Impl & im = *p_;
    if (B < 1 || B > im.cfg.prompt_batch || B > im.NC || pos0 < 0 || pos0 + B > im.cfg.ctx_max) { im.err = "bad batch"; return false; }
    if (!im.upload_inputs(tids, B, pos0)) return false;
    im.pos0 = pos0; im.batch_pos0 = pos0; im.batch = B; im.n = B; im.col0 = 0;
    im.rows_mode = false;
    // per-sub-batch pass_cap (the compressed rows each sub-batch ranks over) - depends only on (pos0, len)
    const int64_t nb = (B + im.S - 1) / im.S;
    im.span_cap.assign((size_t) nb, { 0, 0, 0 });
    for (int64_t i = 0; i < nb; ++i)
        span_pass_caps(im, pos0 + (int) (i * im.S), (int) std::min<int64_t>(im.S, (int64_t) B - i * im.S),
                       im.span_cap[(size_t) i].data());
    // the batch's initial state, one sub-batch at a time (x_state = embd, pre_state = one-hot)
    std::vector<uint8_t> meta(ggml_tensor_overhead() * 256 + ggml_graph_overhead_custom(64, false));
    for (int64_t i = 0; i < nb; ++i) {
        const int64_t c0 = i * im.S, sn = std::min<int64_t>(im.S, (int64_t) B - c0);
        ggml_init_params p = { meta.size(), meta.data(), true };
        ggml_context * gc = ggml_init(p);
        ggml_cgraph * gf = DPROF_BUILD(dprof::K_INIT, -1, build_init(im, gc, sn, c0));
        const bool ok = im.compute(gf, false);
        ggml_free(gc);
        if (!ok) { im.err = "batch init failed"; return false; }
    }
    return true;
}

// The per-span inputs of one layer-major sub-batch: the sub-batch's own pass_cap (the graphs read it) and the
// PASS-LOCAL row maps (the pass's own rows / states sit at the end of the [ring | pass rows] they index, so they are
// recomputed from the sub-batch's own position - upload_inputs laid them out for the whole batch, correct only at
// col0 = 0).  Shared by attn_router_span and attn_router_span_kv.
// A span is a whole sub-batch (col0 % S == 0) or, when a sub-batch's graph did not fit, a narrower piece of one
// (attn_router_span): a piece at an unaligned column gets the pass_cap of its own end position.
static bool upload_span(Ds41Dense::Impl & im, int col0, int n) {
    if (col0 < 0 || n < 1 || im.S < 1 || (int64_t) col0 + n > im.batch) { im.err = "bad span"; return false; }
    if (col0 % im.S == 0) {
        const size_t si = (size_t) (col0 / im.S);
        if (si >= im.span_cap.size()) { im.err = "bad span"; return false; }
        im.pass_cap[0] = im.span_cap[si][0];
        im.pass_cap[1] = im.span_cap[si][1];
        im.pass_cap[2] = im.span_cap[si][2];
    } else {
        span_pass_caps(im, im.batch_pos0 + col0, n, im.pass_cap);
    }
    const int64_t pbase = (int64_t) im.batch_pos0 + col0;
    static const bool memo_on = !(std::getenv("STRATA_SPAN_MEMO") && std::getenv("STRATA_SPAN_MEMO")[0] == '0');
    if (memo_on && im.span_memo_pos0 == (int64_t) im.batch_pos0 && im.span_memo_col0 == col0 && im.span_memo_n == n) return true;
    im.span_memo_pos0 = (int64_t) im.batch_pos0;
    im.span_memo_col0 = col0;
    im.span_memo_n = n;
    {
        std::vector<int32_t> widx((size_t) (im.SWA * n));
        for (int t = 0; t < n; ++t) {
            const int64_t p = pbase + t;
            for (int64_t j = 0; j < im.SWA; ++j) {
                const int64_t q = p - im.SWA + 1 + j;
                widx[(size_t) (t * im.SWA + j)] = q < 0 ? 0 : (int32_t) (q >= pbase ? im.RAW + (q - pbase) : q % im.RAW);
            }
        }
        DPROF_SET(im.i_widx, widx.data(), (size_t) col0 * im.SWA * 4, widx.size() * 4);
    }
    {
        std::vector<int32_t> si((size_t) (2 * n));
        auto row = [&](int64_t q) { return (int32_t) (q >= pbase ? 2 + (q - pbase) : q % 2); };
        for (int t = 0; t < n; ++t) {
            const int64_t p = pbase + t, g0 = (p / 2) * 2;
            si[(size_t) (2 * t)] = row(g0);
            si[(size_t) (2 * t + 1)] = row(p >= g0 + 1 ? g0 + 1 : g0);
        }
        DPROF_SET(im.i_sidx, si.data(), (size_t) col0 * 2 * 4, si.size() * 4);
    }
    {
        const int64_t k = std::min<int64_t>(n, im.RAW);
        std::vector<int32_t> ws((size_t) k);
        for (int64_t j = 0; j < k; ++j) ws[(size_t) j] = (int32_t) ((pbase + n - k + j) % im.RAW);
        DPROF_SET(im.i_wslot, ws.data(), 0, ws.size() * 4);
    }
    {
        const int64_t k = std::min<int64_t>(n, 2);
        std::vector<int32_t> ss((size_t) k);
        for (int64_t j = 0; j < k; ++j) ss[(size_t) j] = (int32_t) ((pbase + n - k + j) % 2);
        DPROF_SET(im.i_sslot, ss.data(), 0, ss.size() * 4);
    }
    return true;
}

// attn_router_span / attn_router_span_kv run their span as pieces of at most span_max columns (the whole span until a
// graph does not fit).  A piece whose GRAPH ALLOCATION fails - nothing has run, the state is untouched - is retried as
// halves down to kMinSpan columns, and span_max keeps the width that fitted for the rest of the prompt, so a tight
// card (a long context's graph, another process taking VRAM mid-prompt) makes the prompt a little slower instead of
// failing it.  The pieces walk the columns in order, as the sub-batches do, so every position-indexed state (window
// ring, compressor state, compressed rows, candidate blocks) advances exactly as before; the per-query arithmetic is
// the same, but the dense matmuls see fewer columns, so a split prompt is not bit-identical to an unsplit one (it only
// happens where the unsplit one would have failed).
static constexpr int kMinSpan = 16;
static bool span_kv_one(Ds41Dense::Impl & im, int il, int col0, int n);
static bool span_attn_one(Ds41Dense::Impl & im, int il, int col0, int n);
static bool run_spans(Ds41Dense::Impl & im, int il, int col0, int n, const std::function<bool(int, int)> & one) {
    int64_t w = im.span_max > 0 ? std::min<int64_t>(im.span_max, n) : n;
    for (int c = col0; c < col0 + n; ) {
        const int m = (int) std::min<int64_t>(w, (int64_t) col0 + n - c);
        im.alloc_failed = false;
        if (one(c, m)) { c += m; continue; }
        if (!im.alloc_failed || m <= kMinSpan) return false;
        w = std::max(kMinSpan, m / 2);
        im.span_max = w;
        std::fprintf(stderr, "ds41_dense: layer %d: the graph of a %d-column span did not fit - %lld-column spans for the rest "
                     "of this prompt\n", il, m, (long long) w);
    }
    return true;
}

bool Ds41Dense::attn_router_span_kv(int il, int col0, int n) {
    Impl & im = *p_;
    return run_spans(im, il, col0, n, [&](int c, int m) { return span_kv_one(im, il, c, m); });
}

static bool span_kv_one(Ds41Dense::Impl & im, int il, int col0, int n) {
    DPROF_SPAN(dprof::C_SPAN_KV);
    if (!upload_span(im, col0, n)) return false;
    if (im.ly[(size_t) il].eslot >= 0 && !engram_ready(im)) return false;   // the Engram add feeds this layer's K/V
    im.pf_meta.resize(ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false));
    ggml_init_params pk = { im.pf_meta.size(), im.pf_meta.data(), true };
    ggml_context * gck = ggml_init(pk);
    ggml_cgraph * gk = DPROF_BUILD(dprof::K_KV, il, build_attn_kv_only(im, gck, il, n, col0));
    const bool okk = im.compute(gk, false);
    ggml_free(gck);
    return okk;
}

// route meter: columns col0..col0+n-1 of the host ids against the exact top-k read beside them
static void meter_add(Ds41Dense::Impl & im, int64_t col0, int64_t n) {
    const int64_t K = im.NU;
    for (int64_t t = col0; t < col0 + n; ++t) {
        const int32_t * c = im.host_ids.data() + t * K, * x = im.host_xids.data() + t * K;
        const float * w = im.host_xwts.data() + t * K;
        for (int64_t j = 0; j < K; ++j) {
            bool kept = false;
            for (int64_t i = 0; i < K && !kept; ++i) kept = c[i] == x[j];
            if (kept) im.rm_kept += w[j]; else ++im.rm_swaps;
        }
        im.rm_slots += K;
        ++im.rm_rows;
    }
}

bool Ds41Dense::attn_router_span(int il, int col0, int n) {
    Impl & im = *p_;
    return run_spans(im, il, col0, n, [&](int c, int m) { return span_attn_one(im, il, c, m); });
}

static bool span_attn_one(Ds41Dense::Impl & im, int il, int col0, int n) {
    DPROF_SPAN(dprof::C_SPAN_ATTN);
    double tp = g_dprof_on ? dprof_now() : 0;
    auto lap = [&](int i) { if (g_dprof_on) { const double t = dprof_now(); g_dprof[i] += t - tp; tp = t; } };
    if (!upload_span(im, col0, n)) return false;
    if (g_dprof_on && dprof_now() - tp > 50) std::fprintf(stderr, "dense prof: slow upload_span layer %d col0 %d n %d: %.0f ms\n", il, col0, n, dprof_now() - tp);
    lap(6);
    if (im.ly[(size_t) il].eslot >= 0 && !engram_ready(im)) return false;   // wait the pass's Engram reads here
    lap(3);
    im.pf_meta.resize(ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false));
    ggml_init_params p = { im.pf_meta.size(), im.pf_meta.data(), true };
    ggml_context * gc = ggml_init(p);
    ggml_cgraph * gf = DPROF_BUILD(dprof::K_ATTN, il, build_attn(im, gc, il, n, col0));
    lap(0);
    g_dprof_slot = 0;
    const bool ok = im.compute(gf, false);
    g_dprof_slot = -1;
    if (g_dprof_on) { tp = dprof_now(); g_dprof[7] += 1; }
    ggml_free(gc);
    if (!ok) return false;
    // the sub-batch's ids / wts / fn land at its own batch columns (token-major: token col0+t at t * NU / t * D)
    DPROF_GET_ASYNC(im.backend, im.o_fn, im.host_fn.data() + (size_t) col0 * im.D, (size_t) col0 * im.D * 4, (size_t) n * im.D * 4);
    DPROF_GET_ASYNC(im.backend, im.o_ids, im.host_ids.data() + (size_t) col0 * im.NU, (size_t) col0 * im.NU * 4, (size_t) n * im.NU * 4);
    DPROF_GET_ASYNC(im.backend, im.o_wts, im.host_wts.data() + (size_t) col0 * im.NU, (size_t) col0 * im.NU * 4, (size_t) n * im.NU * 4);
    if (im.o_xids) {
        DPROF_GET_ASYNC(im.backend, im.o_xids, im.host_xids.data() + (size_t) col0 * im.NU, (size_t) col0 * im.NU * 4, (size_t) n * im.NU * 4);
        DPROF_GET_ASYNC(im.backend, im.o_xwts, im.host_xwts.data() + (size_t) col0 * im.NU, (size_t) col0 * im.NU * 4, (size_t) n * im.NU * 4);
    }
    DPROF_SYNC(im.backend);
    if (im.o_xids) meter_add(im, col0, n);
    lap(4);
    return true;
}

bool Ds41Dense::upload_routed(const float * routed_host, int B) {
    Impl & im = *p_;
    if (!routed_host || B < 1 || B > im.batch) { im.err = "bad routed batch"; return false; }
    DPROF_SET(im.routed, routed_host, 0, (size_t) B * im.D * 4);
    return true;
}

float * Ds41Dense::routed_dev() { return p_->cpu || !p_->routed ? nullptr : (float *) p_->routed->data; }

bool Ds41Dense::finish_span(int il, int col0, int n) {
    Impl & im = *p_;
    (void) il;
    if (col0 < 0 || n < 1 || im.S < 1 || col0 % im.S != 0 || (int64_t) col0 + n > im.batch) { im.err = "bad span"; return false; }
    const double tf0 = g_dprof_on ? dprof_now() : 0;
    std::vector<uint8_t> meta(ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false));
    ggml_init_params p = { meta.size(), meta.data(), true };
    ggml_context * gc = ggml_init(p);
    ggml_cgraph * gf = DPROF_BUILD(dprof::K_FINISH, -1, build_finish(im, gc, n, col0));
    if (g_dprof_on) g_dprof[5] += dprof_now() - tf0;
    g_dprof_slot = 0;
    const bool ok = im.compute(gf, false);
    g_dprof_slot = -1;
    ggml_free(gc);
    return ok;
}

void Ds41Dense::dprof_checkpoint(const char * tag) {
    dprof::checkpoint(tag);
    partprof::report(tag);
    attnprof::report(tag);   // DS41_ATTN_PROF: the core's buckets (inert when unset)
    if (partprof::on) {
        const Impl & im = *p_;
        std::fprintf(stderr, "dense geometry: D %lld DH %lld NH %lld RQ %lld SWA %lld RAW %lld OGD %lld OG %lld OL %lld "
                             "n_ff %lld n_shared %lld NE %lld NU %lld idx %lld/%lld/%lld\n",
                     (long long) im.D, (long long) im.DH, (long long) im.NH, (long long) im.RQ, (long long) im.SWA,
                     (long long) im.RAW, (long long) im.OGD, (long long) im.OG, (long long) im.OL,
                     (long long) im.g.n_ff, (long long) im.g.n_shared, (long long) im.NE, (long long) im.NU,
                     (long long) im.g.idx_heads, (long long) im.g.idx_dim, (long long) im.g.idx_topk);
    }
}

void Ds41Dense::prof_report(const char * tag) {
    if (!g_dprof_on) return;
    std::fprintf(stderr, "dense prof %s: %d attn spans: build %.0f ms, alloc %.0f, gpu compute %.0f, engram wait %.0f, "
                 "readback %.0f, upload %.0f, finish build %.0f\n", tag, (int) g_dprof[7], g_dprof[0], g_dprof[1], g_dprof[2],
                 g_dprof[3], g_dprof[4], g_dprof[6], g_dprof[5]);
    for (double & v : g_dprof) v = 0;
}

int Ds41Dense::batch_size() const { return (int) p_->batch; }
const int32_t * Ds41Dense::batch_ids() const { return p_->host_ids.data(); }
const float * Ds41Dense::batch_wts() const { return p_->host_wts.data(); }
const float * Ds41Dense::batch_fn() const { return p_->host_fn.data(); }

bool Ds41Dense::attn_router(int il, int * ids, float * wts, const float ** fn_host, ggml_tensor ** fn_dev) {
    Impl & im = *p_;
    DPROF_SPAN(dprof::C_ATTN);
    decprof::KindGuard dec_kind(decprof::G_ATTN);   // DS41_DEC_PROF: charge this graph to attn+router+shexp
    if (decprof::g_on) decprof::g_layers = (int) im.ly.size();
    const int64_t n = im.n;
    // Engram layer: wait the pass's reads (started in upload_inputs) before the graph that reads i_eng runs.  For the
    // chunk path this is the first Engram layer too, so the layers below it ran while the disk was reading.
    if (im.ly[(size_t) il].eslot >= 0 && !engram_ready(im)) return false;
    const int64_t r = im.ly[(size_t) il].ratio;
    const int64_t cap = r ? im.pass_cap[r] : 0;
    ggml_cgraph * gf = nullptr;
    ggml_context * gc = nullptr;
    // a batch pass's graph reads its rows' slots: cached by the pass's layout (rows_sig), or built per pass
    const bool cached = (n == 1 && !im.rows_mode) || (im.rows_mode && im.rows_sig != 0);
    // a single sequence's DSpark verify widths (1 < n <= 8): their own cache, keyed by the bound slot (the graph reads
    // that slot's rings / caches; swap_slots / bind_slot reset_graphs() drops it with the rest)
    const bool mcached = !im.rows_mode && pass_graphs_on(n);
    if (cached) {
        const Impl::GKey key = im.rows_mode ? Impl::GKey { il, 8, im.rows_sig, n, -1 } : Impl::GKey { il, 1, cap, 1, im.cur_slot };
        auto it = im.graphs.find(key);
        if (it == im.graphs.end()) {
            if (!im.gctx_room()) im.reset_graphs();   // many slots' decode graphs: start the cache over (rebuilt lazily)
            it = im.graphs.emplace(key, DPROF_BUILD(dprof::K_ATTN, il, build_attn(im, im.gctx, il, n))).first;
        }
        gf = it->second;
    } else if (mcached) {
        gf = pass_graph(im, Impl::GKey { il, 1, cap, n, im.cur_slot }, [&](ggml_context * c) { return DPROF_BUILD(dprof::K_ATTN, il, build_attn(im, c, il, n)); });
    } else {
        im.pf_meta.resize(ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false));
        ggml_init_params p = { im.pf_meta.size(), im.pf_meta.data(), true };
        gc = ggml_init(p);
        gf = DPROF_BUILD(dprof::K_ATTN, il, build_attn(im, gc, il, n));
    }
    const bool ok = im.compute(gf, cached || mcached);
    if (gc) ggml_free(gc);
    if (!ok) return false;
    // one sync for the three readbacks (was one per tensor_get: 3 x 40 stream syncs per decode token)
    DPROF_GET_ASYNC(im.backend, im.o_fn, im.host_fn.data(), 0, (size_t) (im.D * n) * 4);
    DPROF_GET_ASYNC(im.backend, im.o_ids, im.host_ids.data(), 0, (size_t) (im.NU * n) * 4);
    DPROF_GET_ASYNC(im.backend, im.o_wts, im.host_wts.data(), 0, (size_t) (im.NU * n) * 4);
    if (im.o_xids) {
        DPROF_GET_ASYNC(im.backend, im.o_xids, im.host_xids.data(), 0, (size_t) (im.NU * n) * 4);
        DPROF_GET_ASYNC(im.backend, im.o_xwts, im.host_xwts.data(), 0, (size_t) (im.NU * n) * 4);
    }
    DPROF_SYNC(im.backend);
    if (im.o_xids) meter_add(im, 0, n);
    std::memcpy(ids, im.host_ids.data(), (size_t) (im.NU * n) * 4);
    std::memcpy(wts, im.host_wts.data(), (size_t) (im.NU * n) * 4);
    *fn_host = im.host_fn.data();
    if (fn_dev) *fn_dev = im.o_fn;
    return true;
}

bool Ds41Dense::predict(int il, int * top_ids, int n_top) {
    Impl & im = *p_;
    DPROF_SPAN(dprof::C_PREDICT);
    decprof::KindGuard dec_kind(decprof::G_PREDICT);   // DS41_DEC_PROF
    if (im.n != 1 || n_top < 1 || n_top > 16) return false;
    const Impl::GKey key { il, 3, 0, 1 };
    auto it = im.graphs.find(key);
    if (it == im.graphs.end()) {
        ggml_tensor * o = nullptr;
        ggml_cgraph * gf = DPROF_BUILD(dprof::K_PREDICT, il, build_predict(im, im.gctx, il, &o));
        it = im.graphs.emplace(key, gf).first;
        im.predict_out[il] = o;
    }
    if (!im.compute(it->second, true)) return false;
    int32_t buf[16];
    DPROF_GET(im.predict_out[il], buf, 0, sizeof(buf));
    for (int i = 0; i < n_top; ++i) top_ids[i] = buf[i];
    return true;
}

bool Ds41Dense::predict_multi(int il, int * top_ids) {
    Impl & im = *p_;
    DPROF_SPAN(dprof::C_PREDICT);
    const int64_t n = im.n;
    if (n < 1) return false;
    if (n == 1) return predict(il, top_ids, 16);
    const Impl::GKey key { il, 3, 0, (int) n };   // one graph per (layer, pass width): verify passes reuse a few widths
    auto it = im.graphs.find(key);
    ggml_tensor * o = nullptr;
    if (it == im.graphs.end()) {
        ggml_cgraph * gf = DPROF_BUILD(dprof::K_PREDICT, il, build_predict(im, im.gctx, il, &o, n));
        it = im.graphs.emplace(key, gf).first;
        im.predict_out_n[{ il, (int) n }] = o;
    }
    o = im.predict_out_n[{ il, (int) n }];
    if (!im.compute(it->second, true)) return false;
    DPROF_GET(o, top_ids, 0, sizeof(int32_t) * 16 * (size_t) n);
    return true;
}

bool Ds41Dense::finish(int il, const float * routed_sum) {
    Impl & im = *p_;
    DPROF_SPAN(dprof::C_FINISH);
    decprof::KindGuard dec_kind(decprof::G_FINISH);   // DS41_DEC_PROF
    (void) il;
    const int64_t n = im.n;
    if (routed_sum) DPROF_SET(im.routed, routed_sum, 0, (size_t) (im.D * n) * 4);
    ggml_cgraph * gf = nullptr;
    ggml_context * gc = nullptr;
    const bool cached = n == 1 || (im.rows_mode && im.rows_sig != 0);   // (a batch pass: by its width)
    std::vector<uint8_t> meta;
    const bool mcached = !im.rows_mode && pass_graphs_on(n);
    if (cached) {
        const Impl::GKey key = n == 1 ? Impl::GKey { -1, 2, 0, 1 } : Impl::GKey { -1, 9, 0, n, -1 };
        auto it = im.graphs.find(key);
        if (it == im.graphs.end()) {
            if (n > 1 && !im.gctx_room()) im.reset_graphs();
            it = im.graphs.emplace(key, DPROF_BUILD(dprof::K_FINISH, -1, build_finish(im, im.gctx, n))).first;
        }
        gf = it->second;
    } else if (mcached) {
        gf = pass_graph(im, Impl::GKey { -1, 2, 0, n }, [&](ggml_context * c) { return DPROF_BUILD(dprof::K_FINISH, -1, build_finish(im, c, n)); });
    } else {
        meta.resize(ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false));
        ggml_init_params p = { meta.size(), meta.data(), true };
        gc = ggml_init(p);
        gf = DPROF_BUILD(dprof::K_FINISH, -1, build_finish(im, gc, n));
    }
    const bool ok = im.compute(gf, cached || mcached);
    if (gc) ggml_free(gc);
    return ok;
}

bool Ds41Dense::finish_predict(int il, const float * routed_sum, int next_il, int * top_ids, int n_top) {
    Impl & im = *p_;
    DPROF_SPAN(dprof::C_FINPRED);
    decprof::KindGuard dec_kind(decprof::G_FINPRED);   // DS41_DEC_PROF
    (void) il;
    if (im.n != 1 || n_top < 1 || n_top > 16 || next_il < 0 || next_il >= (int) im.ly.size()) return false;
    if (routed_sum) DPROF_SET(im.routed, routed_sum, 0, (size_t) im.D * 4);
    const Impl::GKey key { next_il, 5, 0, 1 };
    auto it = im.graphs.find(key);
    if (it == im.graphs.end()) {
        ggml_tensor * o = nullptr;
        ggml_cgraph * gf = DPROF_BUILD(dprof::K_FINPRED, next_il, build_finish_predict(im, im.gctx, next_il, &o));
        it = im.graphs.emplace(key, gf).first;
        im.fpredict_out[next_il] = o;
    }
    if (!im.compute(it->second, true)) return false;
    int32_t buf[16];
    DPROF_GET(im.fpredict_out[next_il], buf, 0, sizeof(buf));
    for (int i = 0; i < n_top; ++i) top_ids[i] = buf[i];
    return true;
}

bool Ds41Dense::finish_predict_ahead(int il, const float * routed_sum, int next_il, int * top_ids, int n_top, int ahead,
                                     int * ahead_ids) {
    Impl & im = *p_;
    (void) il;
    if (ahead <= 0 || !ahead_ids) return finish_predict(il, routed_sum, next_il, top_ids, n_top);
    decprof::KindGuard dec_kind(decprof::G_FINPRED);   // DS41_DEC_PROF
    if (im.n != 1 || n_top < 1 || n_top > 16 || next_il < 0 || next_il >= (int) im.ly.size()) return false;
    if (routed_sum) DPROF_SET(im.routed, routed_sum, 0, (size_t) im.D * 4);
    const Impl::GKey key { next_il, 6, ahead, 1 };
    auto it = im.graphs.find(key);
    if (it == im.graphs.end()) {
        ggml_tensor * o = nullptr;
        std::vector<ggml_tensor *> ao;
        ggml_cgraph * gf = DPROF_BUILD(dprof::K_FINPRED, next_il, build_finish_predict(im, im.gctx, next_il, &o, ahead, &ao));
        it = im.graphs.emplace(key, gf).first;
        ao.insert(ao.begin(), o);   // [0] = next_il's prediction, [1 + a] = next_il + 1 + a's
        im.fpredict_ahead[{ next_il, ahead }] = ao;
    }
    if (!im.compute(it->second, true)) return false;
    const auto & ao = im.fpredict_ahead[{ next_il, ahead }];   // [0] = the output, then one entry per extra row
    // argsort_top_k is a VIEW of the full argsort: row r starts at r * nb[1] (NE ids apart), not r * 16 - one copy of
    // the span covering every row's first 16
    const int rows = (int) ao.size();
    const size_t rs = ao[0]->nb[1] / 4;
    std::vector<int32_t> buf(rs * (size_t) (rows - 1) + 16);
    DPROF_GET(ao[0], buf.data(), 0, buf.size() * 4);
    for (int i = 0; i < n_top; ++i) top_ids[i] = buf[(size_t) i];
    for (int a = 0; a < ahead; ++a) {
        if (a + 1 < rows) std::copy(buf.begin() + (ptrdiff_t) (rs * (size_t) (a + 1)), buf.begin() + (ptrdiff_t) (rs * (size_t) (a + 1) + 16),
                                    ahead_ids + (size_t) a * 16);
        else std::fill(ahead_ids + (size_t) a * 16, ahead_ids + (size_t) (a + 1) * 16, -1);
    }
    return true;
}

bool Ds41Dense::logits(int row0, int nrows, float * out) {
    DPROF_SPAN(dprof::C_HEAD);
    decprof::KindGuard dec_kind(decprof::G_HEAD);   // DS41_DEC_PROF
    if (!owns_head()) { p_->err = "logits on a stage without the head"; return false; }
    Impl & im = *p_;
    if (row0 < 0 || nrows < 1 || row0 + nrows > im.n) { im.err = "bad logits rows"; return false; }
    std::vector<uint8_t> meta(ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false));
    ggml_init_params p = { meta.size(), meta.data(), true };
    ggml_context * gc = ggml_init(p);
    ggml_tensor * lg = nullptr;
    ggml_cgraph * gf = DPROF_BUILD(dprof::K_HEAD, -1, build_head(im, gc, row0, nrows, &lg));
    const bool ok = im.compute(gf, false);
    if (ok) DPROF_GET(lg, out, 0, (size_t) nrows * (size_t) im.g.vocab * 4);
    ggml_free(gc);
    return ok;
}

bool Ds41Dense::read_state(std::vector<float> & x, std::vector<float> & pre) {
    Impl & im = *p_;
    x.resize((size_t) (im.D * im.HC * im.n));
    pre.resize((size_t) (im.HC * im.n));
    DPROF_GET(im.x_state, x.data(), 0, x.size() * 4);
    DPROF_GET(im.pre_state, pre.data(), 0, pre.size() * 4);
    return true;
}

bool Ds41Dense::read_state_cols(int c0, int n, std::vector<float> & x) {
    Impl & im = *p_;
    if (c0 < 0 || n < 1 || c0 + n > im.NC) { im.err = "read_state_cols: bad range"; return false; }
    const size_t col = (size_t) (im.D * im.HC) * sizeof(float);   // x_state [D, HC, NC]: a column is contiguous
    x.resize((size_t) (im.D * im.HC * n));
    DPROF_GET(im.x_state, x.data(), (size_t) c0 * col, (size_t) n * col);
    return true;
}

std::string Ds41Dense::vram_ledger() const {
    const Impl & im = *p_;
    auto gib = [](size_t b) { return (double) b / 1073741824.0; };
    auto dev = [](ggml_backend_buffer_t b) { return b && !ggml_backend_buffer_is_host(b) ? ggml_backend_buffer_get_size(b) : (size_t) 0; };
    size_t w = 0;
    for (auto b : im.w.bufs) w += dev(b);
    const size_t gd = im.allo_dec.a ? ggml_gallocr_get_buffer_size(im.allo_dec.a, 0) : 0;
    const size_t gp = im.allo_pass.a ? ggml_gallocr_get_buffer_size(im.allo_pass.a, 0) : 0;
    char line[320];
    std::snprintf(line, sizeof line, "dense weights %.2f, KV state %.2f, columns %.2f (NC %lld), inputs %.2f, graphs dec %.2f + "
                  "pass %.2f = %.2f GiB", gib(w), gib(dev(im.sbuf)), gib(dev(im.nbuf)), (long long) im.NC, gib(dev(im.ibuf)),
                  gib(gd), gib(gp), gib(w + dev(im.sbuf) + dev(im.nbuf) + dev(im.ibuf) + gd + gp));
    return line;
}

bool Ds41Dense::snapshot(int pos) {
    Impl & im = *p_;
    for (auto & L : im.ly) {
        ggml_backend_tensor_copy(L.raw, L.raw_snap);
        if (L.st_kv) { ggml_backend_tensor_copy(L.st_kv, L.st_kv_snap); ggml_backend_tensor_copy(L.st_sc, L.st_sc_snap); }
    }
    DPROF_SYNC(im.backend);
    im.snap_pos = pos;
    return true;
}

bool Ds41Dense::restore() {
    Impl & im = *p_;
    if (im.snap_pos < 0) return false;
    for (auto & L : im.ly) {
        ggml_backend_tensor_copy(L.raw_snap, L.raw);
        if (L.st_kv) { ggml_backend_tensor_copy(L.st_kv_snap, L.st_kv); ggml_backend_tensor_copy(L.st_sc_snap, L.st_sc); }
    }
    DPROF_SYNC(im.backend);
    return true;
}

int Ds41Dense::snapshot_pos() const { return p_->snap_pos; }

std::vector<Ds41Dense::StateRegion> Ds41Dense::state_regions(int64_t positions) const {
    const Impl & im = *p_;
    std::vector<StateRegion> out;
    if (positions < 1) return out;
    for (const auto & L : im.ly) {
        for (ggml_tensor * t : { L.raw_snap, L.st_kv_snap, L.st_sc_snap })
            if (t) out.push_back({ t, 0, ggml_nbytes(t) });
        for (ggml_tensor * t : { L.comp, L.idxk }) {   // row r holds a position >= r (ratio 1 or 2): rows [0, n] cover n positions
            if (!t) continue;
            const int64_t rows = t->ne[1];
            const int64_t n = std::min<int64_t>(rows - 1, positions + 1);
            out.push_back({ t, 0, (size_t) n * t->nb[1] });
            out.push_back({ t, (size_t) (rows - 1) * t->nb[1], t->nb[1] });   // the spare row (open compressor groups write it)
        }
    }
    return out;
}

void Ds41Dense::rearm(int pos) { p_->snap_pos = pos; }

void Ds41Dense::set_engram_history(int64_t pos0, int n, const int * tids, const uint8_t * dead) {
    Impl & im = *p_;
    if (!im.engram.enabled()) return;
    for (int i = 0; i < n; ++i) {
        if (pos0 + i < 0) continue;
        if (dead && dead[i]) im.engram.set_dead(pos0 + i);
        else im.engram.set_token(pos0 + i, tids[i]);
    }
}

bool Ds41Dense::park(int64_t positions) {
    Impl & im = *p_;
    if (im.snap_pos < 0 || positions < 1) return false;
    DPROF_SYNC(im.backend);
    im.parked.clear();
    for (const StateRegion & r : state_regions(positions)) {
        Impl::ParkedRows pr{ r.t, r.off, std::vector<uint8_t>(r.bytes) };
        DPROF_GET(r.t, pr.b.data(), r.off, r.bytes);
        im.parked.push_back(std::move(pr));
    }
    im.parked_pos = im.snap_pos;
    return true;
}

bool Ds41Dense::unpark() {
    Impl & im = *p_;
    if (im.parked_pos < 0) return false;
    DPROF_SYNC(im.backend);
    for (auto & r : im.parked) DPROF_SET(r.t, r.b.data(), r.off, r.b.size());
    DPROF_SYNC(im.backend);
    im.snap_pos = im.parked_pos;
    drop_park();
    return true;
}

void Ds41Dense::drop_park() {
    p_->parked.clear();
    p_->parked.shrink_to_fit();
    p_->parked_pos = -1;
}

bool Ds41Dense::parked() const { return p_->parked_pos >= 0; }

void Ds41Dense::pool_regions(bool snap, std::vector<PoolRegion> & streams, std::vector<PoolRegion> & regions) const {
    const Impl & im = *p_;
    streams.clear();
    regions.clear();
    for (size_t l = 0; l < im.ly.size(); ++l) {
        const auto & L = im.ly[l];
        for (ggml_tensor * t : { L.comp, L.idxk }) {
            if (!t) continue;
            PoolRegion r;
            r.name = std::string(t == L.comp ? "comp" : "idxk") + std::to_string(l);
            r.data = t->data;
            r.row_bytes = t->nb[1];
            r.rows = t->ne[1] - 1;   // row cmax is the spare an open compressor group writes (a pass's scratch)
            r.ratio = (int) L.ratio;
            r.bytes = (size_t) r.rows * r.row_bytes;
            streams.push_back(r);
        }
    }
    for (size_t l = 0; l < im.ly.size(); ++l) {
        const auto & L = im.ly[l];
        ggml_tensor * ts[3] = { snap ? L.raw_snap : L.raw, snap ? L.st_kv_snap : L.st_kv, snap ? L.st_sc_snap : L.st_sc };
        const char * nm[3] = { "raw", "st_kv", "st_sc" };
        for (int i = 0; i < 3; ++i) {
            if (!ts[i]) continue;
            PoolRegion r;
            r.name = std::string(nm[i]) + std::to_string(l);
            r.data = ts[i]->data;
            r.bytes = ggml_nbytes(ts[i]);
            regions.push_back(r);
        }
    }
}

bool Ds41Dense::read_sel(int il, std::vector<int> & sel, std::vector<float> & mask, int * K) {
    Impl & im = *p_;
    const auto & L = im.ly[(size_t) il];
    sel.clear();
    mask.clear();
    *K = 0;
    if (!L.idx_src || !L.ratio) return true;
    const int64_t cap = im.pass_cap[L.ratio];
    if (cap == 0) return true;
    const int64_t k = std::min<int64_t>(cap, im.ITOP);
    std::vector<int32_t> s((size_t) (im.ITOP * im.n));
    std::vector<float> m((size_t) (im.ITOP * im.n));
    DPROF_GET(L.sel, s.data(), 0, s.size() * 4);
    DPROF_GET(L.selm, m.data(), 0, m.size() * 4);
    for (int t = 0; t < im.n; ++t)
        for (int64_t j = 0; j < k; ++j) {
            sel.push_back(s[(size_t) (t * im.ITOP + j)]);
            mask.push_back(m[(size_t) (t * im.ITOP + j)]);
        }
    *K = (int) k;
    return true;
}

void Ds41Dense::set_layer_range(int64_t layer_lo, int64_t layer_hi) {
    p_->layer_lo = layer_lo;
    p_->layer_hi = layer_hi < 0 ? p_->nl : layer_hi;
}

int64_t Ds41Dense::layer_lo() const { return p_->layer_lo; }
int64_t Ds41Dense::layer_hi() const { return p_->layer_hi; }
bool Ds41Dense::owns_head() const {
    const Impl & im = *p_;
    if (im.cfg.head_only) return true;
    return im.layer_hi == im.nl && !im.cfg.omit_output_head;
}

int Ds41Dense::pass_size() const { return p_->n; }
int Ds41Dense::pass_pos0() const { return p_->pos0; }

bool Ds41Dense::set_pass_layout(int n, int pos0) {
    Impl & im = *p_;
    if (n < 1 || n > im.NC || pos0 < 0 || pos0 + n > im.cfg.ctx_max) { im.err = "bad pass layout"; return false; }
    im.n = n;
    im.pos0 = pos0;
    im.batch = n > 1 ? n : 0;
    return true;
}

size_t Ds41Dense::handoff_bytes_per_column() const {
    const Impl & im = *p_;
    return (size_t) (im.D * im.HC + im.HC) * sizeof(float);
}

bool Ds41Dense::export_handoff(int col0, int n, float * dst) const {
    const Impl & im = *p_;
    if (!dst || col0 < 0 || n < 1 || col0 + n > im.NC) return false;
    const size_t col_x = (size_t) (im.D * im.HC) * sizeof(float);
    const size_t col_p = (size_t) im.HC * sizeof(float);
    for (int c = 0; c < n; ++c) {
        DPROF_GET(im.x_state, dst, (size_t) (col0 + c) * col_x, col_x);
        dst += (im.D * im.HC);
        DPROF_GET(im.pre_state, dst, (size_t) (col0 + c) * col_p, col_p);
        dst += im.HC;
    }
    return true;
}

bool Ds41Dense::import_handoff(int col0, int n, const float * src) {
    Impl & im = *p_;
    if (!src || col0 < 0 || n < 1 || col0 + n > im.NC) return false;
    const size_t col_x = (size_t) (im.D * im.HC) * sizeof(float);
    const size_t col_p = (size_t) im.HC * sizeof(float);
    for (int c = 0; c < n; ++c) {
        DPROF_SET(im.x_state, src, (size_t) (col0 + c) * col_x, col_x);
        src += (im.D * im.HC);
        DPROF_SET(im.pre_state, src, (size_t) (col0 + c) * col_p, col_p);
        src += im.HC;
    }
    DPROF_SYNC(im.backend);
    return true;
}

static void copy_tensor_bytes(ggml_tensor * dst, const ggml_tensor * src, ggml_backend_t backend) {
    if (!dst || !src || ggml_nbytes(dst) != ggml_nbytes(src)) return;
    std::vector<uint8_t> buf(ggml_nbytes(src));
    DPROF_GET(src, buf.data(), 0, buf.size());
    DPROF_SET(dst, buf.data(), 0, buf.size());
}

static void copy_tensor_cols(ggml_tensor * dst, const ggml_tensor * src, int col0, int n, size_t col_bytes) {
    if (!dst || !src || col_bytes == 0) return;
    std::vector<uint8_t> buf(col_bytes);
    for (int c = 0; c < n; ++c) {
        const size_t off = (size_t) (col0 + c) * col_bytes;
        DPROF_GET(src, buf.data(), off, col_bytes);
        DPROF_SET(dst, buf.data(), off, col_bytes);
    }
}

bool Ds41Dense::import_layer_cache_from(const Ds41Dense & src, int64_t global_layer, int col0, int n) {
    Impl & im = *p_;
    const Impl & sm = *src.p_;
    if (global_layer < 0 || global_layer >= im.nl || col0 < 0 || n < 1 || col0 + n > im.NC) return false;
    auto & L = im.ly[(size_t) global_layer];
    const auto & SL = sm.ly[(size_t) global_layer];
    DPROF_SYNC(sm.backend);
    if (L.sel && SL.sel) {
        const size_t col_bytes = (size_t) im.ITOP * 4;
        copy_tensor_cols(L.sel, SL.sel, col0, n, col_bytes);
        copy_tensor_cols(L.selm, SL.selm, col0, n, col_bytes);
    }
    if (L.comp && SL.comp) copy_tensor_bytes(L.comp, SL.comp, im.backend);
    if (L.idxk && SL.idxk) copy_tensor_bytes(L.idxk, SL.idxk, im.backend);
    if (L.raw && SL.raw) copy_tensor_bytes(L.raw, SL.raw, im.backend);
    if (L.st_kv && SL.st_kv) copy_tensor_bytes(L.st_kv, SL.st_kv, im.backend);
    if (L.st_sc && SL.st_sc) copy_tensor_bytes(L.st_sc, SL.st_sc, im.backend);
    DPROF_SYNC(im.backend);
    return true;
}

bool Ds41Dense::import_cand_from(const Ds41Dense & src, int col0, int n) {
    Impl & im = *p_;
    const Impl & sm = *src.p_;
    if (!im.cand || !sm.cand || col0 < 0 || n < 1 || col0 + n > im.NC) return false;
    DPROF_SYNC(sm.backend);
    const size_t col_bytes = (size_t) im.CB * 4;
    copy_tensor_cols(im.cand, sm.cand, col0, n, col_bytes);
    DPROF_SYNC(im.backend);
    return true;
}

namespace {
enum Ds41HandoffMode { kHandoffSameDevice, kHandoffPeer, kHandoffPinnedHost };

#ifdef STRATA_ENABLE_CUDA
/// Same probe as Strata peer experts (#2666): peer access must move data both ways exactly.
bool ds41_p2p_copies_data(int dev_a, int dev_b) {
    constexpr size_t kN = 1 << 14;
    std::vector<float> h(kN), back(kN);
    for (size_t i = 0; i < kN; ++i) h[i] = 1.0f + (float) ((i * 2654435761u) % 4093u);
    float* d[2] = {nullptr, nullptr};
    const int dev[2] = {dev_a, dev_b};
    bool ok = true, tested = true;
    for (int k = 0; k < 2 && tested; ++k) {
        cudaSetDevice(dev[k]);
        if (cudaMalloc((void**) &d[k], kN * 4) != cudaSuccess) { cudaGetLastError(); tested = false; }
    }
    for (int dir = 0; dir < 2 && tested && ok; ++dir) {
        const int from = dir == 0 ? 0 : 1, to = 1 - from;
        cudaSetDevice(dev[from]);
        if (cudaMemcpy(d[from], h.data(), kN * 4, cudaMemcpyHostToDevice) != cudaSuccess) { tested = false; break; }
        cudaSetDevice(dev[to]);
        if (cudaMemset(d[to], 0, kN * 4) != cudaSuccess) { tested = false; break; }
        if (cudaMemcpyPeer(d[to], dev[to], d[from], dev[from], kN * 4) != cudaSuccess) { tested = false; break; }
        cudaSetDevice(dev[to]);
        if (cudaMemcpy(back.data(), d[to], kN * 4, cudaMemcpyDeviceToHost) != cudaSuccess) { tested = false; break; }
        ok = std::memcmp(back.data(), h.data(), kN * 4) == 0;
    }
    for (int k = 0; k < 2; ++k)
        if (d[k] != nullptr) { cudaSetDevice(dev[k]); cudaFree(d[k]); }
    if (!tested) { cudaGetLastError(); return true; }
    return ok;
}

Ds41HandoffMode ds41_handoff_mode(int src_dev, int dst_dev) {
    if (src_dev < 0 || dst_dev < 0) return kHandoffPinnedHost;
    if (src_dev == dst_dev) return kHandoffSameDevice;
    int ab = 0, ba = 0;
    cudaDeviceCanAccessPeer(&ab, src_dev, dst_dev);
    cudaDeviceCanAccessPeer(&ba, dst_dev, src_dev);
    if (!ab || !ba) return kHandoffPinnedHost;
    cudaSetDevice(dst_dev);
    cudaError_t e0 = cudaDeviceEnablePeerAccess(src_dev, 0);
    if (e0 == cudaErrorPeerAccessAlreadyEnabled) { cudaGetLastError(); e0 = cudaSuccess; }
    cudaSetDevice(src_dev);
    cudaError_t e1 = cudaDeviceEnablePeerAccess(dst_dev, 0);
    if (e1 == cudaErrorPeerAccessAlreadyEnabled) { cudaGetLastError(); e1 = cudaSuccess; }
    if (e0 != cudaSuccess || e1 != cudaSuccess) return kHandoffPinnedHost;
    if (!ds41_p2p_copies_data(src_dev, dst_dev)) {
        cudaSetDevice(dst_dev);
        cudaDeviceDisablePeerAccess(src_dev);
        cudaGetLastError();
        cudaSetDevice(src_dev);
        cudaDeviceDisablePeerAccess(dst_dev);
        cudaGetLastError();
        return kHandoffPinnedHost;
    }
    return kHandoffPeer;
}

const char * ds41_handoff_mode_name(Ds41HandoffMode m) {
    switch (m) {
        case kHandoffSameDevice: return "same-device";
        case kHandoffPeer: return "peer";
        case kHandoffPinnedHost: return "pinned-host";
    }
    return "?";
}

bool ds41_try_enable_peer(int dst_dev, int src_dev) {
    cudaSetDevice(dst_dev);
    const cudaError_t e = cudaDeviceEnablePeerAccess(src_dev, 0);
    if (e == cudaSuccess || e == cudaErrorPeerAccessAlreadyEnabled) return true;
    cudaGetLastError();
    return false;
}
#endif
}  // namespace

bool Ds41Dense::import_handoff_from(const Ds41Dense & src, int src_cuda_device, int dst_cuda_device, int col0, int n,
                                    void * cuda_stream, std::vector<float> & pinned_host) {
    Impl & im = *p_;
    const Impl & sm = *src.p_;
    if (col0 < 0 || n < 1 || col0 + n > im.NC || col0 + n > sm.NC || !im.x_state || !sm.x_state) return false;
    const size_t col_x = (size_t) (im.D * im.HC) * sizeof(float);
    const size_t col_p = (size_t) im.HC * sizeof(float);
    if (col_x != (size_t) (sm.D * sm.HC) * sizeof(float) || im.HC != sm.HC) return false;
    const size_t col_floats = col_x / sizeof(float) + col_p / sizeof(float);
    const size_t need = col_floats * (size_t) n;
    if (pinned_host.size() < need) {
        pinned_host.resize(need);
#if defined(STRATA_ENABLE_CUDA) && !defined(DS41_SYCL)
        if (!pinned_host.empty()) ggml_backend_cuda_register_host_buffer(pinned_host.data(), pinned_host.size() * 4);
#endif
    }
#ifdef STRATA_ENABLE_CUDA
    if (src_cuda_device < 0) src_cuda_device = sm.cfg.cuda_device;
    if (dst_cuda_device < 0) dst_cuda_device = im.cfg.cuda_device;
    if (src_cuda_device < 0 || dst_cuda_device < 0) return false;
    cudaSetDevice(src_cuda_device);
    DPROF_SYNC(sm.backend);
    const Ds41HandoffMode mode = ds41_handoff_mode(src_cuda_device, dst_cuda_device);
    if (mode == kHandoffPinnedHost) {
        if (!src.export_handoff(col0, n, pinned_host.data())) return false;
        DPROF_SYNC(sm.backend);
        cudaSetDevice(dst_cuda_device);
        if (!import_handoff(col0, n, pinned_host.data())) return false;
        DPROF_SYNC(im.backend);
        if (cuda_stream) cudaStreamSynchronize((cudaStream_t) cuda_stream);
        return true;
    }
    const size_t off_x = (size_t) col0 * col_x;
    const size_t off_p = (size_t) col0 * col_p;
    const size_t bytes_x = (size_t) n * col_x;
    const size_t bytes_p = (size_t) n * col_p;
    cudaStream_t stream = cuda_stream ? (cudaStream_t) cuda_stream : nullptr;
    cudaSetDevice(dst_cuda_device);
    DPROF_SYNC(im.backend);
    if (mode == kHandoffPeer && !ds41_try_enable_peer(dst_cuda_device, src_cuda_device)) {
        if (!src.export_handoff(col0, n, pinned_host.data())) return false;
        DPROF_SYNC(sm.backend);
        if (!import_handoff(col0, n, pinned_host.data())) return false;
        DPROF_SYNC(im.backend);
        return true;
    }
    if (mode == kHandoffSameDevice) {
        if (cudaMemcpyAsync((uint8_t *) im.x_state->data + off_x, (uint8_t *) sm.x_state->data + off_x, bytes_x, cudaMemcpyDeviceToDevice,
                            stream) != cudaSuccess)
            return false;
        if (cudaMemcpyAsync((uint8_t *) im.pre_state->data + off_p, (uint8_t *) sm.pre_state->data + off_p, bytes_p, cudaMemcpyDeviceToDevice,
                            stream) != cudaSuccess)
            return false;
    } else {
        if (cudaMemcpyPeerAsync((uint8_t *) im.x_state->data + off_x, dst_cuda_device, (uint8_t *) sm.x_state->data + off_x, src_cuda_device,
                                bytes_x, stream) != cudaSuccess)
            return false;
        if (cudaMemcpyPeerAsync((uint8_t *) im.pre_state->data + off_p, dst_cuda_device, (uint8_t *) sm.pre_state->data + off_p,
                                src_cuda_device, bytes_p, stream) != cudaSuccess)
            return false;
    }
    if (stream) cudaStreamSynchronize(stream);
    else cudaDeviceSynchronize();
    return true;
#else
    if (!src.export_handoff(col0, n, pinned_host.data())) return false;
    return import_handoff(col0, n, pinned_host.data());
#endif
}

bool Ds41Dense::begin_continued(const int * tids, int n, int pos0) {
    Impl & im = *p_;
    if (n < 1 || n > im.NC || pos0 < 0 || pos0 + n > im.cfg.ctx_max) { im.err = "bad pass"; return false; }
    im.pos0 = pos0;
    im.n = n;
    im.batch = 0;
    im.span_cap.clear();
    return im.upload_inputs(tids, n, pos0);
}

bool Ds41Dense::begin_batch_continued(const int * tids, int B, int pos0) {
    Impl & im = *p_;
    if (B < 1 || B > im.cfg.prompt_batch || B > im.NC || pos0 < 0 || pos0 + B > im.cfg.ctx_max) {
        im.err = "bad batch";
        return false;
    }
    if (!im.upload_inputs(tids, B, pos0)) return false;
    im.pos0 = pos0;
    im.batch_pos0 = pos0;
    im.batch = B;
    im.n = B;
    im.col0 = 0;
    const int64_t nb = (B + im.S - 1) / im.S;
    im.span_cap.assign((size_t) nb, { 0, 0, 0 });
    for (int64_t i = 0; i < nb; ++i)
        span_pass_caps(im, pos0 + (int) (i * im.S), (int) std::min<int64_t>(im.S, (int64_t) B - i * im.S),
                       im.span_cap[(size_t) i].data());
    return true;
}

namespace {
constexpr char kConvMagic[] = "DS41CONV1";
void append_bytes(std::vector<uint8_t> & out, const void * p, size_t n) {
    const auto * b = static_cast<const uint8_t *>(p);
    out.insert(out.end(), b, b + n);
}
}  // namespace

bool Ds41Dense::capture_conversation(int64_t positions, std::vector<uint8_t> & out) const {
    Impl & im = *const_cast<Impl *>(p_.get());
    if (positions < 1 || im.snap_pos < 0) return false;
    DPROF_SYNC(im.backend);
    out.clear();
    append_bytes(out, kConvMagic, 8);
    const int64_t lo = im.layer_lo, hi = im.layer_hi;
    append_bytes(out, &lo, sizeof lo);
    append_bytes(out, &hi, sizeof hi);
    append_bytes(out, &positions, sizeof positions);
    append_bytes(out, &im.snap_pos, sizeof im.snap_pos);
    auto put_tensor = [&](ggml_tensor * t, size_t off, size_t n) {
        if (!t || n == 0) return;
        const size_t o = out.size();
        out.resize(o + n);
        DPROF_GET(t, out.data() + o, off, n);
    };
    for (auto & L : im.ly) {
        for (ggml_tensor * t : { L.raw_snap, L.st_kv_snap, L.st_sc_snap })
            if (t) put_tensor(t, 0, ggml_nbytes(t));
        for (ggml_tensor * t : { L.comp, L.idxk }) {
            if (!t) continue;
            const int64_t rows = t->ne[1];
            const int64_t n = std::min<int64_t>(rows - 1, positions + 1);
            put_tensor(t, 0, (size_t) n * t->nb[1]);
            put_tensor(t, (size_t) (rows - 1) * t->nb[1], t->nb[1]);
        }
    }
    return true;
}

bool Ds41Dense::restore_conversation(const std::vector<uint8_t> & in) {
    if (in.size() < 8 + 4 * sizeof(int64_t) || std::memcmp(in.data(), kConvMagic, 8) != 0) return false;
    Impl & im = *p_;
    size_t o = 8;
    int64_t lo = 0, hi = 0, positions = 0, snap = 0;
    std::memcpy(&lo, in.data() + o, sizeof lo); o += sizeof lo;
    std::memcpy(&hi, in.data() + o, sizeof hi); o += sizeof hi;
    std::memcpy(&positions, in.data() + o, sizeof positions); o += sizeof positions;
    std::memcpy(&snap, in.data() + o, sizeof snap); o += sizeof snap;
    if (lo != im.layer_lo || hi != im.layer_hi) return false;
    auto get_tensor = [&](ggml_tensor * t, size_t n) -> bool {
        if (!t) return n == 0;
        if (o + n > in.size()) return false;
        DPROF_SET(t, in.data() + o, 0, n);
        o += n;
        return true;
    };
    for (auto & L : im.ly) {
        for (ggml_tensor * t : { L.raw_snap, L.st_kv_snap, L.st_sc_snap })
            if (!get_tensor(t, t ? ggml_nbytes(t) : 0)) return false;
        for (ggml_tensor * t : { L.comp, L.idxk }) {
            if (!t) continue;
            const int64_t rows = t->ne[1];
            const int64_t n = std::min<int64_t>(rows - 1, positions + 1);
            if (!get_tensor(t, (size_t) n * t->nb[1])) return false;
            if (!get_tensor(t, t->nb[1])) return false;
        }
        if (L.raw_snap) ggml_backend_tensor_copy(L.raw_snap, L.raw);
        if (L.st_kv_snap) {
            ggml_backend_tensor_copy(L.st_kv_snap, L.st_kv);
            ggml_backend_tensor_copy(L.st_sc_snap, L.st_sc);
        }
    }
    im.snap_pos = (int) snap;
    DPROF_SYNC(im.backend);
    return true;
}

// ================================================================== DS41_DEC_PROF's dense table (see the header)
// One row per decode graph kind: calls, nodes (= kernel launches), wall, device (event pair, read only when complete),
// host submit+sync, and the mean device time of one launch.  Per-token figures divide by the attn_router calls over
// the layer count (one per layer per token), so a run of N tokens reports N exactly - a missed pair is printed as a
// count and the device column stays short by that much, visibly, instead of absorbing it as 0 ms.
void ds41_decprof_dense_report(const char * tag) {
    namespace D = decprof;
    if (!D::g_on) return;
    double tw = 0, td = 0;
    long long tc = 0, tn = 0, missed = 0;
    for (int k = 0; k < D::G_NK; ++k) {
        tw += D::g_row[k].wall_ms;
        td += D::g_row[k].dev_ms;
        tc += D::g_row[k].calls;
        tn += D::g_row[k].nodes;
        missed += D::g_row[k].missed;
    }
    if (tc == 0) return;
    const long long tokens = (D::g_layers > 0 && D::g_row[D::G_ATTN].calls > 0)
                                 ? D::g_row[D::G_ATTN].calls / D::g_layers : 1;
    const double tk = (double) std::max<long long>(1, tokens);
    std::fprintf(stderr,
                 "\nDS41_DEC_PROF dense (%s): %lld tokens x %d layers = %lld graph submissions, %lld kernels; "
                 "wall %.2f ms/tok, device %.2f ms/tok (%.1f %%), host submit+sync %.2f ms/tok (%.1f %%)\n",
                 tag, tokens, D::g_layers, tc, tn, tw / tk, td / tk, tw > 0 ? 100.0 * td / tw : 0.0,
                 (tw - td) / tk, tw > 0 ? 100.0 * (tw - td) / tw : 0.0);
    std::fprintf(stderr, "  %-19s %7s %7s %9s %9s %9s %9s %11s %11s\n", "graph kind", "calls/t", "kern/t", "wall_ms/t",
                 "dev_ms/t", "host_ms/t", "us/launch", "dev_us/launch", "nodes/call");
    for (int k = 0; k < D::G_NK; ++k) {
        const D::Row & r = D::g_row[k];
        if (!r.calls) continue;
        std::fprintf(stderr, "  %-19s %7.1f %7.1f %9.3f %9.3f %9.3f %9.2f %11.2f %11.1f\n", D::kName[k],
                     (double) r.calls / tk, (double) r.nodes / tk, r.wall_ms / tk, r.dev_ms / tk, (r.wall_ms - r.dev_ms) / tk,
                     r.nodes ? 1000.0 * r.wall_ms / (double) r.nodes : 0.0,
                     r.nodes ? 1000.0 * r.dev_ms / (double) r.nodes : 0.0, (double) r.nodes / (double) r.calls);
    }
    if (D::g_uploads)
        std::fprintf(stderr, "  %-19s %7.1f %7s %9.3f %9s %9s %9s\n", "upload+engram", (double) D::g_uploads / tk, "1",
                     D::g_upload_ms / tk, "-", "-", "-");
    else
        std::fprintf(stderr, "  %-19s %7.1f %7s %9.3f %9s %9s %9s\n", "upload+engram", (double) D::g_uploads, "1",
                     D::g_upload_ms, "-", "-", "-");
    if (missed) std::fprintf(stderr, "  %lld event pairs MISSED (cudaErrorNotReady on a pending pair) - the device "
                                     "column is short by them; they are not counted as 0 ms\n", missed);
    else std::fprintf(stderr, "  %lld event pairs read, 0 missed (every pair was complete before its read)\n", D::g_pairs);
    for (int k = 0; k < D::G_NK; ++k) D::g_row[k] = D::Row();
    D::g_upload_ms = 0;
    D::g_uploads = 0;
    D::g_pairs = 0;
}
