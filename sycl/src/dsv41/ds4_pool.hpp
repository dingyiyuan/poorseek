// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/ds4/ds4_pool.hpp - the DeepSeek engine's CPU workers: run(n, fn) runs fn(0..n-1) on the workers and the caller and
// returns when every job is done.  The design is glmfast::Workers' (src/core/glm_fast_state.hpp): one 64-bit ticket
// (generation << 32 | next index) claimed by CAS, so a worker that wakes late for an older batch can never take a job
// of the current one; idle workers spin for a while, then sleep on a condition variable.
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "ds4_os.hpp"
#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace strata::ds4 {

inline void cpu_relax() {
#if defined(_MSC_VER)
    _mm_pause();
#elif defined(__x86_64__) || defined(_M_X64)
    __builtin_ia32_pause();
#endif
}

// ------------------------------------------------------------------ [2e16] bounded waits
// The port's waits had no upper bound anywhere: when a host flag the CPU should publish was never published, or a device
// worker's kernel never retired, the host sat in a poll loop or a spin for ever - no output, no way back, and on this
// card freed only by a reboot.  "the prompt past 65,536 tokens deadlocks" is exactly that: a wedge with no witness.
// These helpers put a bound on every host-side wait, readable from the environment so the port can be run exactly as
// before (0: no bound - the old behaviour, kept for a bisect) and so a diagnosis can shorten it to seconds.
/// DS4_WAIT_S: seconds, the bound for every host-side wait in the port (default 180; 0: no bound).  Three minutes is
/// far above any healthy wait (a 72,000-token prompt's chunk is seconds, the flow's ring slot already waits 180 s of its
/// own), so nothing on the normal path is cut; what it stops is the run that would never finish.
inline long long ds4_env_ms(const char* name, long long def_ms) {
    const char* v = std::getenv(name);
    if (v == nullptr || v[0] == 0) return def_ms;
    const long long n = std::atoll(v);
    return n > 0 ? n : 0;
}
inline long long ds4_wait_ms() { return ds4_env_ms("DS4_WAIT_S", 180000); }

class Pool {
public:
    explicit Pool(int n, const std::vector<int>& cpus = {}, int spin_us = 200) : spin_us_(spin_us) {
        for (int i = 0; i < n; ++i) {
            th_.emplace_back([this] { loop(); });
            if (!cpus.empty()) os::pin_thread(th_.back(), cpus[(size_t) i % cpus.size()]);
        }
    }
    ~Pool() {
        quit_.store(true);
        {
            std::lock_guard<std::mutex> lk(mu_);
        }
        cv_.notify_all();
        for (auto& t : th_) t.join();
    }
    int size() const { return (int) th_.size() + 1; }

    bool run(int n, const std::function<void(int)>& fn) {
        start(n, fn);
        help();
        return wait();
    }
    /// The batch goes to the workers; the caller may help() and must wait() before fn goes out of scope.  (Two pools
    /// - one a NUMA node - run at once this way: start both, help one, wait for both.)
    void start(int n, const std::function<void(int)>& fn) {
        n_ = n;
        bad_.store(0, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lk(err_mu_);
            first_err_.clear();
        }
        if (n <= 0) return;
        const uint64_t g = (gen_.load(std::memory_order_relaxed) + 1) & 0xffffffffu;
        ticket_.store((g << 32) | 0x7fffffffu, std::memory_order_seq_cst);
        fn_ = &fn;
        total_.store(n, std::memory_order_release);
        ticket_.store(g << 32, std::memory_order_release);
        {
            // [2e16] done_ and gen_ together, under the lock work() takes for its increment: a job of an abandoned batch
            // that finishes late sees the new gen_ and does not add its increment to this batch's count.
            std::lock_guard<std::mutex> lk(done_mu_);
            done_.store(0, std::memory_order_relaxed);
            gen_.store(g, std::memory_order_release);
        }
        if (sleeping_.load(std::memory_order_acquire) > 0) {
            std::lock_guard<std::mutex> lk(mu_);
            cv_.notify_all();
        }
    }
    void help() {
        if (n_ > 0) work(gen_.load(std::memory_order_acquire));
    }
    /// Wait for the batch: true when every job ran, false when a job threw or the batch did not finish inside
    /// DS4_WAIT_S (first_error() then says which).  A caller that ignores the result keeps the old behaviour, minus the
    /// spin: the batch is abandoned with a printed reason instead of the caller burning a core for ever.
    bool wait() {
        if (n_ <= 0) return bad_.load(std::memory_order_acquire) == 0;
        const int n = n_;
        const long long bound_ms = ds4_wait_ms();
        const auto t0 = std::chrono::steady_clock::now();
        int spins = 0;
        while (done_.load(std::memory_order_acquire) < n) {
            // [2e16] the old body was `while (done_ < n_) cpu_relax();` and nothing else: a job that threw skipped its
            // done_ increment (see work()) and the caller spun on this line for ever - the 100 % CPU thread of the gdb
            // snapshot - while every device worker waiting on the batch's output hung with it.  Spin first (the common
            // batch is microseconds, and the spin is what keeps it cheap), then sleep on the condition variable so a
            // wedged batch costs a sleeping thread, not a core, and give up at the bound with a named reason.
            if (++spins < 20000) {
                cpu_relax();
                continue;
            }
            std::unique_lock<std::mutex> lk(done_mu_);
            done_cv_.wait_for(lk, std::chrono::milliseconds(20),
                              [&] { return done_.load(std::memory_order_acquire) >= n; });
            if (done_.load(std::memory_order_acquire) >= n) break;
            if (bound_ms <= 0) continue;
            if (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count() <= bound_ms)
                continue;
            const int left = n - done_.load(std::memory_order_acquire);
            {
                std::lock_guard<std::mutex> g(err_mu_);
                if (bad_.load(std::memory_order_relaxed) == 0) {
                    bad_.store(2, std::memory_order_release);
                    first_err_ = "job(s) " + std::to_string(left) + " of " + std::to_string(n) +
                                 " never returned (a job's fn did not return: a wedged device or a lost flag) - " +
                                 "gave up after " + std::to_string(bound_ms) + " ms (DS4_WAIT_S; 0: no bound)";
                    std::fprintf(stderr, "ds4 pool: %s\n", first_err_.c_str());
                    std::fflush(stderr);
                }
            }
            break;
        }
        n_ = 0;
        return bad_.load(std::memory_order_acquire) == 0;
    }
    /// whether the last batch ran whole (false: a job threw or the wait timed out)
    bool ok() const { return bad_.load(std::memory_order_acquire) == 0; }
    /// the first error the last batch saw (a thrown job's message, or the wait's timeout), empty when it was clean
    std::string first_error() const {
        std::lock_guard<std::mutex> g(err_mu_);
        return first_err_;
    }

private:
    /// the first failure of the batch wins (the others are usually its consequence)
    void note_fail(const std::string& what) {
        std::lock_guard<std::mutex> g(err_mu_);
        if (bad_.load(std::memory_order_relaxed) != 0) return;
        bad_.store(1, std::memory_order_release);
        first_err_ = what;
        std::fprintf(stderr, "ds4 pool: a job of the batch failed: %s\n", what.c_str());
        std::fflush(stderr);
    }
    void work(uint64_t g) {
        for (;;) {
            uint64_t t = ticket_.load(std::memory_order_acquire);
            if ((t >> 32) != g) return;
            const int n = total_.load(std::memory_order_acquire);
            if ((int) (t & 0xffffffffu) >= n) return;
            if (!ticket_.compare_exchange_weak(t, t + 1, std::memory_order_acq_rel)) continue;
            const int idx = (int) (t & 0xffffffffu);
            // [2e16] a job that throws must still advance done_ and leave its message behind: an escaping exception
            // used to reach std::terminate in a worker thread (a silent death the caller then waited on for ever).
            try {
                (*fn_)(idx);
            } catch (const std::exception& e) {
                note_fail("job " + std::to_string(idx) + " threw: " + e.what());
            } catch (...) {
                note_fail("job " + std::to_string(idx) + " threw an unknown exception");
            }
            {
                std::lock_guard<std::mutex> lk(done_mu_);
                // (a late job of an abandoned batch: its increment must not make the next batch look finished)
                if (gen_.load(std::memory_order_acquire) == g) done_.fetch_add(1, std::memory_order_acq_rel);
            }
            done_cv_.notify_all();
        }
    }
    void loop() {
        uint64_t seen = gen_.load();
        for (;;) {
            const auto t0 = std::chrono::steady_clock::now();
            uint64_t g;
            for (;;) {
                g = gen_.load(std::memory_order_acquire);
                if (g != seen || quit_.load(std::memory_order_relaxed)) break;
                cpu_relax();
                if (std::chrono::steady_clock::now() - t0 > std::chrono::microseconds(spin_us_)) {
                    std::unique_lock<std::mutex> lk(mu_);
                    sleeping_.fetch_add(1);
                    cv_.wait(lk, [&] { return gen_.load() != seen || quit_.load(); });
                    sleeping_.fetch_sub(1);
                }
            }
            if (quit_.load()) return;
            seen = g;
            work(g);
        }
    }

    std::vector<std::thread> th_;
    int spin_us_;
    int n_ = 0;   // the current batch's size (the caller's side)
    const std::function<void(int)>* fn_ = nullptr;
    alignas(64) std::atomic<uint64_t> ticket_{0};
    alignas(64) std::atomic<uint64_t> gen_{0};
    alignas(64) std::atomic<int> total_{0};
    alignas(64) std::atomic<int> done_{0};
    std::atomic<int> sleeping_{0};
    std::atomic<bool> quit_{false};
    std::mutex mu_;
    std::condition_variable cv_;
    // [2e16] the batch's completion and its first failure (see wait() / work())
    mutable std::mutex err_mu_;
    std::string first_err_;
    std::mutex done_mu_;
    std::condition_variable done_cv_;
    std::atomic<int> bad_{0};   // 0: clean, 1: a job threw, 2: the wait timed out
};

/// File reads on threads of their own, in the order they were pushed, the pusher never waiting: the prompt path's
/// streamed experts (each part of an expert a job), unbuffered where the job has a direct descriptor and its range is
/// not all in the page cache already (the disk straight into dst, the cache neither used nor disturbed), buffered
/// otherwise or when the unbuffered read fails - the bytes land at dst + off % os::kDirectAlign either way.  A job's `left` is
/// counted down when it is done (`bad` set if it failed); wait_any() sleeps until some job finished or the timeout.
class IoQueue {
public:
    struct Job {
        os::fd_t fd = os::kBadFd, dfd = os::kBadFd;
        uint64_t off = 0, len = 0;
        uint8_t* dst = nullptr;
        std::atomic<int>* left = nullptr;
        std::atomic<bool>* bad = nullptr;
        const uint8_t* mem = nullptr;   // non-null: a copy of len bytes from memory to dst instead of a read
        const void* map = nullptr;      // the range in the file's mapping: all in the page cache, it is read buffered
        uint8_t* keep = nullptr;        // non-null (a read): the len bytes read copied here as well
        bool exact = false;             // the len bytes land exactly at dst (a ring slot: no staging slack around them)
        std::atomic<bool>* cancel = nullptr;   // set before the job starts: it is dropped unread (bad is set, left counted down)
    };
    explicit IoQueue(int n) {
        for (int i = 0; i < n; ++i) th_.emplace_back([this] { loop(); });
    }
    ~IoQueue() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            quit_ = true;
        }
        cv_.notify_all();
        for (auto& t : th_) t.join();
    }
    /// low: a read that can wait (a guess) - the jobs of the first queue are all taken before any of this one's
    void push(const Job* jobs, int n, bool low = false) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (int i = 0; i < n; ++i) (low ? ql_ : q_).push_back(jobs[i]);
        }
        if (n == 1) cv_.notify_one();
        else cv_.notify_all();
    }
    void wait_any(int us) {
        std::unique_lock<std::mutex> lk(done_mu_);
        const uint64_t seen = done_;
        done_cv_.wait_for(lk, std::chrono::microseconds(us), [&] { return done_ != seen; });
    }
    double busy_ms() const { return busy_us_.load(std::memory_order_relaxed) / 1000.0; }
    uint64_t direct_bytes() const { return direct_.load(std::memory_order_relaxed); }
    uint64_t buffered_bytes() const { return buffered_.load(std::memory_order_relaxed); }

private:
    void loop() {
        for (;;) {
            Job j;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return quit_ || !q_.empty() || !ql_.empty(); });
                if (quit_ && q_.empty() && ql_.empty()) return;
                std::deque<Job>& qq = q_.empty() ? ql_ : q_;
                j = qq.front();
                qq.pop_front();
            }
            if (j.cancel != nullptr && j.cancel->load(std::memory_order_acquire)) {   // (dropped unread)
                if (j.bad != nullptr) j.bad->store(true, std::memory_order_release);
                if (j.left != nullptr) j.left->fetch_sub(1, std::memory_order_acq_rel);
                {
                    std::lock_guard<std::mutex> lk(done_mu_);
                    ++done_;
                }
                done_cv_.notify_all();
                continue;
            }
            const auto t0 = std::chrono::steady_clock::now();
            const uint64_t lead = j.off % os::kDirectAlign;
            bool ok = false;
            if (j.mem != nullptr) {
                std::memcpy(j.dst, j.mem, (size_t) j.len);
                ok = true;
            } else if (j.exact) {
                // unbuffered through a bounce buffer (the disk reads aligned spans), then the bytes to where they belong
                thread_local std::vector<uint8_t> bounce;
                const uint64_t span = os::direct_span(j.off, j.len);
                if (bounce.size() < span + os::kDirectAlign) bounce.resize((size_t) span + 2 * os::kDirectAlign);
                uint8_t* bp = (uint8_t*) (((uintptr_t) bounce.data() + os::kDirectAlign - 1) & ~(uintptr_t) (os::kDirectAlign - 1));
                if (j.dfd != os::kBadFd && !(j.map != nullptr && os::resident(j.map, (size_t) j.len)) &&
                    os::pread_direct(j.dfd, bp, j.len, j.off)) {
                    std::memcpy(j.dst, bp + lead, (size_t) j.len);
                    ok = true;
                    direct_.fetch_add(j.len, std::memory_order_relaxed);
                } else if (os::pread_par(j.fd, j.dst, j.len, j.off)) {
                    ok = true;
                    buffered_.fetch_add(j.len, std::memory_order_relaxed);
                }
            } else if (j.dfd != os::kBadFd && !(j.map != nullptr && os::resident(j.map, (size_t) j.len)) &&
                       os::pread_direct(j.dfd, j.dst, j.len, j.off)) {   // (the page cache's ones: buffered, below)
                ok = true;
                direct_.fetch_add(j.len, std::memory_order_relaxed);
            } else if (os::pread_par(j.fd, j.dst + lead, j.len, j.off)) {
                ok = true;
                buffered_.fetch_add(j.len, std::memory_order_relaxed);
            }
            if (ok && j.keep != nullptr && j.mem == nullptr) std::memmove(j.keep, j.exact ? j.dst : j.dst + lead, (size_t) j.len);   // (keep may overlap dst: the store reads in place and moves up)
            busy_us_.fetch_add((uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                                   std::chrono::steady_clock::now() - t0).count(),
                               std::memory_order_relaxed);
            if (!ok && j.bad != nullptr) j.bad->store(true, std::memory_order_release);
            if (j.left != nullptr) j.left->fetch_sub(1, std::memory_order_acq_rel);
            {
                std::lock_guard<std::mutex> lk(done_mu_);
                ++done_;
            }
            done_cv_.notify_all();
        }
    }
    std::vector<std::thread> th_;
    std::mutex mu_, done_mu_;
    std::condition_variable cv_, done_cv_;
    std::deque<Job> q_, ql_;
    uint64_t done_ = 0;
    bool quit_ = false;
    std::atomic<uint64_t> busy_us_{0}, direct_{0}, buffered_{0};
};

}  // namespace strata::ds4
