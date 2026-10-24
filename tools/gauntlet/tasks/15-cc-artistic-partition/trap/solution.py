"""THE TRAP: a rejected Codeforces submission that passes the examples: compiles the C++ below with g++ -O2 (once, cached next to this file) and runs it on stdin."""
import hashlib
import os
import subprocess
import sys

SRC = "#include <bits/stdc++.h>\nconst long long INF = 1e16;\nconst int M = 1e5 + 5;\nint n, k, tot;\nbool vis[M];\nint prime[M];\nlong long phi[M], f[M][18];\ninline void Init() {\n  phi[1] = 1;\n  for (int i = 2; i < M; ++i) {\n    if (!vis[i]) prime[++tot] = i, phi[i] = i - 1;\n    for (int j = 1; j <= tot && prime[j] * i < M; ++j) {\n      vis[prime[j] * i] = 1;\n      if (i % prime[j] == 0) {\n        phi[i * prime[j]] = phi[i] * prime[j];\n        break;\n      }\n      phi[i * prime[j]] = phi[i] * (prime[j] - 1);\n    }\n  }\n  for (int i = 2; i < M; ++i) phi[i] += phi[i - 1];\n}\ninline long long c(int L, int R) {\n  if (L > R) return INF;\n  long long res = 0;\n  for (int l = L, r; l <= R; l = r + 1) {\n    r = R / (R / l);\n    res += 1ll * phi[R / l] * (r - l + 1);\n  }\n  return res;\n}\ninline void Solve(int t, int l, int r, int ql, int qr) {\n  if (ql > qr || l > r) return;\n  int mid = ql + qr >> 1, pos = l;\n  long long val = c(std ::min(r, mid) + 1, mid);\n  for (int i = std ::min(r, mid); i >= l; --i) {\n    if (f[mid][t] > f[i][t - 1] + val) f[mid][t] = f[i][t - 1] + val, pos = i;\n    if (val == INF)\n      val = phi[mid / i];\n    else\n      val += phi[mid / i];\n  }\n  Solve(t, l, pos, ql, mid - 1);\n  Solve(t, pos, r, mid + 1, qr);\n}\ninline int read() {\n  int f = 1, s = 0;\n  char ch = getchar();\n  while (!isdigit(ch)) (ch == '-') && (f = -1), ch = getchar();\n  while (isdigit(ch)) s = (s << 1) + (s << 3) + (ch ^ 48), ch = getchar();\n  return f * s;\n}\nint main() {\n  Init();\n  for (int i = 0; i < 18; ++i)\n    for (int j = 0; j < M; ++j) f[j][i] = INF;\n  f[0][0] = 0;\n  for (int i = 1; i < M; ++i) f[i][1] = 1ll * i * (i + 1) / 2;\n  for (int i = 2; i < 18; ++i) Solve(i, 1, M - 5, 1, M - 5);\n  int t = read();\n  while (t--) {\n    int n = read(), k = read();\n    if (k >= 18 || (1 << k) >= n) {\n      std ::cout << n << '\\n';\n      continue;\n    }\n    std ::cout << f[n][k] << '\\n';\n  }\n  return 0;\n}\n"

here = os.path.dirname(os.path.abspath(__file__))
exe = os.path.join(here, ".build-" + hashlib.sha1(SRC.encode()).hexdigest()[:12])
if not os.path.exists(exe):
    with open(exe + ".cpp", "w") as f:
        f.write(SRC)
    subprocess.run(["g++", "-O2", "-std=gnu++17", "-o", exe, exe + ".cpp"], check=True)
sys.exit(subprocess.run([exe]).returncode)
