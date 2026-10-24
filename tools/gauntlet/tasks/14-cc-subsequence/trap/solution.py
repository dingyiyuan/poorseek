"""THE TRAP: a rejected Codeforces submission that passes the examples: compiles the C++ below with g++ -O2 (once, cached next to this file) and runs it on stdin."""
import hashlib
import os
import subprocess
import sys

SRC = '#include <bits/stdc++.h>\nnamespace wxy {\nconst long long N = 4005, inf = 1ll << 31;\nlong long f[N][N], tot;\nlong long a[N], size[N];\nlong long n, m, ls[N], rs[N];\nvoid calc(long long &u, long long l, long long r) {\n  if (!u) u = ++tot;\n  long long rt = 0;\n  size[u] = 1;\n  if (l == r) {\n    f[u][1] = (m - 1) * a[l];\n    return;\n  }\n  for (long long i = l; i <= r; i++)\n    if (a[rt] > a[i]) rt = i;\n  if (rt != l)\n    calc(ls[u], l, rt - 1);\n  else\n    f[ls[u]][0] = 0;\n  if (rt != r)\n    calc(rs[u], rt + 1, r);\n  else\n    f[rs[u]][0] = 0;\n  for (long long i = 0; i <= std::min(size[ls[u]], m); i++)\n    for (long long j = 0; j <= std::min(size[rs[u]], m); j++) {\n      if (i + j == 0) continue;\n      long long val = f[ls[u]][i] + f[rs[u]][j];\n      if (i + j <= m)\n        f[u][i + j] = std::max(f[u][i + j], val - 2 * i * j * a[rt]);\n      else\n        break;\n      if (i + j + 1 <= m)\n        f[u][i + j + 1] =\n            std::max(f[u][i + j + 1], val + m * a[rt] - 2 * i * j * a[rt] -\n                                          2 * (i + j) * a[rt] - a[rt]);\n    }\n  size[u] += size[ls[u]] + size[rs[u]];\n}\ninline long long get(long long l, long long r) {\n  long long res = 0;\n  for (long long i = l; i <= r; i++) {\n    res += m * a[i];\n    for (long long j = l; j <= r; j++) {\n      long long val = inf;\n      for (long long k = std::min(i, j); k <= std::max(i, j); k++)\n        val = std::min(val, a[k]);\n      std::cout << val << std::endl;\n      res -= val;\n    }\n  }\n  return res;\n}\ninline void main() {\n  memset(f, 0xcf, sizeof f);\n  std::cin >> n >> m;\n  for (long long i = 1; i <= n; i++) std::cin >> a[i];\n  a[0] = inf;\n  long long rt = 0;\n  calc(rt, 1, n);\n  std::cout << f[1][m];\n}\n}  // namespace wxy\nsigned main() {\n  wxy::main();\n  return 0;\n}\n'

here = os.path.dirname(os.path.abspath(__file__))
exe = os.path.join(here, ".build-" + hashlib.sha1(SRC.encode()).hexdigest()[:12])
if not os.path.exists(exe):
    with open(exe + ".cpp", "w") as f:
        f.write(SRC)
    subprocess.run(["g++", "-O2", "-std=gnu++17", "-o", exe, exe + ".cpp"], check=True)
sys.exit(subprocess.run([exe]).returncode)
