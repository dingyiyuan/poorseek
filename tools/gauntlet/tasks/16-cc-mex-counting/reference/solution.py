"""REFERENCE: an accepted Codeforces submission: compiles the C++ below with g++ -O2 (once, cached next to this file) and runs it on stdin."""
import hashlib
import os
import subprocess
import sys

SRC = '#include <bits/stdc++.h>\nusing namespace std;\ninline int read() {\n  int x = 0, f = 1;\n  char c = getchar();\n  while (!isdigit(c)) {\n    if (c == \'-\') f = -1;\n    c = getchar();\n  }\n  while (isdigit(c)) {\n    x = (x << 3) + (x << 1) + (c ^ 48);\n    c = getchar();\n  }\n  return f == -1 ? ~x + 1 : x;\n}\nint n, k;\nint b[2010];\nint dp[2][110][2010];\nint D[5310];\nint fac[4010], ifac[4010];\nconst int mod = 998244353;\ninline int qpow(int x, int k = mod - 2) {\n  int res = 1;\n  while (k) {\n    if (k & 1) res = 1ll * res * x % mod;\n    x = 1ll * x * x % mod;\n    k >>= 1;\n  }\n  return res;\n}\nint main() {\n  n = read(), k = read();\n  for (int i = 1; i <= n; ++i) b[i] = read();\n  dp[0][k][0] = 1;\n  bool fl = 0;\n  fac[0] = 1;\n  for (int i = 1; i <= n; ++i) fac[i] = 1ll * fac[i - 1] * i % mod;\n  ifac[n] = qpow(fac[n]);\n  for (int i = n - 1; i >= 0; --i) ifac[i] = 1ll * ifac[i + 1] * (i + 1) % mod;\n  for (int i = 1; i <= n; ++i) {\n    fl ^= 1;\n    memset(dp[fl], 0, sizeof(dp[fl]));\n    memset(D, 0, sizeof(D));\n    int it = -min(k, b[i - 1]);\n    for (int j = -min(b[i], k); j <= k; ++j) {\n      int val = b[i] + j;\n      while (it < val - b[i - 1] && it <= k) {\n        for (int l = 0; l <= i; ++l)\n          (D[l + b[i - 1] + it] +=\n           1ll * dp[fl ^ 1][it + k][l] * fac[l] % mod) %= mod;\n        ++it;\n      }\n      for (int l = 0; l <= i; ++l) {\n        if (l + val > 0)\n          dp[fl][j + k][l] = 1ll * D[l + val - 1] * ifac[l] % mod;\n        if (abs(val - b[i - 1]) <= k) {\n          (dp[fl][j + k][l] +=\n           1ll * (val + l) * dp[fl ^ 1][k + val - b[i - 1]][l] % mod) %= mod;\n          if (l)\n            (dp[fl][j + k][l] += dp[fl ^ 1][k + val - b[i - 1]][l - 1]) %= mod;\n        }\n      }\n    }\n  }\n  int ans = 0;\n  for (int i = -min(b[n], k); i <= k && i + b[n] <= n; ++i) {\n    for (int l = 0; l <= n && b[n] + i + l <= n; ++l) {\n      (ans += 1ll * dp[fl][i + k][l] * fac[n - (b[n] + i)] % mod *\n              ifac[n - (b[n] + i + l)] % mod) %= mod;\n    }\n  }\n  printf("%d\\n", ans);\n}\n'

here = os.path.dirname(os.path.abspath(__file__))
exe = os.path.join(here, ".build-" + hashlib.sha1(SRC.encode()).hexdigest()[:12])
if not os.path.exists(exe):
    with open(exe + ".cpp", "w") as f:
        f.write(SRC)
    subprocess.run(["g++", "-O2", "-std=gnu++17", "-o", exe, exe + ".cpp"], check=True)
sys.exit(subprocess.run([exe]).returncode)
