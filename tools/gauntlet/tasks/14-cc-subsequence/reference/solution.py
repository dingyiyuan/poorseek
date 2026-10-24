"""REFERENCE: an accepted Codeforces submission: compiles the C++ below with g++ -O2 (once, cached next to this file) and runs it on stdin."""
import hashlib
import os
import subprocess
import sys

SRC = '#include <bits/stdc++.h>\nusing namespace std;\nconst int MOD = 998244353;\ntemplate <class T>\nusing vi2 = vector<vector<T>>;\nusing ll = long long;\nusing pii = pair<int, int>;\nint main() {\n  auto solve = [&]() {\n    int n, m;\n    cin >> n >> m;\n    vector<int> a(n + 1), per(n + 1), fa(n + 1), sz(n + 1);\n    for (int i = 1, _i = int(n); i <= _i; i++) cin >> a[i];\n    function<int(int)> root = [&](int x) {\n      return fa[x] ? fa[x] = root(fa[x]) : x;\n    };\n    auto un = [&](int a, int b) {\n      a = root(a), b = root(b);\n      fa[a] = b;\n    };\n    for (int i = 1, _i = int(n); i <= _i; i++) per[i] = i;\n    sort(per.begin() + 1, per.end(), [&](int x, int y) { return a[x] > a[y]; });\n    vi2<int> g(n + 1);\n    for (int i = 1, _i = int(n); i <= _i; i++) {\n      int x = per[i];\n      if (x != 1 && a[x - 1] > a[x]) {\n        int r = root(x - 1);\n        g[x].emplace_back(r);\n        un(r, x);\n      }\n      if (x != n && a[x + 1] > a[x]) {\n        int r = root(x + 1);\n        g[x].emplace_back(r);\n        un(r, x);\n      }\n    }\n    int r = per[n];\n    vi2<ll> dp(n + 1);\n    function<void(int, int)> dfs = [&](int x, int fa) {\n      vector<ll> dp1(2, -LLONG_MAX / 3);\n      dp1[0] = 0;\n      dp1[1] = 1LL * (m - 1) * a[x];\n      sz[x] = 1;\n      for (int to : g[x]) {\n        dfs(to, x);\n        vector<ll> tmp1(sz[x] + sz[to] + 1, -LLONG_MAX / 3);\n        for (int l = 0, _l = int(sz[x]); l <= _l; l++)\n          for (int r = 0, _r = int(sz[to]); r <= _r; r++) {\n            tmp1[l + r] =\n                max(dp1[l] + dp[to][r] - 2LL * l * r * a[x], tmp1[l + r]);\n          }\n        sz[x] += sz[to];\n        dp1 = tmp1;\n      }\n      dp[x] = dp1;\n    };\n    dfs(r, r);\n    cout << dp[r][m] << endl;\n  };\n  int t = 1;\n  while (t--) {\n    solve();\n  }\n  return 0;\n}\n'

here = os.path.dirname(os.path.abspath(__file__))
exe = os.path.join(here, ".build-" + hashlib.sha1(SRC.encode()).hexdigest()[:12])
if not os.path.exists(exe):
    with open(exe + ".cpp", "w") as f:
        f.write(SRC)
    subprocess.run(["g++", "-O2", "-std=gnu++17", "-o", exe, exe + ".cpp"], check=True)
sys.exit(subprocess.run([exe]).returncode)
