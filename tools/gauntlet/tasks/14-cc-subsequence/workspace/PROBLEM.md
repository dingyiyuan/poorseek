# 1580_D. Subsequence  (Codeforces, rating 2900)

Time limit: 1 s per test on Codeforces (C++). **Here: 10 s per test**, measured on `solution.py` run as `python3 solution.py < input`. Memory: 244 MB.

Write `solution.py`: read the input from stdin, print the answer to stdout. Any approach is allowed: a pure Python solution of the intended complexity may be too slow on the largest tests, so `solution.py` may also write a C++ source file and compile it with `g++ -O2` (on PATH) and run it - cache the binary next to solution.py. Output is compared token by token.

---

Alice has an integer sequence a of length n and all elements are different. She will choose a subsequence of a of length m, and defines the value of a subsequence a_{b_1},a_{b_2},…,a_{b_m} as $∑_{i = 1}^m (m ⋅ a_{b_i}) - ∑_{i = 1}^m ∑_{j = 1}^m f(min(b_i, b_j), max(b_i, b_j)), where f(i, j) denotes \min(a_i, a_{i + 1}, \ldots, a_j)$.

Alice wants you to help her to maximize the value of the subsequence she choose.

A sequence s is a subsequence of a sequence t if s can be obtained from t by deletion of several (possibly, zero or all) elements.

Input

The first line contains two integers n and m (1 ≤ m ≤ n ≤ 4000).

The second line contains n distinct integers a_1, a_2, …, a_n (1 ≤ a_i < 2^{31}).

Output

Print the maximal value Alice can get.

Examples

Input


6 4
15 2 18 12 13 4


Output


100


Input


11 5
9 3 7 1 8 12 10 20 15 18 5


Output


176


Input


1 1
114514


Output


0


Input


2 1
666 888


Output


0

Note

In the first example, Alice can choose the subsequence [15, 2, 18, 13], which has the value 4 ⋅ (15 + 2 + 18 + 13) - (15 + 2 + 2 + 2) - (2 + 2 + 2 + 2) - (2 + 2 + 18 + 12) - (2 + 2 + 12 + 13) = 100. In the second example, there are a variety of subsequences with value 176, and one of them is [9, 7, 12, 20, 18].
