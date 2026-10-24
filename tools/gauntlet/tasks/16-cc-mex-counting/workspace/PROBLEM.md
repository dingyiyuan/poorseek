# 1608_F. MEX counting  (Codeforces, rating 3200)

Time limit: 4 s per test on Codeforces (C++). **Here: 10 s per test**, measured on `solution.py` run as `python3 solution.py < input`. Memory: 244 MB.

Write `solution.py`: read the input from stdin, print the answer to stdout. Any approach is allowed: a pure Python solution of the intended complexity may be too slow on the largest tests, so `solution.py` may also write a C++ source file and compile it with `g++ -O2` (on PATH) and run it - cache the binary next to solution.py. Output is compared token by token.

---

For an array c of nonnegative integers, MEX(c) denotes the smallest nonnegative integer that doesn't appear in it. For example, MEX([0, 1, 3]) = 2, MEX([42]) = 0.

You are given integers n, k, and an array [b_1, b_2, …, b_n].

Find the number of arrays [a_1, a_2, …, a_n], for which the following conditions hold:

  * 0 ≤ a_i ≤ n for each i for each i from 1 to n.

  * |MEX([a_1, a_2, …, a_i]) - b_i| ≤ k for each i from 1 to n.




As this number can be very big, output it modulo 998 244 353.

Input

The first line of the input contains two integers n, k (1 ≤ n ≤ 2000, 0 ≤ k ≤ 50).

The second line of the input contains n integers b_1, b_2, …, b_n (-k ≤ b_i ≤ n+k) — elements of the array b.

Output

Output a single integer — the number of arrays which satisfy the conditions from the statement, modulo 998 244 353.

Examples

Input


4 0
0 0 0 0


Output


256


Input


4 1
0 0 0 0


Output


431


Input


4 1
0 0 1 1


Output


509


Input


5 2
0 0 2 2 0


Output


6546


Input


3 2
-2 0 4


Output


11
