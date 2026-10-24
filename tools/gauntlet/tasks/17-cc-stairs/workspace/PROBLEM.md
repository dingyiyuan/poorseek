# 1553_I. Stairs  (Codeforces, rating 3400)

Time limit: 10 s per test on Codeforces (C++). **Here: 10 s per test**, measured on `solution.py` run as `python3 solution.py < input`. Memory: 976 MB.

Write `solution.py`: read the input from stdin, print the answer to stdout. Any approach is allowed: a pure Python solution of the intended complexity may be too slow on the largest tests, so `solution.py` may also write a C++ source file and compile it with `g++ -O2` (on PATH) and run it - cache the binary next to solution.py. Output is compared token by token.

---

For a permutation p of numbers 1 through n, we define a stair array a as follows: a_i is length of the longest segment of permutation which contains position i and is made of consecutive values in sorted order: [x, x+1, …, y-1, y] or [y, y-1, …, x+1, x] for some x ≤ y. For example, for permutation p = [4, 1, 2, 3, 7, 6, 5] we have a = [1, 3, 3, 3, 3, 3, 3]. 

You are given the stair array a. Your task is to calculate the number of permutations which have stair array equal to a. Since the number can be big, compute it modulo 998 244 353. Note that this number can be equal to zero.

Input

The first line of input contains integer n (1 ≤ n ≤ 10^5) — the length of a stair array a.

The second line of input contains n integers a_1, a_2, …, a_n (1 ≤ a_i ≤ n).

Output

Print the number of permutations which have stair array equal to a. Since the number can be big, compute it modulo 998 244 353.

Examples

Input


6
3 3 3 1 1 1


Output


6


Input


7
4 4 4 4 3 3 3


Output


6


Input


1
1


Output


1


Input


8
2 2 2 2 2 2 1 1


Output


370


Input


4
3 2 3 1


Output


0
