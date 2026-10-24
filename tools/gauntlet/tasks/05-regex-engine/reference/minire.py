"""Reference: fullmatch for the Gauntlet spec - longest-match commitment (no backtracking),
line-wise `^`/`$` with `.` matching a newline, and the invented odd-offset operator `~`.

Parse -> AST -> one non-backtracking pass, memoised on (node, offset): polynomial in
len(pattern) * len(text).
"""


class _P:
    def __init__(self, p):
        self.p, self.i = p, 0

    def peek(self):
        return self.p[self.i] if self.i < len(self.p) else None

    def take(self):
        c = self.p[self.i]
        self.i += 1
        return c

    def alt(self):
        branches = [self.concat()]
        while self.peek() == "|":
            self.take()
            branches.append(self.concat())
        return ("alt", tuple(branches)) if len(branches) > 1 else branches[0]

    def concat(self):
        items = []
        while self.peek() is not None and self.peek() not in "|)":
            items.append(self.item())
        return ("cat", tuple(items))

    def item(self):
        c = self.peek()
        if c == "^":
            self.take()
            self._bare()
            return ("bol",)
        if c == "$":
            self.take()
            self._bare()
            return ("eol",)
        a = self.atom()
        while self.peek() is not None and self.peek() in "*+?~":
            a = (self.take(), a)
        return a

    def _bare(self):
        if self.peek() is not None and self.peek() in "*+?~":
            raise ValueError("nothing to repeat")

    def atom(self):
        c = self.take()
        if c == "(":
            inner = self.alt()
            if self.peek() != ")":
                raise ValueError("missing )")
            self.take()
            return inner
        if c in "*+?~":
            raise ValueError("nothing to modify")
        if c == ")":
            raise ValueError("unbalanced )")
        if c == ".":
            return ("any",)
        if c == "\\":
            if self.peek() is None:
                raise ValueError("trailing backslash")
            return ("lit", self.take())
        if c == "[":
            return self.cls()
        return ("lit", c)

    def cls(self):
        neg = False
        if self.peek() == "^":
            self.take()
            neg = True
        items, first = [], True
        while True:
            c = self.peek()
            if c is None:
                raise ValueError("missing ]")
            if c == "]" and not first:
                self.take()
                break
            c = self.take()
            if c == "\\":
                if self.peek() is None:
                    raise ValueError("trailing backslash")
                c = self.take()
            if self.peek() == "-" and self.i + 1 < len(self.p) and self.p[self.i + 1] != "]":
                self.take()
                hi = self.take()
                if hi == "\\":
                    hi = self.take()
                if hi < c:
                    raise ValueError("bad range")
                items.append((c, hi))
            else:
                items.append((c, c))
            first = False
        return ("cls", neg, tuple(items))


def _in_class(node, c):
    hit = any(lo <= c <= hi for lo, hi in node[2])
    return hit != node[1]


def fullmatch(pattern, text):
    p = _P(pattern)
    ast = p.alt()
    if p.i != len(pattern):
        raise ValueError("unbalanced )")
    n = len(text)
    memo = {}

    def run(node, i):
        key = (node, i)
        if key in memo:
            return memo[key]
        r = step(node, i)
        memo[key] = r
        return r

    def step(node, i):
        k = node[0]
        if k == "lit":
            return i + 1 if i < n and text[i] == node[1] else None
        if k == "any":
            return i + 1 if i < n else None
        if k == "cls":
            return i + 1 if i < n and _in_class(node, text[i]) else None
        if k == "bol":
            return i if i == 0 or text[i - 1] == "\n" else None
        if k == "eol":
            return i if i == n or text[i] == "\n" else None
        if k == "~":
            return run(node[1], i) if i % 2 == 0 else None
        if k == "cat":
            for sub in node[1]:
                i = run(sub, i)
                if i is None:
                    return None
            return i
        if k == "alt":
            best = None
            for b in node[1]:
                j = run(b, i)
                if j is not None and (best is None or j > best):
                    best = j
            return best
        if k == "?":
            j = run(node[1], i)
            return i if j is None else j
        j, got = i, False
        while True:
            r = run(node[1], j)
            if r is None:
                break
            got = True
            if r == j:
                break
            j = r
        return j if (k == "*" or got) else None

    return run(ast, 0) == n
