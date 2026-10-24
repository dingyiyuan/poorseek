"""THE TRAP: what an agent writes from memory of Python's `re`.

Leftmost-first with full give-back (backtracking), `.` excluding a newline, `^`/`$` as
string anchors, and no invented `~` operator (it is a literal here).  Passes every visible
test, fails the hidden ones - and is exponential on (a*)*b / (a|aa)*c.
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
        return ("alt", branches) if len(branches) > 1 else branches[0]

    def concat(self):
        items = []
        while self.peek() is not None and self.peek() not in "|)":
            items.append(self.item())
        return ("cat", items)

    def item(self):
        c = self.peek()
        if c in "^$":                      # Python's string anchors
            self.take()
            return ("bol",) if c == "^" else ("eol",)
        a = self.atom()
        while self.peek() is not None and self.peek() in "*+?":
            a = (self.take(), a)
        return a

    def atom(self):
        c = self.take()
        if c == "(":
            inner = self.alt()
            if self.peek() != ")":
                raise ValueError("missing )")
            self.take()
            return inner
        if c in "*+?":
            raise ValueError("nothing to repeat")
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
        return ("lit", c)                  # `~` lands here: no such operator

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
        return ("cls", neg, items)


def _match_char(node, c):
    if node[0] == "any":
        return c != "\n"                   # Python's default
    if node[0] == "lit":
        return node[1] == c
    hit = any(lo <= c <= hi for lo, hi in node[2])
    return hit != node[1]


def _m(node, text, i, k):
    kind = node[0]
    if kind == "bol":
        return i == 0 and k(i)
    if kind == "eol":
        return i == len(text) and k(i)
    if kind in ("lit", "any", "cls"):
        return i < len(text) and _match_char(node, text[i]) and k(i + 1)
    if kind == "cat":
        def go(j, idx):
            return k(j) if idx == len(node[1]) else _m(node[1][idx], text, j, lambda jj: go(jj, idx + 1))
        return go(i, 0)
    if kind == "alt":
        return any(_m(b, text, i, k) for b in node[1])
    q, sub = kind, node[1]
    if q == "?":
        return _m(sub, text, i, k) or k(i)
    if q == "*":
        def star(j):
            return _m(sub, text, j, lambda jj: jj != j and star(jj)) or k(j)
        return star(i)
    return _m(sub, text, i, lambda jj: _m(("*", sub), text, jj, k))


def fullmatch(pattern, text):
    p = _P(pattern)
    ast = p.alt()
    if p.i != len(pattern):
        raise ValueError("unbalanced )")
    return _m(ast, text, 0, lambda j: j == len(text))
