"""A small regular-expression engine.  Implement fullmatch(pattern, text) -> bool.

The whole text must match the pattern (as `re.fullmatch` does); an empty pattern matches
only the empty text; a malformed pattern raises ValueError.  THIS ENGINE IS NOT PYTHON'S
`re`: the three rules below are deliberate, and the hidden cases pin every one of them.
Where your memory of `re` and this docstring disagree, the docstring wins.

Grammar - the entire syntax; anything not listed is a literal or a malformed pattern
  c          a literal character (anything not special below)
  \\x         the next character, literally: \\. \\* \\( \\| \\[ \\\\ \\~ \\^ \\$
  .          ANY single character, a newline included
  [abc] [a-z0-9] [^abc]   a character class: ranges, a leading ^ negates; inside a class
             only `]`, `\\`, `^` (first) and `-` (between two characters) are special, so
             `] ^ $ . * ( ) |` are ordinary characters there
  X* X+ X?   zero-or-more / one-or-more / zero-or-one of the preceding atom
  X~         (rule 3) the preceding atom, but only where it starts at an ODD offset
  ( ... )    group (nothing is captured)
  A|B        alternation, the lowest precedence; an empty branch matches the empty string

Rule 1 - every choice is made once, and it is the LONGEST one.  Python's `re` backtracks
and gives choices back; this engine never does.  Matching is a single left-to-right pass:
at every choice point the engine commits to the candidate that consumes the most
characters and never revisits it.  A repetition applies its atom as long as the atom can
match, and an iteration that matches the empty string ends the repetition.  In A|B|... every
branch that can match at the current offset is a candidate and the one consuming the most
characters is taken (equal lengths keep pattern order, which nothing can observe: only
True/False is returned).  Whether the rest of the pattern can then succeed is never
reconsidered - that is what "commits" means, so this engine rejects text that a
backtracking engine accepts:
  fullmatch("a|ab", "ab")     True    (the branch "ab" is longer than "a")
  fullmatch("(a|aa)a", "aa")  False   ("aa" is the longer branch, and leaves no "a")
  fullmatch("a*a", "aaa")     False   (a* commits to all three "a"s)
  fullmatch("a*ab", "ab")     False   (a* takes the one "a" and will not give it back)
  fullmatch("(ab|a)(bc|b)c", "abbc")  False   ("bc" commits, "c" has nothing left)
  fullmatch("(a?)(a)", "a")   False   (a? takes that "a" for itself)
Python's `re` returns the other value for all of these except the first.

Rule 2 - `^`, `$` and `.`.  Both assertions are line-wise everywhere in the pattern, with
no flag to set: `^` holds at offset 1 and just after any newline, `$` holds at the end of
the text and just before any newline.  Neither consumes a character, and `.` matches a
newline.  Python's defaults are the other way round on all three (its `.` wants re.S, its
anchors want re.M):
  fullmatch("a.c", "a\\nc")    True     fullmatch("ab^c", "abc")     False
  fullmatch("a\\n^b", "a\\nb")  True     fullmatch("ab$c", "abc")     False
  fullmatch("a$\\nb", "a\\nb")  True     fullmatch("a$b", "a\\nb")    False  ($ is zero width)
  fullmatch("^\\n^", "\\n")     True     fullmatch("$^", "")          True  (both hold at 0)

Rule 3 - `~`, an operator of this engine only.  `X~` matches X but only where X begins at an
odd 1-based offset of the text - the 1st, 3rd, 5th character.  It is written after an atom
or after a repetition and is an assertion like `^`: it constrains the offset and consumes
nothing.  `X*~` constrains the offset where the repetition starts; `X~*` repeats the
offset-constrained atom:
  fullmatch("a~a", "aa")    True     fullmatch("a~a~", "aa")    False  (the 2nd a starts at 2)
  fullmatch("a*~", "aaa")   True     fullmatch("a~*", "aaa")   False  (the 2nd a starts at 2)
  fullmatch("a~*", "a")     True     fullmatch("(a~)*", "aa")  False  (iteration 2 starts at 2)
  fullmatch("(ab)~", "ab")  True     fullmatch("b(a)~", "ba")  False  (the group starts at 2)

Malformed patterns raise ValueError: "(", "a)", "*a", "~a", "[a", "[^", "a|*", "(?", "\\",
a class range that runs backwards ("[z-a]"), and a quantifier or a `~` with nothing to
modify or after an assertion - "^", "$" and "~" are assertions, not atoms, so "^*", "$+"
and "a^*" are malformed too.  "a**" is NOT malformed, but you will not be tested on
doubled quantifiers.

PERFORMANCE IS PART OF THE SPEC: matching must take polynomial time in len(pattern) *
len(text) - fullmatch("(a*)*b", "a" * 40), fullmatch("(a|aa)*c", "a" * 60),
fullmatch("(a*)*(b*)*c", "ab" * 25), fullmatch("((a+)+)+d", "a" * 35) and
fullmatch("(.*)*x", "y" * 50) must each return False in well under a second.
"""


def fullmatch(pattern, text):
    raise NotImplementedError
