"""
parse tree from text using spaCy + benepar; adds tree traversals and bracketed string output.
It also provides a `BeneparParser` wrapper for convenience, which is used in `scripts/02_parse.py`.
Provides useful tree attributes and methods for traversing the tree, such as `parent`, `children`, `depth`, and `subtree`.
The `ParseTree` class is a wrapper around the benepar parse tree, providing additional functionality
"""

from nltk import tree
import spacy

class ParseTree:
    #inputs a benepar doc object
    def __init__(self, sent, parent = None):
        self.tree = sent
        self.label = self.tree._.labels[0] if len(self.tree._.labels) > 0 else ""

        self.parent = parent

        self._children = [ParseTree(child, parent=self) for child in self.tree._.children]

        self.tokens = [(token.tag_, token.text) for token in self.tree]

    def parent(self):
        return self.parent

    def children(self):
        return self._children

    def depth(self):
        if not self.children():
            return 1
        return 1 + max(child.depth() for child in self.children())
        

    def __str__(self):
        if len(self.children()) == 0 and self.label != "":
            return self.label + " [" + " ".join(str(token) for token in self.tokens) + "]"
        elif len(self.children()) == 0 and self.label == "":
            return " ".join(str(token) for token in self.tokens)
        else:
            return str(self.label) + " [" + " ".join(str(child) for child in self.children()) + "]"

   