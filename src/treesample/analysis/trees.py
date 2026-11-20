"""Tree utilities on nltk.Tree: depth, branching factor, constituent counts, production rules."""
"""
parse tree from text using spaCy + benepar; adds tree traversals and bracketed string output.
It also provides a `BeneparParser` wrapper for convenience, which is used in `scripts/02_parse.py`.
Provides useful tree attributes and methods for traversing the tree, such as `parent`, `children`, `depth`, and `subtree`.
The `ParseTree` class is a wrapper around the benepar parse tree, providing additional functionality
"""

from nltk import tree
import spacy

class ParseTree:
    def __init__(self, node, parent = None):
        self.node = node

        self.text = node["text"]

        self.parent = parent
        self.children = [ParseTree(child, self) for child in self.node["children"]]

    def depth(self):
        if not self.children:
            return 1
        return 1 + max(child.depth() for child in self.children)

    def count_label(self, label):
        if label in self.node["labels"]:
            return 1 + sum(child.count_label(label) for child in self.children)
        else:
            return sum(child.count_label(label) for child in self.children)

    def count_spacy_tag(self, tag):
        if "token" not in self.node:
            return sum(child.count_spacy_tag(tag) for child in self.children)
        elif tag == self.node["token"]["spacy_tag"]:
            return 1
        else:
            return 0

    def count_spacy_pos(self, pos):
        if "token" not in self.node:
            return sum(child.count_spacy_pos(pos) for child in self.children)
        elif pos == self.node["token"]["spacy_pos"]:
            return 1
        else:
            return 0

    def __str__(self):
        return str(self.node)

