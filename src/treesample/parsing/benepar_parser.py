"""Wrap spaCy + benepar; return bracketed parse strings per sentence."""

from spacy.ml import Doc

from .parse_tree import ParseTree

class BeneparParser:
    def __init__(self, model_name: str = "benepar_en3"):
        import benepar
        import spacy

        self.nlp = spacy.load("en_core_web_sm")

        self.nlp.add_pipe("benepar", config={"model": model_name})

    def parse(self, text: str) -> Doc:
        doc = self.nlp(text)
        return doc 

    def span_to_dict(self, span):
        children = list(span._.children)

        node = {
            "text": span.text,
            "labels": list(span._.labels),
            "start": span.start,
            "end": span.end,
            "children": []
        }

        # If this is a terminal node (individual token)
        if not children and len(span) == 1:
            token = span[0]

            node["token"] = {
                "text": token.text,
                "benepar_pos": list(token._.labels),
                "spacy_pos": token.pos_,
                "spacy_tag": token.tag_,
                "lemma": token.lemma_
            }

        else:
            node["children"] = [
                self.span_to_dict(child)
                for child in children
            ]

        return node

    def doc_to_dict(self, doc):
        return [self.span_to_dict(sent) for sent in doc.sents]
