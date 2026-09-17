import tempfile
import unittest
from pathlib import Path

from scripts.build_flickr_entities_manifest import boxes_by_chain, phrase_records


class FlickrEntitiesParserTests(unittest.TestCase):
    def test_matches_visual_phrase_with_chain_box(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            xml = root / "sample.xml"
            xml.write_text("<annotation><object><name>1</name><bndbox><xmin>1</xmin><ymin>2</ymin><xmax>10</xmax><ymax>20</ymax></bndbox></object></annotation>")
            sentence = root / "sample.txt"
            sentence.write_text("A [/EN#1/person person] walks with [/EN#0/notvisual someone].\n")
            records = phrase_records(sentence, boxes_by_chain(xml))
        self.assertEqual(records, [(0, 0, "1", "person", [[1, 2, 10, 20]])])
