import json
from pathlib import Path


class ICDCCSMapper:
    """Loads the ICD-9 -> CCS mappings supplied with EHR-KnowGen.

    The original repository contains mapping JSON files, but their exact JSON
    representation may vary. This class accepts either:
      {"401.9": "98"}
      {"401.9": ["98"]}
    and normalizes keys/values to strings.
    """

    def __init__(self, mapping_path=None):
        self.mapping = {}
        if mapping_path and Path(mapping_path).exists():
            with open(mapping_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for k, v in raw.items():
                key = self.normalize_code(k)
                if isinstance(v, list):
                    self.mapping[key] = [str(x) for x in v]
                else:
                    self.mapping[key] = [str(v)]

    @staticmethod
    def normalize_code(code):
        if code is None:
            return ""
        return str(code).strip().replace(".", "")

    def map_code(self, code):
        key = self.normalize_code(code)
        return self.mapping.get(key, [])

    def map_codes(self, codes):
        result = []
        for code in codes:
            result.extend(self.map_code(code))
        return list(dict.fromkeys(result))
