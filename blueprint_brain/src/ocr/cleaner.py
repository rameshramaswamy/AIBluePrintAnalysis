import re
from typing import Optional, Tuple
from enum import Enum
from rapidfuzz import process, fuzz

class TextType(Enum):
    ROOM_LABEL = "room_label"
    DIMENSION = "dimension"
    SCALE_MARKER = "scale"
    NOISE = "noise"

class TextCleaner:
    """
    Enterprise Text Classifier using Fuzzy Logic.
    """
    
    # Canonical Mapping: Detected -> Standard
    ROOM_VOCAB = {
        "MASTER BEDROOM": ["MASTER", "MSTR", "MBED", "MAIN BED"],
        "BEDROOM": ["BED", "BDRM", "GUEST", "SLEEP"],
        "KITCHEN": ["KITCHEN", "KIT", "KITCH", "GALLEY"],
        "BATHROOM": ["BATH", "WC", "TOILET", "POWDER", "ENSUITE"],
        "LIVING ROOM": ["LIVING", "LVRM", "GREAT ROOM", "LOUNGE", "FAMILY"],
        "DINING ROOM": ["DINING", "DINE", "BREAKFAST"],
        "CLOSET": ["CLOSET", "WIC", "W.I.C", "STORAGE", "WIR"],
        "GARAGE": ["GARAGE", "CARPORT"],
        "HALLWAY": ["HALL", "CORRIDOR", "ENTRY", "FOYER"],
        "BALCONY": ["BALCONY", "TERRACE", "PATIO", "DECK"]
    }

    @staticmethod
    def classify_text(text: str) -> Tuple[TextType, Optional[str]]:
        """
        Returns (Type, Canonical_Name)
        """
        clean_text = text.upper().strip().replace(".", "")
        
        # 1. Scale Markers (High Priority)
        # Matches: "Scale 1:100", "1/4\" = 1'"
        if "SCALE" in clean_text or re.search(r"SCALE\s*[:]\s*\d", clean_text):
            return TextType.SCALE_MARKER, clean_text

        # 2. Dimensions
        # Matches: 12'6", 12x14, 3400mm
        if re.search(r"\d+\s*['’]\s*\d+", clean_text) or re.search(r"\d+\s*[xX]\s*\d+", clean_text):
            return TextType.DIMENSION, clean_text

        # 3. Fuzzy Room Matching
        # Flatten the vocab for search
        choices = {}
        for canonical, variants in TextCleaner.ROOM_VOCAB.items():
            choices[canonical] = canonical
            for v in variants:
                choices[v] = canonical
        
        # Extract best match
        # score_cutoff=85 means "85% similar"
        match = process.extractOne(
            clean_text, 
            choices.keys(), 
            scorer=fuzz.token_sort_ratio, 
            score_cutoff=80
        )
        
        if match:
            found_key, score, _ = match
            canonical_name = choices[found_key]
            return TextType.ROOM_LABEL, canonical_name
            
        # 4. Check length (Room labels usually < 20 chars, Notes are longer)
        if 2 < len(clean_text) < 20 and clean_text.isalpha():
            return TextType.ROOM_LABEL, clean_text # Unknown room type

        return TextType.NOISE, None