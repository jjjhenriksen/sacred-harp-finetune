from __future__ import annotations

import re
import unicodedata

CANONICAL_PHRASE_ALIASES = {
    "come oh thou traveler unknown": "come o thou traveler unknown",
    "deep in a cold a joyless cell": "deep in a cold and joyless cell",
    "from whence doth this union arise": "from whence does this union arise",
    "hail the blest morn see the great mediator": "hail the blest morn when the great mediator",
    "sinner oh why so thoughtless grown": "sinner o why so thoughtless grown",
    "salvation oh the joyful sound": "salvation o the joyful sound",
    "mercy oh thou son of david": "mercy o thou son of david",
    "i love to steal awhile away": "i love to steal a while away",
    "there is a fountain fill d with blood": "there is a fountain filled with blood",
    "when thro the torn sail the wild tempest is streaming": "when through the torn sail the wild tempest is streaming",
    "when god reveal d his gracious name": "when god revealed his gracious name",
    "im dying mother dying now": "i am dying mother dying now",
    "i m dying mother dying now": "i am dying mother dying now",
    "awak d by sinais awful sound": "awaked by sinai s awful sound",
    "o happy day that fix d my choice": "o happy day that fixed my choice",
    "o what of all my suffrings here": "o what of all my suff rings here",
    "my breth ren all on you i call": "my brethren all on you i call",
    "vital spark of heaven ly flame": "vital spark of heavenly flame",
    "i m on my journey home": "i am on my journey home",
    "lift up your heads immanuels friends": "lift up your heads immanuel s friends",
    "the glorious plan of mans redemption": "the glorious plan of man s redemption",
    "awaked by sinais awful sound": "awaked by sinai s awful sound",
    "while shepherds watch d their flocks by night": "while shepherds watched their flocks by night",
    "that glor ous day is drawing nigh": "that glorious day is drawing nigh",
    "the glor ous light of zion": "the glorious light of zion",
    "o once i had a glor ous view": "o once i had a glorious view",
    "how pleas d and blest was i": "how pleased and blest was i",
    "how happys every child of grace": "how happy every child of grace",
    "awak d by sinai s awful sound": "awaked by sinai s awful sound",
    "and must i be to judgment brot": "and must i be to judgment brought",
    "lord when my raptured tho t surveys": "lord when my raptured thought surveys",
    "thro all the world below": "through all the world below",
    "wrapped in the silence of the night": "wrapt in the silence of the night",
    "the char ot the char ot its wheels roll in fire": "the chariot the chariot its wheels roll in fire",
    "while shepherds watch d their flocks": "while shepherds watched their flocks",
    "thou art gone to the gravebut we will not deplore thee": "thou art gone to the grave but we will not deplore thee",
    "hosanna to jesus my souls filled with praises": "hosanna to jesus my soul s filled with praises",
    "o tell me no more of this worlds vain store": "o tell me no more of this world s vain store",
    "vain man thy found pursuits forbear": "vain man thy fond pursuits forbear",
    "when ever i cross the stream of death": "whenever i cross the stream of death",
    "farewell vain world im going home": "farewell vain world i m going home",
    "heres my heart my loving jesus": "here s my heart my loving jesus",
    "how long dear savior oh how long": "how long dear savior o how long",
    "how long dear saviour o how long": "how long dear savior o how long",
    "love divine all loves excelling": "love divine all love excelling",
    "the lord into his garden comes": "the lord into his garden come",
    "thou man of griefs remember me": "thou man of grief remember me",
}


def norm_first_line(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = s.strip().lower()
    s = (
        s.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
    )

    replacements = {
        "ev'ry": "every",
        "e'er": "ever",
        "ne'er": "never",
        "o'er": "over",
        "oer": "over",
        "saviour": "savior",
        "heav'n": "heaven",
        "heav’n": "heaven",
        "pow'r": "power",
        "powr": "power",
        "int'rest": "interest",
        "trav'ling": "traveling",
        "em'rald": "emerald",
        "em’rald": "emerald",
        "emrald": "emerald",
        "heavnly": "heavenly",
        "watch d": "watched",
        "watch d their": "watched their",
        "heaven s": "heavens",
        "gravebut": "grave but",
        "evo ry": "every",
        "evry": "every",
        "orderd": "ordered",
        "watchd": "watched",
        "risn": "ris n",
        "drivn": "driv n",
        "genrous": "generous",
        "mans": "man s",
        "worlds": "world s",
        "immanuels": "immanuel s",
        "fixd": "fixed",
        "heavn": "heaven",
        "jordans": "jordan s",
        "glor ous": "glorious",
        "brot": "brought",
        "tho t": "thought",
        "wrapt": "wrapped",
        "char ot": "chariot",
        "that steals that steals": "that steals",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)

    s = re.sub(r"^oh\b", "o", s)
    s = re.sub(r"\boh my\b", "o my", s)
    s = re.sub(r"\boh lord\b", "o lord", s)
    s = re.sub(r"\bheav[\"']?n[\"']?s[\"']?\b", "heavens", s)

    s = re.sub(r"\b([a-z]+)\s+s\b", r"\1s", s)
    s = re.sub(r"\btombs\b", "tomb", s)
    s = re.sub(r"\bguards\b", "guard", s)
    s = re.sub(r"\bheavens\b", "heaven", s)

    s = s.replace("-", " ")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\bheaven s\b", "heaven", s)
    s = re.sub(r"\bgen\s+rous\b", "generous", s)
    s = re.sub(r"\bthro\b", "through", s)
    s = re.sub(r"\bthat\s+steals\s+that\s+steals\b", "that steals", s)
    s = re.sub(r"\s+", " ", s).strip()
    return CANONICAL_PHRASE_ALIASES.get(s, s)
