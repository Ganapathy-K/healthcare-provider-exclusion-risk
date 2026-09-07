"""Say each record in the words a person would use, as well as the words the LEIE uses.

Measured 2026-09-07. Three golden-set questions were refused with the answer sitting in the
file, and all three failed the same way -- the question and the record used different words
for the same thing:

    question                            record
    "cardiologists"                     CARDIOLOGY
    "proctologist"                      PROCTOLOGY
    "community mental health centers"   COMM MNTL HLTH CNTR

Rewording the question to match the file found every one of them (0/2 -> 2/2, 0/1 -> 1/1),
which is what proves this is a vocabulary problem and not a ranking one.

Neither retrieval leg can close that gap on its own. The embedding does not reliably place
CARDIOLOGY next to "cardiologists", and BM25 compares whole tokens with no stemmer, so
`cardiologists` and `cardiology` are simply two different words to it. Stemming would not
rescue it either: -ology and -ologist are two words, not two endings of one word.

So the gap is closed at INDEX time instead, by writing both forms into the sentence. Doing it
here rather than at query time means it is fixed once for every future question, and nobody
has to guess the file's spelling.
"""

# LEIE specialty fields are truncated to fit a fixed width. Only unambiguous ones are listed;
# tokens whose intent is not obvious from the corpus (T, K, BELO, GRADE) are left alone,
# because a wrong expansion indexes a lie.
ABBREVIATIONS = {
    "ACF": "adult care facility",
    "CNTR": "center",
    "CO": "company",
    "COMM": "community",
    "CONGLOM": "conglomerate",
    "CTR": "center",
    "DME": "durable medical equipment",
    "EQ": "equipment",
    "FAC": "facility",
    "FACI": "facility",
    "FACIL": "facility",
    "FP": "family practice",
    "GEN": "general",
    "GOV": "government",
    "GYN": "gynecology",
    "HC": "healthcare",
    "HE": "health",
    "HLTH": "health",
    "IDTF": "independent diagnostic testing facility",
    "MANUF": "manufacturer",
    "MGMT": "management",
    "MNTL": "mental",
    "OBS": "obstetrics",
    "ORGANIZAT": "organization",
    "PHYS": "physician",
    "PHYSIATRIS": "physiatrist",
    "PRACT": "practice",
    "PRACTITIONE": "practitioner",
    "PROSTHETIS": "prosthetist",
    "PROVID": "provider",
    "PROVIDE": "provider",
    "RECIPT": "recipient",
    "REHA": "rehabilitation",
    "REHAB": "rehabilitation",
    "SUPLIER": "supplier",
    "SUPP": "supplier",
    "SUPPL": "supplier",
    "SVCS": "services",
    "TRANS": "transportation",
    "UNK": "unknown",
}

# A question names the PERSON ("was a proctologist excluded?"); the file names the FIELD
# (PROCTOLOGY). These are the ones no suffix rule gets right.
IRREGULAR_PERSON_FORMS = {
    "ACUPUNCTURE": "acupuncturist",
    "CHIROPRACTIC": "chiropractor",
    "COUNSELING": "counselor",
    "DENTAL": "dentist",
    "GENETICS": "geneticist",
    "MEDICINE": "physician",
    "NURSING": "nurse",
    "ORTHOPEDICS": "orthopedist",
    "PEDIATRICS": "pediatrician",
    "PHARMACY": "pharmacist",
    "SURGERY": "surgeon",
    "THERAPY": "therapist",
}

# Everything else that follows a rule. Ordered longest-first so OMETRY beats OTOMY-style
# overlaps and nothing matches on a shorter tail by accident.
PERSON_FORM_SUFFIXES = (
    ("OMETRY", "ometrist"),
    ("OTOMY", "otomist"),
    ("IATRY", "iatrist"),
    ("OLOGY", "ologist"),
    ("PATHY", "path"),
)

STATE_NAMES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
    "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan",
    "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
    "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota",
    "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee",
    "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
    "AS": "American Samoa", "GU": "Guam", "MP": "Northern Mariana Islands",
    "PR": "Puerto Rico", "VI": "Virgin Islands",
}


def _text(value):
    """The field as a string. LEIE leaves SPECIALTY and STATE empty on some rows, and pandas
    hands those back as a float NaN rather than as a missing string."""
    return value if isinstance(value, str) else ""


def person_form(specialty):
    """The word for the PERSON, given the file's word for the field. None when there isn't one.

    Only whole specialties are converted, not tokens inside them: "MENTAL/BEHAVIORAL HE" has
    no person form, and inventing one puts a word in the index that no record supports.
    """
    key = _text(specialty).strip().upper()
    if key in IRREGULAR_PERSON_FORMS:
        return IRREGULAR_PERSON_FORMS[key]
    for suffix, replacement in PERSON_FORM_SUFFIXES:
        if key.endswith(suffix) and len(key) > len(suffix):
            return key[: -len(suffix)].lower() + replacement
    return None


def spell_out(specialty):
    """The specialty with its truncated words written out. None when nothing was truncated."""
    tokens = _text(specialty).replace("/", " ").split()
    if not tokens:
        return None
    expanded = [ABBREVIATIONS.get(token.upper(), token.lower()) for token in tokens]
    spelled = " ".join(expanded)
    return spelled if spelled != " ".join(token.lower() for token in tokens) else None


def also_written_as(specialty, state):
    """Every other way a person might write this record's specialty and state.

    Returned as a list so the caller decides the phrasing, and empty when the file's own
    wording is already the wording a person would use -- most records need nothing.
    """
    forms = []

    spelled = spell_out(specialty)
    if spelled:
        forms.append(spelled)

    person = person_form(specialty)
    if person:
        forms.append(person)
        forms.append(person + "s")

    state_name = STATE_NAMES.get(_text(state).strip().upper())
    if state_name:
        forms.append(state_name)

    return forms
