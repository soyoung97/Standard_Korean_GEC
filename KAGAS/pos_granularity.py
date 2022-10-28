from pprint import pprint

NOUN = 'NOUN'
VERB = 'VERB'
ADJECTIVE = 'ADJECTIVE'
PARTICLE = 'PARTICLE'

MODIFIER = 'MODIFIER'
ENDING = 'ENDING'
PUNCT = 'PUNCT'

#AFFIX = 'AFFIX'
ROOT = 'ROOT'
CONJUGATION = 'CONJUGATION'

UNCLASSIFIED = 'UNCLASSIFIED'

GRANULARITY_MAPPING = {
    'NNG': NOUN,
    'NNP': NOUN,
    'NNB': NOUN,
    'NNM': NOUN,
    'NP': NOUN,
    'NR': NOUN,
    'NNA': NOUN,
    'VV': VERB,
    'VXV': VERB,
    'VA': ADJECTIVE,
    'VXA': ADJECTIVE,
    'VCP': ADJECTIVE,
    'VCN': ADJECTIVE,
    'VX': UNCLASSIFIED,
    'MDT': MODIFIER,
    'MDN': MODIFIER,
    'MD': MODIFIER,
    'MAG': MODIFIER,
    'MAC': MODIFIER,
    'IC': UNCLASSIFIED,
    'JKS': PARTICLE,
    'JKC': PARTICLE,
    'JKG': PARTICLE,
    'JKO': PARTICLE,
    'JKM': PARTICLE,
    'JKI': PARTICLE,
    'JKQ': PARTICLE,
    'JX': PARTICLE,
    'JC': PARTICLE,
    'EPH': ENDING,
    'EPT': ENDING,
    'EPP': ENDING,
    'EP': ENDING,
    'EFN': ENDING,
    'EFQ': ENDING,
    'EFO': ENDING,
    'EFA': ENDING,
    'EFI': ENDING,
    'EFR': ENDING,
    'EF': ENDING,
    'ECE': ENDING,
    'ECD': ENDING,
    'ECS': ENDING,
    'EC': ENDING,
    'ETN': ENDING,
    'ETD': ENDING,
    'XPN': UNCLASSIFIED, #AFFIX,
    'XPV': UNCLASSIFIED, #AFFIX,
    'XSN': NOUN, #AFFIX,
    'XSV': VERB, #AFFIX,
    'XSA': ADJECTIVE, #AFFIX,
    'XR': UNCLASSIFIED,
    'SF': PUNCT,
    'SP': PUNCT,
    'SS': PUNCT,
    'SE': PUNCT,
    'SO': PUNCT,
    'SW': PUNCT,
    'UN': UNCLASSIFIED,
    'OL': NOUN,
    'OH': NOUN,
    'ON': NOUN,
    'EMO': UNCLASSIFIED,
}

def adjust_pos_granularity(pos):
  if pos not in GRANULARITY_MAPPING:
    return UNCLASSIFIED
  return GRANULARITY_MAPPING[pos]

def composite_error_type(error_types):
  if error_types == sorted([NOUN, PARTICLE]):
    return UNCLASSIFIED #NOUN + ':' + PARTICLE
  if error_types == sorted([VERB, ENDING]):
    return CONJUGATION
  if error_types == sorted([ADJECTIVE, ENDING]):
    return CONJUGATION
  return None

