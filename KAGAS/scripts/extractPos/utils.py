from jamo import h2j, j2hcj

def break_down_to_jamo(s):
  return j2hcj(h2j(s))
