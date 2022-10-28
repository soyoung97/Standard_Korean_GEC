from scripts.extractPos.utils import break_down_to_jamo

POS_WHITESPACE = 'WS'

class Token:
  def __init__(self, token, pos):
    self.token = token
    self.pos = pos

    self.jamo = break_down_to_jamo(token)

  def __repr__(self):
    return f'{self.token} ({self.pos})'

class AlignedToken:
  def __init__(self, original_token, corrected_token):
    self.original_token = original_token
    self.corrected_token = corrected_token
