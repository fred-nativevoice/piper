""" from https://github.com/keithito/tacotron

Defines the set of symbols used in text input to the model.
"""
_bos = "^"
_eos = "$"
_pad = "%"
_numerals = "1234567890"
_punctuation = ';:,.!?"\'-_\n '
_letters = "#@ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_ipa = (
    "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
)

# Export all symbols:
symbols = [_pad] + [_bos] + [_eos] + list(_numerals) + list(_punctuation) + list(_letters)

# Special symbol ids
SPACE_ID = symbols.index(" ")
