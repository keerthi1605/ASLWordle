"""
Word list for SignWordle.

This is a small, hand-picked list of common, everyday 5-letter English
words (no proper nouns, no obscure words) — not a research dataset, so
there is nothing to cite here. It is used for two things:

1. Picking a random target word for a new game.
2. Checking that a submitted guess is a "real" word (like NYT Wordle
   rejecting gibberish), so `wordle_engine.py` doesn't have to embed
   any word data itself.

Kept deliberately small (a few hundred words) so the game is
predictable and easy to test/demo — not an attempt at a complete
dictionary.
"""

import random

# fmt: off
WORDS = [
    "ABOUT", "ABOVE", "ABUSE", "ACTOR", "ACUTE", "ADMIT", "ADOPT", "ADULT",
    "AFTER", "AGAIN", "AGENT", "AGREE", "AHEAD", "ALARM", "ALBUM", "ALERT",
    "ALIKE", "ALIVE", "ALLOW", "ALONE", "ALONG", "ALTER", "AMONG", "ANGER",
    "ANGLE", "ANGRY", "APPLE", "APPLY", "ARENA", "ARGUE", "ARISE", "ARRAY",
    "ASIDE", "ASSET", "AVOID", "AWAKE", "AWARD", "AWARE", "BADLY", "BAKER",
    "BASIC", "BEACH", "BEGAN", "BEGIN", "BEING", "BELOW", "BENCH", "BILLY",
    "BIRTH", "BLACK", "BLAME", "BLANK", "BLAST", "BLIND", "BLOCK", "BLOOD",
    "BOARD", "BOAST", "BONUS", "BOOST", "BOOTH", "BOUND", "BRAIN", "BRAND",
    "BRAVE", "BREAD", "BREAK", "BREED", "BRIEF", "BRING", "BROAD", "BROKE",
    "BROWN", "BUILD", "BUILT", "BUYER", "CABLE", "CANDY", "CARRY", "CATCH",
    "CAUSE", "CHAIN", "CHAIR", "CHAOS", "CHARM", "CHART", "CHASE", "CHEAP",
    "CHECK", "CHESS", "CHEST", "CHIEF", "CHILD", "CHOSE", "CIVIL", "CLAIM",
    "CLASS", "CLEAN", "CLEAR", "CLICK", "CLIMB", "CLOCK", "CLOSE", "CLOUD",
    "COACH", "COAST", "COULD", "COUNT", "COURT", "COVER", "CRAFT", "CRASH",
    "CRAZY", "CREAM", "CRIME", "CROSS", "CROWD", "CROWN", "CRUDE", "CURVE",
    "CYCLE", "DAILY", "DANCE", "DEALT", "DEATH", "DEBUT", "DELAY", "DEPTH",
    "DOING", "DOUBT", "DOZEN", "DRAFT", "DRAMA", "DRANK", "DRAWN", "DREAM",
    "DRESS", "DRIED", "DRINK", "DRIVE", "DROVE", "DYING", "EAGER", "EARLY",
    "EARTH", "EIGHT", "ELITE", "EMPTY", "ENEMY", "ENJOY", "ENTER", "ENTRY",
    "EQUAL", "ERROR", "EVENT", "EVERY", "EXACT", "EXIST", "EXTRA", "FAITH",
    "FALSE", "FAULT", "FIBER", "FIELD", "FIFTH", "FIFTY", "FIGHT", "FINAL",
    "FIRST", "FIXED", "FLASH", "FLEET", "FLOOR", "FLUID", "FOCUS", "FORCE",
    "FORTH", "FORTY", "FORUM", "FOUND", "FRAME", "FRANK", "FRAUD", "FRESH",
    "FRONT", "FROST", "FRUIT", "FULLY", "FUNNY", "GIANT", "GIVEN", "GLASS",
    "GLOBE", "GOING", "GRACE", "GRADE", "GRAND", "GRANT", "GRASS", "GREAT",
    "GREEN", "GROSS", "GROUP", "GROWN", "GUARD", "GUESS", "GUEST", "GUIDE",
    "HAPPY", "HARSH", "HEART", "HEAVY", "HELLO", "HENCE", "HORSE", "HOTEL",
    "HOUSE", "HUMAN", "IDEAL", "IMAGE", "INDEX", "INNER", "INPUT", "ISSUE",
    "JOINT", "JUDGE", "KNIFE", "KNOWN", "LABEL", "LARGE", "LASER", "LATER",
    "LAUGH", "LAYER", "LEARN", "LEAST", "LEAVE", "LEGAL", "LEVEL", "LIGHT",
    "LIMIT", "LINKS", "LIVED", "LOCAL", "LOGIC", "LOOSE", "LOWER", "LOYAL",
    "LUCKY", "LUNCH", "LYING", "MAGIC", "MAJOR", "MAKER", "MARCH", "MATCH",
    "MAYBE", "MAYOR", "MEANT", "MEDIA", "METAL", "MIGHT", "MINOR", "MINUS",
    "MIXED", "MODEL", "MONEY", "MONTH", "MORAL", "MOTOR", "MOUNT", "MOUSE",
    "MOUTH", "MOVIE", "MUSIC", "NAKED", "NAMED", "NEEDS", "NERVE", "NEVER",
    "NEWLY", "NIGHT", "NOISE", "NORTH", "NOTED", "NOVEL", "NURSE", "OCCUR",
    "OCEAN", "OFFER", "OFTEN", "ORDER", "OTHER", "OUGHT", "OUTER", "OWNER",
    "PANEL", "PAPER", "PARTY", "PEACE", "PHASE", "PHONE", "PHOTO", "PIECE",
    "PILOT", "PITCH", "PLACE", "PLAIN", "PLANE", "PLANT", "PLATE", "POINT",
    "POUND", "POWER", "PRESS", "PRICE", "PRIDE", "PRIME", "PRINT", "PRIOR",
    "PRIZE", "PROOF", "PROUD", "PROVE", "QUEEN", "QUICK", "QUIET", "QUITE",
    "RADIO", "RAISE", "RANGE", "RAPID", "RATIO", "REACH", "READY", "REFER",
    "RIGHT", "RIVAL", "RIVER", "ROBOT", "ROUGH", "ROUND", "ROUTE", "ROYAL",
    "RURAL", "SCALE", "SCENE", "SCOPE", "SCORE", "SENSE", "SERVE", "SEVEN",
    "SHALL", "SHAPE", "SHARE", "SHARP", "SHEET", "SHELF", "SHELL", "SHIFT",
    "SHINE", "SHIRT", "SHOCK", "SHOOT", "SHORT", "SHOWN", "SIGHT", "SINCE",
    "SIXTH", "SIXTY", "SIZED", "SKILL", "SLEEP", "SLIDE", "SMALL",
    "SMART", "SMELL", "SMILE", "SMOKE", "SOLID", "SOLVE", "SORRY", "SOUND",
    "SOUTH", "SPACE", "SPARE", "SPEAK", "SPEED", "SPEND", "SPENT", "SPLIT",
    "SPOKE", "SPORT", "STAFF", "STAGE", "STAKE", "STAND", "START", "STATE",
    "STEAM", "STEEL", "STEEP", "STICK", "STILL", "STOCK", "STONE", "STOOD",
    "STORE", "STORM", "STORY", "STRIP", "STUCK", "STUDY", "STUFF", "STYLE",
    "SUGAR", "SUPER", "SWEET", "TABLE", "TAKEN", "TASTE", "TEACH",
    "TEETH", "THANK", "THEFT", "THEIR", "THEME", "THERE", "THESE",
    "THICK", "THING", "THINK", "THIRD", "THOSE", "THREE", "THREW", "THROW",
    "TIGHT", "TIMER", "TITLE", "TODAY", "TOKEN", "TOPIC", "TOTAL", "TOUCH",
    "TOUGH", "TOWER", "TRACK", "TRADE", "TRAIL", "TRAIN", "TREAT", "TREND",
    "TRIAL", "TRIBE", "TRICK", "TRIED", "TRIES", "TRUCK", "TRULY", "TRUNK",
    "TRUST", "TRUTH", "TWICE", "UNDER", "UNION", "UNITY", "UNTIL", "UPPER",
    "UPSET", "URBAN", "USAGE", "USUAL", "VALID", "VALUE", "VIDEO", "VIRUS",
    "VISIT", "VITAL", "VOCAL", "VOICE", "WASTE", "WATCH", "WATER", "WHEEL",
    "WHERE", "WHICH", "WHILE", "WHITE", "WHOLE", "WHOSE", "WOMAN", "WOMEN",
    "WORLD", "WORRY", "WORSE", "WORST", "WORTH", "WOULD", "WOUND", "WRITE",
    "WRONG", "WROTE", "YIELD", "YOUNG", "YOUTH",
]
# fmt: on

# De-duplicate defensively and make sure everything is a clean 5-letter
# uppercase alphabetic word (guards against copy/paste typos above).
WORDS = sorted({w for w in WORDS if len(w) == 5 and w.isalpha()})

_WORD_SET = set(WORDS)


def is_valid_word(word: str) -> bool:
    """Return True if `word` (any case) is in the accepted word list."""
    return word.upper() in _WORD_SET


def get_random_target(rng: random.Random | None = None) -> str:
    """Pick a random target word. Pass `rng` for deterministic tests."""
    rng = rng or random
    return rng.choice(WORDS)
