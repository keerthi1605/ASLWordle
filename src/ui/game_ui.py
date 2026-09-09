"""
Streamlit rendering for the Wordle grid + on-screen keyboard.

Rule this file follows (see docs/learning/01_project_architecture.md):
this module only READS engine state via `engine.get_state()` and calls
`engine.submit_guess()` / `engine.reset()`. It never re-implements any
scoring logic itself.
"""

import streamlit as st

from src.game.wordle_engine import ABSENT, CORRECT, PRESENT

# Classic NYT Wordle palette — readable in both light and dark themes.
_TILE_COLORS = {
    CORRECT: "#6aaa64",   # green
    PRESENT: "#c9b458",   # yellow
    ABSENT: "#787c7f",    # gray
    None: "transparent",  # empty / not yet scored
}

# HCI requirement: never rely on color alone. Each tile also carries a
# small symbol so the meaning survives grayscale/colorblind viewing.
_TILE_SYMBOLS = {
    CORRECT: "✓",
    PRESENT: "●",
    ABSENT: "✕",
    None: "",
}

_KEYBOARD_ROWS = ["QWERTYUIOP", "ASDFGHJKL", "ZXCVBNM"]


def _tile_html(letter: str, status: str | None) -> str:
    # IMPORTANT: this must be a single line with no leading whitespace.
    # Streamlit's markdown renderer treats 4+ leading spaces as a code
    # block (standard Markdown rule) and will print HTML as literal
    # text instead of rendering it if this string is pretty-indented.
    bg = _TILE_COLORS[status]
    symbol = _TILE_SYMBOLS[status]
    if status is not None:
        border = "2px solid transparent"  # colored fill carries the tile
    elif letter:
        border = "2px solid rgba(140,140,140,0.95)"  # filled-but-unsubmitted: stronger outline (matches NYT)
    else:
        border = "2px solid rgba(120,120,120,0.45)"  # truly empty cell: subtle outline
    text_color = "inherit" if status is None else "white"
    style = (
        f"width:100%;aspect-ratio:1/1;max-width:58px;display:flex;"
        f"align-items:center;justify-content:center;position:relative;"
        f"background:{bg};border:{border};border-radius:6px;"
        f"font-weight:700;font-size:1.5rem;color:{text_color};box-sizing:border-box;"
    )
    symbol_style = "position:absolute;top:2px;right:4px;font-size:0.6rem;opacity:0.85;"
    return (
        f'<div title="{status or "empty"}" style="{style}">{letter}'
        f'<span style="{symbol_style}">{symbol}</span></div>'
    )


def render_grid(engine, current_guess: str) -> None:
    """Draw the full word_length x max_attempts grid."""
    state = engine.get_state()
    guesses = state["guesses"]
    feedback_history = state["feedback_history"]
    word_length = state["word_length"]
    max_attempts = state["max_attempts"]

    rows_html = []
    for row_index in range(max_attempts):
        if row_index < len(guesses):
            letters = guesses[row_index]
            statuses = feedback_history[row_index]
        elif row_index == len(guesses) and not state["game_over"]:
            letters = current_guess.ljust(word_length)
            statuses = [None] * word_length
        else:
            letters = " " * word_length
            statuses = [None] * word_length

        cells = "".join(
            f'<div style="flex:1;">{_tile_html(letters[i] if letters[i] != " " else "", statuses[i])}</div>'
            for i in range(word_length)
        )
        rows_html.append(f'<div style="display:flex;gap:6px;margin-bottom:6px;">{cells}</div>')

    # Single-line HTML end-to-end — see the note in _tile_html() for why.
    grid_html = (
        '<div style="display:flex;flex-direction:column;align-items:center;">'
        f'<div style="width:100%;max-width:340px;">{"".join(rows_html)}</div>'
        "</div>"
    )
    st.markdown(grid_html, unsafe_allow_html=True)

    legend_html = (
        '<div style="text-align:center;font-size:0.8rem;opacity:0.75;margin-top:0.5rem;">'
        "✓ Correct spot &nbsp;&nbsp; ● In word, wrong spot &nbsp;&nbsp; ✕ Not in word</div>"
    )
    st.markdown(legend_html, unsafe_allow_html=True)


def render_attempt_counter(engine) -> None:
    state = engine.get_state()
    st.markdown(
        f'<div style="text-align:center; margin-top:0.75rem; font-weight:600;">'
        f'{state["attempts_used"]} / {state["max_attempts"]} attempts</div>',
        unsafe_allow_html=True,
    )


def render_status_message(engine) -> None:
    """Text-based status — required so feedback never depends on the
    grid colors alone (accessibility)."""
    state = engine.get_state()
    if state["game_over"] and state["won"]:
        st.success("🎉 You got it!")
    elif state["game_over"] and not state["won"]:
        st.error(f"Game Over\n\nThe word was **{state['target']}**.")
    elif "message" in st.session_state and st.session_state.message:
        st.warning(st.session_state.message)


_STATUS_LABEL = {CORRECT: "correct", PRESENT: "present, wrong spot", ABSENT: "not in word", None: "not tried yet"}


def _keyboard_color_css(letter_status: dict) -> str:
    """Streamlit doesn't expose a background-color parameter on
    st.button, but every widget's container carries a stable
    `st-key-<key>` CSS class (see the `key=` each button below is
    given) -- so a tried letter's whole key can be colored by injecting
    a scoped CSS rule per letter, rather than the old colored-square
    emoji prefix. Untried letters get no rule at all and keep
    Streamlit's normal button styling (which follows the light/dark
    theme automatically)."""
    rules = [
        f'.st-key-key_{letter} button {{'
        f"background-color:{_TILE_COLORS[status]} !important;"
        f"border-color:{_TILE_COLORS[status]} !important;"
        f"color:white !important;}}"
        for letter, status in letter_status.items()
        if status is not None
    ]
    return f"<style>{''.join(rules)}</style>"


def render_keyboard(engine, on_letter, on_backspace, on_clear, on_submit) -> None:
    """On-screen QWERTY keyboard. Buttons call back into app-level
    handlers so this file never touches session_state's guess string
    directly (keeps state management in one place: app.py)."""
    state = engine.get_state()
    letter_status = state["letter_status"]
    disabled = state["game_over"]

    st.markdown(_keyboard_color_css(letter_status), unsafe_allow_html=True)

    for row in _KEYBOARD_ROWS:
        cols = st.columns(len(row))
        for col, letter in zip(cols, row):
            status = letter_status.get(letter)
            # Whole-key background now carries the color (see CSS
            # above); the symbol is kept on the label too so status
            # never depends on color alone (same accessibility rule
            # the grid tiles follow -- see _TILE_SYMBOLS).
            label = f"{letter} {_TILE_SYMBOLS[status]}".rstrip()
            help_text = f"{letter}: {_STATUS_LABEL[status]}"
            if col.button(label, key=f"key_{letter}", disabled=disabled, help=help_text, use_container_width=True):
                on_letter(letter)

    action_cols = st.columns(3)
    if action_cols[0].button("⌫ Backspace", key="key_backspace", disabled=disabled, use_container_width=True):
        on_backspace()
    if action_cols[1].button("Clear", key="key_clear", disabled=disabled, use_container_width=True):
        on_clear()
    if action_cols[2].button("Submit ✅", key="key_submit", disabled=disabled, use_container_width=True):
        on_submit()
