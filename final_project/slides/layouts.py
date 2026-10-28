"""Figure + takeaway layouts for the result slides (content band of the UTA content layout)."""
from PIL import Image

import uta as U


def _fit(path, max_w, max_h):
    w, h = Image.open(path).size
    r = w / h
    return (max_w, max_w / r) if max_w / r <= max_h else (max_h * r, max_h)


def _notes_append(s, tiles):
    """Tile details are spoken, not shown: append them to the speaker notes."""
    extra = [f"{t[0]}: {t[1]}" + (f" ({t[2]})" if len(t) > 2 and t[2] else "") for t in tiles]
    tf = s.notes_slide.notes_text_frame
    tf.text = (tf.text + "\n\n" if tf.text else "") + "Numbers on this slide: " + "; ".join(extra) + "."


def stat_tile(s, x, y, w, h, value, label, note=None, value_color=U.NAVY):
    """Big number + one short label on a light card (readable when projected)."""
    U.box(s, x, y, w, h, fill=U.LIGHT, radius=0.08)
    U.text(s, x + 0.14, y + 0.06, w - 0.24, 0.42, [value], size=22, bold=True, color=value_color, anchor="m")
    U.text(s, x + 0.14, y + 0.46, w - 0.24, h - 0.5, [label], size=12, color=U.TEXT, anchor="t")


def figure_with_tiles_right(s, fig, tiles, alt, fig_w=6.6, tile_w=None):
    """Figure on the left, a column of stat tiles on the right."""
    fw, fh = _fit(fig, fig_w, U.BOT - U.TOP)
    U.picture(s, fig, 0.5, U.TOP + (U.BOT - U.TOP - fh) / 2, w=fw, alt=alt)
    x = 0.5 + fw + 0.15
    w = tile_w or (9.5 - x)
    gap = 0.1
    h = (U.BOT - U.TOP - gap * (len(tiles) - 1)) / len(tiles)
    for i, t in enumerate(tiles):
        stat_tile(s, x, U.TOP + i * (h + gap), w, h, *t)
    _notes_append(s, tiles)


def figure_with_tiles_below(s, fig, tiles, alt, tile_h=0.86):
    """Wide figure on top, a row of stat tiles underneath."""
    max_h = U.BOT - U.TOP - tile_h - 0.1
    fw, fh = _fit(fig, 9.0, max_h)
    U.picture(s, fig, 0.5 + (9.0 - fw) / 2, U.TOP, w=fw, alt=alt)
    n = len(tiles)
    w = (9.0 - 0.15 * (n - 1)) / n
    y = U.BOT - tile_h
    for i, t in enumerate(tiles):
        stat_tile(s, 0.5 + i * (w + 0.15), y, w, tile_h, *t)
    _notes_append(s, tiles)
