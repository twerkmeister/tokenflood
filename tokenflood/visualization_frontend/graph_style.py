import colorsys

from tokenflood.analysis import Mean, PERCENTILE_PREFIX


def brighten_color(hex_color: str, steps: int, max_steps=50):
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    h, lightness, s = colorsys.rgb_to_hls(r, g, b)

    max_lightness = 0.80
    lightness_range = max_lightness - lightness
    steps = max(0, min(steps, max_steps))
    new_l = lightness + (lightness_range * (steps / max_steps))

    new_r, new_g, new_b = colorsys.hls_to_rgb(h, new_l, s)
    return "#{:02x}{:02x}{:02x}".format(
        int(new_r * 255), int(new_g * 255), int(new_b * 255)
    )


BASE_COLORS = [
    "#2e4c8f",
    "#4d7358",
    "#9e2638",
    "#616161",
    "#75279c",
    "#8b4937",
    "#1a919c",
    "#5b6170",
    "#bb370f",
    "#748e2f",
]


def aggregation_name_to_color_step(aggregation_name: str) -> int:
    if aggregation_name == Mean.name:
        return 0
    if aggregation_name.startswith(PERCENTILE_PREFIX) and len(aggregation_name) <= 3:
        try:
            percentile = int(aggregation_name[1:])
        except ValueError:
            return 0
        step = abs(50 - percentile)
        return step
    return 0


def aggregation_name_to_line_style(aggregation_name: str) -> str:
    if aggregation_name == Mean.name:
        return "dash"
    return "solid"
