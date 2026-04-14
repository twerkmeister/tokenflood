import colorsys
from typing import Tuple


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

METRIC_SEPERATOR = "__"
METRIC_PARTS_SEPERATOR = " "
MEAN_METRIC_PREFIX = "mean"
PERCENTILE_METRIC_PREFIX = "p"



def split_measurement_name(name: str) -> Tuple[str, str]:
    split_name = name.split(METRIC_SEPERATOR)
    metric = split_name[-1]
    name_part = METRIC_SEPERATOR.join(split_name[:-1])
    return name_part, metric


# 3. Logic to assign colors based on metric names
def assign_metric_colors(metric_names: list[str]) -> dict[str, str]:
    def map_metric_to_step(metric_name: str) -> int:
        if metric_name == MEAN_METRIC_PREFIX:
            return 0
        if metric_name.startswith(PERCENTILE_METRIC_PREFIX) and len(metric_name) <= 3:
            try:
                percentile = int(metric_name[1:])
            except ValueError:
                return 0
            step = abs(50 - percentile)
            return step
        return 0

    color_assignments = {}
    unique_experiment_names = []

    # First pass: Identify unique prefixes to assign base colors
    for name in metric_names:
        experiment_name, _ = split_measurement_name(name)
        if experiment_name not in unique_experiment_names:
            unique_experiment_names.append(experiment_name)

    experiment_name_to_color = {
        experiment_name: BASE_COLORS[i % len(BASE_COLORS)]
        for i, experiment_name in enumerate(unique_experiment_names)
    }

    # Second pass: Determine the specific color per metric
    for name in metric_names:
        experiment_name, metric = split_measurement_name(name)

        # Extract identifier (e.g., 'p25' from 'p25 response time')
        # This assumes identifier is at the start of the suffix
        metric_name = metric.split(METRIC_PARTS_SEPERATOR)[0]

        base_hex = experiment_name_to_color[experiment_name]
        step = map_metric_to_step(metric_name)  # Default to step 10 if unknown

        color_assignments[name] = brighten_color(base_hex, step)

    return color_assignments


def assign_metric_line_style(metric_names: list[str]):
    def map_metric_to_line_style(plot_suffix: str) -> str:
        if plot_suffix.startswith(
            f"{MEAN_METRIC_PREFIX}{METRIC_PARTS_SEPERATOR}network"
        ):
            return "dot"
        elif plot_suffix.startswith(MEAN_METRIC_PREFIX):
            return "dash"
        return "solid"

    style_assignments = {}

    for name in metric_names:
        _, suffix = split_measurement_name(name)
        style_assignments[name] = map_metric_to_line_style(suffix)

    return style_assignments