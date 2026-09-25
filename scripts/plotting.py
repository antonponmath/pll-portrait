import matplotlib.pyplot as plt

from pll_portrait.drawing import draw_comparison_portrait
from pll_portrait.models import (
    ComparisonSRFPLL,
    ForcedSRFPLL,
    LinearPendulum,
    VanDerPol,
)


def main() -> None:
    _, ax = plt.subplots(figsize=[5, 5])
    k = 0.22  # unbalance factor
    draw_comparison_portrait(
        system_left=ComparisonSRFPLL(left_or_right="left", unbalance_factor=k),
        system_right=ComparisonSRFPLL(left_or_right="right", unbalance_factor=k),
        t_max=50.0,
        axes=ax,
    )

    ax.set_box_aspect(1)
    plt.show()


if __name__ == "__main__":
    main()
