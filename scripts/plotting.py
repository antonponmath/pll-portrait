import matplotlib.pyplot as plt

from pll_portrait.drawing import draw_comparison_portrait
from pll_portrait.models import (
    ComparisonSRFPLL,
    ForcedSRFPLL,
    LinearPendulum,
    VanDerPol,
)


def main():
    _, ax = plt.subplots(figsize=[5, 5])
    k = 0.26
    draw_comparison_portrait(
        ComparisonSRFPLL(left_or_right="left", unbalance_factor=k),
        ComparisonSRFPLL(left_or_right="right", unbalance_factor=k),
        50.0,
        ax,
    )

    ax.set_box_aspect(1)
    plt.show()


if __name__ == "__main__":
    main()
