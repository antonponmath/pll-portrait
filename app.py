import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import streamlit as st

from pll_portrait.drawing import draw_comparison_srf_pll_portrait
from pll_portrait.models import ComparisonSRFPLL


@st.cache_data
def draw(unbalance_factor: float) -> Figure:
    fig, ax = plt.subplots(figsize=[5, 5])
    draw_comparison_srf_pll_portrait(
        system_left=ComparisonSRFPLL(
            left_or_right="left", unbalance_factor=unbalance_factor
        ),
        system_right=ComparisonSRFPLL(
            left_or_right="right", unbalance_factor=unbalance_factor
        ),
        t_max=50.0,
        axes=ax,
    )
    ax.set_box_aspect(1)
    return fig


st.title("Portrait of SRF-PLL")

unbalance_factor_percent = st.slider("Unbalance factor (%)", 0, 30, 1)

fig = draw(unbalance_factor_percent / 100.0)
st.pyplot(fig)
