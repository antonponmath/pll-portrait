import matplotlib.pyplot as plt
import streamlit as st

from pll_portrait.drawing import draw_comparison_portrait
from pll_portrait.models import ComparisonSRFPLL


@st.cache_data
def draw(unbalance_factor):
    fig, ax = plt.subplots(figsize=[5, 5])
    draw_comparison_portrait(
        ComparisonSRFPLL(left_or_right="left", unbalance_factor=unbalance_factor),
        ComparisonSRFPLL(left_or_right="right", unbalance_factor=unbalance_factor),
        50.0,
        ax,
    )
    ax.set_box_aspect(1)
    return fig


st.title("Portrait of SRF-PLL")

unbalance_factor_percent = st.slider("Unbalance factor (%)", 0, 20, 1)

# if st.button("Draw"):
fig = draw(unbalance_factor_percent / 100.0)
st.pyplot(fig)
