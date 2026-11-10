
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="ReLU Activation Function",
    page_icon="📈",
    layout="centered"
)

# Title
st.title("Rectified Linear Unit (ReLU)")
st.subheader("Neural Network Activation Function")

st.write(
    """
    ReLU is an activation function that allows only positive
    values to pass through it. Negative values are mapped to zero.
    """
)

# Mathematical formula
st.latex(r"f(x) = \max(0,x)")

st.markdown("---")

# Interactive input
st.subheader("Interactive ReLU Calculator")

x_value = st.slider(
    "Select Input Value (x)",
    min_value=-10.0,
    max_value=10.0,
    value=2.0,
    step=0.1
)

# ReLU calculation
y_value = max(0, x_value)

# Display result
col1, col2 = st.columns(2)

with col1:
    st.metric("Input (x)", f"{x_value:.2f}")

with col2:
    st.metric("ReLU Output", f"{y_value:.2f}")

# Generate graph data
x = np.linspace(-10, 10, 400)
y = np.maximum(0, x)

# Plot graph
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(x, y, label="ReLU Function", linewidth=2.5)
ax.scatter(x_value, y_value, color="red", s=100,
           label="Selected Input")

ax.axhline(0, color="black", linewidth=0.8)
ax.axvline(0, color="black", linewidth=0.8)

ax.set_title("ReLU Activation Function")
ax.set_xlabel("Input (x)")
ax.set_ylabel("Output f(x)")
ax.grid(True, alpha=0.3)
ax.legend()

st.pyplot(fig)

# Explanation
st.subheader("Function Characteristics")

st.markdown(
    """
    - If x is negative, the output is 0.
    - If x is positive, the output equals x.
    - The output range is from 0 to positive infinity.
    """
)

st.success(f"Result: ReLU({x_value:.2f}) = {y_value:.2f}")
