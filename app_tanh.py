
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Tanh Activation Function",
    page_icon="📈",
    layout="centered"
)

# Title
st.title("Hyperbolic Tangent (Tanh)")
st.subheader("Neural Network Activation Function")

st.write(
    """
    Tanh is an activation function similar to Sigmoid,
    but it maps input values into an output range between -1 and 1.
    """
)

# Mathematical formula
st.latex(r"f(x) = \tanh(x)")

st.markdown("---")

# Interactive input
st.subheader("Interactive Tanh Calculator")

x_value = st.slider(
    "Select Input Value (x)",
    min_value=-10.0,
    max_value=10.0,
    value=0.0,
    step=0.1
)

# Tanh calculation
y_value = np.tanh(x_value)

# Display result
col1, col2 = st.columns(2)

with col1:
    st.metric("Input (x)", f"{x_value:.2f}")

with col2:
    st.metric("Tanh Output", f"{y_value:.4f}")

# Generate graph data
x = np.linspace(-10, 10, 400)
y = np.tanh(x)

# Plot graph
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(x, y, label="Tanh Function", linewidth=2.5)
ax.scatter(x_value, y_value, color="red", s=100,
           label="Selected Input")

ax.axhline(0, color="black", linewidth=0.8)
ax.axvline(0, color="black", linewidth=0.8)

ax.set_title("Tanh Activation Function")
ax.set_xlabel("Input (x)")
ax.set_ylabel("Output f(x)")
ax.set_ylim(-1.1, 1.1)
ax.grid(True, alpha=0.3)
ax.legend()

st.pyplot(fig)

# Explanation
st.subheader("Function Characteristics")

st.markdown(
    """
    - The output is between -1 and 1.
    - When x = 0, the output is 0.
    - Large positive inputs produce outputs close to 1.
    - Large negative inputs produce outputs close to -1.
    """
)

st.success(
    f"Result: Tanh({x_value:.2f}) = {y_value:.4f}"
)

