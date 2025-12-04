import streamlit as st
import matplotlib.pyplot as plt

def circular_progress(score, color, label):
    fig, ax = plt.subplots(figsize=(2,2))
    ax.pie([score, 100 - score], colors=[color, "#222"], startangle=90, counterclock=False, wedgeprops=dict(width=0.3))
    ax.text(0, 0, f"{int(score)}%", ha='center', va='center', fontsize=18, color=color, fontweight='bold')
    plt.axis('equal')
    plt.title(label, fontsize=14)
    return fig

def st_circular_progress(score, color, label):
    fig = circular_progress(score, color, label)
    st.pyplot(fig)
