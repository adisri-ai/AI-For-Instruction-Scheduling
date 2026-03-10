import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import networkx as nx
from flow_inference_utils import load_artifacts
from training import build_cfg
from instruction_scheduling_full_part_2 import predict_flow_for_code
from rule_based_flow import simulate_rule_based_flow
from pipeline_simulator import simulate_pipeline_table
from random_code_generator import generate_code_snippet

# ---------------------------
# Streamlit Interface
# ---------------------------
st.title("🧩 5-Stage Pipeline Simulator with Control Flow Graph (CFG)")
st.markdown("---")

# Example pseudo-codes for users
examples = {
    "Simple If-Else": "a=3\nif(a<4){\n  b=a+1\n}\nelse{\n  b=a-1\n}\nc=b+2\nEND",
    "For Loop": "sum=0\nfor(i=0;i<3;i++){\n  sum=sum+i\n}\nEND",
    "Nested Conditions": "x=5\ny=2\nif(x>y){\n  if(y<3){\n    z=x+y\n  }\n  else{\n    z=x-y\n  }\n}\nEND",
}

choice = st.selectbox(
    "Choose a sample or write your own:", list(examples.keys()) + ["Custom"]
)

# MODIFIED: Handle code_input logic with session state for Custom
if choice != "Custom":
    code_input = examples[choice]
else:
    # NEW: Initialize session state if not present
    if "custom_code" not in st.session_state:
        st.session_state.custom_code = ""

    # NEW: Use columns for layout: text area + Random button
    col1, col2 = st.columns([3, 1])  # 75% for text, 25% for button

    with col1:
        st.session_state.custom_code = st.text_area(
            "Enter your pseudo-code:",
            value=st.session_state.custom_code,
            height=220,
            placeholder="Example:\na=3\nif(a<4){\n  b=a+1\n}\nelse{\n  b=a-1\n}\nc=b+2\nEND",
            key="code_input_area",
        )

    with col2:
        if st.button("Random", key="random_btn"):
            # Generate random code
            full_code, _ = generate_code_snippet()
            # Clean: Remove trailing "END" line to match expected format
            lines = full_code.split("\n")
            if lines and lines[-1].strip() == "END":
                lines = lines[:-1]
            st.session_state.custom_code = "\n".join(lines)
            st.rerun()  # Trigger rerun to update the text area immediately

    code_input = st.session_state.custom_code

# MODIFIED: Removed the st.number_input and fixed the value to 2
flush_penalty = 2

run = st.button("Simulate Pipeline", key="simulate_button")

if run:
    st.info("🔍 Building Control Flow Graph (CFG) and simulating...")

    # --- Build CFG --- (UNCHANGED)
    node_dict, adj_list = build_cfg(code_input)

    # --- Display Node and Adjacency Info --- (MODIFIED)
    st.subheader("🧠 Control Flow Graph Details")

    # --- ADD THIS ---
    st.write("**Original Code:**")
    st.code(code_input, language="cpp")
    # --- END ADD ---

    st.write("**Node Dictionary:**")
    st.json(node_dict)
    st.write("**Adjacency List:**")
    st.json(adj_list)

    # --- Visualize CFG (IMPROVED: Better layout for less congestion) ---
    st.subheader("📊 Control Flow Graph Visualization")
    G = nx.DiGraph()
    for node_id in node_dict.keys():
        G.add_node(node_id)
    for src, dests in adj_list.items():
        for dest in dests:
            G.add_edge(src, dest)

    pos = nx.spring_layout(G, k=0.8, iterations=50)

    plt.figure(figsize=(16, 12))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=800,
        node_color="lightgreen",
        arrowsize=15,
        font_size=8,
        font_weight="bold",
        edge_color="gray",
        width=1.0,
    )
    st.pyplot(plt)

    # --- Load LSTM model ---
    model, tokenizer, meta = load_artifacts("cfg_lstm")

    # --- Run rule-based & LSTM flow ---
    rule_path, _ = simulate_rule_based_flow(code_input)
    lstm_path = predict_flow_for_code(code_input, model, tokenizer, meta)

    truth, pred = {}, {}
    for i, n in enumerate(rule_path):
        if "if(" in n[1] or "for(" in n[1]:
            truth[i] = n[2]
    for i, n in enumerate(lstm_path):
        if "if(" in n[1] or "for(" in n[1]:
            pred[i] = "ENTER"

    rule_instructions = [n[1] for n in rule_path]
    lstm_instructions = [n[1] for n in lstm_path]

    # --- Simulate both pipelines ---
    table_rule, cycles_rule, _ = simulate_pipeline_table(
        rule_instructions, truth, truth, flush_penalty
    )
    table_lstm, cycles_lstm, _ = simulate_pipeline_table(
        lstm_instructions, truth, pred, flush_penalty
    )

    # --- Display results ---
    st.subheader("📘 Rule-Based Execution (Ground Truth)")
    st.dataframe(table_rule, use_container_width=True, height=400)
    st.markdown(f"**Total Cycles:** {cycles_rule}")

    st.subheader("🤖 LSTM-Predicted Execution")
    st.dataframe(table_lstm, use_container_width=True, height=400)
    st.markdown(f"**Total Cycles:** {cycles_lstm}")

    reduction = cycles_rule - cycles_lstm
    percent = (reduction / cycles_rule * 100) if cycles_rule else 0
    st.success(f"✅ Cycle Reduction: {reduction} cycles ({percent:.1f}%)")
