"""
ASL Ablation Study Interactive Dashboard
=========================================
A comprehensive research workbench for exploring ablation study results.

Features:
- 🏆 Leaderboard: Aggregate MAE scores with bar charts
- 📈 Curve Explorer: Multi-scenario stacking, auto-rescaling, "Difference vs LS" mode
- ⚙️ Config Inspector: Side-by-side hyperparameter comparison
- 🎨 Consistent colors: Same experiment = same color everywhere
- 🔍 Hover comparison: See all model values at any X-coordinate

Usage:
    streamlit run asl_interactive_dashboard.py

Requirements:
    pip install streamlit plotly pandas pyyaml
"""

import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import numpy as np
import itertools

# Try to import yaml - optional for config inspection
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="ASL Ablation Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================
st.markdown("""
<style>
    /* Main container */
    .block-container {padding-top: 1rem; padding-bottom: 5rem;}
    
    /* Headers */
    h1 {font-size: 2.2rem; color: #1f77b4;}
    h3 {font-size: 1.5rem; color: #444; border-bottom: 2px solid #ddd; padding-bottom: 5px;}
    
    /* Metric cards */
    .metric-card {
        background-color: #f8f9fa; 
        padding: 15px; 
        border-radius: 8px; 
        border-left: 5px solid #4e8cff;
        margin: 10px 0;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {background-color: #f0f2f6;}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

@st.cache_data
def load_all_experiments(root_dir: str) -> dict:
    """
    Recursively find and load all interactive_plot_data.json files.
    Also loads config.yaml for each experiment if available.
    
    Expected structure:
        root_dir/
            01_Baseline_Naive/
                validation_results/
                    interactive_plot_data.json
                config.yaml
            02_Feature_Peak/
                ...
    """
    experiments = {}
    root = Path(root_dir)
    
    if not root.exists():
        return {}
    
    # Find all interactive plot data files
    json_files = list(root.rglob("interactive_plot_data.json"))
    
    for jf in json_files:
        # Extract experiment name from path
        # Structure: root/EXP_NAME/validation_results/json
        try:
            exp_name = jf.parent.parent.name
        except:
            exp_name = jf.parent.name
        
        # Load plot data
        try:
            with open(jf, 'r') as f:
                plot_data = json.load(f)
        except Exception as e:
            st.warning(f"Failed to load {jf}: {e}")
            continue
        
        # Try to load config.yaml
        config = {}
        config_path = jf.parent.parent / "config.yaml"
        if config_path.exists() and YAML_AVAILABLE:
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            except:
                pass
        
        experiments[exp_name] = {
            "data": plot_data,
            "config": config,
            "path": str(jf)
        }
    
    return experiments


def get_color_map(experiment_names: list) -> dict:
    """
    Assign a unique, consistent color to each experiment.
    Colors persist even when experiments are filtered.
    """
    # Professional color palette (Plotly D3 + extended)
    palette = [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#7f7f7f",  # Gray
        "#bcbd22",  # Olive
        "#17becf",  # Cyan
        "#3366cc",  # Dark Blue
        "#dc3912",  # Dark Red
        "#ff9900",  # Gold
        "#109618",  # Forest Green
        "#990099",  # Magenta
    ]
    color_cycle = itertools.cycle(palette)
    return {name: next(color_cycle) for name in sorted(experiment_names)}


# ============================================================================
# LEADERBOARD TAB
# ============================================================================

def build_leaderboard_df(experiments: dict, selected_exps: list) -> pd.DataFrame:
    """Build a DataFrame with aggregate metrics for the leaderboard."""
    rows = []
    
    for exp in selected_exps:
        if exp not in experiments:
            continue
            
        data = experiments[exp]['data']
        config = experiments[exp].get('config', {})
        
        row = {"Experiment": exp}
        
        # Aggregate MAE across all scenarios
        cbf_maes = []
        att_maes = []
        
        for scenario, content in data.items():
            metrics = content.get('metrics', {})
            if metrics.get('CBF_MAE_NN') is not None:
                cbf_maes.append(metrics['CBF_MAE_NN'])
            if metrics.get('ATT_MAE_NN') is not None:
                att_maes.append(metrics['ATT_MAE_NN'])
        
        if cbf_maes:
            row["Avg CBF MAE"] = np.mean(cbf_maes)
        if att_maes:
            row["Avg ATT MAE"] = np.mean(att_maes)
        
        # Extract key config parameters
        training = config.get('training', {})
        row["LR"] = training.get('learning_rate', 'N/A')
        row["Features"] = len(config.get('active_features', []))
        row["Noise Types"] = len(config.get('data_noise_components', []))
        row["MSE Weight"] = training.get('mse_weight', 'N/A')
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def render_leaderboard_tab(experiments: dict, selected_exps: list):
    """Render the Leaderboard tab content."""
    st.subheader("🏆 Global Performance Summary")
    
    df_scores = build_leaderboard_df(experiments, selected_exps)
    
    if df_scores.empty:
        st.info("No metric data found. Did you run the updated validation script?")
        return
    
    # Display styled table
    numeric_cols = [c for c in ["Avg CBF MAE", "Avg ATT MAE"] if c in df_scores.columns]
    if numeric_cols:
        st.dataframe(
            df_scores.style.background_gradient(subset=numeric_cols, cmap="RdYlGn_r"),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.dataframe(df_scores, use_container_width=True, hide_index=True)
    
    # Bar charts
    col1, col2 = st.columns(2)
    
    with col1:
        if "Avg CBF MAE" in df_scores.columns:
            st.markdown("#### 🩸 CBF Error Comparison")
            fig = px.bar(
                df_scores, 
                x="Experiment", 
                y="Avg CBF MAE", 
                color="Experiment",
                text_auto='.2f'
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if "Avg ATT MAE" in df_scores.columns:
            st.markdown("#### ⏱️ ATT Error Comparison")
            fig = px.bar(
                df_scores, 
                x="Experiment", 
                y="Avg ATT MAE", 
                color="Experiment",
                text_auto='.2f'
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# CURVE EXPLORER TAB
# ============================================================================

def create_metric_plot(
    all_data: dict,
    scenario: str,
    metric_key: str,
    title: str,
    models_to_plot: list,
    color_map: dict,
    view_mode: str,
    show_ls: bool
) -> go.Figure:
    """Create a Plotly figure for a specific metric."""
    fig = go.Figure()
    
    x_label = "X-Axis"
    ref_x = None
    
    # Plot Neural Net curves for each selected model
    for mod in models_to_plot:
        if mod not in all_data:
            continue
        
        exp_data = all_data[mod]['data']
        if scenario not in exp_data:
            continue
        
        try:
            scen_data = exp_data[scenario]
            x_vals = scen_data['x_axis']
            x_label = scen_data.get('x_label', 'X-Axis')
            
            curve_data = scen_data['curves'][metric_key]
            y_nn = np.array(curve_data['nn'], dtype=float)
            y_ls = np.array(curve_data['ls'], dtype=float)
            
            ref_x = x_vals  # Store for LS baseline
            
            if view_mode == "Difference vs LS (Δ)":
                # Negative = NN is better, Positive = LS is better
                y_plot = np.abs(y_nn) - np.abs(y_ls)
                hover_lbl = "Δ Error"
            else:
                y_plot = y_nn
                hover_lbl = "Value"
            
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_plot,
                mode='lines+markers',
                name=mod,
                line=dict(color=color_map.get(mod, '#999'), width=2),
                marker=dict(size=5),
                hovertemplate=f"<b>{mod}</b><br>{x_label}: %{{x:.1f}}<br>{hover_lbl}: %{{y:.3f}}<extra></extra>"
            ))
            
        except (KeyError, TypeError) as e:
            continue
    
    # Add LS Baseline (only in Absolute mode)
    if show_ls and view_mode == "Absolute Values" and ref_x is not None and len(models_to_plot) > 0:
        try:
            # Use LS from first valid model (physics is constant)
            ref_mod = models_to_plot[0]
            ref_ls = all_data[ref_mod]['data'][scenario]['curves'][metric_key]['ls']
            
            fig.add_trace(go.Scatter(
                x=ref_x,
                y=ref_ls,
                mode='lines',
                name="LS Baseline",
                line=dict(color='black', width=2, dash='dot'),
                opacity=0.4,
                hoverinfo='skip'
            ))
        except:
            pass
    
    # Zero line for Difference mode
    if view_mode == "Difference vs LS (Δ)":
        fig.add_hline(y=0, line_color="green", line_width=1.5, opacity=0.6,
                      annotation_text="LS Equivalence", annotation_position="bottom right")
    elif "Bias" in metric_key:
        fig.add_hline(y=0, line_color="gray", line_width=0.5, opacity=0.3)
    
    # Layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title=x_label,
        yaxis_title=metric_key.replace("_", " "),
        margin=dict(l=10, r=10, t=40, b=10),
        height=350,
        hovermode="x unified",  # Compare all models at same X
        legend=dict(orientation="h", y=-0.2, x=0),
        template="plotly_white"
    )
    
    return fig


def render_curve_explorer_tab(
    experiments: dict, 
    selected_exps: list, 
    color_map: dict
):
    """Render the Curve Explorer tab content."""
    
    if not selected_exps:
        st.info("Please select at least one experiment.")
        return
    
    # Get available scenarios from first experiment
    first_exp_data = experiments[selected_exps[0]]['data']
    available_scenarios = sorted(list(first_exp_data.keys()))
    
    # Controls row
    c1, c2, c3 = st.columns([1, 1, 2])
    
    with c1:
        selected_scenarios = st.multiselect(
            "📊 Select Scenarios:",
            available_scenarios,
            default=[available_scenarios[0]] if available_scenarios else []
        )
    
    with c2:
        view_mode = st.radio(
            "View Mode:",
            ["Absolute Values", "Difference vs LS (Δ)"],
            horizontal=True
        )
    
    with c3:
        show_ls = st.checkbox("Show LS Baseline", value=True)
    
    if not selected_scenarios:
        st.warning("Please select at least one scenario.")
        return
    
    st.divider()
    
    # Render plots for each selected scenario
    for scenario in selected_scenarios:
        st.markdown(f"### 🔬 Scenario: {scenario}")
        
        # 2x2 grid: CBF row, ATT row
        col1, col2 = st.columns(2)
        
        with col1:
            fig = create_metric_plot(
                experiments, scenario, "CBF_Bias", "🩸 CBF Bias (Accuracy)",
                selected_exps, color_map, view_mode, show_ls
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = create_metric_plot(
                experiments, scenario, "CBF_CoV", "🩸 CBF CoV (Precision)",
                selected_exps, color_map, view_mode, show_ls
            )
            st.plotly_chart(fig, use_container_width=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            fig = create_metric_plot(
                experiments, scenario, "ATT_Bias", "⏱️ ATT Bias (Accuracy)",
                selected_exps, color_map, view_mode, show_ls
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            fig = create_metric_plot(
                experiments, scenario, "ATT_CoV", "⏱️ ATT CoV (Precision)",
                selected_exps, color_map, view_mode, show_ls
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()


# ============================================================================
# CONFIG INSPECTOR TAB
# ============================================================================

def render_config_inspector_tab(experiments: dict, selected_exps: list):
    """Render the Config Inspector tab content."""
    st.subheader("⚙️ Hyperparameter Comparison")
    
    if not YAML_AVAILABLE:
        st.warning("Install PyYAML (`pip install pyyaml`) to enable config inspection.")
        return
    
    # Build comparison table
    config_rows = []
    for exp in selected_exps:
        if exp not in experiments:
            continue
        
        c = experiments[exp].get('config', {})
        training = c.get('training', {})
        
        flat = {
            "Experiment": exp,
            "Learning Rate": training.get('learning_rate'),
            "MSE Weight": training.get('mse_weight', 0),
            "Reg Lambda": training.get('log_var_reg_lambda', 0),
            "Epochs": training.get('n_epochs'),
            "Ensembles": training.get('n_ensembles'),
            "Hidden Sizes": str(training.get('hidden_sizes', [])),
            "Features": str(c.get('active_features', [])),
            "Noise Types": str(c.get('data_noise_components', []))
        }
        config_rows.append(flat)
    
    if config_rows:
        df = pd.DataFrame(config_rows)
        st.dataframe(df.set_index("Experiment"), use_container_width=True)
    else:
        st.info("No config data available.")
        return
    
    # Raw YAML view
    st.markdown("### 📄 Raw Config View")
    
    cols = st.columns(min(len(selected_exps), 3))
    for i, exp in enumerate(selected_exps[:3]):
        with cols[i]:
            st.markdown(f"**{exp}**")
            config = experiments.get(exp, {}).get('config', {})
            if config:
                st.json(config)
            else:
                st.info("No config found")


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.title("🧠 ASL Ablation Study Dashboard")
    st.markdown("*Interactive exploration of neural network vs. least squares performance*")
    
    # Sidebar
    st.sidebar.title("🎛️ Controls")
    
    # Data source
    data_path = st.sidebar.text_input(
        "📁 Data Directory:",
        value="hpc_ablation_jobs",
        help="Path to folder containing experiment results"
    )
    
    # Load data
    experiments = load_all_experiments(data_path)
    
    if not experiments:
        st.error(f"❌ No data found in '{data_path}'")
        st.markdown("""
        **Expected structure:**
        ```
        hpc_ablation_jobs/
        ├── 01_Baseline_Naive/
        │   ├── validation_results/
        │   │   └── interactive_plot_data.json
        │   └── config.yaml
        ├── 02_Feature_Peak/
        │   └── ...
        ```
        
        **To generate data:**
        ```bash
        python validate.py --run_dir hpc_ablation_jobs/01_Baseline_Naive \\
                          --output_dir hpc_ablation_jobs/01_Baseline_Naive/validation_results
        ```
        """)
        st.stop()
    
    # Success message
    st.sidebar.success(f"✅ Loaded {len(experiments)} experiments")
    
    # Get all experiment names and create color map
    all_exp_names = sorted(list(experiments.keys()))
    COLOR_MAP = get_color_map(all_exp_names)
    
    # Model selection
    st.sidebar.divider()
    st.sidebar.subheader("🔬 Experiment Selection")
    
    selected_exps = st.sidebar.multiselect(
        "Select models to compare:",
        all_exp_names,
        default=all_exp_names[:min(3, len(all_exp_names))],
        help="Uncheck models to remove outliers (Y-axis auto-rescales)"
    )
    
    # Show color legend
    if selected_exps:
        st.sidebar.markdown("**Color Legend:**")
        for exp in selected_exps:
            color = COLOR_MAP.get(exp, '#999')
            st.sidebar.markdown(
                f'<span style="color:{color}; font-weight:bold;">●</span> {exp}',
                unsafe_allow_html=True
            )
    
    if not selected_exps:
        st.warning("Please select at least one experiment from the sidebar.")
        st.stop()
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs([
        "🏆 Leaderboard",
        "📈 Curve Explorer", 
        "⚙️ Config Inspector"
    ])
    
    with tab1:
        render_leaderboard_tab(experiments, selected_exps)
    
    with tab2:
        render_curve_explorer_tab(experiments, selected_exps, COLOR_MAP)
    
    with tab3:
        render_config_inspector_tab(experiments, selected_exps)


if __name__ == "__main__":
    main()
