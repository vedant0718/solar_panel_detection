import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import os

# Page configuration with wide layout
st.set_page_config(page_title="Solar Panel Detection Dashboard", layout="wide")

# Inject custom CSS for a modern dark UI with black and orange
st.markdown("""
    <style>
        /* Overall background and font styling */
        body {
            background-color: #000000;
            color: #ffffff;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        /* Container padding */
        .reportview-container .main .block-container{
            padding: 2rem 2rem;
        }
        /* Sidebar styling */
        .css-1d391kg, .css-1d391kg .sidebar-content {
            background-color: #000000;
            color: #FFA500;
        }
        /* Sidebar text color */
        .css-1d391kg .sidebar-content * {
            color: #FFA500;
        }
        /* Header and title styling */
        h1, h2, h3, h4, h5, h6 {
            color: #FFA500;
        }
        /* Card style for content sections */
        .card {
            background: #111111;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(255, 165, 0, 0.4);
            margin-bottom: 1rem;
        }
        /* Button styling */
        .stButton>button {
            background-color: #FFA500;
            color: #000000;
            border: none;
            border-radius: 5px;
            padding: 0.5rem 1rem;
        }
        /* Tab container override */
        .css-1outpf7 {
            padding: 1rem;
        }
        /* Expander style */
        .streamlit-expanderHeader {
            color: #FFA500;
        }
    </style>
    """, unsafe_allow_html=True)

# Header and introduction
st.title("Solar Panel Detection Dashboard")
st.markdown("""
Welcome to the Solar Panel Detection Dashboard. Use the sidebar to navigate through modules.
Each section is designed with a modern, dark theme using black and orange to minimize scrolling and present key information clearly.
""")

# Sidebar Navigation
module = st.sidebar.radio("Modules", 
                          ["Data Exploration", "Fundamental Functions", "Model Training", "Model Evaluation", "Split Dataset"])

# ----------------- Data Exploration Module -----------------
if module == "Data Exploration":
    st.header("Data Exploration")
    st.markdown("Explore dataset statistics. Use the tabs below to view the histogram of solar panel areas (with observations) or the labels per image.")
    
    from data_exploration import explore_data, value_counts_labels

    tabs = st.tabs(["Histogram & Observations", "Labels per Image"])
    
    with tabs[0]:
        with st.expander("Generate Area Histogram"):
            if st.button("Show Histogram", key="hist"):
                fig, stats = explore_data()
                st.subheader("Histogram of Solar Panel Areas")
                st.pyplot(fig)
                st.markdown("""
                **Observations:**
                - The histogram displays the distribution of solar panel areas in m².
                - The histogram shows a highly right-skewed distribution, with most solar panel areas being small.
                - A dominant peak is observed at lower area values, indicating that small-scale installations are the most common.
                - A long tail extends towards larger values, suggesting the presence of a few large-scale solar farms.
                - The rapid drop in frequency as area increases indicates that very large installations are rare.
                - Outliers might indicate unusual panel sizes.
                """)
                st.info(f"**Mean Area:** {stats['mean_area']:.4f} m²  \n**Std Deviation:** {stats['std_area']:.4f} m²  \n**Total Instances:** {stats['total_instances']}")
    
    with tabs[1]:
        with st.expander("Show Labels per Image"):
            if st.button("View Label Counts", key="labels"):
                counts = value_counts_labels()
                df_counts = pd.DataFrame(list(counts.items()), columns=["Number of Labels", "Number of Images"])
                st.subheader("Labels per Image")
                st.dataframe(df_counts.style.format({"Number of Images": "{:.0f}"}))
                
# ----------------- Fundamental Functions Module -----------------
elif module == "Fundamental Functions":
    st.header("Fundamental Functions")
    st.markdown("Test core functions such as IoU computation, AP metrics, and generate precision-recall curves.")
    
    from fundamental_functions import generate_test_data, compute_precisionRecall, ap_voc11, ap_coco101, area_under_curve_ap, plot_precision_recall_curve
    if st.button("Run Fundamental Functions Test", key="ff_test"):
        with st.spinner("Processing test data and computing metrics..."):
            gt_data, pred_data, pred_scores_data = generate_test_data()
            precision, recall, scores = compute_precisionRecall(gt_data, pred_data, pred_scores_data, iou_threshold=0.5)
            ap_voc = ap_voc11(precision, recall)
            ap_coco = ap_coco101(precision, recall)
            ap_area = area_under_curve_ap(precision, recall)
        st.success("Test completed!")
        with st.container():
            st.markdown("#### Metrics")
            st.write(f"**AP (VOC 11-point):** {ap_voc:.4f}")
            st.write(f"**AP (COCO 101-point):** {ap_coco:.4f}")
            st.write(f"**AP (Area under PR curve):** {ap_area:.4f}")
        with st.container():
            st.markdown("#### Precision-Recall Curve")
            fig = plot_precision_recall_curve(precision, recall, ap_voc, ap_coco, ap_area)
            st.pyplot(fig)

# ----------------- Model Training Module -----------------
elif module == "Model Training":
    st.header("Model Training")
    st.markdown("Train the YOLO object detection model using the provided dataset configuration.")
    from model_training import train_model
    if st.button("Train Model", key="train"):
        with st.spinner("Training model... Check console for details."):
            message = train_model()
        st.success(message)
    else:
        st.info("Click the button to start training.")

# ----------------- Model Evaluation Module -----------------
elif module == "Model Evaluation":
    st.header("Model Evaluation")
    st.markdown("Visualize predictions on test images (ground truth in green and predictions in red) and evaluate performance metrics.")
    
    from model_evaluation import visualize_predictions, evaluate_model
    
    tab1, tab2 = st.tabs(["Visualize Predictions", "Evaluate Metrics"])
    
    with tab1:
        if st.button("Show Predictions", key="pred_vis"):
            with st.spinner("Loading predictions..."):
                figs = visualize_predictions()
            st.success("Predictions loaded!")
            for fig in figs:
                with st.container():
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.pyplot(fig)
                    st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        if st.button("Evaluate Model", key="eval"):
            with st.spinner("Evaluating model..."):
                precision_df, recall_df, f1_df, precision_default, recall_default = evaluate_model()
            st.markdown("### Default Metrics (IoU=0.5, Conf=0.5)")
            st.write(f"**Precision:** {precision_default:.3f}  \n**Recall:** {recall_default:.3f}")
            st.markdown("### Precision Table")
            st.dataframe(precision_df)
            st.markdown("### Recall Table")
            st.dataframe(recall_df)
            st.markdown("### F1 Score Table")
            st.dataframe(f1_df)

# ----------------- Split Dataset Module -----------------
elif module == "Split Dataset":
    st.header("Split Dataset")
    st.markdown("Split the dataset into training, validation, and test sets.")
    from split_dataset import split_dataset
    if st.button("Split Dataset", key="split"):
        with st.spinner("Splitting dataset..."):
            message = split_dataset()
        st.success(message)
