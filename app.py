import streamlit as st
import pandas as pd
import cv2
import io
import json
import time
import os
from tempfile import NamedTemporaryFile
from pathlib import Path

# Try importing optional packages
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly not available - using basic charts")

from omr import evaluate, validate_answer_key, SUBJECTS

# Page configuration
st.set_page_config(
    page_title="OMR Evaluation System",
    page_icon="📝",
    layout="wide"
)

# Title and description
st.title("📝 Automated OMR Evaluation System")
st.write("### 🎯 Hackathon Project: Instant OMR Sheet Processing")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Load answer keys
    try:
        answer_keys_path = "answer_keys.json"
        if not os.path.exists(answer_keys_path):
            # Create default answer keys if not found
            default_keys = {
                "A": {str(i): i % 4 for i in range(100)},
                "R": {str(i): (3 - i) % 4 for i in range(100)}
            }
            with open(answer_keys_path, "w") as f:
                json.dump(default_keys, f)
            st.warning("⚠️ Using default answer keys. Please update answer_keys.json with actual keys.")
        
        with open(answer_keys_path) as f:
            answer_keys = json.load(f)
        st.success("✅ Answer keys loaded")
    except Exception as e:
        st.error(f"❌ Error loading answer keys: {str(e)}")
        st.stop()
    
    # Version selection
    version = st.selectbox(
        "📋 Select Exam Version", 
        list(answer_keys.keys()),
        help="Choose the correct answer key version"
    )
    
    # Convert answer key
    try:
        answer_key = {int(k): v for k, v in answer_keys[version].items()}
        if not validate_answer_key(answer_key):
            st.error("❌ Invalid answer key format!")
            st.stop()
        st.success(f"✅ Version '{version}' ready")
    except ValueError:
        st.error("❌ Error processing answer key!")
        st.stop()
    
    # Processing options
    st.subheader("🔧 Processing Options")
    show_detailed_results = st.checkbox("Show detailed analysis", value=True)
    show_overlay_images = st.checkbox("Show bubble detection", value=True)

# File upload
st.header("📤 Upload OMR Sheets")
uploaded_files = st.file_uploader(
    "Choose OMR image files",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    help="Upload clear images of filled OMR sheets"
)

# Show upload info
if uploaded_files:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📁 Files", len(uploaded_files))
    with col2:
        st.metric("📋 Questions", len(answer_key))
    with col3:
        st.metric("🎯 Max Score", len(answer_key))

# Results storage
if 'results_data' not in st.session_state:
    st.session_state.results_data = []
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

# Processing
if uploaded_files and not st.session_state.processing_complete:
    st.header("🔄 Processing Results")
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Clear previous results
    st.session_state.results_data = []
    
    successful_processing = 0
    failed_processing = 0
    
    for idx, uploaded_file in enumerate(uploaded_files):
        progress = (idx) / len(uploaded_files)
        progress_bar.progress(progress)
        status_text.text(f"Processing {uploaded_file.name} ({idx+1}/{len(uploaded_files)})...")
        
        # Save uploaded file temporarily
        with NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        
        try:
            # Process the image
            start_time = time.time()
            student_id = f"Student_{idx+1:03d}"
            
            overlay, results, score, subject_scores, _ = evaluate(
                tmp_path, answer_key, student_id=student_id
            )
            
            processing_time = time.time() - start_time
            
            if overlay is not None and results is not None:
                # Success
                successful_processing += 1
                result_data = {
                    "Student ID": student_id,
                    "File Name": uploaded_file.name,
                    "Total Score": score,
                    "Percentage": round((score / len(answer_key)) * 100, 2),
                    "Processing Time (s)": round(processing_time, 2),
                    **subject_scores
                }
                
                # Add detailed results
                result_data["detailed_results"] = results
                result_data["overlay_image"] = overlay
                
                st.session_state.results_data.append(result_data)
                
            else:
                # Error in processing
                failed_processing += 1
                st.error(f"❌ Failed to process {uploaded_file.name}")
                
        except Exception as e:
            failed_processing += 1
            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
        
        # Clean up temporary file
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except:
            pass
    
    progress_bar.progress(1.0)
    status_text.text("✅ Processing complete!")
    st.session_state.processing_complete = True
    
    # Show summary
    if st.session_state.results_data:
        st.success(f"🎉 Successfully processed {successful_processing}/{len(uploaded_files)} sheets!")
    
    time.sleep(1)
    
    # Refresh page - compatible with all Streamlit versions
    try:
        st.rerun()
    except AttributeError:
        try:
            st.experimental_rerun()
        except AttributeError:
            st.write("Please refresh the page to see results")

# Display results if available
if st.session_state.results_data:
    st.header("📊 Evaluation Results")
    
    # Summary statistics
    df = pd.DataFrame([{k: v for k, v in result.items() 
                      if k not in ['detailed_results', 'overlay_image']} 
                     for result in st.session_state.results_data])
    
    # Overall statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_score = df["Total Score"].mean()
        st.metric("📈 Average Score", f"{avg_score:.1f}/{len(answer_key)}")
    with col2:
        avg_percentage = df["Percentage"].mean()
        st.metric("📊 Average %", f"{avg_percentage:.1f}%")
    with col3:
        max_score = df["Total Score"].max()
        st.metric("🏆 Highest Score", f"{max_score}/{len(answer_key)}")
    with col4:
        avg_time = df["Processing Time (s)"].mean()
        st.metric("⏱️ Avg. Time", f"{avg_time:.2f}s")
    
    # Subject-wise performance
    st.subheader("📈 Subject-wise Performance")
    subject_cols = [col for col in df.columns if col in SUBJECTS.keys()]
    if subject_cols:
        subject_avg = df[subject_cols].mean()
        
        if PLOTLY_AVAILABLE:
            # Advanced chart with Plotly
            fig = px.bar(
                x=subject_avg.index,
                y=subject_avg.values,
                title="Average Score by Subject (out of 20)",
                labels={'x': 'Subject', 'y': 'Average Score'},
                text=subject_avg.values.round(1)
            )
            fig.update_traces(texttemplate='%{text}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Simple chart with Streamlit
            st.bar_chart(subject_avg)
    
    # Score distribution
    if PLOTLY_AVAILABLE:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x="Percentage", title="Score Distribution", nbins=10)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.box(df, y="Percentage", title="Score Statistics")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.subheader("📊 Score Distribution")
        st.bar_chart(df["Percentage"].value_counts().sort_index())
    
    # Detailed results table
    st.subheader("📋 Summary Results")
    display_df = df.drop(columns=['Processing Time (s)'], errors='ignore')
    st.dataframe(display_df, use_container_width=True)
    
    # Individual student results
    if show_detailed_results:
        st.subheader("👤 Individual Student Analysis")
        
        student_selection = st.selectbox(
            "Select a student:",
            options=range(len(st.session_state.results_data)),
            format_func=lambda x: f"{st.session_state.results_data[x]['Student ID']} ({st.session_state.results_data[x]['Percentage']}%)"
        )
        
        selected_result = st.session_state.results_data[student_selection]
        
        # Student summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Student", selected_result["Student ID"])
        with col2:
            st.metric("Score", f"{selected_result['Total Score']}/{len(answer_key)}")
        with col3:
            st.metric("Percentage", f"{selected_result['Percentage']}%")
        with col4:
            grade = "A" if selected_result['Percentage'] >= 90 else "B" if selected_result['Percentage'] >= 80 else "C" if selected_result['Percentage'] >= 70 else "D" if selected_result['Percentage'] >= 60 else "F"
            st.metric("Grade", grade)
        
        # Show overlay image
        if show_overlay_images and 'overlay_image' in selected_result:
            st.subheader("🎯 Bubble Detection Results")
            overlay_img = selected_result['overlay_image']
            if overlay_img is not None:
                overlay_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
                st.image(overlay_rgb, caption="Green: Correct, Red: Empty, Blue: Multiple", use_column_width=True)
    
    # Export functionality
    st.header("💾 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV Export
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download as CSV",
            data=csv_data,
            file_name=f"omr_results_{version}_{len(st.session_state.results_data)}_students.csv",
            mime="text/csv"
        )
    
    with col2:
        # JSON Export
        json_data = {
            "exam_version": version,
            "total_students": len(st.session_state.results_data),
            "average_score": df["Total Score"].mean(),
            "results": df.to_dict("records")
        }
        json_str = json.dumps(json_data, indent=2)
        st.download_button(
            label="🔧 Download as JSON",
            data=json_str,
            file_name=f"omr_data_{version}.json",
            mime="application/json"
        )

# Reset functionality
if st.session_state.results_data:
    if st.button("🆕 Process New Images"):
        st.session_state.results_data = []
        st.session_state.processing_complete = False
        try:
            st.rerun()
        except AttributeError:
            try:
                st.experimental_rerun()
            except AttributeError:
                st.write("Click 'Process New Images' again or refresh the page")

# Footer
st.markdown("---")
st.markdown("**🏆 OMR Evaluation System - Hackathon Project** | Built with Streamlit & OpenCV")