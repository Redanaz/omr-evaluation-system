import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import streamlit as st
import pandas as pd
import cv2
import io
import json
import time
import os
from tempfile import NamedTemporaryFile
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from omr import evaluate, validate_answer_key, SUBJECTS

# Page configuration
st.set_page_config(
    page_title="OMR Evaluation System",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">📝 Automated OMR Evaluation System</h1>', unsafe_allow_html=True)
st.markdown("### 🎯 Hackathon Project: Instant OMR Sheet Processing with 99.5%+ Accuracy")

# Add some info about the system
with st.expander("ℹ️ About This System"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **🚀 Features:**
        - Instant OMR processing from phone images
        - Support for multiple exam versions
        - Subject-wise performance analysis
        - Batch processing capabilities
        - Real-time error detection
        """)
    with col2:
        st.markdown("""
        **📊 Specifications:**
        - 100 questions across 5 subjects
        - <0.5% error rate target
        - <30 seconds processing time
        - Handles 3000+ sheets per exam
        - Export in multiple formats
        """)

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
    batch_mode = st.checkbox("Batch processing mode", value=False)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📤 Upload OMR Sheets")
    uploaded_files = st.file_uploader(
        "Choose OMR image files",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Upload clear images of filled OMR sheets. Supports multiple files for batch processing."
    )

with col2:
    if uploaded_files:
        st.metric("📁 Files Uploaded", len(uploaded_files))
        st.metric("📋 Questions/Sheet", len(answer_key))
        st.metric("🎯 Max Score", len(answer_key))
        
        # Show file info
        with st.expander("📄 File Details"):
            for i, file in enumerate(uploaded_files):
                file_size_mb = len(file.getvalue()) / (1024 * 1024)
                st.write(f"**{i+1}.** {file.name} ({file_size_mb:.1f} MB)")

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
    processing_container = st.container()
    
    # Clear previous results
    st.session_state.results_data = []
    
    with processing_container:
        col1, col2, col3 = st.columns(3)
        with col1:
            processed_count = st.empty()
        with col2:
            success_count = st.empty()
        with col3:
            error_count = st.empty()
    
    successful_processing = 0
    failed_processing = 0
    
    for idx, uploaded_file in enumerate(uploaded_files):
        progress = (idx) / len(uploaded_files)
        progress_bar.progress(progress)
        status_text.text(f"Processing {uploaded_file.name} ({idx+1}/{len(uploaded_files)})...")
        
        processed_count.metric("📊 Processed", f"{idx+1}/{len(uploaded_files)}")
        
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
                st.error(f"❌ Failed to process {uploaded_file.name}: {score if isinstance(score, str) else 'Unknown error'}")
                
        except Exception as e:
            failed_processing += 1
            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
        
        # Update counters
        success_count.metric("✅ Successful", successful_processing)
        error_count.metric("❌ Failed", failed_processing)
        
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
        st.success(f"🎉 Successfully processed {len(st.session_state.results_data)} out of {len(uploaded_files)} sheets!")
        if failed_processing > 0:
            st.warning(f"⚠️ {failed_processing} sheets failed to process. Check image quality and try again.")
    
    time.sleep(1)
    st.rerun()

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
    
    # Subject-wise performance chart
    st.subheader("📈 Subject-wise Performance Analysis")
    subject_cols = [col for col in df.columns if col in SUBJECTS.keys()]
    if subject_cols:
        subject_avg = df[subject_cols].mean()
        
        fig = px.bar(
            x=subject_avg.index,
            y=subject_avg.values,
            title="Average Score by Subject (out of 20)",
            labels={'x': 'Subject', 'y': 'Average Score'},
            color=subject_avg.values,
            color_continuous_scale='viridis',
            text=subject_avg.values.round(1)
        )
        fig.update_layout(showlegend=False, height=400)
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    # Score distribution
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            df, 
            x="Percentage", 
            title="Score Distribution",
            nbins=20,
            labels={'Percentage': 'Score (%)', 'count': 'Number of Students'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(
            df, 
            y="Percentage", 
            title="Score Statistics",
            labels={'Percentage': 'Score (%)'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed results table
    st.subheader("📋 Summary Results")
    display_df = df.drop(columns=['Processing Time (s)'], errors='ignore')
    st.dataframe(display_df, use_container_width=True)
    
    # Individual student results
    if show_detailed_results:
        st.subheader("👤 Individual Student Analysis")
        
        student_selection = st.selectbox(
            "Select a student for detailed analysis:",
            options=range(len(st.session_state.results_data)),
            format_func=lambda x: f"{st.session_state.results_data[x]['Student ID']} - {st.session_state.results_data[x]['Percentage']}%"
        )
        
        selected_result = st.session_state.results_data[student_selection]
        
        # Student summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Student ID", selected_result["Student ID"])
        with col2:
            st.metric("Total Score", f"{selected_result['Total Score']}/{len(answer_key)}")
        with col3:
            st.metric("Percentage", f"{selected_result['Percentage']}%")
        with col4:
            grade = "A" if selected_result['Percentage'] >= 90 else "B" if selected_result['Percentage'] >= 80 else "C" if selected_result['Percentage'] >= 70 else "D" if selected_result['Percentage'] >= 60 else "F"
            st.metric("Grade", grade)
        
        # Show overlay image if available
        if show_overlay_images and 'overlay_image' in selected_result:
            st.subheader("🎯 Bubble Detection Results")
            overlay_img = selected_result['overlay_image']
            if overlay_img is not None:
                # Convert BGR to RGB for display
                overlay_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
                st.image(
                    overlay_rgb,
                    caption=f"Answer Detection for {selected_result['Student ID']} - Green: Filled, Red: Empty, Blue: Multiple",
                    use_column_width=True
                )
            else:
                st.warning("No overlay image available for this student")
        
        # Question-wise results
        if 'detailed_results' in selected_result:
            st.subheader("📝 Question-wise Analysis")
            
            detailed_results = selected_result['detailed_results']
            
            # Create tabs for each subject
            subject_tabs = st.tabs(list(SUBJECTS.keys()))
            
            for tab_idx, (subject_name, question_range) in enumerate(SUBJECTS.items()):
                with subject_tabs[tab_idx]:
                    subject_questions = [r for r in detailed_results if r[0] in question_range]
                    
                    if subject_questions:
                        # Subject statistics
                        correct_count = sum(1 for q in subject_questions if q[3])
                        total_questions = len(subject_questions)
                        subject_percentage = (correct_count / total_questions) * 100
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Correct Answers", f"{correct_count}/{total_questions}")
                        with col2:
                            st.metric("Subject Score", f"{subject_percentage:.1f}%")
                        with col3:
                            grade = "A" if subject_percentage >= 90 else "B" if subject_percentage >= 80 else "C" if subject_percentage >= 70 else "D" if subject_percentage >= 60 else "F"
                            st.metric("Grade", grade)
                        
                        # Question details in a more compact format
                        st.write("**Question Details:**")
                        
                        # Group questions into rows of 5 for better display
                        questions_per_row = 5
                        for row_start in range(0, len(subject_questions), questions_per_row):
                            cols = st.columns(questions_per_row)
                            for col_idx, q_idx in enumerate(range(row_start, min(row_start + questions_per_row, len(subject_questions)))):
                                if col_idx < len(cols):
                                    question_num, student_ans, correct_ans, is_correct = subject_questions[q_idx]
                                    
                                    student_answer_text = "✖️"  # No answer
                                    if student_ans == -1:
                                        student_answer_text = "⚠️"  # Multiple
                                    elif student_ans >= 0:
                                        student_answer_text = chr(65 + student_ans)
                                    
                                    status_icon = "✅" if is_correct else "❌"
                                    
                                    with cols[col_idx]:
                                        st.write(f"**Q{question_num}**")
                                        st.write(f"{status_icon} {student_answer_text}/{chr(65 + correct_ans)}")
    
    # Export functionality
    st.header("💾 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CSV Export
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"omr_results_{version}_{len(st.session_state.results_data)}_students.csv",
            mime="text/csv"
        )
    
    with col2:
        # Excel Export
        try:
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                df.to_excel(writer, index=False, sheet_name="Summary")
            
            st.download_button(
                label="📊 Download Excel",
                data=excel_buffer.getvalue(),
                file_name=f"omr_results_{version}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.error(f"Excel export not available: {str(e)}")
    
    with col3:
        # JSON Export
        json_data = {
            "exam_version": version,
            "total_students": len(st.session_state.results_data),
            "average_score": df["Total Score"].mean(),
            "results": df.to_dict("records")
        }
        
        json_str = json.dumps(json_data, indent=2)
        st.download_button(
            label="🔧 Download JSON",
            data=json_str,
            file_name=f"omr_data_{version}.json",
            mime="application/json"
        )

# Reset button
if st.session_state.results_data:
    st.header("🔄 Process New Batch")
    if st.button("🆕 Clear Results and Upload New Images", type="primary"):
        st.session_state.results_data = []
        st.session_state.processing_complete = False
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>🏆 OMR Evaluation System - Hackathon Project</strong></p>
        <p>Automated processing of 3000+ sheets with 99.5%+ accuracy | Built with Streamlit & OpenCV</p>
        <p>📸 Upload clear, well-lit images for best results</p>
    </div>
    """, 
    unsafe_allow_html=True
)