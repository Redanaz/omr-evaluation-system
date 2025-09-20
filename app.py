import streamlit as st
import pandas as pd
import cv2
import io
import json
import time
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
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">📝 Automated OMR Evaluation System</h1>', unsafe_allow_html=True)
st.markdown("### Upload OMR sheets, get instant scores, and export comprehensive results")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Load answer keys
    try:
        with open("answer_keys.json") as f:
            answer_keys = json.load(f)
        st.success("✅ Answer keys loaded successfully")
    except FileNotFoundError:
        st.error("❌ answer_keys.json not found!")
        st.stop()
    except json.JSONDecodeError:
        st.error("❌ Invalid answer_keys.json format!")
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
        st.success(f"✅ Answer key '{version}' validated")
    except ValueError:
        st.error("❌ Error processing answer key!")
        st.stop()
    
    # Processing options
    st.subheader("🔧 Processing Options")
    show_detailed_results = st.checkbox("Show detailed question-wise results", value=True)
    show_overlay_images = st.checkbox("Show bubble detection overlay", value=True)
    auto_export = st.checkbox("Auto-export results", value=False)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📤 Upload OMR Sheets")
    uploaded_files = st.file_uploader(
        "Choose OMR image files",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Upload clear images of filled OMR sheets"
    )

with col2:
    if uploaded_files:
        st.metric("📁 Files Uploaded", len(uploaded_files))
        st.metric("📋 Questions per Sheet", len(answer_key))
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
    
    for idx, uploaded_file in enumerate(uploaded_files):
        progress = (idx) / len(uploaded_files)
        progress_bar.progress(progress)
        status_text.text(f"Processing {uploaded_file.name}...")
        
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
                st.error(f"❌ Failed to process {uploaded_file.name}: {score}")
                
        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
        
        # Clean up temporary file
        Path(tmp_path).unlink(missing_ok=True)
    
    progress_bar.progress(1.0)
    status_text.text("✅ Processing complete!")
    st.session_state.processing_complete = True
    
    # Show summary
    if st.session_state.results_data:
        st.success(f"🎉 Successfully processed {len(st.session_state.results_data)} out of {len(uploaded_files)} sheets")
    
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
        st.metric("⏱️ Avg. Process Time", f"{avg_time:.2f}s")
    
    # Subject-wise performance chart
    st.subheader("📈 Subject-wise Performance Analysis")
    subject_cols = [col for col in df.columns if col in SUBJECTS.keys()]
    if subject_cols:
        subject_avg = df[subject_cols].mean()
        
        fig = px.bar(
            x=subject_avg.index,
            y=subject_avg.values,
            title="Average Score by Subject",
            labels={'x': 'Subject', 'y': 'Average Score'},
            color=subject_avg.values,
            color_continuous_scale='viridis'
        )
        fig.update_layout(showlegend=False)
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
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(
            df, 
            y="Percentage", 
            title="Score Distribution Box Plot",
            labels={'Percentage': 'Score (%)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed results table
    st.subheader("📋 Detailed Results")
    display_df = df.drop(columns=['Processing Time (s)'], errors='ignore')
    st.dataframe(display_df, use_container_width=True)
    
    # Individual student results
    if show_detailed_results:
        st.subheader("👤 Individual Student Analysis")
        
        student_selection = st.selectbox(
            "Select a student for detailed view:",
            options=range(len(st.session_state.results_data)),
            format_func=lambda x: st.session_state.results_data[x]["Student ID"]
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
            processing_time = selected_result.get("Processing Time (s)", "N/A")
            st.metric("Process Time", f"{processing_time}s" if processing_time != "N/A" else "N/A")
        
        # Show overlay image if available
        if show_overlay_images and 'overlay_image' in selected_result:
            st.subheader("🎯 Bubble Detection Results")
            overlay_img = selected_result['overlay_image']
            if overlay_img is not None:
                # Convert BGR to RGB for display
                overlay_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
                st.image(
                    overlay_rgb,
                    caption=f"Answer Detection for {selected_result['Student ID']}",
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
                        
                        # Question details
                        question_data = []
                        for question_num, student_ans, correct_ans, is_correct in subject_questions:
                            student_answer_text = "No Answer"
                            if student_ans == -1:
                                student_answer_text = "Multiple Answers"
                            elif student_ans >= 0:
                                student_answer_text = chr(65 + student_ans)
                            
                            question_data.append({
                                "Question": question_num,
                                "Student Answer": student_answer_text,
                                "Correct Answer": chr(65 + correct_ans),
                                "Result": "✅ Correct" if is_correct else "❌ Incorrect",
                                "Status": "✅" if is_correct else "❌"
                            })
                        
                        question_df = pd.DataFrame(question_data)
                        st.dataframe(question_df, use_container_width=True, hide_index=True)
    
    # Export functionality
    st.header("💾 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CSV Export
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download as CSV",
            data=csv_data,
            file_name=f"omr_results_{version}_{len(st.session_state.results_data)}_students.csv",
            mime="text/csv",
            help="Download summary results as CSV file"
        )
    
    with col2:
        # Excel Export
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
            # Summary sheet
            df.to_excel(writer, index=False, sheet_name="Summary")
            
            # Detailed results for each student
            for idx, result in enumerate(st.session_state.results_data):
                if 'detailed_results' in result:
                    detailed_data = []
                    for q_num, s_ans, c_ans, is_correct in result['detailed_results']:
                        s_ans_text = "No Answer" if s_ans == -2 else "Multiple" if s_ans == -1 else chr(65 + s_ans)
                        detailed_data.append({
                            "Question": q_num,
                            "Student_Answer": s_ans_text,
                            "Correct_Answer": chr(65 + c_ans),
                            "Correct": is_correct,
                            "Subject": next((subj for subj, qrange in SUBJECTS.items() if q_num in qrange), "Unknown")
                        })
                    
                    if detailed_data:
                        detail_df = pd.DataFrame(detailed_data)
                        sheet_name = f"Student_{idx+1:03d}"
                        detail_df.to_excel(writer, index=False, sheet_name=sheet_name)
        
        st.download_button(
            label="📊 Download as Excel",
            data=excel_buffer.getvalue(),
            file_name=f"omr_results_detailed_{version}_{len(st.session_state.results_data)}_students.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Download detailed results with individual student sheets"
        )
    
    with col3:
        # JSON Export (for further processing)
        json_data = {
            "exam_version": version,
            "total_students": len(st.session_state.results_data),
            "summary_statistics": {
                "average_score": df["Total Score"].mean(),
                "average_percentage": df["Percentage"].mean(),
                "highest_score": df["Total Score"].max(),
                "lowest_score": df["Total Score"].min(),
                "subject_averages": {subj: df[subj].mean() for subj in SUBJECTS.keys() if subj in df.columns}
            },
            "detailed_results": [
                {
                    "student_id": result["Student ID"],
                    "file_name": result["File Name"],
                    "scores": {subj: result[subj] for subj in SUBJECTS.keys() if subj in result},
                    "total_score": result["Total Score"],
                    "percentage": result["Percentage"]
                }
                for result in st.session_state.results_data
            ]
        }
        
        json_str = json.dumps(json_data, indent=2)
        st.download_button(
            label="🔧 Download as JSON",
            data=json_str,
            file_name=f"omr_results_data_{version}.json",
            mime="application/json",
            help="Download structured data for further processing"
        )

# Reset button
if st.session_state.results_data:
    st.header("🔄 Process New Batch")
    if st.button("Clear Results and Process New Images", type="primary"):
        st.session_state.results_data = []
        st.session_state.processing_complete = False
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>OMR Evaluation System</strong> | Built for Hackathon</p>
        <p>Upload clear, well-lit images for best results | Supported formats: JPG, JPEG, PNG</p>
    </div>
    """, 
    unsafe_allow_html=True
)

# Auto-export functionality
if auto_export and st.session_state.results_data:
    # Auto-save results to local files
    try:
        results_dir = Path("auto_results")
        results_dir.mkdir(exist_ok=True)
        
        # Save CSV
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = results_dir / f"omr_results_{timestamp}.csv"
        df.to_csv(csv_filename, index=False)
        
        st.sidebar.success(f"✅ Auto-exported to {csv_filename}")
        
    except Exception as e:
        st.sidebar.error(f"❌ Auto-export failed: {str(e)}")


    """### 📊 Understanding Results:
    - Green circles: Correctly detected filled bubbles
    - Red circles: Unfilled bubbles
    - Blue circles: Multiple answers detected (marked incorrect)
    - Subject scores are calculated automatically based on question ranges
    """