import streamlit as st
import requests
import pandas as pd
from PIL import Image
import io
import time

# Config
API_URL = "http://localhost:8000/analyze"
IMG_BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="RadiScribe AI Engine", layout="wide", page_icon="🩻")

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3050/3050525.png", width=100)
    st.title("RadiScribe AI")
    st.caption("Engine v1.1 (CheXNet + Local/Open-Weight Inference)")
    st.divider()
    
    st.header("Upload X-Ray")
    uploaded_file = st.file_uploader("Choose a Chest X-Ray (PA/AP)", type=["png", "jpg", "jpeg", "dcm"])
    
    if st.button("Download Sample X-Ray"):
        st.info("Downloading sample...")
        # Redirect to external sample
        st.markdown("[Click to get Sample](https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/01E392EE-69F9-4E33-BFCE-E5C968654078.jpeg)")

    st.divider()
    st.header("Demo Video")
    demo_video_url = st.text_input("Demo video URL", value="")
    if demo_video_url:
        st.video(demo_video_url)

    st.header("Partners")
    partners_raw = st.text_area("Add partner names (one per line)", value="Hospital Research Lab\nImaging Center")
    partners = [p.strip() for p in partners_raw.splitlines() if p.strip()]

# Main
st.title("🩻 Medical Image Analysis Console")

if uploaded_file is not None:
    # 1. Layout: Image on Left, Results on Right
    col_img, col_results = st.columns([1, 2])
    
    with col_img:
        st.subheader("Patient Scan")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True, caption="Original X-Ray (PA/AP)")

    # Analysis Trigger
    if col_results.button("⚡ Analyze Scan", type="primary", use_container_width=True):
        with st.spinner("Analyzing anatomy... Generating insights..."):
            try:
                # Reset pointer
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file, "image/png")}
                
                t0 = time.time()
                response = requests.post(API_URL, files=files)
                latency = time.time() - t0
                
                if response.status_code == 200:
                    data = response.json()
                    status = data.get("status")
                    
                    # Status Banner
                    if status == "SUCCESS":
                        col_results.success(f"Analysis Complete ({latency:.2f}s)")
                    elif status == "MANUAL_REVIEW_REQUIRED":
                        col_results.warning(f"⚠️ High Sensitivity Triggered: Manual Review Required")
                    else:
                        col_results.error(f"Analysis Failed: {status}")

                    # TABS for Organized View
                    tab_report, tab_visuals, tab_data = col_results.tabs(["📄 Radiology Report", "👁️ Heatmap & Evidence", "📊 Technical Findings"])
                    
                    # --- TAB 1: REPORT (Readable Terms) ---
                    with tab_report:
                        reasoning = data.get("reasoning")
                        if reasoning:
                            st.markdown("### Clinical Impression")
                            st.info(reasoning.get('impression') or "No specific impression generated.")
                            
                            st.markdown("### Findings Narrative")
                            st.write(reasoning.get('findings_narrative'))
                            
                            st.markdown("### Recommendations")
                            for rec in reasoning.get('recommendations', []):
                                st.markdown(f"- {rec}")
                                
                            st.caption("Auto-Generated Draft. Not a final diagnosis.")
                            
                            # Export Feature
                            full_text = f"IMPRESSION:\n{reasoning.get('impression')}\n\nFINDINGS:\n{reasoning.get('findings_narrative')}\n\nRECOMMENDATIONS:\n{', '.join(reasoning.get('recommendations', []))}"
                            st.text_area("📋 Copyable Report Text", value=full_text, height=150)
                            
                        else:
                            st.warning("No textual report could be generated due to safety constraints.")

                    # --- TAB 2: VISUALS (Heatmap/Annotations) ---
                    with tab_visuals:
                        st.markdown("### AI Attention Map")
                        heatmap_rel = data.get("explainability", {}).get("heatmap_url")
                        
                        if heatmap_rel:
                            heatmap_full_url = f"{IMG_BASE_URL}{heatmap_rel}"
                            # Side-by-side comparison inside table
                            c1, c2 = st.columns(2)
                            with c1:
                                st.image(image, use_column_width=True, caption="Original")
                            with c2:
                                st.image(heatmap_full_url, use_column_width=True, caption="Grad-CAM Overlay")
                            
                            st.info("The heatmap highlights regions contributing most to the AI's prediction.")
                        else:
                            st.warning("Heatmap generation failed or blocked.")

                    # --- TAB 3: DATA (Technical) ---
                    with tab_data:
                        st.markdown("### Detected Abnormalities")
                        findings = data.get("vision_findings", [])
                        if findings:
                            df = pd.DataFrame(findings)
                            st.dataframe(
                                df[["label", "probability", "confidence"]].style.format({"probability": "{:.1%}"}),
                                use_container_width=True,
                                hide_index=True
                            )
                        else:
                            st.write("No significant pathologies detected above threshold.")
                            
                        st.divider()
                        st.json(data.get("metadata"))

                    # --- METRICS & ACCURACY CONTEXT ---
                    st.divider()
                    meta = data.get("metadata", {})
                    conf_score = meta.get('confidence_score', 0)
                    
                    # Layout Metrics
                    c1, c2, c3 = st.columns(3)
                    c1.metric("End-to-End Latency", f"{latency:.2f}s")
                    c2.metric("Processing Time", f"{meta.get('processing_time_ms', 0):.0f}ms")
                    c3.metric("Model Confidence", f"{conf_score:.2f}", 
                              help="0.0-1.0 Score. High (>0.8) means the model successfully identified standard Chest X-Ray anatomy.")
                    
                    # Accuracy Explanation for User
                    if conf_score < 0.6:
                        st.warning(f"⚠️ **Low Confidence ({conf_score:.2f})**: The model is unsure. This often happens with **Non-Chest Images**, Rotated Scans, or Children. **Accuracy is likely reduced.**")
                    elif conf_score > 0.8:
                        st.success(f"✅ **High Confidence**: Analysis is likely accurate (Standard Chest X-Ray detected).")
                    else:
                        st.info(f"ℹ️ **Moderate Confidence**: Usable for screening, but verify findings.")

                else:
                    col_results.error(f"Server Error {response.status_code}: {response.text}")
                    
            except Exception as e:
                col_results.error(f"Connection Error: {e}")

else:
    st.info("Please upload an X-ray from the sidebar to begin.")
    
    # Quick Stats/Dashboard view when empty
    st.markdown("### System Status")
    c1, c2, c3 = st.columns(3)
    c1.metric("Status", "Online", delta="Healthy")
    c2.metric("Active Model", "CheXNet (XRV)")
    c3.metric("Safety Mode", "Strict (Blocking)")
    st.markdown("### Product Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("Supported Modalities", "Chest X-Ray")
    m2.metric("Embedding Dimension", "768")
    m3.metric("Inference Stack", "Local/OSS")

    st.markdown("### Partnered With")
    if partners:
        for partner in partners:
            st.markdown(f"- {partner}")
    else:
        st.caption("Add partners from the sidebar.")
