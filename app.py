import os
import asyncio
import logging
import streamlit as st
from src.pdf_extractor import extract_text_from_pdf
from src.document_processor import LegalDocumentProcessor
import json
from datetime import datetime
from fpdf import FPDF, XPos, YPos
from dotenv import load_dotenv

# Load environment variables (for local development)
load_dotenv()

# Get API key - prioritize Streamlit secrets, fallback to environment variable
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except (KeyError, FileNotFoundError):
    api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("⚠️ OpenAI API key not found! Please add it to Streamlit secrets or .env file.")
    st.info("For Streamlit Cloud: Add OPENAI_API_KEY in your app settings under 'Secrets'")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key

# Set up logging
logging.basicConfig(level=logging.INFO)

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "documents" not in st.session_state:
        st.session_state.documents = {}  # Store multiple documents
    if "current_doc" not in st.session_state:
        st.session_state.current_doc = None
    if "document_processed" not in st.session_state:
        st.session_state.document_processed = False

def clear_chat():
    st.session_state.messages = []

def new_chat():
    st.session_state.messages = []
    st.session_state.documents = {}
    st.session_state.current_doc = None
    st.session_state.document_processed = False

def create_pdf_from_text(text: str, title: str) -> bytes:
    """Convert text to PDF and return as bytes"""
    try:
        pdf = FPDF()
        pdf.add_page()
        
        # Use Helvetica (built-in font) instead of Arial
        pdf.set_font("helvetica", "B", 16)
        # Update cell parameters to use new positioning
        pdf.cell(0, 10, title.encode('latin-1', 'replace').decode('latin-1'), 
                new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        pdf.ln(10)
        
        # Use Helvetica for content
        pdf.set_font("helvetica", size=12)
        safe_text = text.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 10, safe_text)
        
        return bytes(pdf.output())
    except Exception as e:
        logging.error(f"Error creating PDF: {e}")
        raise

def get_download_filename(request_type: str, doc_name: str, ext: str = "pdf") -> str:
    """Generate a filename for downloaded content"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{request_type}_{doc_name}_{timestamp}.{ext}"

def process_request(request_type, question=None):
    try:
        processor = LegalDocumentProcessor()
        with st.spinner("Processing..."):
            result = asyncio.run(processor.process_document(
                st.session_state.documents[st.session_state.current_doc]["text"], 
                request_type,
                question
            ))
            
            if "error" in result:
                response = f"Error: {result['error']}"
                st.error(response)
            else:
                response = result["result"]
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
                
                # Add spacing after buttons
                st.write("\n")
                st.divider()
                
                # Create full-width response container
                st.markdown("### Generated Response")
                st.write(response)
                
                # Add download button in a separate row
                st.write("")  # Add space
                doc_name = st.session_state.current_doc.split('.')[0]
                
                # Ensure PDF extension
                filename = get_download_filename(request_type, doc_name, "pdf")
                
                try:
                    # Get friendly name for the action type
                    action_names = {
                        "summary": "Summary",
                        "appeal": "Appeal",
                        "review": "Review",
                        "lawsuit": "Lawsuit",
                        "lawsuit_response": "Lawsuit Response",
                        "contract_analysis": "Contract Analysis",
                        "chat": "Chat Response"
                    }
                    action_name = action_names.get(request_type, request_type.title())
                    
                    # Convert response to PDF with error handling
                    title = f"{action_name} - {doc_name}"
                    pdf_content = create_pdf_from_text(response, title)
                    
                    # Center the download button with wider columns
                    left_col, center_col, right_col = st.columns([2, 3, 2])
                    with center_col:
                        st.download_button(
                            label=f"📥 Download {action_name}",
                            data=pdf_content,
                            file_name=filename,
                            mime="application/pdf",
                            key=f"download_{request_type}_{datetime.now().strftime('%H%M%S')}",
                            use_container_width=True
                        )
                except Exception as pdf_error:
                    st.error(f"Error creating PDF: {pdf_error}")
                    
                # Add final divider
                st.write("")
                st.divider()
                
            return response
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        logging.error(error_msg)
        st.error(error_msg)
        return error_msg

def show_user_manual():
    with st.expander("📖 User Manual - How to Use Smart Legal Solutions"):
        st.markdown("""
        # Smart Legal Solutions - User Manual

        ## 🚀 Getting Started
        1. **Upload Documents**
           * Use the sidebar's "Upload legal documents" section
           * Supports PDF files only
           * Multiple documents can be uploaded simultaneously

        2. **Process Documents**
           * Click "Process" button next to each uploaded document
           * Wait for confirmation message
           * Document is ready for analysis when processing completes

        ## 💡 Available Features

        ### 1. Document Summary
        * Generates comprehensive legal document summaries
        * Includes: Case overview, key facts, legal issues, arguments, and conclusions
        * Perfect for quick document understanding

        ### 2. Legal Appeal Generation
        * Creates structured legal appeals
        * Includes proper formatting and legal terminology
        * Follows Serbian legal standards

        ### 3. Legal Document Review
        * Performs detailed document analysis
        * Highlights key clauses and obligations
        * Identifies potential risks and compliance issues

        ### 4. Lawsuit Generation
        * Creates formal lawsuit documents
        * Includes all necessary legal sections
        * Follows proper legal formatting

        ### 5. Lawsuit Response
        * Generates formal responses to lawsuits
        * Addresses key arguments
        * Includes counter-arguments and evidence sections

        ### 6. Contract Analysis
        * Detailed contract review
        * Identifies risks and obligations
        * Provides improvement recommendations

        ### 7. Interactive Chat Assistant
        * Ask specific questions about documents
        * Get clarifications on legal terms
        * Receive guided assistance

        ## 🔍 How to Use Each Feature
        1. **Select Document**: Choose your processed document
        2. **Choose Action**: Use the dropdown menu to select desired analysis
        3. **Review Results**: Results appear in main window
        4. **Download**: Use download buttons for saving results

        ## 💾 Saving Your Work
        * All generated documents can be downloaded as PDFs
        * Use the download buttons below each analysis
        * Files are named with timestamp for easy organization

        ## ⚠️ Important Notes
        * Keep documents under 200MB
        * Supported format: PDF only
        * Processing time varies with document size
        * All data is processed securely
        """)

def main():
    st.set_page_config(
        page_title="Smart Legal Solutions",
        page_icon="⚖️",
        layout="wide"
    )
    
    st.title("⚖️ Smart Legal Solutions")
    st.markdown("#### Automated legal document analysis, lawsuit drafting, and contract review - all in one platform")
    
    # Add user manual expander
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        show_user_manual()
    
    st.divider()
    
    initialize_session_state()

    # Sidebar
    with st.sidebar:
        st.header("📄 Document Upload")
        uploaded_files = st.file_uploader("Upload legal documents (PDF)", type=["pdf"], accept_multiple_files=True)
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state.documents:
                    pdf_path = f"temp_{uploaded_file.name}"
                    try:
                        with open(pdf_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        if st.button(f"Process {uploaded_file.name}"):
                            with st.spinner(f"Processing {uploaded_file.name}..."):
                                try:
                                    text = extract_text_from_pdf(pdf_path)
                                    st.session_state.documents[uploaded_file.name] = {
                                        "text": text,
                                        "processed": True
                                    }
                                    st.session_state.current_doc = uploaded_file.name
                                    st.session_state.document_processed = True
                                    st.success(f"✅ {uploaded_file.name} processed successfully!")
                                    st.session_state.messages.append({
                                        "role": "assistant",
                                        "content": f"I've processed {uploaded_file.name}. You can use the dropdown below to select an action."
                                    })
                                except Exception as e:
                                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                                finally:
                                    # Clean up temp file
                                    if os.path.exists(pdf_path):
                                        os.remove(pdf_path)
                    except Exception as e:
                        st.error(f"Error handling {uploaded_file.name}: {str(e)}")

        # Document selector
        if st.session_state.documents:
            st.divider()
            st.subheader("📋 Select Document")
            selected_doc = st.selectbox(
                "Choose a processed document:",
                options=[doc for doc, info in st.session_state.documents.items() if info["processed"]],
                key="doc_selector"
            )
            if selected_doc:
                st.session_state.current_doc = selected_doc
                st.session_state.document_processed = True

        st.divider()
        st.header("💬 Chat Controls")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat"):
                clear_chat()
                st.rerun()
        with col2:
            if st.button("🔄 New Chat"):
                new_chat()
                st.rerun()

    # Chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Display action dropdown when document is processed
    if st.session_state.document_processed and st.session_state.current_doc:
        st.divider()
        st.subheader(f"🎯 Actions for: {st.session_state.current_doc}")
        
        # Dropdown menu for actions
        action = st.selectbox(
            "Select an action:",
            ["Instant Legal Document Summary", "Appeal Builder Pro", "Smart Document Review", "Lawsuit Builder", "Defense Builder", "Smart Contract Analyzer"]
        )
        
        if st.button("▶️ Execute Action"):
            action_map = {
                "Instant Legal Document Summary": "summary",
                "Appeal Builder Pro": "appeal",
                "Smart Document Review": "review",
                "Lawsuit Builder": "lawsuit",
                "Defense Builder": "lawsuit_response",
                "Smart Contract Analyzer": "contract_analysis"
            }
            process_request(action_map[action])

    # Update chat input handler
    if prompt := st.chat_input("Ask any question about the document..." if st.session_state.document_processed else "Please upload and process a document first"):
        if not st.session_state.current_doc:
            st.error("Please select a processed document first!")
            return

        # Display user message immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Check for specific commands first
        if any(keyword in prompt.lower() for keyword in ["summary", "appeal", "review", "lawsuit", "contract analysis"]):
            action_map = {
                "summary": "summary",
                "appeal": "appeal",
                "review": "review",
                "lawsuit": "lawsuit",
                "contract analysis": "contract_analysis"
            }
            for keyword, action in action_map.items():
                if keyword in prompt.lower():
                    process_request(action, prompt)
                    break
        else:
            # Use chat helper for general questions
            processor = LegalDocumentProcessor()
            with st.chat_message("assistant"):
                with st.spinner("Analyzing document and preparing response..."):
                    result = asyncio.run(processor.process_document(
                        st.session_state.documents[st.session_state.current_doc]["text"],
                        "chat",
                        prompt
                    ))
                    
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.markdown(result["result"])
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result["result"]
                        })

    # Add credits at the bottom of sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📤 Upload Your Legal Documents here for Automation**")


if __name__ == "__main__":
    main()
