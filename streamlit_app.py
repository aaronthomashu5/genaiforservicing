import streamlit as st
import os
import tempfile
import base64
from PIL import Image
import io
import glob
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from streamlit_extras.stylable_container import stylable_container
import time
import fitz  # PyMuPDF
# Add these imports for video processing
import cv2
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# Set page config
st.set_page_config(page_title="ESAB Welding Assistant", layout="wide", page_icon="🔧")

# Add custom CSS for better styling
st.markdown("""
<style>
    .stApp {
        background-color: #181818;
        color: #f5f5f5;
    }
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #FF9800;
        margin-bottom: 20px;
        text-align: center;
    }
    .feature-header {
        font-size: 24px;
        font-weight: bold;
        color: #FFB74D;
        margin-top: 30px;
        margin-bottom: 10px;
    }
    .chat-message {
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .user-message {
        background-color: #263238;
        border-left: 5px solid #29B6F6;
        color: #f5f5f5;
    }
    .assistant-message {
        background-color: #212121;
        border-left: 5px solid #FF9800;
        color: #f5f5f5;
    }
    .stButton>button {
        background-color: #FF9800;
        color: #181818;
        font-weight: bold;
    }
    .info-box {
        background-color: #333333;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #FF9800;
        color: #f5f5f5;
    }
    .stProgress > div > div {
        background-color: #FF9800;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Gemini API (you'll need to replace with your API key)
# In production, use st.secrets or environment variables
def init_genai():
    try:
        # You should store your API key in Streamlit secrets or environment variables
        api_key = st.secrets["GOOGLE_API_KEY"] if "GOOGLE_API_KEY" in st.secrets else os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            st.error("Google API key not found. Please add it to your Streamlit secrets or environment variables.")
            st.stop()
        
        genai.configure(api_key=api_key)
        
        # Configure safety settings for the model
        safety_settings = {
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }
        
        # Generation config
        generation_config = {
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
        }
        
        # Create the model
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        # Create a flash model for streaming
        flash_model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        return model, flash_model
    except Exception as e:
        st.error(f"Error initializing Gemini API: {e}")
        st.stop()

# Load the manuals
def load_manuals():
    """Find and load all available welding machine manuals"""
    manual_files = glob.glob("manuals/*.pdf")
    manuals = {}
    
    for manual_path in manual_files:
        filename = os.path.basename(manual_path)
        # Extract machine name from filename - can be customized based on actual naming convention
        machine_name = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ")
        manuals[machine_name] = manual_path
    
    return manuals

# Extract text from a PDF
def extract_pdf_text(pdf_path):
    """Extract text from a PDF file"""
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

# Extract images from a PDF
def extract_pdf_images(pdf_path):
    """Extract images from a PDF file"""
    images = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Convert image bytes to PIL Image
                image = Image.open(io.BytesIO(image_bytes))
                images.append({
                    "page": page_num + 1,
                    "image": image,
                    "bytes": image_bytes
                })
                
    except Exception as e:
        st.error(f"Error extracting images from PDF: {e}")
    
    return images

# Create a system prompt for the selected machine
def create_system_prompt(machine_name, manual_path):
    """Create a system prompt with the manual content"""
    manual_text = extract_pdf_text(manual_path)
    
    system_prompt = f"""
    You are an expert ESAB welding machine assistant specializing in the {machine_name} model.
    You have been trained on the manual for this machine and can provide accurate information
    about error codes, component identification, troubleshooting, and repair procedures.
    
    Here is the manual content to reference:
    {manual_text[:100000]}  # We limit to first 100K chars as prompt context
    
    When answering questions:
    1. Always base your answers on the manual content.
    2. If asked about error codes, provide the code meaning and recommended actions.
    3. If asked about components, provide the part name, part number, and location if available.
    4. For troubleshooting, give step-by-step guidance with safety precautions.
    5. Always prioritize safety in your recommendations.
    6. If asked about something not in the manual, admit that you don't have that specific information.
    
    Format your responses clearly with headings, bullet points, and numbered steps where appropriate.
    """
    
    return system_prompt

# Function to analyze uploaded component image
def analyze_component_image(image, machine_name, manual_path, model):
    """Analyze an uploaded image of a component and identify it"""
    # Extract images from the manual for reference
    manual_images = extract_pdf_images(manual_path)
    manual_text = extract_pdf_text(manual_path)
    
    # Convert the uploaded image to bytes for the model
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # Create a prompt for the model
    prompt = f"""
    I have uploaded an image of a component from an ESAB {machine_name} welding machine.
    Please identify this component by:
    1. The name of the component
    2. The part number if visible or if you can determine it
    3. Its function in the welding machine
    4. Which board or assembly it belongs to
    
    Use the manual content to assist in identification:
    {manual_text[:50000]}
    
    If you cannot identify the exact component, suggest the closest matches and explain why.
    """
    
    # Send the image and prompt to the model
    try:
        response = model.generate_content([
            prompt,
            {"mime_type": "image/jpeg", "data": img_str}
        ])
        
        return response.text
    except Exception as e:
        return f"Error analyzing image: {e}"

# Function to process fault list and recommend spare parts
def process_fault_list(fault_pdf, machine_name, manual_path, model):
    """Process a fault list PDF and recommend spare parts"""
    # Extract text from the uploaded fault PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        tmp.write(fault_pdf.getvalue())
        tmp_path = tmp.name
    
    fault_text = extract_pdf_text(tmp_path)
    manual_text = extract_pdf_text(manual_path)
    
    # Clean up the temp file
    os.unlink(tmp_path)
    
    # Create a prompt for the model
    prompt = f"""
    I have a fault list for an ESAB {machine_name} welding machine.
    The fault list contains the following information:
    
    {fault_text}
    
    Based on the manual for this machine:
    {manual_text[:50000]}
    
    Please:
    1. Extract and list all fault codes from the uploaded document
    2. For each fault code, identify the most likely components that could cause this fault
    3. Rank the top 3 spare parts recommended for inspection or replacement
    4. Provide the part numbers for these components if available
    5. Suggest a troubleshooting approach for each fault
    
    Format your response as a structured report with sections for each fault code.
    """
    
    # Send the prompt to the model
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error processing fault list: {e}"

# Function to generate a repair playbook
def generate_repair_playbook(issue, machine_name, manual_path, model):
    """Generate a step-by-step repair playbook for the selected issue"""
    manual_text = extract_pdf_text(manual_path)
    
    # Create a prompt for the model
    prompt = f"""
    Create a detailed repair playbook for fixing the following issue on an ESAB {machine_name} welding machine:
    
    Issue: {issue}
    
    Using the manual content:
    {manual_text[:50000]}
    
    Generate a comprehensive, step-by-step repair guide that includes:
    
    1. Required tools and safety equipment
    2. Safety precautions and warnings specific to this repair
    3. Preparation steps (power off, disconnect, etc.)
    4. Detailed numbered steps for the repair process with clear instructions
    5. Testing procedures to verify the repair was successful
    6. Reassembly instructions if needed
    7. Additional tips or common mistakes to avoid
    
    Format the playbook with clear headings, numbered steps, and emphasize all safety warnings.
    Reference specific parts by their correct names and part numbers whenever possible.
    Include any relevant troubleshooting steps that should be performed before component replacement.
    """
    
    # Send the prompt to the model
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating repair playbook: {e}"

# Function to handle streaming responses
def stream_response(prompt, flash_model):
    """Stream a response from the model"""
    try:
        response = flash_model.generate_content(prompt, stream=True)
        
        # Create a placeholder for the streamed response
        message_placeholder = st.empty()
        full_response = ""
        
        # Stream the response
        for chunk in response:
            if chunk.text:
                full_response += chunk.text
                message_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
            
        # Show the final response without the cursor
        message_placeholder.markdown(full_response, unsafe_allow_html=True)
        return full_response
    
    except Exception as e:
        st.error(f"Error streaming response: {e}")
        return None

# Video processor class for the webcam stream
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.last_screenshot = None
        self.take_screenshot = False
    
    def toggle_screenshot(self):
        self.take_screenshot = True
    
    def get_screenshot(self):
        return self.last_screenshot
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Take screenshot when requested
        if self.take_screenshot:
            self.last_screenshot = img.copy()
            self.take_screenshot = False
        
        # Add frame counter
        self.frame_count += 1
        
        # Optional: Add text overlay to show it's live
        cv2.putText(
            img,
            f"ESAB Live Feed - Frame: {self.frame_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 165, 255),  # Orange color
            2,
        )
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Main app
def main():
    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'selected_machine' not in st.session_state:
        st.session_state.selected_machine = None
        
    if 'model' not in st.session_state or 'flash_model' not in st.session_state:
        model, flash_model = init_genai()
        st.session_state.model = model
        st.session_state.flash_model = flash_model
    
    # Add a callback function to clear the chat input
    def clear_chat_input():
        st.session_state.chat_input = ""
    
    # Load available manuals
    manuals = load_manuals()
    
    # Header
    st.markdown('<div class="main-header">ESAB Welding Machine Assistant</div>', unsafe_allow_html=True)
    
    # Display ESAB logo or a welding image here if desired
    # st.image("esab_logo.png", width=200)
    
    # 1. Machine Selector
    st.markdown('<div class="feature-header">1. Select Your Welding Machine</div>', unsafe_allow_html=True)
    
    if not manuals:
        st.warning("No machine manuals found. Please add PDF manuals to the 'manuals' folder.")
    else:
        machine_options = list(manuals.keys())
        selected_machine = st.selectbox(
            "Choose your ESAB welding machine model",
            options=machine_options,
            index=0 if machine_options else None
        )
        
        if selected_machine:
            st.session_state.selected_machine = selected_machine
            manual_path = manuals[selected_machine]
            st.session_state.manual_path = manual_path
            
            st.markdown(f'<div class="info-box">Selected machine: <b>{selected_machine}</b><br>Manual: {os.path.basename(manual_path)}</div>', unsafe_allow_html=True)
            
            # Create tabs for different features
            tabs = st.tabs([
                "AI Chat Assistant", 
                "Component Image Recognition", 
                "Fault-to-Spare Recommendation", 
                "Repair Playbook Generator",
                "Live Video Stream"  # New tab
            ])
            
            # 2. AI Chat Assistant Tab
            with tabs[0]:
                st.markdown('<div class="feature-header">AI Chat Assistant</div>', unsafe_allow_html=True)
                
                # Quick access buttons for common queries
                st.markdown("#### Quick Access")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if st.button("Error Codes"):
                        st.session_state.chat_history.append({"role": "user", "content": "What are the common error codes for this machine and what do they mean?"})
                
                with col2:
                    if st.button("Parts List"):
                        st.session_state.chat_history.append({"role": "user", "content": "Show me the main parts list for this machine with part numbers"})
                
                with col3:
                    if st.button("Safety Precautions"):
                        st.session_state.chat_history.append({"role": "user", "content": "What are the important safety precautions for this machine?"})
                
                with col4:
                    if st.button("Maintenance Schedule"):
                        st.session_state.chat_history.append({"role": "user", "content": "What is the recommended maintenance schedule for this machine?"})
                
                # Chat interface
                st.markdown("#### Chat with the AI Assistant")
                
                # Display chat history
                for message in st.session_state.chat_history:
                    if message["role"] == "user":
                        st.markdown(f'<div class="chat-message user-message"><b>You:</b><br>{message["content"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="chat-message assistant-message"><b>Assistant:</b><br>{message["content"]}</div>', unsafe_allow_html=True)
                
                # Initialize chat_input value in session state if it doesn't exist
                if "chat_input" not in st.session_state:
                    st.session_state.chat_input = ""
                
                # Chat input - use a callback to handle submission
                def submit_chat():
                    if st.session_state.chat_input:
                        # Store current input in a temporary variable
                        user_message = st.session_state.chat_input
                        
                        # Add user message to chat history
                        st.session_state.chat_history.append({"role": "user", "content": user_message})
                        
                        # Clear the input by setting its value to empty in session state
                        st.session_state.chat_input = ""
                        
                        # Create system prompt based on the selected machine
                        system_prompt = create_system_prompt(selected_machine, manual_path)
                        
                        # Process and generate response
                        prompt_parts = [
                            system_prompt,
                            "\n\nI need information about the ESAB " + selected_machine + ":\n" + user_message
                        ]
                        
                        if st.session_state.get("stream_chat", True):
                            response_text = stream_response(prompt_parts, st.session_state.flash_model)
                        else:
                            try:
                                response = st.session_state.model.generate_content(prompt_parts)
                                response_text = response.text
                            except Exception as e:
                                response_text = f"Error generating response: {e}"
                                st.error(response_text)
                        
                        # Add assistant response to chat history
                        st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                
                # Text input with on_change callback
                user_input = st.text_input(
                    "Ask about error codes, parts, troubleshooting, etc.",
                    key="chat_input",
                    on_change=submit_chat
                )
                
                # Stream option
                st.checkbox("Stream response", value=True, key="stream_chat")
                
                # Add a separate send button for users who prefer clicking
                if st.button("Send"):
                    # The submit_chat function will be triggered if there's text in the input
                    if st.session_state.chat_input:
                        submit_chat()
            
            # 3. Component Image Recognition Tab
            with tabs[1]:
                st.markdown('<div class="feature-header">Component Image Recognition</div>', unsafe_allow_html=True)
                st.markdown("""
                Upload a photo of a component or circuit board, and I'll identify it 
                and provide relevant information from the manual.
                """)
                
                uploaded_image = st.file_uploader("Upload component image", type=["jpg", "jpeg", "png"])
                
                if uploaded_image is not None:
                    # Display the uploaded image
                    image = Image.open(uploaded_image)
                    st.image(image, caption="Uploaded Component", width=400)
                    
                    if st.button("Identify Component"):
                        with st.spinner("Analyzing image..."):
                            # Process the image with the model
                            result = analyze_component_image(image, selected_machine, manual_path, st.session_state.model)
                            
                            # Display the result
                            st.markdown("### Component Analysis Result")
                            st.markdown(result)
            
            # 4. Fault-to-Spare Recommendation Tab
            with tabs[2]:
                st.markdown('<div class="feature-header">Fault-to-Spare Recommendation</div>', unsafe_allow_html=True)
                st.markdown("""
                Upload a fault list PDF, and I'll analyze it to recommend the most likely 
                spare parts needed for repair.
                """)
                
                uploaded_fault_list = st.file_uploader("Upload fault list PDF", type=["pdf"])
                
                if uploaded_fault_list is not None:
                    st.success(f"Uploaded: {uploaded_fault_list.name}")
                    
                    if st.button("Analyze Faults & Recommend Parts"):
                        with st.spinner("Processing fault list..."):
                            # Process the fault list
                            recommendations = process_fault_list(
                                uploaded_fault_list, 
                                selected_machine, 
                                manual_path, 
                                st.session_state.model
                            )
                            
                            # Display the recommendations
                            st.markdown("### Spare Parts Recommendations")
                            st.markdown(recommendations)
            
            # 5. Repair Playbook Generator Tab
            with tabs[3]:
                st.markdown('<div class="feature-header">Repair Playbook Generator</div>', unsafe_allow_html=True)
                st.markdown("""
                Describe the issue or select a component, and I'll generate a 
                step-by-step repair guide with safety precautions.
                """)
                
                issue_description = st.text_area(
                    "Describe the issue or component to repair",
                    placeholder="E.g., 'E21 error code on display', 'Replace the main PCB', 'Wire feed motor not working'"
                )
                
                if st.button("Generate Repair Playbook") and issue_description:
                    with st.spinner("Generating comprehensive repair guide..."):
                        # Stream the playbook generation
                        st.markdown("### Repair Playbook")
                        
                        # Generate the repair playbook with streaming
                        system_prompt = create_system_prompt(selected_machine, manual_path)
                        prompt = f"{system_prompt}\n\nPlease create a repair playbook for: {issue_description}"
                        
                        stream_response(prompt, st.session_state.flash_model)
            
            # 6. Live Video Streaming Tab
            with tabs[4]:
                st.markdown('<div class="feature-header">Live Video Stream</div>', unsafe_allow_html=True)
                
                st.markdown("""
                Use your camera to stream live video for remote diagnostics or share your screen 
                to show the machine's display. You can also take screenshots for AI analysis.
                """)
                
                # Create two columns for the video options
                stream_col1, stream_col2 = st.columns(2)
                
                # Option 1: Live Webcam Stream
                with stream_col1:
                    st.markdown("### Live Camera Feed")
                    st.markdown('<div class="stream-options">', unsafe_allow_html=True)
                    
                    # Configure WebRTC
                    rtc_configuration = RTCConfiguration(
                        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
                    )
                    
                    # Create a unique key for the webrtc component
                    webrtc_ctx = webrtc_streamer(
                        key="esab-welding-live",
                        video_processor_factory=VideoProcessor,
                        rtc_configuration=rtc_configuration,
                        media_stream_constraints={"video": True, "audio": True},
                    )
                    
                    # Add a screenshot button
                    if webrtc_ctx.video_processor:
                        if st.button("Take Screenshot for Analysis", key="take_screenshot", 
                                    help="Capture the current frame for AI analysis"):
                            webrtc_ctx.video_processor.toggle_screenshot()
                            st.session_state.screenshot_taken = True
                    
                    # Display and analyze the screenshot if taken
                    if webrtc_ctx.video_processor and hasattr(st.session_state, 'screenshot_taken') and st.session_state.screenshot_taken:
                        screenshot = webrtc_ctx.video_processor.get_screenshot()
                        if screenshot is not None:
                            # Convert from OpenCV format to PIL Image
                            screenshot_rgb = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
                            pil_image = Image.fromarray(screenshot_rgb)
                            
                            st.image(pil_image, caption="Captured Screenshot", width=400)
                            
                            # Add a button to analyze the captured image
                            if st.button("Analyze Captured Component"):
                                with st.spinner("Analyzing screenshot..."):
                                    # Process the image with the model
                                    result = analyze_component_image(pil_image, selected_machine, manual_path, st.session_state.model)
                                    
                                    # Display the result
                                    st.markdown("### Component Analysis Result")
                                    st.markdown(result)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Option 2: Screen Sharing
                with stream_col2:
                    st.markdown("### Screen Sharing")
                    st.markdown('<div class="stream-options">', unsafe_allow_html=True)
                    
                    st.markdown("""
                    For desktop screen sharing:
                    
                    1. Click the "Start Screen Share" button below
                    2. Select which window or screen to share
                    3. The technician will see your screen in real-time
                    """)
                    
                    # NOTE: Due to Streamlit's limitations, we're implementing this with a placeholder button
                    # that would normally launch a screen sharing session using a JavaScript library
                    
                    if st.button("Start Screen Share", key="start_screen_share"):
                        st.info("""
                        Screen sharing initiated! 
                        
                        NOTE: In a production application, this would launch a WebRTC or similar screen sharing session.
                        Due to Streamlit's limitations, full screen sharing requires additional JavaScript components
                        which would be implemented when deploying this app.
                        """)
                        
                        # Here you would normally include JavaScript to start screen sharing
                        # This is just a placeholder for the actual implementation
                        
                    st.markdown("### Upload Screen Recording")
                    
                    # Allow users to upload a screen recording as an alternative
                    screen_recording = st.file_uploader("Upload a video recording of your screen", 
                                                       type=["mp4", "mov", "avi", "webm"])
                    
                    if screen_recording is not None:
                        # Save the uploaded video to a temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{screen_recording.name.split(".")[-1]}') as tmp:
                            tmp.write(screen_recording.getvalue())
                            temp_path = tmp.name
                        
                        # Display the video
                        st.video(temp_path)
                        
                        # Option to analyze a frame from the video
                        if st.button("Extract Frame for Analysis"):
                            # Open the video file
                            cap = cv2.VideoCapture(temp_path)
                            
                            # Move to the middle frame of the video
                            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
                            
                            # Read the frame
                            ret, frame = cap.read()
                            cap.release()
                            
                            if ret:
                                # Convert from OpenCV format to PIL Image
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                pil_frame = Image.fromarray(frame_rgb)
                                
                                st.image(pil_frame, caption="Extracted Frame", width=400)
                                
                                # Add a button to analyze the frame
                                if st.button("Analyze Extracted Frame"):
                                    with st.spinner("Analyzing frame..."):
                                        # Process the image with the model
                                        result = analyze_component_image(pil_frame, selected_machine, manual_path, st.session_state.model)
                                        
                                        # Display the result
                                        st.markdown("### Component Analysis Result")
                                        st.markdown(result)
                            else:
                                st.error("Failed to extract frame from the video.")
                        
                        # Clean up the temporary file
                        os.unlink(temp_path)
                    
                    st.markdown('</div>', unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()